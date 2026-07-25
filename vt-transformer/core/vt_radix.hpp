#ifndef _VT_RADIX_HPP_
#define _VT_RADIX_HPP_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vt.hpp"

namespace vt {

// Page Size 对齐的 Radix 前缀匹配树（KV cache slot 索引）。
//
// 职责切分（本文件的设计不变式）：
//   - Query   (match_prefix)  : 完全只读，不允许修改树。
//   - Insert  (insert_prefix) : 独占所有结构修改（节点拆分 + 挂接新孩子）。
//   - Lock/Unlock             : 只调整 ref_count，不维护统计信息。
//   - Evict                   : 只淘汰叶子，不维护其它状态。
//   - RadixStats              : protected_size / evictable_size 的唯一来源。
//
// 非线程安全，调用方需自行串行化（与 Python 参考实现一致）。
//
// 模板参数：
//   Token  token id 类型（通常 int32_t）。
//   Index  slot 索引类型（通常 int32_t，对应 KV cache pool 的下标）。

template <typename T>
struct VectorHash {
    size_t operator()(const std::vector<T>& v) const noexcept {
        size_t h = 1469598103934665603ULL;   // FNV-1a 64-bit offset basis
        for (const auto& x : v) {
            h ^= std::hash<T>{}(x);
            h *= 1099511628211ULL;            // FNV-1a 64-bit prime
        }
        return h;
    }
};

template <typename Token = int32_t, typename Index = int32_t>
struct RadixNode {
    using TokenVec = std::vector<Token>;
    using IndexVec = std::vector<Index>;
    using ChildKey = std::vector<Token>;
    using ChildMap = std::unordered_map<ChildKey, std::unique_ptr<RadixNode>, VectorHash<Token>>;

    TokenVec   key_;
    IndexVec   value_;
    RadixNode* parent_     = nullptr;
    ChildMap   children_;
    int        ref_count_  = 0;
    uint64_t   timestamp_  = 0;   // 越小越旧；evict 用最小堆出队
    uint64_t   uuid_       = 0;

    size_t length()  const { return key_.size(); }
    bool   is_root() const { return parent_ == nullptr; }
    bool   is_leaf() const { return children_.empty(); }
};

// 统计模块：protected_size / evictable_size 的唯一来源。
// 每条修改路径（insert / evict / lock / unlock）把自己的 delta 报到这里。
class RadixStats {
public:
    void on_insert (size_t len) { evictable_ += len; }
    void on_evict  (size_t len) { evictable_ -= len; }
    void on_lock   (size_t len) { evictable_ -= len; protected_ += len; }
    void on_unlock (size_t len) { protected_ -= len; evictable_ += len; }

    size_t evictable_size() const { return evictable_; }
    size_t protected_size() const { return protected_; }
    size_t total_size()     const { return evictable_ + protected_; }

private:
    size_t evictable_ = 0;
    size_t protected_ = 0;
};

template <typename Token = int32_t, typename Index = int32_t>
class RadixTree {
public:
    using Node     = RadixNode<Token, Index>;
    using TokenVec = std::vector<Token>;
    using IndexVec = std::vector<Index>;

    struct MatchResult {
        Node*  node;          // 匹配到达的 deepest 节点
        size_t prefix_len;    // 已匹配的 token 数（page 对齐）
    };
    struct InsertResult {
        Node*  node;          // 插入后的 deepest 节点（可能新建）
        size_t prefix_len;    // 插入前树中已存在的 token 数
        size_t inserted_len;  // 本次实际生效（page 对齐后）的写入长度
    };

    explicit RadixTree(size_t page_size = 1) : page_size_(page_size) {
        if (page_size_ == 0) vt_panic("RadixTree: page_size must be > 0");
        root_ = std::make_unique<Node>();
        root_->ref_count_ = 1;          // root 永远 protected，不会被淘汰
        root_->uuid_      = next_uuid_++;
    }
    ~RadixTree() = default;

    RadixTree(const RadixTree&)            = delete;
    RadixTree& operator=(const RadixTree&) = delete;

    // ---- (1) 查询：返回 tokens 在树中的最长前缀。完全只读。----
    MatchResult match_prefix(const TokenVec& tokens) const {
        WalkResult wr = walk(tokens);
        return {wr.node, wr.total_matched};
    }

    // ---- (2) 写入：tokens -> indices（向下 page 对齐）。独占所有结构修改。----
    InsertResult insert_prefix(const TokenVec& tokens, const IndexVec& indices) {
        if (tokens.size() != indices.size())
            vt_panic("RadixTree::insert_prefix: tokens/indices length mismatch");

        const size_t insert_len = align_down(tokens.size(), page_size_);
        if (insert_len == 0) return {root_.get(), 0, 0};

        // 1. 只读 descent，定位已有前缀在哪里结束。
        WalkResult wr = walk(tokens);

        // 2. LRU 刷新：把路径上节点的时间戳置为最新。
        const uint64_t ts = ++tick_;
        for (Node* n : wr.path) n->timestamp_ = ts;

        // 3. 若停在了某节点内部（partial < length），在 page 边界处拆开。
        Node*  parent     = wr.node;
        size_t prefix_len = wr.total_matched;
        if (wr.partial < parent->length()) {
            parent = split_at(parent, wr.partial);
        }

        // 4. 把未命中的后缀挂成一个新孩子。
        if (prefix_len < insert_len) {
            Node* child = attach_child(parent, tokens, indices, prefix_len, insert_len, ts);
            return {child, prefix_len, insert_len};
        }
        return {parent, prefix_len, insert_len};
    }

    // ---- (3) 加锁 / 解锁：只调整 ref_count。统计由 RadixStats 维护。----
    void lock(Node* node) {
        for (; !node->is_root(); node = node->parent_) {
            if (node->ref_count_ == 0) stats_.on_lock(node->length());
            ++node->ref_count_;
        }
    }
    void unlock(Node* node) {
        for (; !node->is_root(); node = node->parent_) {
            vt_assert(node->ref_count_ > 0, "RadixTree::unlock: ref_count underflow");
            --node->ref_count_;
            if (node->ref_count_ == 0) stats_.on_unlock(node->length());
        }
    }

    // ---- (4) 淘汰：按 timestamp 从旧到新回收至少 `size` 个 token 的叶子。----
    IndexVec evict(size_t size) {
        IndexVec freed;
        if (size == 0) return freed;
        if (size > stats_.evictable_size())
            vt_panic("RadixTree::evict: not enough evictable tokens");

        auto cmp = [](const Node* a, const Node* b) { return a->timestamp_ > b->timestamp_; };
        std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> heap(cmp);
        for (Node* leaf : collect_evictable_leaves()) heap.push(leaf);

        size_t freed_size = 0;
        while (freed_size < size) {
            if (heap.empty()) vt_panic("RadixTree::evict: ran out of leaves");
            Node* n = heap.top(); heap.pop();
            vt_assert(n->is_leaf() && !n->is_root() && n->ref_count_ == 0,
                      "RadixTree::evict: integrity violated");

            freed_size += n->length();
            for (Index idx : n->value_) freed.push_back(idx);
            stats_.on_evict(n->length());

            // 摘掉自己；父节点可能因此变成新的可淘汰叶子。
            Node* parent = n->parent_;
            parent->children_.erase(make_child_key(n->key_));
            if (parent->is_leaf() && !parent->is_root() && parent->ref_count_ == 0)
                heap.push(parent);
        }
        return freed;
    }

    // ---- 便于上层调度的状态查询 ----
    size_t page_size()      const { return page_size_; }
    size_t evictable_size() const { return stats_.evictable_size(); }
    size_t protected_size() const { return stats_.protected_size(); }
    size_t total_size()     const { return stats_.total_size(); }
    Node*  root()           const { return root_.get(); }

private:
    // 只读 descent 的结果。`partial` 是在 `node->key_` 中已消费的 token 数
    // （下钻后 ≥ page_size_；若等于 node->length() 表示整段命中）。`path`
    // 是 root..node 的访问序列，供 Insert 做 LRU 刷新使用。
    struct WalkResult {
        Node*              node;
        size_t             total_matched;
        size_t             partial;
        std::vector<Node*> path;
    };

    size_t                page_size_ = 1;
    RadixStats            stats_;
    std::unique_ptr<Node> root_;
    uint64_t              tick_      = 0;   // 单调时间戳，每次 insert 递增
    uint64_t              next_uuid_ = 0;

    static size_t align_down(size_t x, size_t p) { return (x / p) * p; }

    // 子节点在父节点 children_ 中的 key = 子节点 key 的前 page_size 个 token。
    std::vector<Token> make_child_key(const TokenVec& key) const {
        vt_assert(key.size() >= page_size_,
                  "RadixTree: node key shorter than page_size");
        return std::vector<Token>(key.begin(), key.begin() + page_size_);
    }

    // 纯只读 descent。停止条件：剩余 token 不足一页 | 孩子未命中 |
    // 在孩子内部 page 边界处分叉。永远不修改树。
    WalkResult walk(const TokenVec& tokens) const {
        WalkResult r{root_.get(), 0, 0, {root_.get()}};
        const size_t total = tokens.size();
        size_t       i      = 0;

        while (i + page_size_ <= total) {
            std::vector<Token> key(tokens.begin() + i, tokens.begin() + i + page_size_);
            auto it = r.node->children_.find(key);
            if (it == r.node->children_.end()) break;

            Node*  child = it->second.get();
            size_t ncmp  = std::min(child->length(), total - i);
            size_t m     = 0;
            for (; m < ncmp; ++m) {
                if (child->key_[m] != tokens[i + m]) break;
            }
            m = align_down(m, page_size_);

            r.node          = child;
            r.partial       = m;
            r.total_matched = i + m;
            r.path.push_back(child);
            i += m;
            if (m < child->length()) break;   // 在孩子内部停住
        }
        return r;
    }

    // 把 node 在 page 对齐位置 pos 处拆成 head [0,pos) + tail [pos,L)。
    // head 接管 node 在父节点中的位置，继承 ref_count / timestamp；tail
    // 成为 head 唯一的孩子。两侧总长度不变（pos + (L-pos) = L），stats 无需更新。
    Node* split_at(Node* node, size_t pos) {
        vt_assert(pos > 0 && pos < node->length(),
                  "RadixTree::split_at: pos out of range");
        vt_assert(pos % page_size_ == 0,
                  "RadixTree::split_at: pos not page-aligned");

        Node* parent    = node->parent_;
        auto  slot_key  = make_child_key(node->key_);

        // 先把 node 从父节点摘下，独占所有权，避免后续 erase 把它一起销毁。
        std::unique_ptr<Node> node_owner = std::move(parent->children_[slot_key]);
        parent->children_.erase(slot_key);

        // 建 head，承接前缀和原节点的锁/时间戳。
        auto head = std::make_unique<Node>();
        head->key_.assign  (node_owner->key_.begin(),   node_owner->key_.begin()   + pos);
        head->value_.assign(node_owner->value_.begin(), node_owner->value_.begin() + pos);
        head->ref_count_ = node_owner->ref_count_;
        head->timestamp_ = node_owner->timestamp_;
        head->uuid_      = next_uuid_++;
        head->parent_    = parent;
        Node* head_raw   = head.get();

        // 原 node 截断为 tail [pos, L)，挂在 head 下。
        TokenVec tail_key  (node_owner->key_.begin()   + pos, node_owner->key_.end());
        IndexVec tail_value(node_owner->value_.begin() + pos, node_owner->value_.end());
        node_owner->key_    = std::move(tail_key);
        node_owner->value_  = std::move(tail_value);
        node_owner->parent_ = head_raw;

        // 重新接线：head <- node，parent <- head（同一 slot）。
        head->children_[make_child_key(node_owner->key_)] = std::move(node_owner);
        parent->children_[std::move(slot_key)]            = std::move(head);
        return head_raw;
    }

    // 在 parent 下挂一个新叶子，承载 tokens[prefix_len, insert_len)。
    Node* attach_child(Node* parent, const TokenVec& tokens, const IndexVec& indices,
                       size_t prefix_len, size_t insert_len, uint64_t ts) {
        auto child = std::make_unique<Node>();
        child->key_.assign  (tokens.begin()  + prefix_len, tokens.begin()  + insert_len);
        child->value_.assign(indices.begin() + prefix_len, indices.begin() + insert_len);
        child->parent_    = parent;
        child->timestamp_ = ts;
        child->uuid_      = next_uuid_++;
        Node* raw = child.get();
        parent->children_[make_child_key(child->key_)] = std::move(child);
        stats_.on_insert(raw->length());
        return raw;
    }

    // 当前可淘汰叶子快照（ref_count == 0 且不是 root），供 evict 建堆。
    std::vector<Node*> collect_evictable_leaves() const {
        std::vector<Node*> out, stack{root_.get()};
        while (!stack.empty()) {
            Node* n = stack.back(); stack.pop_back();
            if (n->is_leaf()) {
                if (!n->is_root() && n->ref_count_ == 0) out.push_back(n);
            } else {
                for (auto& kv : n->children_) stack.push_back(kv.second.get());
            }
        }
        return out;
    }
};

} // namespace vt

#endif
