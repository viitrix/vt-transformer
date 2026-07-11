// radix_cache.hpp — Header-only C++ Radix Tree prefix KV cache.
//
// Mirrors the semantics of python/minisgl/kvcache/radix_cache.py:
//   * match_prefix / insert_prefix / evict / lock_handle
//   * Page-aligned keying and insertion (align_down to page_size)
//   * Lazy node splitting when a partial prefix matches
//   * LRU eviction via per-node timestamp (oldest leaves first)
//   * check_integrity invariant verification
//
// The "value" array of each node (page indices) is stored via a user-supplied
// Allocator policy, so the same code drives either host memory or CUDA-managed
// memory. The tree topology itself (parent/child pointers, key arrays, ref
// counts) always lives on the host — only the value arrays migrate.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef MINISGL_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace minisgl {

// ============================================================================
// Allocator policies
// ============================================================================
//
// A valid Allocator<T> exposes three static functions:
//   static T*   allocate(size_t n);              // throws on failure
//   static void deallocate(T* p) noexcept;       // accepts nullptr
//   static void copy_from_host(T* dst, const T* src, size_t n);
//   static void copy_to_host(T* dst, const T* src, size_t n);

template <typename T>
struct HostAllocator {
    static T* allocate(size_t n) {
        if (n == 0) return nullptr;
        T* p = new (std::nothrow) T[n];
        if (!p) throw std::runtime_error("HostAllocator: out of memory");
        return p;
    }
    static void deallocate(T* p) noexcept { delete[] p; }
    static void copy_from_host(T* dst, const T* src, size_t n) {
        if (n) std::memcpy(dst, src, n * sizeof(T));
    }
    static void copy_to_host(T* dst, const T* src, size_t n) {
        if (n) std::memcpy(dst, src, n * sizeof(T));
    }
};

#ifdef MINISGL_USE_CUDA
struct CudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename T>
struct CudaAllocator {
    static T* allocate(size_t n) {
        void* raw = nullptr;
        cudaError_t err = cudaMallocManaged(&raw, n * sizeof(T));
        if (err != cudaSuccess)
            throw CudaError(std::string("CudaAllocator::allocate: ") + cudaGetErrorString(err));
        return static_cast<T*>(raw);
    }
    static void deallocate(T* p) noexcept { cudaFree(p); }
    static void copy_from_host(T* dst, const T* src, size_t n) {
        if (n == 0) return;
        cudaError_t err = cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw CudaError(std::string("CudaAllocator::copy_from_host: ") + cudaGetErrorString(err));
    }
    static void copy_to_host(T* dst, const T* src, size_t n) {
        if (n == 0) return;
        cudaError_t err = cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw CudaError(std::string("CudaAllocator::copy_to_host: ") + cudaGetErrorString(err));
    }
};
#endif

// ============================================================================
// RAII Buffer wrapping an Allocator-managed span
// ============================================================================

template <typename T, template <typename> class Allocator>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t n) : data_(n ? Allocator<T>::allocate(n) : nullptr), size_(n) {}

    Buffer(const std::vector<T>& host_src) : data_(nullptr), size_(host_src.size()) {
        if (size_) {
            data_ = Allocator<T>::allocate(size_);
            try {
                Allocator<T>::copy_from_host(data_, host_src.data(), size_);
            } catch (...) {
                Allocator<T>::deallocate(data_);
                throw;
            }
        }
    }

    ~Buffer() {
        if (data_) Allocator<T>::deallocate(data_);
    }

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) {
            if (data_) Allocator<T>::deallocate(data_);
            data_     = o.data_;
            size_     = o.size_;
            o.data_   = nullptr;
            o.size_   = 0;
        }
        return *this;
    }

    T*       data()       { return data_; }
    const T* data() const { return data_; }
    size_t   size() const { return size_; }

    std::vector<T> to_host() const {
        std::vector<T> out(size_);
        if (size_) Allocator<T>::copy_to_host(out.data(), data_, size_);
        return out;
    }

private:
    T*     data_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// FNV-1a hasher for std::vector<T> (used as child map key)
// ============================================================================

template <typename T>
struct VectorHasher {
    size_t operator()(const std::vector<T>& v) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        for (const auto& x : v) {
            h ^= std::hash<T>{}(x);
            h *= 0x100000001b3ULL;
        }
        return h;
    }
};

// ============================================================================
// RadixNode
// ============================================================================

template <typename TokenT, typename IndexT, template <typename> class Allocator>
struct RadixNode {
    using Key          = std::vector<TokenT>;
    using IndexBuffer  = Buffer<IndexT, Allocator>;
    using Children     = std::unordered_map<Key, RadixNode*, VectorHasher<TokenT>>;

    Key          key;        // token segment owned by this node (host)
    IndexBuffer  value;      // page indices for this segment (Allocator-backed)
    Children     children;
    RadixNode*   parent    = nullptr;
    uint64_t     ref_count = 0;
    uint64_t     timestamp = 0;
    uint64_t     uuid      = 0;

    bool   is_root() const { return parent == nullptr; }
    bool   is_leaf() const { return children.empty(); }
    size_t length()  const { return key.size(); }
};

// ============================================================================
// RadixCache
// ============================================================================

template <typename TokenT,
          typename IndexT   = int32_t,
          template <typename> class Allocator = HostAllocator>
class RadixCache {
public:
    using Node        = RadixNode<TokenT, IndexT, Allocator>;
    using Key         = std::vector<TokenT>;
    using IndexBuffer = Buffer<IndexT, Allocator>;

    struct CacheHandle {
        size_t     cached_len;
        Node*      node;
    };

    struct SizeInfo {
        size_t evictable_size;
        size_t protected_size;
        size_t total_size() const { return evictable_size + protected_size; }
    };

    struct MatchResult {
        CacheHandle handle;
    };

    struct InsertResult {
        size_t      cached_len;   // length already cached before this insert
        CacheHandle handle;
    };

    explicit RadixCache(size_t page_size)
        : page_size_(page_size == 0 ? 1 : page_size) {
        root_ = create_node();
        root_->ref_count = 1;          // root is always protected
        root_->timestamp = ++tic_;
    }

    RadixCache(const RadixCache&)            = delete;
    RadixCache& operator=(const RadixCache&) = delete;
    RadixCache(RadixCache&&) noexcept            = default;
    RadixCache& operator=(RadixCache&&) noexcept = default;

    // ---------- match_prefix ----------
    MatchResult match_prefix(const std::vector<TokenT>& input_ids) {
        auto [node, prefix_len] = tree_walk(input_ids);
        return MatchResult{CacheHandle{prefix_len, node}};
    }

    // ---------- insert_prefix ----------
    InsertResult insert_prefix(const std::vector<TokenT>& input_ids,
                               const std::vector<IndexT>& indices) {
        if (input_ids.size() != indices.size())
            throw std::runtime_error("insert_prefix: input_ids/indices size mismatch");

        const size_t insert_len = align_down(input_ids.size(), page_size_);
        std::vector<TokenT> ids(input_ids.begin(), input_ids.begin() + insert_len);
        std::vector<IndexT> idx(indices.begin(), indices.begin() + insert_len);

        auto [node, prefix_len] = tree_walk(ids);

        if (prefix_len != insert_len) {
            Node* child = create_node();
            child->key.assign(ids.begin() + prefix_len, ids.end());
            child->value = IndexBuffer(
                std::vector<IndexT>(idx.begin() + prefix_len, idx.end()));
            child->parent    = node;
            child->timestamp = tic_;
            node->children[first_page_of(child->key, page_size_)] = child;
            evictable_size_ += child->length();
            node = child;
        }
        return InsertResult{prefix_len, CacheHandle{insert_len, node}};
    }

    // ---------- evict ----------
    // Returns the indices that were freed, concatenated in arbitrary order.
    IndexBuffer evict(size_t size) {
        if (size == 0) return IndexBuffer{};
        if (size > evictable_size_)
            throw std::runtime_error(
                "evict: cannot evict " + std::to_string(size) +
                ", only " + std::to_string(evictable_size_) + " is evictable");

        std::vector<Node*> leaves = collect_evictable_leaves();
        auto cmp = [](const Node* a, const Node* b) {
            return a->timestamp > b->timestamp;     // min-heap by timestamp
        };
        std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> heap(cmp);
        for (Node* n : leaves) heap.push(n);

        std::vector<IndexT> evicted_indices;
        size_t evicted_size = 0;
        while (evicted_size < size) {
            if (heap.empty())
                throw std::runtime_error("evict: heap exhausted before target reached");
            Node* n = heap.top(); heap.pop();
            if (!(n->ref_count == 0 && n->is_leaf() && !n->is_root()))
                throw std::runtime_error(
                    "evict: invariant violated on node " + std::to_string(n->uuid));

            evicted_size += n->length();
            auto v = n->value.to_host();
            evicted_indices.insert(evicted_indices.end(), v.begin(), v.end());
            evictable_size_ -= n->length();

            Node* parent = n->parent;
            size_t removed = parent->children.erase(first_page_of(n->key, page_size_));
            if (removed == 0)
                throw std::runtime_error(
                    "evict: node " + std::to_string(n->uuid) +
                    " missing from parent's children map");
            destroy_node(n);

            if (parent->is_leaf() && parent->ref_count == 0 && !parent->is_root())
                heap.push(parent);
        }
        return IndexBuffer(evicted_indices);
    }

    // ---------- lock_handle ----------
    void lock_handle(const CacheHandle& h, bool unlock = false) {
        Node* node = h.node;
        if (unlock) {
            while (node && !node->is_root()) {
                if (node->ref_count == 0)
                    throw std::runtime_error(
                        "lock_handle(unlock): ref_count underflow on node " +
                        std::to_string(node->uuid));
                node->ref_count -= 1;
                if (node->ref_count == 0) {
                    evictable_size_ += node->length();
                    protected_size_ -= node->length();
                }
                node = node->parent;
            }
        } else {
            while (node && !node->is_root()) {
                if (node->ref_count == 0) {
                    evictable_size_  -= node->length();
                    protected_size_  += node->length();
                }
                node->ref_count += 1;
                node = node->parent;
            }
        }
    }

    // ---------- get_matched_indices ----------
    // Walks from the matched node up to root, concatenating each node's value
    // (parent-first). The result lives on whatever backend the Allocator picks.
    IndexBuffer get_matched_indices(const CacheHandle& h) const {
        std::vector<std::vector<IndexT>> parts;
        Node* node = h.node;
        while (node && !node->is_root()) {
            parts.push_back(node->value.to_host());
            node = node->parent;
        }
        std::vector<IndexT> result;
        for (auto it = parts.rbegin(); it != parts.rend(); ++it)
            result.insert(result.end(), it->begin(), it->end());
        return IndexBuffer(result);
    }

    // ---------- size_info ----------
    SizeInfo size_info() const noexcept {
        return SizeInfo{evictable_size_, protected_size_};
    }

    // ---------- reset ----------
    // Python version explicitly raises NotImplementedError; we mirror it.
    void reset() {
        throw std::runtime_error("RadixCache::reset is not implemented");
    }

    // ---------- check_integrity ----------
    void check_integrity() const {
        if (root_->ref_count < 1)
            throw std::runtime_error("integrity: root must be protected");
        if (root_->parent != nullptr)
            throw std::runtime_error("integrity: root must not have a parent");

        size_t computed_evictable = 0;
        size_t computed_protected = 0;

        std::vector<const Node*> stack;
        stack.push_back(root_);
        while (!stack.empty()) {
            const Node* n = stack.back();
            stack.pop_back();
            if (!n->is_root()) {
                if (n->ref_count > 0) computed_protected += n->length();
                else                  computed_evictable += n->length();
            }
            for (const auto& kv : n->children) {
                if (kv.second->parent != n)
                    throw std::runtime_error(
                        "integrity: child parent pointer mismatch on node " +
                        std::to_string(kv.second->uuid));
                stack.push_back(kv.second);
            }
        }
        if (computed_evictable != evictable_size_)
            throw std::runtime_error(
                "integrity: evictable_size mismatch (" +
                std::to_string(computed_evictable) + " vs " +
                std::to_string(evictable_size_) + ")");
        if (computed_protected != protected_size_)
            throw std::runtime_error(
                "integrity: protected_size mismatch (" +
                std::to_string(computed_protected) + " vs " +
                std::to_string(protected_size_) + ")");
    }

    // ---------- root accessor (for tests/debug) ----------
    Node* root() { return root_; }

private:
    // ---------- internal helpers ----------
    Node* create_node() {
        auto up = std::make_unique<Node>();
        Node* raw = up.get();
        raw->uuid = counter_++;
        all_nodes_[raw] = std::move(up);
        return raw;
    }

    void destroy_node(Node* n) { all_nodes_.erase(n); }

    static size_t align_down(size_t a, size_t b) { return a / b * b; }

    static Key first_page_of(const std::vector<TokenT>& ids, size_t page_size) {
        const size_t end = std::min(page_size, ids.size());
        return Key(ids.begin(), ids.begin() + end);
    }
    static Key first_page_of(const std::vector<TokenT>& ids,
                             size_t offset, size_t page_size) {
        const size_t end = std::min(offset + page_size, ids.size());
        return Key(ids.begin() + offset, ids.begin() + end);
    }

    size_t get_match_len(const Node* node,
                         const std::vector<TokenT>& input_ids,
                         size_t offset) const {
        const size_t n = std::min(node->key.size(), input_ids.size() - offset);
        for (size_t i = 0; i < n; ++i)
            if (node->key[i] != input_ids[offset + i]) return i;
        return n;
    }

    // Split `node` so that the prefix [0, pos) becomes a new internal node
    // holding the original node's parent slot, and `node` keeps the tail
    // [pos, end). Returns the new internal node.
    Node* split_at(Node* node, size_t pos) {
        if (!(pos > 0 && pos < node->key.size()))
            throw std::runtime_error("split_at: position out of range");

        Node* parent = node->parent;

        // Slice the original value to host once, then split.
        auto v_host = node->value.to_host();
        std::vector<IndexT> head_val(v_host.begin(), v_host.begin() + pos);
        std::vector<IndexT> tail_val(v_host.begin() + pos, v_host.end());

        // New internal node inherits ref_count and timestamp from the original.
        Node* new_internal = create_node();
        new_internal->key.assign(node->key.begin(), node->key.begin() + pos);
        new_internal->value     = IndexBuffer(head_val);
        new_internal->parent    = parent;
        new_internal->ref_count = node->ref_count;
        new_internal->timestamp = node->timestamp;

        // Rewire parent: replace old child slot with new_internal.
        if (parent) {
            size_t removed = parent->children.erase(first_page_of(node->key, page_size_));
            if (removed == 0)
                throw std::runtime_error(
                    "split_at: node missing from parent's children map");
            parent->children[first_page_of(new_internal->key, page_size_)] = new_internal;
        }

        // Shrink `node` to the tail and reparent under new_internal.
        std::vector<TokenT> tail_key(node->key.begin() + pos, node->key.end());
        node->key    = std::move(tail_key);
        node->value  = IndexBuffer(tail_val);
        node->parent = new_internal;
        new_internal->children[first_page_of(node->key, page_size_)] = node;

        return new_internal;
    }

    std::pair<Node*, size_t> tree_walk(const std::vector<TokenT>& input_ids) {
        size_t prefix_len  = 0;
        const size_t indice_len = input_ids.size();
        Node* node         = root_;
        const uint64_t tic = ++tic_;

        while (prefix_len < indice_len) {
            Key lookup_key = first_page_of(input_ids, prefix_len, page_size_);
            auto it = node->children.find(lookup_key);
            if (it == node->children.end()) return {node, prefix_len};
            node = it->second;

            size_t raw_match = get_match_len(node, input_ids, prefix_len);
            size_t match_len = align_down(raw_match, page_size_);
            prefix_len += match_len;

            if (match_len != node->length()) {
                node = split_at(node, match_len);
                node->timestamp = tic;
                return {node, prefix_len};
            }
            node->timestamp = tic;
        }
        return {node, prefix_len};
    }

    std::vector<Node*> collect_evictable_leaves() {
        std::vector<Node*> leaves;
        std::vector<Node*> stack;
        stack.push_back(root_);
        while (!stack.empty()) {
            Node* n = stack.back();
            stack.pop_back();
            if (n->is_leaf()) {
                if (n->ref_count == 0) leaves.push_back(n);
            } else {
                for (const auto& kv : n->children) stack.push_back(kv.second);
            }
        }
        return leaves;
    }

    // ---------- state ----------
    size_t                                                 page_size_;
    uint64_t                                               counter_         = 0;
    uint64_t                                               tic_              = 0;
    size_t                                                 evictable_size_   = 0;
    size_t                                                 protected_size_   = 0;
    std::unordered_map<Node*, std::unique_ptr<Node>>       all_nodes_;
    Node*                                                  root_;
};

}  // namespace minisgl
