#ifndef _VT_CACHE_HPP_
#define _VT_CACHE_HPP_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "vt.hpp"
#include "vt_pages.hpp"
#include "vt_radix.hpp"

namespace vt {

// CacheManager = Radix 协议层 + PageTable 协调者。
//
// 持有：
//   - PageTable  （拥有，unique_ptr）跟 CacheManager 同生共死
//   - RadixTree  （拥有，unique_ptr）跟 CacheManager 同生共死
//
//   - 所有 match / lock / evict 操作走 radix_
//   - 所有 page_id 的"来源 / 去向"走 pt_
//
// 单独的 RadixTree 不知道 page_id 从哪里来、谁拥有它们；
// CacheManager 知道——这是它存在的唯一理由。
//
// 非线程安全（与 RadixTree / PageTable 一致）。
template <typename TokenT = int32_t, typename IndexT = int32_t>
class CacheManager {
public:
    using Token = TokenT;
    using Index = IndexT;
    using Tree  = RadixTree<Token, Index>;
    using Node  = typename Tree::Node;

    // 构造时一并创建 PageTable（pt_ 由本类拥有）。
    // page_size > 1 时：radix 按 page 边界对齐；PageTable 发 page-aligned 的 page_id
    // （0, page_size, 2*page_size, ...），同一 page 内的 page_size 个 token 位置共享同一个 page_id。
    CacheManager(int max_running_reqs, int num_pages, int max_seq_len, int page_size = 1)
        : pt_(std::make_unique<PageTable<Index>>(max_running_reqs, num_pages,
                                                  max_seq_len, page_size)),
          radix_(std::make_unique<Tree>(page_size)) {}

    // 暴露 PageTable 指针给外部（kernel 读 page_row、调用方读容量等）。
    // 返回非 const 版本：调用方可能需要在 pt 上做 alloc_row / write_pages 等操作。
          PageTable<Index>* pt()       { return pt_.get(); }
    const PageTable<Index>* pt() const { return pt_.get(); }

    // ---- 容量查询 ----
    // 还能装多少 token。两个来源的单位不同：
    //   - radix evictable_size 是 token 数（value_ 仍是 per-token，跟 mini-sglang radix 一致）
    //   - pt available_pages 是 page 数；每 page 含 page_size 个 token（1 page = page_size tokens）
    // 所以 pt 那部分要 × page_size 换算成 token。等价于 mini-sglang cache.py:33-34。
    size_t available_size() const {
        return radix_->evictable_size()
             + (size_t)pt_->available_pages() * radix_->page_size();
    }

    // ---- 调用顺序契约（req 生命周期）----
    //   alloc_row()       ← 调用方从 PageTable 拿 table_idx（CacheManager 不管）
    //   prepare()         ← 锁命中前缀，写 [0, cached_len) 段；返回 PrepareResult
    //   allocate_pages()  ← 写 [cached_len, ...) 新段（forward 提交前）
    //   ... forward / decode 期间使用 pt_->page_row() 读 page_id ...
    //   finished()        ← 写回 radix，归还行号，PrepareResult.node 失效
    //
    // Node* 生命周期：
    //   prepare() 返回的 node 在 [prepare, finished) 区间内有效——
    //   prepare 时路径已 lock，evict 不能动它。finished() 后路径 unlock，
    //   node 可能被后续 evict 删除；不要跨 finished() 持有 node。

    // ---- prepare：Prefill 入选时调用一次 ----
    // 内部做三件事：
    //   a) radix_->match_prefix 找命中段
    //   b) radix_->lock 把命中路径标为 protected（防止被 evict）
    //   c) 把命中段的 page_id 和 token 写到 PageTable 的新行 [0, cached_len)
    // 调用方拿到结果后填 Request::node / Request::prefill_pos。
    //
    // 注意：table_idx 由调用方先 pt_->alloc_row() 拿到，CacheManager 不代劳。
    //       容量检查（够不够入选）也由调用方做（查 available_size），prepare 只负责"拿到就装好"。
    struct PrepareResult {
        // node 仅在 [prepare, finished) 期间有效（见类顶调用顺序契约）。
        Node* node       = nullptr;   // 命中节点（radix 树里的锚点）
        int   cached_len = 0;         // 命中段长度（page_size=1 时 = page_id 数）
    };
    PrepareResult prepare(Index table_idx, const std::vector<Token>& input_ids) {
        vt_assert(table_idx >= 0 && table_idx < pt_->max_running_reqs(),
                  "CacheManager::prepare: table_idx out of range");
        vt_assert(input_ids.size() <= (size_t)pt_->max_seq_len(),
                  "CacheManager::prepare: input_ids longer than max_seq_len");

        // (a)(b) match + lock
        auto m = radix_->match_prefix(input_ids);
        radix_->lock(m.node);

        // (c) 命中段写进 PageTable：page_id 来自 radix 路径
        if (m.prefix_len > 0) {
            std::vector<Index> indices = collect_path_indices(m.node, m.prefix_len);
            vt_assert(indices.size() == m.prefix_len,
                      "CacheManager::prepare: path indices length mismatch");
            pt_->write_pages (table_idx, /*first=*/0, /*last=*/(int)m.prefix_len, indices.data());
        }

        return {m.node, (int)m.prefix_len};
    }

    // ---- finished：req Finished 后调用 ----
    // 跟 prepare 对称，是"反向操作"：把 PageTable 行里的 page_id 写回 radix 树，
    // 让后续 req 能复用这段前缀。同时释放"重复段"和"tail 段"的 page，归还行号。
    //
    // 区间划分（沿用 mini-sglang cache.py:55-79 的语义）：
    //   [0,                       prepare.cached_len)  —— prepare 时复用的，已在树里，不动
    //   [prepare.cached_len,      ins.prefix_len)      —— 发现"别人已插过"，重复 → free
    //   [ins.prefix_len,          ins.inserted_len)    —— 这次新插入，留在树里
    //   [ins.inserted_len,        cur_cached_len)      —— tail（page 对齐没对上） → free
    struct FinishInput {
        Index              table_idx;       // req 的行
        PrepareResult      prepare_result;  // prepare 当时的快照
        int                cur_cached_len;  // 当前 req.cached_len = prefill_pos + decode_pos
        std::vector<Token> tokens;          // 已拼好；长度必须 == cur_cached_len
    };
    void finished(const FinishInput& in) {
        vt_assert(in.table_idx >= 0 && in.table_idx < pt_->max_running_reqs(),
                  "CacheManager::finished: table_idx out of range");
        vt_assert(in.cur_cached_len >= in.prepare_result.cached_len,
                  "CacheManager::finished: cur_cached_len < prepare's cached_len");
        vt_assert(in.cur_cached_len <= pt_->max_seq_len(),
                  "CacheManager::finished: cur_cached_len > max_seq_len");
        vt_assert(in.tokens.size() == (size_t)in.cur_cached_len,
                  "CacheManager::finished: tokens.size() must == cur_cached_len");

        const Index* row = pt_->page_row(in.table_idx);

        // (1) radix insert：把 [0, cur_cached_len) 段的 (tokens, page_ids) 写进树
        std::vector<Index> indices(row, row + in.cur_cached_len);
        auto ins = radix_->insert_prefix(in.tokens, indices);

        // 不变量：prepare 时 lock 了 [0, old) 这段，radix 保证 lock 段不被 evict，
        // 所以 finished 时这段一定还在树里 → prefix_len ≥ old。
        // 若不成立说明 lock 协议被破坏，下面的"重复段"释放会静默泄漏 page。
        vt_assert(ins.prefix_len >= (size_t)in.prepare_result.cached_len,
                  "CacheManager::finished: prefix_len < prepare.cached_len "
                  "(lock protocol broken?)");

        // (2) unlock prepare 时 lock 的旧路径（不再需要保护）
        radix_->unlock(in.prepare_result.node);

        // (3) 释放"重复段" [prepare.cached_len, prefix_len)
        //     这段是"prepare 后别人也插入了相同前缀"，我们这次 insert 发现已在树里。
        //     row 是 per-token 视图（同一 page 内 page_id 重复 page_size 次），
        //     所以 free 前要先 dedup 成 per-page 列表，否则同一 page_id 会 free 多次。
        int old = in.prepare_result.cached_len;
        if (ins.prefix_len > (size_t)old) {
            auto to_free = extract_pages(row, old, (int)ins.prefix_len);
            pt_->free_pages(to_free.data(), (int)to_free.size());
        }

        // (4) 释放"tail 段" [inserted_len, cur_cached_len)
        //     page 对齐没对上的尾巴（finished=true 时整段归还）。同样要 dedup。
        if ((size_t)in.cur_cached_len > ins.inserted_len) {
            auto to_free = extract_pages(row, (int)ins.inserted_len, in.cur_cached_len);
            pt_->free_pages(to_free.data(), (int)to_free.size());
        }

        // (5) 归还行号（req 已死，PageTable 行回池）
        pt_->free_row(in.table_idx);
    }

    // ---- allocate_pages：Forward 提交前调用，给 batch 里所有 req 的新 token 段分配 page ----
    // 跟 prepare/finished 不同：本接口完全不碰 radix，只动 PageTable 的 page 池 + 2D 表。
    //
    // 语义：对每个 item，确保覆盖 [first_pos, last_pos) 的 page 区间都已分配。
    //   - 已经分配过的 page（上次 allocate_pages 留下的，比如 decode 走进了一个未填满的 page）
    //     直接跳过，不重复分配（对齐 mini-sglang cache.py:46-48 的 if last_page > first_page 逻辑）。
    //   - 新分配的 page_id（page-aligned）broadcast 到该 page 的 page_size 个 token 位置。
    //
    // 失败语义：池子不够时**整批失败**（不部分分配），返回 false。
    //          调用方决定是缩 batch、还是先调 evict 腾空间再重试。
    //          整批失败的好处：调用方不用关心"哪些 req 拿到了、哪些没拿到"。
    //
    // 注意：first_pos/last_pos 是 token 位置（不是 page 索引）。
    //       调用方按 req.cached_len() / req.total_len() 直接传即可，单位换算在内部做。
    struct AllocItem {
        Index table_idx;
        int   first_pos;
        int   last_pos;
    };
    bool allocate_pages(const std::vector<AllocItem>& items) {
        const int page_size = (int)pt_->page_size();

        // (1) 校验 + 计算每 item 真正需要新分配的 page 区间（跳过已分配的）
        struct Plan { Index table_idx; int first_page; int last_page; };
        std::vector<Plan> plans;
        plans.reserve(items.size());

        int needed_pages = 0;
        for (const auto& it : items) {
            vt_assert(it.table_idx >= 0 && it.table_idx < pt_->max_running_reqs(),
                      "CacheManager::allocate_pages: table_idx out of range");
            vt_assert(0 <= it.first_pos && it.first_pos <= it.last_pos
                      && it.last_pos <= pt_->max_seq_len(),
                      "CacheManager::allocate_pages: pos out of range");

            int first_page = it.first_pos / page_size;
            int last_page  = (it.last_pos + page_size - 1) / page_size;  // ceil

            // 跳过开头已分配的 page：上次 allocate 留下的"半满 page"不应该重复分配。
            // （page 内任意一个 slot 非 kInvalid 就认为该 page 已分配；broadcast 不变量
            //   保证同 page 内所有 slot 同步写入，所以只看 page 的第一个 slot 即可。）
            const Index* row = pt_->page_row(it.table_idx);
            while (first_page < last_page
                   && row[first_page * page_size] != PageTable<Index>::kInvalid) {
                ++first_page;
            }

            needed_pages += last_page - first_page;
            plans.push_back({it.table_idx, first_page, last_page});
        }

        // (2) 容量检查（不部分分配）
        if (needed_pages > pt_->available_pages()) {
            return false;
        }

        // (3) 逐 item 分配 + broadcast 写入
        //     page_buf 收新分配的 page_id（page-aligned）；token_buf 是 broadcast 后的 per-token 视图。
        std::vector<Index> page_buf;
        std::vector<Index> token_buf;
        for (const auto& p : plans) {
            int want_pages = p.last_page - p.first_page;
            if (want_pages == 0) continue;

            page_buf.resize(want_pages);
            int got = pt_->alloc_pages(want_pages, page_buf.data());
            vt_assert(got == want_pages,
                      "CacheManager::allocate_pages: alloc short of want "
                      "(should not happen, capacity was checked)");

            // broadcast：每个 page_id 写到该 page 的 page_size 个 token 位置
            int token_count = want_pages * page_size;
            token_buf.resize(token_count);
            for (int pp = 0; pp < want_pages; ++pp) {
                for (int t = 0; t < page_size; ++t) {
                    token_buf[pp * page_size + t] = page_buf[pp];
                }
            }

            int first_pos = p.first_page * page_size;
            int last_pos  = p.last_page  * page_size;
            pt_->write_pages(p.table_idx, first_pos, last_pos, token_buf.data());
        }
        return true;
    }

    // ---- evict：从 radix 树淘汰 num_tokens 个 token，返回对应的 page_id 列表 ----
    // 调用方拿到 page_id 后**自己**调 pt_->free_pages(...) 归还池子。
    // 职责分离：radix 决定"淘汰谁"，PageTable 管"自己的池子"。
    //
    // 失败语义：不够 num_tokens 个 evictable（lock 段不能淘汰）时，
    //          返回空 vector（不部分淘汰）。调用方决策。
    //
    // 注意：radix 的 value_ 是 per-token（同 page 内 page_id 重复 page_size 次），
    //       所以这里返回前要先 dedup 成 per-page 列表，否则调用方会把同一 page_id free 多次。
    //       lock 段是安全的——req 正在用的前缀（prepare lock 未释放）不会被淘汰。
    std::vector<Index> evict(size_t num_tokens) {
        if (num_tokens == 0) return {};
        if (num_tokens > radix_->evictable_size()) return {};
        std::vector<Index> raw = radix_->evict(num_tokens);
        return extract_pages(raw.data(), 0, (int)raw.size());
    }

private:
    std::unique_ptr<PageTable<Index>> pt_;     // 拥有
    std::unique_ptr<Tree>             radix_;  // 拥有

    // 从 per-token 视图抽取 per-page 列表（dedup）。
    // 同一 page 内 page_id 在 row 上重复 page_size 次，这里取每 page 第一个 slot 的值。
    // 用于 finished() 和 evict() 在调 pt_->free_pages 前把 per-token 收缩成 per-page，
    // 否则同一 page_id 会被 free 多次，污染池子。
    std::vector<Index> extract_pages(const Index* row, int first_pos, int last_pos) const {
        const int page_size = (int)pt_->page_size();
        int first_page = first_pos / page_size;
        int last_page  = (last_pos + page_size - 1) / page_size;  // ceil
        std::vector<Index> out;
        out.reserve(last_page - first_page);
        for (int p = first_page; p < last_page; ++p) {
            out.push_back(row[p * page_size]);
        }
        return out;
    }

    // 从 root 走到 node，沿路径收集 value_，截到 prefix_len 个为止。
    // root 自己没有 value_（见 vt_radix.hpp 构造函数），所以从 node 向上爬，
    // 跳过 root，再反转成 root→node 顺序拼接。
    std::vector<Index> collect_path_indices(Node* node, size_t prefix_len) const {
        std::vector<Node*> path;
        for (Node* n = node; n != nullptr && !n->is_root(); n = n->parent_) {
            path.push_back(n);
        }
        std::reverse(path.begin(), path.end());

        std::vector<Index> out;
        out.reserve(prefix_len);
        size_t taken = 0;
        for (Node* n : path) {
            size_t want = std::min(n->value_.size(), prefix_len - taken);
            out.insert(out.end(), n->value_.begin(), n->value_.begin() + want);
            taken += want;
            if (taken == prefix_len) break;
        }
        return out;
    }
};

} // namespace vt

#endif
