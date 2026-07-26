#ifndef _VT_PAGES_HPP_
#define _VT_PAGES_HPP_

#include <cstdint>
#include <vector>

#include "vt.hpp"

namespace vt {

// 物理资源池：行号 (table_idx) + 页号 (page_id) + 一张 2D 表。
//
// 模板参数：
//   IndexT  page_id / table_idx 等下标类型，存进 pages_ / free_pages_ / free_rows_
// 类内再 alias 成 Index 供外部使用（PageTable<>::Index）。
//
// 物理布局（扁平 1D，行连续；page_id 按 page 边界对齐，对齐 mini-sglang cache.py:16-25）：
//   pages_[row * max_seq_len + pos] = page_id   （per-token 存储；同一 page 的 page_size 个位置共享同一个 page_id）
//
// page_id 是 page-aligned 的（0, page_size, 2*page_size, ...）—— 一个 page_id 代表 KV pool
// 里 page_size 个连续物理 slot 的起点。Kernel 地址翻译：
//   physical_slot = page_id + (pos % page_size)
//
// 不变量：
//   - 同一 table_idx 同一时间至多归属一个活跃 req（alloc_row/free_row）
//   - 同一 page_id 同一时间至多被一行引用（alloc_pages/free_pages）
//   - 同一 page 内 page_size 个 token 位置的 page_id 必须相等（broadcast 不变量）
//
// 非线程安全，调用方串行化（与 RadixTree 一致）。
template <typename IndexT = int32_t>
class PageTable {
public:
    using Index    = IndexT;
    using IndexVec = std::vector<Index>;

    // -1 既表达"page 未写过"（pages_ 上的洞），也表达"alloc_row 池空"。
    static constexpr Index kInvalid = (Index)-1;

    PageTable(int max_running_reqs, int num_pages, int max_seq_len, int page_size = 1)
        : max_running_reqs_(max_running_reqs),
          num_pages_(num_pages),
          max_seq_len_(max_seq_len),
          page_size_(page_size),
          pages_ ((size_t)max_running_reqs * max_seq_len, kInvalid) {
        vt_assert(max_running_reqs > 0, "PageTable: max_running_reqs must be > 0");
        vt_assert(num_pages       > 0, "PageTable: num_pages must be > 0");
        vt_assert(max_seq_len     > 0, "PageTable: max_seq_len must be > 0");
        vt_assert(page_size       > 0, "PageTable: page_size must be > 0");

        // LIFO：倒序入栈，分配时从 page_id=0 开始往外发。
        // page_id 是 page-aligned 的（i * page_size），代表 KV pool 里第 i 个 page 的起点 slot。
        free_rows_.reserve(max_running_reqs);
        for (int i = max_running_reqs - 1; i >= 0; --i) free_rows_.push_back((Index)i);
        free_pages_.reserve(num_pages);
        for (int i = num_pages - 1; i >= 0; --i)
            free_pages_.push_back((Index)(i * page_size));
    }

    // ---- 行号池 ----
    Index alloc_row() {
        if (free_rows_.empty()) return kInvalid;
        Index r = free_rows_.back();
        free_rows_.pop_back();
        return r;
    }
    void free_row(Index table_idx) {
        vt_assert(table_idx >= 0 && table_idx < max_running_reqs_,
                  "PageTable::free_row: table_idx out of range");
        free_rows_.push_back(table_idx);
    }

    // ---- 页号池 ----
    // 申请 n 个 page，写入 out[0..ret)，ret <= n。池不够就返回实际拿到的数量。
    int alloc_pages(int n, Index* out) {
        vt_assert(n >= 0, "PageTable::alloc_pages: n must be >= 0");
        int got = 0;
        while (got < n && !free_pages_.empty()) {
            out[got++] = free_pages_.back();
            free_pages_.pop_back();
        }
        return got;
    }
    void free_pages(const Index* ids, int n) {
        vt_assert(n >= 0, "PageTable::free_pages: n must be >= 0");
        for (int i = 0; i < n; ++i) {
            vt_assert(ids[i] >= 0 && ids[i] < (Index)(num_pages_ * page_size_),
                      "PageTable::free_pages: page_id out of range");
            vt_assert(ids[i] % (Index)page_size_ == 0,
                      "PageTable::free_pages: page_id not page-aligned");
            free_pages_.push_back(ids[i]);
        }
    }

    // ---- 2D 表写 ----
    // 把 page_ids[0..last_pos-first_pos) 拷进 pages_[table_idx, first_pos..last_pos)。
    // 半开区间，与 STL 风格一致。
    void write_pages(Index table_idx, int first_pos, int last_pos,
                     const Index* page_ids) {
        vt_assert(table_idx >= 0 && table_idx < max_running_reqs_,
                  "PageTable::write_pages: table_idx out of range");
        vt_assert(0 <= first_pos && first_pos <= last_pos && last_pos <= max_seq_len_,
                  "PageTable::write_pages: pos out of range");
        Index* dst = pages_.data() + (size_t)table_idx * max_seq_len_ + first_pos;
        for (int i = 0; i < last_pos - first_pos; ++i) {
            vt_assert(page_ids[i] >= 0 && page_ids[i] < (Index)(num_pages_ * page_size_),
                      "PageTable::write_pages: page_id out of range");
            vt_assert(page_ids[i] % (Index)page_size_ == 0,
                      "PageTable::write_pages: page_id not page-aligned");
            dst[i] = page_ids[i];
        }
    }

    // ---- 行视图（kernel 切片 / cache_manager 回收读都用它）----
          Index* page_row(Index table_idx)       { return pages_.data() + (size_t)table_idx * max_seq_len_; }
    const Index* page_row(Index table_idx) const { return pages_.data() + (size_t)table_idx * max_seq_len_; }

    // ---- 容量 ----
    int max_running_reqs() const { return max_running_reqs_; }
    int num_pages()         const { return num_pages_; }
    int max_seq_len()       const { return max_seq_len_; }
    int page_size()         const { return page_size_; }
    int available_rows()    const { return (int)free_rows_.size(); }
    int available_pages()   const { return (int)free_pages_.size(); }

private:
    int                max_running_reqs_;
    int                num_pages_;
    int                max_seq_len_;
    int                page_size_;

    IndexVec           pages_;
    IndexVec           free_rows_;
    IndexVec           free_pages_;
};

} // namespace vt

#endif
