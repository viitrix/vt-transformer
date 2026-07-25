#ifndef _VT_PAGES_HPP_
#define _VT_PAGES_HPP_

#include <cstdint>
#include <vector>

#include "vt.hpp"

namespace vt {

// 物理资源池：行号 (table_idx) + 页号 (page_id) + 两张 2D 表。
//
// 模板参数（与 vt_radix.hpp 对齐）：
//   TokenT  token id 类型，存进 tokens_
//   IndexT  page_id / table_idx 等下标类型，存进 pages_ / free_pages_ / free_rows_
// 类内再 alias 成 Token / Index 供外部使用（PageTable<>::Index / ::Token）。
//
// 物理布局（扁平 1D，行连续）：
//   pages_[row * max_seq_len + pos] = page_id   （地址翻译：(req, pos) -> page）
//   tokens_[row * max_seq_len + pos] = token_id （兄弟池：kernel input / detokenize）
//
// 不变量：
//   - 同一 table_idx 同一时间至多归属一个活跃 req（alloc_row/free_row）
//   - 同一 page_id 同一时间至多被一行引用（alloc_pages/free_pages）
//
// 非线程安全，调用方串行化（与 RadixTree 一致）。
template <typename TokenT = int32_t, typename IndexT = int32_t>
class PageTable {
public:
    using Token    = TokenT;
    using Index    = IndexT;
    using TokenVec = std::vector<Token>;
    using IndexVec = std::vector<Index>;

    // -1 既表达"page 未写过"（pages_ 上的洞），也表达"alloc_row 池空"。
    static constexpr Index kInvalid = (Index)-1;

    PageTable(int max_running_reqs, int num_pages, int max_seq_len)
        : max_running_reqs_(max_running_reqs),
          num_pages_(num_pages),
          max_seq_len_(max_seq_len),
          pages_ ((size_t)max_running_reqs * max_seq_len, kInvalid),
          tokens_((size_t)max_running_reqs * max_seq_len, 0) {
        vt_assert(max_running_reqs > 0, "PageTable: max_running_reqs must be > 0");
        vt_assert(num_pages       > 0, "PageTable: num_pages must be > 0");
        vt_assert(max_seq_len     > 0, "PageTable: max_seq_len must be > 0");

        // LIFO：倒序入栈，分配时从 0 开始往外发。
        free_rows_.reserve(max_running_reqs);
        for (int i = max_running_reqs - 1; i >= 0; --i) free_rows_.push_back((Index)i);
        free_pages_.reserve(num_pages);
        for (int i = num_pages - 1; i >= 0; --i) free_pages_.push_back((Index)i);
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
            vt_assert(ids[i] >= 0 && ids[i] < num_pages_,
                      "PageTable::free_pages: page_id out of range");
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
            vt_assert(page_ids[i] >= 0 && page_ids[i] < num_pages_,
                      "PageTable::write_pages: page_id out of range");
            dst[i] = page_ids[i];
        }
    }

    // 把 tokens[0..last_pos-first_pos) 拷进 tokens_[table_idx, first_pos..last_pos)。
    // 不做 token id 范围检查——vocab_size 不归 PageTable 管，由 tokenizer / sampling 层保证。
    void write_tokens(Index table_idx, int first_pos, int last_pos,
                      const Token* tokens) {
        vt_assert(table_idx >= 0 && table_idx < max_running_reqs_,
                  "PageTable::write_tokens: table_idx out of range");
        vt_assert(0 <= first_pos && first_pos <= last_pos && last_pos <= max_seq_len_,
                  "PageTable::write_tokens: pos out of range");
        Token* dst = tokens_.data() + (size_t)table_idx * max_seq_len_ + first_pos;
        for (int i = 0; i < last_pos - first_pos; ++i) dst[i] = tokens[i];
    }

    // ---- 行视图（kernel 切片 / prefill 拷贝 / cache_manager 回收读都用它）----
          Index* page_row (Index table_idx)       { return pages_.data()  + (size_t)table_idx * max_seq_len_; }
          Token* token_row(Index table_idx)       { return tokens_.data() + (size_t)table_idx * max_seq_len_; }
    const Index* page_row (Index table_idx) const { return pages_.data()  + (size_t)table_idx * max_seq_len_; }
    const Token* token_row(Index table_idx) const { return tokens_.data() + (size_t)table_idx * max_seq_len_; }

    // ---- 容量 ----
    int max_running_reqs() const { return max_running_reqs_; }
    int num_pages()         const { return num_pages_; }
    int max_seq_len()       const { return max_seq_len_; }
    int available_rows()    const { return (int)free_rows_.size(); }
    int available_pages()   const { return (int)free_pages_.size(); }

private:
    int                max_running_reqs_;
    int                num_pages_;
    int                max_seq_len_;

    IndexVec           pages_;
    TokenVec           tokens_;
    IndexVec           free_rows_;
    IndexVec           free_pages_;
};

} // namespace vt

#endif
