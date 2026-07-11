// table.hpp — TableManager
//
// 端口 mini-sglang/python/minisgl/scheduler/table.py。
// 原 Python 版拥有 page_table tensor[max_running_reqs, max_seq_len] 和
// token_pool tensor 同形状。C++ host 版不需要等价的 GPU tensor:
//   * 每 req 的 page_indices 直接存在 Req 里(Req::page_indices)
//   * 每 req 的 input_ids 直接存在 Req 里
// 所以 TableManager 首版只管 slot 池(给每 req 分配一个 table_idx),
// 不持有 per-slot 数据。

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace minisgl {

class TableManager {
public:
    explicit TableManager(size_t max_running_reqs)
        : max_running_reqs_(max_running_reqs) {
        free_slots_.reserve(max_running_reqs);
        // 逆序 push,allocate() 从 back pop,这样首次拿到的是 slot 0,符合直觉。
        for (size_t i = 0; i < max_running_reqs; ++i) {
            free_slots_.push_back(static_cast<int64_t>(i));
        }
    }

    TableManager(const TableManager&)            = delete;
    TableManager& operator=(const TableManager&) = delete;
    TableManager(TableManager&&) noexcept            = default;
    TableManager& operator=(TableManager&&) noexcept = default;

    size_t available_size() const noexcept { return free_slots_.size(); }

    int64_t allocate() {
        if (free_slots_.empty()) {
            throw std::runtime_error("TableManager: no free slot");
        }
        int64_t idx = free_slots_.back();
        free_slots_.pop_back();
        return idx;
    }

    void free(int64_t slot) {
        if (slot < 0) return;
        free_slots_.push_back(slot);
    }

    size_t capacity() const noexcept { return max_running_reqs_; }

private:
    size_t              max_running_reqs_;
    std::vector<int64_t> free_slots_;
};

}  // namespace minisgl
