// req.hpp — Req / Batch / PendingReq / BatchPhase
//
// 端口 mini-sglang/python/minisgl/core.py 的 Req 和 Batch,以及 scheduler/utils.py
// 的 PendingReq。把 torch.Tensor 全部换掉:
//   * input_ids: torch.Tensor -> std::vector<int32_t>(host,随 append_host 增长)
//   * cache_handle: 直接持有 RadixCache::CacheHandle(value 类型,内含 node 指针)
//   * page_table[table_idx]: 折叠进 Req.page_indices(每 token 一个 page id)
//
// 所有权(关键):Req 用 shared_ptr 管理。
//   * live_reqs_ 持 shared_ptr<Req>(主所有权)
//   * Batch.reqs 持 shared_ptr<Req>(让 batch 在 overlap 中安全存活)
//   * DecodeManager.running_reqs_ 持 shared_ptr<Req>
// 这样在 overlap 模式下,即使一个 Req 已在 batch_N 的 process_last_data 里被
// "finished",它在 batch_N+1 中仍存活(shared_ptr 引用计数 > 0),直到所有引用
// 消失才析构。对应 Python 的隐式 GC 行为。

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "message.hpp"    // SamplingParams
#include "radix.hpp"      // RadixCache::CacheHandle

namespace minisgl {

// 默认实例化的 RadixCache 类型(HostAllocator + int32_t)。
using ReqRadixCache   = RadixCache<int32_t, int32_t, HostAllocator>;
using ReqCacheHandle  = ReqRadixCache::CacheHandle;

enum class BatchPhase { Prefill, Decode };

struct Req {
    std::vector<int32_t> input_ids;        // host,append_host 推进
    std::vector<int32_t> page_indices;     // 每 token 的 page id,长度 == max_device_len
    int64_t  table_idx      = -1;          // TableManager slot 下标
    size_t   cached_len     = 0;           // 已经被 RadixCache 收纳的前缀长度(页对齐)
    size_t   device_len     = 0;           // 当前 KV 长度(== "已写过的最大位置 + 1")
    size_t   max_device_len = 0;           // input_len + output_len
    int64_t  uid            = 0;
    bool     chunked        = false;       // 预留:chunked prefill;首版恒 false
    SamplingParams sampling_params{};
    ReqCacheHandle cache_handle{};

    // mini-sglang core.py:43-46 的 property
    size_t extend_len() const noexcept { return device_len - cached_len; }
    size_t remain_len() const noexcept { return max_device_len - device_len; }
    bool   can_decode() const noexcept { return remain_len() > 0 && !chunked; }

    // forward 写完 [cached_len, device_len) 的 KV 后调用:把 cached_len 推到
    // 当前 device_len,再为下一个 token 预留位置。
    // 对应 mini-sglang core.py:52-54 的 complete_one(那里在 engine.forward_batch
    // 里调;C++ 端移到 Scheduler.process_last_data 里调,backend 不必关心)。
    void complete_one() noexcept {
        cached_len = device_len;
        device_len += 1;
    }

    // 把采样出的 token 追加到 input_ids 末尾。device_len 不在这里动 ——
    // complete_one 才推进 device_len。对应 core.py:56-57。
    void append_host(int32_t tok) {
        input_ids.push_back(tok);
    }
};

struct Batch {
    // shared_ptr:见文件头注释。
    std::vector<std::shared_ptr<Req>> reqs;
    BatchPhase        phase = BatchPhase::Prefill;

    // _prepare_batch 填充的每 batch 状态:
    std::vector<int32_t> input_ids;        // gather 后总输入,len == sum(extend_len)
    std::vector<int64_t> positions;        // len == input_ids.size()

    bool is_prefill() const noexcept { return phase == BatchPhase::Prefill; }
    bool is_decode()  const noexcept { return phase == BatchPhase::Decode; }
    size_t size() const noexcept { return reqs.size(); }
};

// mini-sglang scheduler/utils.py:PendingReq 的 C++ 版。
// chunked_req 字段首版省略。
struct PendingReq {
    int64_t uid = 0;
    std::vector<int32_t> input_ids;
    SamplingParams sampling_params{};

    size_t input_len() const noexcept { return input_ids.size(); }
    size_t output_len() const noexcept {
        return static_cast<size_t>(std::max<int64_t>(0, sampling_params.max_tokens));
    }
};

}  // namespace minisgl
