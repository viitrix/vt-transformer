// forward.hpp — ForwardInput / ForwardOutput / ForwardData + IForwardBackend
//
// Scheduler 与 GPU backend 的接缝。Scheduler 把准备好的 batch 打包成
// ForwardInput 交给 backend,backend 返回 ForwardOutput(每 req 一个 token
// + 一个 copy_done SyncEvent)。
//
// 接口最小化:只暴露 forward() 和 engine_stream_event()。pad_batch /
// prepare_attn_metadata / sampler.prepare 等都封进 backend 的 forward 实现
// 里,Scheduler 看不到。

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "pinned_allocator.hpp"
#include "req.hpp"
#include "sync.hpp"

namespace minisgl {

// 每 req 一个 token 的容器。MINISGL_USE_CUDA 时 backed by pinned host memory
// (cudaMemcpyAsync 真 async);否则退化为普通堆内存。
using PinnedInt32Vector = std::vector<int32_t, PinnedAllocator<int32_t>>;

// 每个 batch 的采样参数(按 req 展开)。首版直接从 req.sampling_params 拷。
struct BatchSamplingArgs {
    std::vector<double>  temperatures;
    std::vector<int64_t> top_k;
    std::vector<double>  top_p;
    std::vector<bool>    ignore_eos;
};

struct ForwardInput {
    Batch             batch;             // batch.input_ids 已 gather 好
    BatchSamplingArgs sample_args;
    // 每 token 一项:这个 token 来自哪个 req 的 table_idx。len == batch.input_ids.size()。
    std::vector<int64_t> input_mapping;
    // 每 req 一项:这个 req 的 table_idx,用于回写 next_token。
    std::vector<int64_t> write_mapping;
    // 每 req 一项:req.device_len(可 decode 时);不可 decode(chunked,首版不会出现)给 -1。
    std::vector<int64_t> write_lens;
};

struct ForwardOutput {
    // 每 req 一个 token(按 batch.reqs 顺序)。chunked req 也算一位,首版不会出现。
    // pinned-backed,供 backend forward() 直接 cudaMemcpyAsync D2H 写入。
    PinnedInt32Vector next_tokens_cpu;
    // next_tokens_cpu 可读前的同步点。CUDA backend 用 CudaContextSyncEvent。
    std::unique_ptr<SyncEvent> copy_done;
};

// 一轮 overlap 的完整数据:输入 + 输出配对。
// 对应 mini-sglang scheduler.py:42 的 ForwardData。
using ForwardData = std::pair<ForwardInput, ForwardOutput>;

class IForwardBackend {
public:
    virtual ~IForwardBackend() = default;

    // 单一 GPU 入口。input.batch.input_ids 已经 gather 完成;backend 负责
    // 把 next_tokens_cpu 填好(每 req 一个 token),并设置 copy_done。
    // copy_done 必须在 next_tokens_cpu 真正可读前被同步过一次。
    virtual ForwardOutput forward(const ForwardInput& in) = 0;

    // 返回 backend 自己的 stream events 给 Scheduler 做 overlap 同步。
    // 指针归 backend 所有,Scheduler 只持 non-owning 引用。两个都必须非空 ——
    // Scheduler 构造时直接取用,不再有 fallback。
    virtual SyncEvent* sched_stream_event()  { return nullptr; }
    virtual SyncEvent* engine_stream_event() { return nullptr; }
};

}  // namespace minisgl
