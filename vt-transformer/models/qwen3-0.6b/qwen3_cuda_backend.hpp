// qwen3_cuda_backend.hpp — Qwen3CudaBackend : IForwardBackend
//
// 真 CUDA backend 的 v1 骨架。kernel 内部不在本阶段范围:
//   * 不跑 model forward(权重的 ptr 都在 qwen3::g_weights,这版不读)
//   * 不跑 sampler(sampled_gpu_ 直接 cudaMemset 0)
//   * 不写 page_table(stub kernel 不读 KV)
// 本阶段要验证的 plumbing:
//   * vt::CudaContext 启动 + 2 stream + 3 event 池
//   * CudaContextSyncEvent 跨 stream 同步(record / wait_on / synchronize)
//   * H2D ForwardInput + D2H ForwardOutput(走 pinned allocator 真 async)
//   * sched_stream / engine_stream 与 Scheduler 的 overlap_loop 配合不挂
// 真实 model + sampler 接进来时,只改 forward() 内部;接口与外部行为不变。

#ifndef MINISGL_QWEN3_CUDA_BACKEND_HPP
#define MINISGL_QWEN3_CUDA_BACKEND_HPP

#include "config.hpp"
#include "forward.hpp"
#include "sync.hpp"

#ifdef MINISGL_USE_CUDA
#include "vt_cuda.hpp"

namespace minisgl {

class Qwen3CudaBackend : public IForwardBackend {
public:
    explicit Qwen3CudaBackend(const SchedulerConfig& cfg);
    ~Qwen3CudaBackend() override;

    Qwen3CudaBackend(const Qwen3CudaBackend&)            = delete;
    Qwen3CudaBackend& operator=(const Qwen3CudaBackend&) = delete;

    ForwardOutput forward(const ForwardInput& in) override;

    SyncEvent* sched_stream_event()  override { return sched_event_.get(); }
    SyncEvent* engine_stream_event() override { return engine_event_.get(); }

private:
    // CudaContext 内部 stream / event 池的下标。
    static constexpr size_t kSchedStream   = 0;
    static constexpr size_t kEngineStream  = 1;
    static constexpr size_t kSchedEvent    = 0;  // sched_stream 上 record,engine wait_on 它
    static constexpr size_t kEngineEvent   = 1;  // 占位;Scheduler 当前不通过它同步
    static constexpr size_t kCopyDoneEvent = 2;  // engine_stream 上每 forward 末尾 record

    SchedulerConfig                       cfg_;
    std::unique_ptr<vt::CudaContext>      ctx_;
    std::unique_ptr<SyncEvent>            sched_event_;
    std::unique_ptr<SyncEvent>            engine_event_;

    // 预分配 GPU buffer,每 forward 复用。size 取 cfg 给的上界。
    vt::CudaTensor input_ids_gpu_;   // int32 [cfg_.max_extend_tokens]
    vt::CudaTensor positions_gpu_;   // int64 [cfg_.max_extend_tokens]
    vt::CudaTensor sampled_gpu_;     // int32 [cfg_.max_running_reqs]
};

}  // namespace minisgl

#endif  // MINISGL_USE_CUDA

#endif  // MINISGL_QWEN3_CUDA_BACKEND_HPP
