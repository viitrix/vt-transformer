// qwen3_cuda_backend.cpp — see qwen3_cuda_backend.hpp
//
// Compile only under MINISGL_USE_CUDA (Makefile cuda target)。

#include "qwen3_cuda_backend.hpp"

#ifdef MINISGL_USE_CUDA

#include <cstdint>
#include <iostream>

namespace minisgl {

Qwen3CudaBackend::Qwen3CudaBackend(const SchedulerConfig& cfg)
    : cfg_(cfg),
      ctx_(std::make_unique<vt::CudaContext>(/*dev=*/0,
                                              /*num_streams=*/2,
                                              /*num_events=*/3)),
      sched_event_(std::make_unique<CudaContextSyncEvent>(ctx_.get(),
                                                           kSchedStream, kSchedEvent)),
      engine_event_(std::make_unique<CudaContextSyncEvent>(ctx_.get(),
                                                            kEngineStream, kEngineEvent)),
      input_ids_gpu_(vt::CudaMemoryType::Device,
                     cfg.max_extend_tokens * sizeof(int32_t)),
      positions_gpu_(vt::CudaMemoryType::Device,
                     cfg.max_extend_tokens * sizeof(int64_t)),
      sampled_gpu_(vt::CudaMemoryType::Device,
                   cfg.max_running_reqs * sizeof(int32_t)) {
    std::cerr << "[qwen3_cuda] CudaContext ready: "
              << ctx_->num_streams() << " streams, "
              << ctx_->num_events() << " events, "
              << "sm_" << ctx_->sm_version() << ", "
              << ctx_->num_sms() << " SMs\n";
    std::cerr << "[qwen3_cuda] GPU buffers: "
              << "input_ids=" << input_ids_gpu_.size() << "B, "
              << "positions=" << positions_gpu_.size() << "B, "
              << "sampled="   << sampled_gpu_.size()   << "B\n";
}

Qwen3CudaBackend::~Qwen3CudaBackend() {
    // 析构前确保所有排队工作完成,避免 stream/event 句柄销毁后 CUDA 仍访问。
    if (ctx_) ctx_->sync_all();
}

ForwardOutput Qwen3CudaBackend::forward(const ForwardInput& in) {
    ctx_->set_current(kEngineStream);
    cudaStream_t stream = ctx_->current_stream();

    const size_t n_tokens = in.batch.input_ids.size();
    const size_t n_reqs   = in.batch.reqs.size();

    // ---- H2D: 输入 token ids + positions(stub kernel 不读,但跑通 plumbing)----
    if (n_tokens > 0) {
        CUDA_CHECK(cudaMemcpyAsync(input_ids_gpu_.data(),
                                    in.batch.input_ids.data(),
                                    n_tokens * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(positions_gpu_.data(),
                                    in.batch.positions.data(),
                                    n_tokens * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
    }

    // ---- STUB KERNEL ----
    // 真实 model forward + sampler 留给后续 PR。这里 cudaMemsetAsync 把 sampled_gpu_
    // 置 0,sglfront 那侧收到一串 token 0 直到 max_tokens 触发 EOS。
    // TODO: 接 model.forward() + sampler.sample() 时换掉这一段(读 qwen3::g_weights,
    // 用 input_ids_gpu_ / positions_gpu_ / 未来的 page_table_gpu_)。
    if (n_reqs > 0) {
        CUDA_CHECK(cudaMemsetAsync(sampled_gpu_.data(), 0,
                                    n_reqs * sizeof(int32_t), stream));
    }

    // ---- D2H: 每 req 一个 token,写入 pinned-backed ForwardOutput.next_tokens_cpu ----
    ForwardOutput out;
    out.next_tokens_cpu.resize(n_reqs);
    if (n_reqs > 0) {
        // next_tokens_cpu 是 pinned-backed(见 pinned_allocator.hpp),cudaMemcpyAsync 真 async。
        CUDA_CHECK(cudaMemcpyAsync(out.next_tokens_cpu.data(),
                                    sampled_gpu_.data(),
                                    n_reqs * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, stream));
    }

    // ---- Record copy_done on engine_stream ----
    // 下一次 process_last_data 会 copy_done->synchronize() 阻塞到这次 D2H 完成。
    ctx_->record_event(kCopyDoneEvent);
    out.copy_done = std::make_unique<CudaContextSyncEvent>(ctx_.get(),
                                                            kEngineStream, kCopyDoneEvent);

    return out;
}

}  // namespace minisgl

#endif  // MINISGL_USE_CUDA
