// sync.hpp — SyncEvent abstraction for scheduler/engine stream overlap.
//
// Overlap loop 的核心是两条 stream:scheduler stream(CPU 准备 Req / 写 page_table
// 等)与 engine stream(GPU forward)。两条 stream 通过 event 同步:
//   * sched_stream.record() —— Scheduler CPU 活干完时,sched 上 record 一个 event
//   * engine_stream.wait_on(sched_stream) —— engine 排队等 sched 的 event,
//     保证 forward 启动前 scheduler 的写入(若 sched_stream 上有 CUDA 工作的话)已可见
//   * forward 内部 record 一个 copy_done event,GPU→CPU 拷贝完成时触发
//   * 下一轮 _process_last_data 调 copy_done.synchronize() 等 token 可读
//
// 接口语义:
//   * record()         在本 event 所属 stream 上 record event_idx
//   * wait_on(other)   让本 event 所属 stream 等 other 的最近一次 record
//                      (对应 cudaStreamWaitEvent(my_stream, other_event, 0))
//   * synchronize()    host 阻塞直到本 event 完成
//
// 复用 vt::CudaContext 已有的 stream/event 池,不重新发明抽象。

#pragma once

#include <utility>

#ifdef MINISGL_USE_CUDA
#include "vt_cuda.hpp"
#endif

namespace minisgl {

class SyncEvent {
public:
    virtual ~SyncEvent() = default;
    // 在本 event 所属 stream 上 record event_idx。
    virtual void record() {}
    // 让本 event 所属 stream 等待 other 最近一次 record(跨 stream 同步原语,
    // 对应 cudaStreamWaitEvent(my_stream, other_event, 0))。
    virtual void wait_on(SyncEvent* /*other*/) {}
    // host 阻塞直到本 event 完成。
    virtual void synchronize() {}
};

#ifdef MINISGL_USE_CUDA
// 包一层 vt::CudaContext 的 (stream_idx, event_idx) 二元组。
// 不持有 CudaContext 的生命周期(由调用方保证 ctx 存活)。
// 同一 backend 内所有 CudaContextSyncEvent 应共享同一 ctx(否则 wait_on 跨 ctx 无意义)。
struct CudaContextSyncEvent final : SyncEvent {
    vt::CudaContext* ctx       = nullptr;
    size_t           stream_idx = 0;
    size_t           event_idx  = 0;

    CudaContextSyncEvent(vt::CudaContext* c, size_t s, size_t e)
        : ctx(c), stream_idx(s), event_idx(e) {}

    void record() override {
        if (!ctx) return;
        ctx->set_current(stream_idx);
        ctx->record_event(event_idx);
    }
    void wait_on(SyncEvent* other) override {
        if (!ctx || !other) return;
        // 假设 other 也是 CudaContextSyncEvent 且共享同一 ctx(单 backend 内必然成立)。
        auto* o = dynamic_cast<CudaContextSyncEvent*>(other);
        if (!o || !o->ctx) return;
        ctx->set_current(stream_idx);    // 本 event 的 stream 是等待方
        ctx->wait_event(o->event_idx);   // cudaStreamWaitEvent(my_stream, other_event, 0)
    }
    void synchronize() override {
        if (!ctx) return;
        ctx->sync_event(event_idx);
    }
};
#endif

}  // namespace minisgl
