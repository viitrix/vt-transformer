#ifndef _VT_BACKEND_HPP_
#define _VT_BACKEND_HPP_

#include <atomic>
#include <string>

#include "opt/zmq.hpp"

#include "vt_endpoint.hpp"
#include "vt_scheduler.hpp"

namespace vt {

// Backend = Endpoint + FullScheduler + (caller-provided Engine)，驱动一次完整推理循环。
//
// 职责切分（对齐 mini-sglang scheduler.run_forever）：
//   - Endpoint   ：与 frontend / tokenizer 的 ZMQ 双向 IO
//   - FullScheduler  ：req 状态机、batch 调度、CacheManager 协调
//   - EngineBase ：实际 model forward（由调用方注入；测试用 DummyEngine，真实
//                  部署用 CudaEngine 等子类）
//   - Backend    ：把三者拼起来——drain event → step → reply
//
// 调用方 owns Engine；Backend owns Endpoint + FullScheduler。
// 单线程驱动——Endpoint / FullScheduler / EngineBase 都不是线程安全的，跟 mini-sglang
// scheduler 的单线程模型一致。
//
// 两种调度模式（对齐 mini-sglang scheduler.py:121-131 run_forever 的分支）：
//   - Mode::Sync   ：run_once 走 FullScheduler::step()，forward 完立刻 reply。
//                    跟 DummyEngine / MockEngine 等 sync engine 配合。
//   - Mode::Overlap：run_once 走 FullScheduler::step_overlap()——把上一次 forward
//                    的 wait+record+reply 与本次 forward 的 submit 重叠到一次调用里
//                    （详见 scheduler overlap_loop 注释）。和真实 async engine（自己
//                    重写 EngineBase::forward_async/wait 用 CUDA stream/event）配合时，
//                    能隐藏 CPU 元数据处理时延、提高 GPU 利用率。两种模式不可混用，
//                    因为 step / step_overlap 的 commit/record 时序不同——mode 在 Backend
//                    构造时定下来后整个生命周期不变。
//
// 默认 zmq 拓扑（对齐 sgl-frontend）：
//   PULL bind recv_addr（frontend connect 进来）
//   PUSH connect send_addr（detokenizer 那头 bind）
// 单元测试需要两端都 bind / 都 connect 时，靠 bind_recv / bind_send 切换。
class Backend {
public:
    // 调度模式（详见类顶部注释）。
    enum class Mode { Sync, Overlap };

    // run_once 的产出。
    struct TickOutcome {
        bool exit;     // 收到 ExitMsg——调用方应跳出循环
        bool did_work; // drained 至少一个 event，或跑了至少一次 forward batch
                       // false 表示这一轮空转，调用方可以选择 sleep
    };

    // engine 必须活到 Backend 析构之后；不会接管所有权。
    // cfg.num_pages 默认 1024 == default_max_output 会让单条 req 永远装不下，
    // 调用方应给 num_pages 留 headroom（如 16 * default_max_output）。
    Backend(zmq::context_t&  ctx,
            const std::string& recv_addr,
            const std::string& send_addr,
            SchedulerConfig<>  cfg,
            EngineBase<>*      engine,
            Mode               mode = Mode::Sync,
            bool               bind_recv = true,
            bool               bind_send = false);

    // 跑一次：drain 所有 in-flight event → 按 mode 走 step / step_overlap → reply。
    // 不 sleep——idle 策略由调用方决定（run() 会 sleep，run_once() 不会）。
    TickOutcome run_once();

    // 跑到 ExitMsg 或 stop_flag 变 true。Idle 时 sleep 1ms。
    // 用于真实长跑（vt_backend_test 之外的真实部署 / 集成测试）。
    void run(const std::atomic<bool>& stop_flag);

    // 诊断 / 测试用。
    FullScheduler<>& scheduler() { return sched_; }
    Endpoint<>&  endpoint()  { return ep_; }

private:
    // 把 scheduler 的 ReqResult 翻译成 endpoint 的 Result 批量回送。
    // 失败仅打日志，不抛——保持单线程循环不被 IO 异常打断。
    void reply(const std::vector<FullScheduler<>::ReqResult>& results);

    // 处理一个 Endpoint event。返回 false 表示收到 Exit。
    static bool handle_event(FullScheduler<>& sched, const Endpoint<>::Event& ev);

    // run_once 按 mode_ 分派到下面两个实现。
    //   Sync   ：step()——forward 完立刻 reply，简单。
    //   Overlap：step_overlap()——单次调用里既 wait 上一次 inflight（reply），
    //            又 submit 本次的 batch。冷启动（无 inflight）跳过 wait；
    //            has_work() && !has_inflight() 时再调一次 drain 最后一批。
    TickOutcome run_once_sync();
    TickOutcome run_once_overlap();

    Endpoint<>    ep_;
    EngineBase<>* engine_;   // 不拥有
    FullScheduler<>   sched_;
    Mode          mode_;
};

} // namespace vt

#endif
