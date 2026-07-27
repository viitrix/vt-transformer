#ifndef _VT_BACKEND_HPP_
#define _VT_BACKEND_HPP_

#include <atomic>
#include <string>

#include "opt/zmq.hpp"

#include "vt_endpoint.hpp"
#include "vt_scheduler.hpp"

namespace vt {

// Backend = Endpoint + Scheduler + (caller-provided Engine)，驱动一次完整推理循环。
//
// 职责切分（对齐 mini-sglang scheduler.run_forever 的 normal_loop）：
//   - Endpoint   ：与 frontend / tokenizer 的 ZMQ 双向 IO
//   - Scheduler  ：req 状态机、batch 调度、CacheManager 协调
//   - EngineBase ：实际 model forward（由调用方注入；测试用 DummyEngine，真实
//                  部署用 CudaEngine 等子类）
//   - Backend    ：把三者拼起来——drain event → step → reply
//
// 调用方 owns Engine；Backend owns Endpoint + Scheduler。
// 单线程驱动——Endpoint / Scheduler / EngineBase 都不是线程安全的，跟 mini-sglang
// scheduler 的单线程模型一致。
//
// 默认 zmq 拓扑（对齐 sgl-frontend）：
//   PULL bind recv_addr（frontend connect 进来）
//   PUSH connect send_addr（detokenizer 那头 bind）
// 单元测试需要两端都 bind / 都 connect 时，靠 bind_recv / bind_send 切换。
class Backend {
public:
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
            bool               bind_recv = true,
            bool               bind_send = false);

    // 跑一次：drain 所有 in-flight event → 若有 work 跑一步 step → reply。
    // 不 sleep——idle 策略由调用方决定（run() 会 sleep，run_once() 不会）。
    TickOutcome run_once();

    // 跑到 ExitMsg 或 stop_flag 变 true。Idle 时 sleep 1ms。
    // 用于真实长跑（vt_backend_test 之外的真实部署 / 集成测试）。
    void run(const std::atomic<bool>& stop_flag);

    // 诊断 / 测试用。
    Scheduler<>& scheduler() { return sched_; }
    Endpoint<>&  endpoint()  { return ep_; }

private:
    // 把 scheduler 的 ReqResult 翻译成 endpoint 的 Result 批量回送。
    // 失败仅打日志，不抛——保持单线程循环不被 IO 异常打断。
    void reply(const std::vector<Scheduler<>::ReqResult>& results);

    // 处理一个 Endpoint event。返回 false 表示收到 Exit。
    static bool handle_event(Scheduler<>& sched, const Endpoint<>::Event& ev);

    Endpoint<>    ep_;
    EngineBase<>* engine_;   // 不拥有
    Scheduler<>   sched_;
};

} // namespace vt

#endif
