// vt_backend.cpp — Backend 实现（推理循环驱动器）。
//
// 详见 vt_backend.hpp 顶部注释中的职责切分与主循环语义。

#include "vt_backend.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace vt {

Backend::Backend(zmq::context_t&   ctx,
                 const std::string& recv_addr,
                 const std::string& send_addr,
                 SchedulerConfig<> cfg,
                 EngineBase<>*     engine,
                 Mode              mode,
                 bool              bind_recv,
                 bool              bind_send)
    : ep_(ctx, recv_addr, send_addr, bind_recv, bind_send),
      engine_(engine),
      sched_(std::move(cfg), engine_),
      mode_(mode) {
    vt_assert(engine != nullptr, "Backend: engine must not be null");
}

Backend::TickOutcome Backend::run_once() {
    return mode_ == Mode::Overlap ? run_once_overlap() : run_once_sync();
}

Backend::TickOutcome Backend::run_once_sync() {
    bool did_work = false;

    // 1. drain 入站事件——非阻塞，有多少取多少
    while (auto ev = ep_.try_recv()) {
        did_work = true;
        if (!handle_event(sched_, *ev)) {
            return {/*exit=*/true, did_work};
        }
    }

    // 2. 有 work 就跑一步。step() 走 sync 路径，forward 完拿到结果就 reply。
    if (sched_.has_work()) {
        auto sr = sched_.step();
        if (sr.ran_batch) {
            reply(sr.results);
            did_work = true;
        }
    }

    return {/*exit=*/false, did_work};
}

Backend::TickOutcome Backend::run_once_overlap() {
    bool did_work = false;

    // 1. drain 入站事件——非阻塞，有多少取多少（与 sync 一致）
    while (auto ev = ep_.try_recv()) {
        did_work = true;
        if (!handle_event(sched_, *ev)) {
            return {/*exit=*/true, did_work};
        }
    }

    // 2. 有 work 或有 in-flight 时跑一步 step_overlap。
    //    - has_work()       ：有 req 等 prefill / decode——phase 2 会 submit
    //    - has_inflight()   ：上次 step_overlap 提交的 forward 还没被 phase 1 处理——
    //                         本次 phase 1 会 wait + record + reply
    //    两者皆空 = 真正 idle，让上层 sleep（对齐 mini-sglang overlap_loop 里
    //    `blocking = not (last_data or runnable)` 的判断）。
    if (sched_.has_work() || sched_.has_inflight()) {
        auto sr = sched_.step_overlap();

        // phase 1 产出：上一批 inflight 的 reply——立即回送。
        if (!sr.results.empty()) {
            reply(sr.results);
            did_work = true;
        }
        // phase 2 产出：本次 submit 了一个新 forward——reply 要等下一步的 phase 1。
        // 算作 did_work 以免上层在 async engine 等待期间空转 sleep。
        if (sr.ran_batch) {
            did_work = true;
        }
    }

    return {/*exit=*/false, did_work};
}

void Backend::run(const std::atomic<bool>& stop_flag) {
    while (!stop_flag.load()) {
        auto r = run_once();
        if (r.exit) return;
        if (!r.did_work) {
            // 空转：歇 1ms 别把 CPU 烧满。下一轮 run_once 会立刻处理新事件。
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Backend::reply(const std::vector<FullScheduler<>::ReqResult>& results) {
    if (results.empty()) return;
    std::vector<Endpoint<>::Result> out;
    out.reserve(results.size());
    for (const auto& r : results) {
        out.push_back({r.uid, r.next_token, r.finished});
    }
    try {
        ep_.send(out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vt_backend: send failed: %s\n", e.what());
    }
}

bool Backend::handle_event(FullScheduler<>& sched, const Endpoint<>::Event& ev) {
    if (auto* nr = std::get_if<Endpoint<>::NewRequests>(&ev)) {
        for (auto& r : nr->requests) {
            // Request 默认 state=Waiting / output 空，符合 add_req 的契约。
            // ev 是 const&，r 是 const Req&——move ctor 不会被选中（const rvalue
            // 回退到 copy），所以这里不写 std::move。
            sched.add_req(r);
        }
        return true;
    }
    if (auto* a = std::get_if<Endpoint<>::Abort>(&ev)) {
        sched.abort_req(a->uid);
        return true;
    }
    if (std::get_if<Endpoint<>::Exit>(&ev)) {
        return false;
    }
    return true;  // 不可能到这（Endpoint::decode 失败会抛）
}

} // namespace vt
