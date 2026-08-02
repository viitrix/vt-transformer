// main.cpp — Qwen3-0.6B 推理后端入口：Qwen3Engine 框架 + ZMQ 事件循环驱动。

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <opt/zmq.hpp>
#include <core/vt.hpp>
#include <core/vt_backend.hpp>
#include <core/vt_cuda.hpp>
#include <core/vt_endpoint.hpp>
#include <core/vt_pages.hpp>
#include <core/vt_request.hpp>
#include <core/vt_scheduler.hpp>

#include "qwen3.hpp"

namespace {
    // SIGINT 仅置位 stop flag；真正清理交给 main 的栈析构。
    std::atomic<bool> g_should_stop{false};
    void on_sigint(int) { g_should_stop.store(true); }

    // ZMQ → scheduler 事件循环（对齐 mini-sglang scheduler.normal_loop）：
    //   1. drain 入站：NewRequests → sched.add_req；Abort → sched.abort_req；Exit → 退出
    //   2. 推进一步：把 StepResult 翻译成 Endpoint::Result 回送 frontend
    //   3. 没活儿时短暂 sleep，避免空转
    // 单线程驱动，跟 scheduler / endpoint 的非线程安全约束一致。
    void run_event_loop(vt::FullScheduler<>& sched,
                        vt::Endpoint<>&      ep,
                        std::atomic<bool>&   should_stop) {
        while (!should_stop.load()) {
            bool did_work = false;

            // ---- 1. drain 入站事件 ----
            while (auto ev = ep.try_recv()) {
                did_work = true;
                if (auto* nr = std::get_if<vt::Endpoint<>::NewRequests>(&*ev)) {
                    for (auto& r : nr->requests) sched.add_req(std::move(r));
                } else if (auto* a = std::get_if<vt::Endpoint<>::Abort>(&*ev)) {
                    sched.abort_req(a->uid);
                } else if (std::get_if<vt::Endpoint<>::Exit>(&*ev)) {
                    should_stop.store(true);
                }
            }

            // ---- 2. 推进一步 ----
            if (sched.has_work()) {
                auto sr = sched.step();
                if (sr.ran_batch && !sr.results.empty()) {
                    std::vector<vt::Endpoint<>::Result> out;
                    out.reserve(sr.results.size());
                    for (const auto& rr : sr.results) {
                        out.push_back({rr.uid, rr.next_token, rr.finished});
                    }
                    ep.send(out);
                }
                did_work = true;
            }

            // ---- 3. 空转时让出 CPU ----
            if (!did_work) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
} // namespace

static const char* kInitPath = "dag/init.vt";

int main(int argc, char** argv) {
    // 命令行可覆盖 ZMQ 地址；默认对齐 sglfront shared 拓扑。
    //   argv[1] = recv_addr（frontend connect 进来）
    //   argv[2] = send_addr（detokenizer 那头 bind）
    const std::string recv_addr = (argc > 1) ? argv[1] : "ipc:///tmp/minisgl_0";
    const std::string send_addr = (argc > 2) ? argv[2] : "ipc:///tmp/minisgl_1";

    std::signal(SIGINT, on_sigint);

    vt::Enviroment* env = vt::create_vt_cuda(0);

    // 方便释放资源
    {
        // CUDA 推理计算执行
        qwen3::Qwen3Engine eng(*env);
        vt_assert(eng.run_dag(kInitPath), "Qwen3Engine: cannot open dag/init.vt");

        // 注册本引擎对外暴露的所有 DAG words（qwen3.prefill 等）。
        // 必须在 run_dag 之后——prefill_forward 依赖 env hash 里的 kv_cache。
        eng.register_all();

        // Engine 由外部注入：scheduler 只持有非拥有指针，不参与 engine 生命周期。
        vt::SchedulerConfig<> cfg;

        // 从 env->hash 提取 init.vt 里算出的 KV 池布局：
        //   page_size / num_pages 必须与 DAG 分配的 kv_cache 一致——
        //   否则 PageTable 引用的 page_id 会在 kv_cache 范围外，或浪费已分配的 page。
        //   其它字段（max_running_reqs / max_seq_len / max_extend_tokens / default_max_output）
        //   是调度策略选择，跟 DAG 解耦，保留 SchedulerConfig 默认值。
        auto& h = env->hash();
        cfg.page_size = static_cast<int>(h.find_number("kBlockSize"));
        cfg.num_pages = static_cast<int>(h.find_number("kNumBlocks"));

        vt::FullScheduler<> sched(cfg, &eng);

        // ZMQ 端点：Endpoint 默认 bind_recv=true / bind_send=false，对齐 sglfront
        // shared 拓扑（frontend connect 进 recv_addr；detokenizer 那头 bind send_addr）。
        zmq::context_t ctx(1);
        vt::Endpoint<> ep(ctx, recv_addr, send_addr);

        std::cerr << "qwen3-0.6b backend ready\n"
                  << "  recv (PULL bind):    " << recv_addr << "\n"
                  << "  send (PUSH connect): " << send_addr << "\n"
                  << "  Ctrl-C / ExitMsg to exit\n\n";

        run_event_loop(sched, ep, g_should_stop);
    }

    delete env;
    return 0;
}