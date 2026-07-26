// vt_endpoint_test.cpp — Dummy backend loop for sglfront interop.
//
// 启动后会：
//   1. 在 sglfront 默认地址上 PULL-bind / PUSH-connect（见下）
//   2. 循环 drain 入站事件：
//        - NewRequests → 加入 live 池池（每个给一个随机输出预算 [4,12]）
//        - Abort    → 从池中剔除
//        - Exit     → 退出循环
//   3. 每步为 live 池中每个 Request 产一个随机 token（[1,100)），预算-1；预算到 0 时
//      标记 finished=true 并移出池。
//   4. 把这一步的 Result 批量 send() 回送。
//
// 用法（默认地址与 sglfront 一致）：
//   ./vt_endpoint_test
// 自定义地址：
//   ./vt_endpoint_test ipc:///tmp/minisgl_0 ipc:///tmp/minisgl_1
//
// 编译：
//   g++ -std=c++17 -O2 -Wall -I.. -I../opt
//       vt_endpoint_test.cpp -o /tmp/vt_endpoint_test -lzmq -lpthread
//
// 配合 sglfront 跑：
//   1. 启 sglfront（默认 num_tokenizer=0 / shared 模式，会用 minisgl_0 / minisgl_1）
//   2. 启本测试
//   3. 用 HTTP 客户端打 sglfront 的 API 发请求

#include <core/vt_endpoint.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {

zmq::context_t g_ctx(1);
std::atomic<bool>  g_should_stop{false};

void on_sigint(int) { g_should_stop.store(true); }

struct LiveReq {
    vt::Endpoint<>::Req req;
    int remaining;       // 还能产几个 token；到 0 即 finished
};

// 简单 xorshift PRNG，每个 Request 一进来取一个偏移做预算随机化。
std::mt19937 g_rng{std::random_device{}()};

int randint(int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(g_rng);
}

} // anon namespace

int main(int argc, char** argv) {
    const std::string recv_addr = (argc > 1) ? argv[1] : "ipc:///tmp/minisgl_0";
    const std::string send_addr = (argc > 2) ? argv[2] : "ipc:///tmp/minisgl_1";

    std::signal(SIGINT,  on_sigint);

    // sglfront shared 模式：backend 是 PULL bind / PUSH connect，正好对应 Endpoint 默认。
    vt::Endpoint<> ep(g_ctx, recv_addr, send_addr);

    std::unordered_map<uint64_t, LiveReq> live;
    int step = 0;

    std::cerr << "vt_endpoint_test ready\n"
              << "  recv (PULL bind):    " << recv_addr << "\n"
              << "  send (PUSH connect): " << send_addr << "\n"
              << "  Ctrl-C to exit\n\n";

    while (!g_should_stop.load()) {
        bool did_work = false;

        // ---- 1. drain 入站事件 ----
        while (auto ev = ep.try_recv()) {
            did_work = true;
            if (auto* nr = std::get_if<vt::Endpoint<>::NewRequests>(&*ev)) {
                for (auto& r : nr->requests) {
                    const int budget = randint(4, 12);
                    live[r.id] = LiveReq{ std::move(r), budget };
                    std::printf("[+] new     uid=%-6llu input_len=%-4zu budget=%d\n",
                                (unsigned long long)r.id, r.input.size(), budget);
                }
            } else if (auto* a = std::get_if<vt::Endpoint<>::Abort>(&*ev)) {
                const uint64_t uid = a->uid;
                if (live.erase(uid)) {
                    std::printf("[-] abort   uid=%-6llu (removed from live)\n",
                                (unsigned long long)uid);
                }
            } else if (std::get_if<vt::Endpoint<>::Exit>(&*ev)) {
                std::printf("[x] ExitMsg received — draining\n");
                g_should_stop.store(true);
            }
        }

        // ---- 2. 推进每个 live Request 一步 ----
        if (!live.empty()) {
            ++step;
            std::vector<vt::Endpoint<>::Result> results;
            results.reserve(live.size());

            std::vector<uint64_t> finished_uids;

            for (auto& [uid, lr] : live) {
                const int tok = randint(1, 100);
                --lr.remaining;

                vt::Endpoint<>::Result r;
                r.uid        = uid;
                r.next_token = static_cast<vt::Endpoint<>::Token>(tok);
                r.finished   = (lr.remaining <= 0);
                results.push_back(r);

                if (r.finished) finished_uids.push_back(uid);
            }

            try {
                ep.send(results);
            } catch (const std::exception& e) {
                std::cerr << "send failed: " << e.what() << "\n";
            }

            std::printf("[step %4d] sent %zu result(s)", step, results.size());
            for (const auto& r : results) {
                std::printf("  uid=%-6llu tok=%-3d%s",
                           (unsigned long long)r.uid,
                           (int)r.next_token,
                           r.finished ? " (finished)" : "");
            }
            std::printf("\n");

            // 把 finished 的剔除
            for (uint64_t uid : finished_uids) live.erase(uid);
            did_work = true;
        }

        // ---- 3. 没事干就 sleep 一会儿 ----
        if (!did_work) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // 退出前把 live 池里未完成的请求收尾（不再发包，避免 tokenizer 卡死）。
    // 真实场景应由 scheduler 在收到 Exit 前自行 flush；这里只是 dummy 测试，不模拟该路径。
    std::cerr << "\nshutting down. live=" << live.size() << " unfinished\n";
    return 0;
}
