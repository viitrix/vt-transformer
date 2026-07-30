// vt_backend_test.cpp — 端到端集成测试 for vt::Backend。
//
// 拓扑（两端都在测试进程里）：
//   peer PUSH  -- connect --> recv_addr  <-- PULL bind --  Backend
//   peer PULL  <-- bind ----  send_addr  >-- PUSH connect -- Backend
//
// 场景覆盖：
//   1. 单条 UserMsg → 一条 DetokenizeMsg reply（prefill step）
//   2. BatchBackendMsg 携带 2 条 UserMsg → 2 条 reply（BatchTokenizerMsg 包裹）
//   3. 多步 decode：同一 req 连续两轮 run_once 各收一条 reply，finished 都为 false
//   4. AbortBackendMsg：send → 一轮 reply → abort → 之后 run_once 不再为它产 reply
//   5. ExitMsg：run_once().exit == true
//   6. run() 空闲：没有事件、没有 work 时 stop_flag 能干净退出
//
// 编译见 ../test/Makefile（`make vt_backend_test`）。
// 该测试自终止——`make check` 会跑。

#include <core/vt_backend.hpp>
#include <core/vt_endpoint.hpp>
#include <core/vt_scheduler.hpp>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "opt/msgpack.hpp"
#include "opt/zmq.hpp"

namespace {

using namespace std::chrono_literals;

int g_failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n",                       \
                         __func__, __LINE__, msg);                          \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

// ---- 测试用 DummyEngine：每个 req 给一个 [1,100) 的随机 token。----
// 跟生产 path 同形——证明 Backend 接受任意 EngineBase 子类。
class DummyEngine : public vt::EngineBase<> {
public:
    vt::ForwardOutput<> forward(const vt::Batch<>& batch) override {
        vt::ForwardOutput<> out;
        out.next_tokens.reserve(batch.reqs.size());
        std::uniform_int_distribution<int> d(1, 99);
        for (std::size_t i = 0; i < batch.reqs.size(); ++i) {
            out.next_tokens.push_back(static_cast<vt::FullScheduler<>::Token>(d(rng_)));
        }
        return out;
    }
    int eos_token_id() const override { return -1; }

private:
    std::mt19937 rng_{std::random_device{}()};
};

// ---- 测试 fixture：建好 Backend + 配对 socket ----
// 拓扑（与 vt::Backend 默认对齐）：
//   Backend.in_ (PULL bind)  ← push (connect)
//   Backend.out_(PUSH conn)  → pull (bind)
// 顺序很重要：Backend 先 bind，peer 后 connect —— 避免 PUSH 端在 bind 完成前
// 把第一条消息丢掉（PUSH socket 在没有 connected peer 时会丢弃）。
struct Peer {
    zmq::socket_t push;  // connect recv_addr → 喂 UserMsg 给 Backend
    zmq::socket_t pull;  // bind send_addr    ← 收 Backend 的 DetokenizeMsg

    Peer(zmq::context_t& ctx,
         const std::string& recv_addr,
         const std::string& send_addr)
        : push(ctx, zmq::socket_type::push),
          pull(ctx, zmq::socket_type::pull) {
        push.connect(recv_addr);
        pull.bind(send_addr);
        // 给 zmq 一点时间 handshake——两端 bind/connect 都已就位。
        std::this_thread::sleep_for(100ms);
    }
};

// 共享 context + 创建顺序封装：先 Backend，再 Peer，最后 settle。
// 每个 scenario 用独立 recv/send 地址，避免互相干扰。
struct Fixture {
    zmq::context_t ctx{1};
    std::string    recv_addr;
    std::string    send_addr;
    std::unique_ptr<DummyEngine>            engine;
    std::unique_ptr<vt::Backend>            backend;
    std::unique_ptr<Peer>                   peer;

    Fixture(int n, vt::Backend::Mode mode = vt::Backend::Mode::Sync)
        : recv_addr("ipc:///tmp/vtbe_test_in_"  + std::to_string(n)),
          send_addr("ipc:///tmp/vtbe_test_out_" + std::to_string(n)),
          engine(std::make_unique<DummyEngine>()) {
        vt::SchedulerConfig<> cfg;
        cfg.num_pages = 16 * cfg.default_max_output;
        backend = std::make_unique<vt::Backend>(ctx, recv_addr, send_addr,
                                                 cfg, engine.get(), mode);
        peer = std::make_unique<Peer>(ctx, recv_addr, send_addr);
    }
};

// DetokenizeMsg / BatchTokenizerMsg 的极简解析——只关心 (uid, next_token, finished) 三元组。
struct Reply {
    uint64_t uid;
    int64_t  next_token;
    bool     finished;
};

// msgpack pack 辅助
void pk_str(msgpack::packer<msgpack::sbuffer>& pk, const char* s) {
    pk.pack_str(static_cast<uint32_t>(std::strlen(s)));
    pk.pack_str_body(s, std::strlen(s));
}

// 构造一条 UserMsg 的 wire 字节。
msgpack::sbuffer build_user_msg(uint64_t uid, const std::vector<int32_t>& tokens) {
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk(&buf);
    pk.pack_map(3);
    pk_str(pk, "__type__"); pk_str(pk, "UserMsg");
    pk_str(pk, "uid");      pk.pack_uint64(uid);
    pk_str(pk, "input_ids");
    pk.pack_map(3);
    pk_str(pk, "__type__"); pk_str(pk, "Tensor");
    pk_str(pk, "dtype");    pk_str(pk, "torch.int32");
    pk_str(pk, "buffer");
    pk.pack_bin(tokens.size() * sizeof(int32_t));
    pk.pack_bin_body(reinterpret_cast<const char*>(tokens.data()),
                     tokens.size() * sizeof(int32_t));
    return buf;
}

msgpack::sbuffer build_abort_msg(uint64_t uid) {
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk(&buf);
    pk.pack_map(2);
    pk_str(pk, "__type__"); pk_str(pk, "AbortBackendMsg");
    pk_str(pk, "uid");      pk.pack_uint64(uid);
    return buf;
}

msgpack::sbuffer build_exit_msg() {
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk(&buf);
    pk.pack_map(1);
    pk_str(pk, "__type__"); pk_str(pk, "ExitMsg");
    return buf;
}

msgpack::sbuffer build_batch_backend_msg(const std::vector<msgpack::sbuffer>& user_msgs) {
    // BatchBackendMsg = { __type__: "BatchBackendMsg", data: [<UserMsg>, ...] }
    // data 里每个元素是已打包好的 UserMsg——重新解析然后 repack 太麻烦，
    // 这里直接重新构造一遍（每个 UserMsg 是 map(3)，结构稳定）。
    msgpack::sbuffer buf;
    msgpack::packer<msgpack::sbuffer> pk(&buf);
    pk.pack_map(2);
    pk_str(pk, "__type__"); pk_str(pk, "BatchBackendMsg");
    pk_str(pk, "data");
    pk.pack_array(user_msgs.size());
    for (const auto& um : user_msgs) {
        // 把每条 UserMsg 的字节当作 raw msgpack object 嵌入。
        // 用 unpack→repack 的笨办法重新序列化进 data 字段——量小，可以接受。
        auto oh = msgpack::unpack(um.data(), um.size());
        pk << oh.get();
    }
    return buf;
}

void send_raw(zmq::socket_t& s, const msgpack::sbuffer& buf) {
    s.send(zmq::message_t(buf.data(), buf.size()), zmq::send_flags::none);
    // PUSH→PULL 在 ipc:// 上是异步投递——send() 返回时消息可能还在 PUSH 端 outgoing
    // 队列里没到 PULL。给 zmq IO 线程一点时间，否则紧跟着的 run_once 会 drain 不到。
    std::this_thread::sleep_for(10ms);
}

// 从 peer.pull 取一条 DetokenizeMsg / BatchTokenizerMsg 并展开成 Reply 列表。
// timeout_ms 内没消息返回空（用于断言"不应再收到 reply"）。
std::vector<Reply> recv_replies(zmq::socket_t& s, int timeout_ms) {
    std::vector<Reply> out;
    zmq::message_t frame;
    zmq_pollitem_t item{};
    item.socket = static_cast<void*>(s);
    item.events = ZMQ_POLLIN;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        int rc = zmq_poll(&item, 1, 10);  // 10ms 一次
        if (rc < 0) {
            std::fprintf(stderr, "recv_replies: zmq_poll rc=%d errno=%d\n", rc, errno);
            return out;
        }
        if (rc > 0 && (item.revents & ZMQ_POLLIN)) {
            if (s.recv(frame, zmq::recv_flags::dontwait).has_value()) break;
        }
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() >= timeout_ms) {
            return out;  // 超时，没消息
        }
    }

    auto oh = msgpack::unpack(static_cast<const char*>(frame.data()), frame.size());
    const msgpack::object& top = oh.get();
    if (top.type != msgpack::type::MAP) return out;

    // 找 __type__
    std::string tname;
    for (uint32_t i = 0; i < top.via.map.size; ++i) {
        const auto& k = top.via.map.ptr[i].key;
        if (k.type == msgpack::type::STR
            && std::string_view(k.via.str.ptr, k.via.str.size) == "__type__") {
            const auto& v = top.via.map.ptr[i].val;
            tname.assign(v.via.str.ptr, v.via.str.size);
            break;
        }
    }

    // 解 DetokenizeMsg（单条）或 BatchTokenizerMsg（数组）
    auto parse_detok = [&](const msgpack::object& o, std::vector<Reply>& sink) {
        Reply r{};
        for (uint32_t i = 0; i < o.via.map.size; ++i) {
            const auto& k = o.via.map.ptr[i].key;
            const auto& v = o.via.map.ptr[i].val;
            std::string_view key(k.via.str.ptr, k.via.str.size);
            if      (key == "uid")        r.uid = v.via.u64;
            else if (key == "next_token") r.next_token = v.via.i64;
            else if (key == "finished")   r.finished = v.via.boolean;
        }
        sink.push_back(r);
    };

    if (tname == "DetokenizeMsg") {
        parse_detok(top, out);
    } else if (tname == "BatchTokenizerMsg") {
        // 找 data
        for (uint32_t i = 0; i < top.via.map.size; ++i) {
            const auto& k = top.via.map.ptr[i].key;
            if (k.type == msgpack::type::STR
                && std::string_view(k.via.str.ptr, k.via.str.size) == "data") {
                const auto& arr = top.via.map.ptr[i].val;
                for (uint32_t j = 0; j < arr.via.array.size; ++j) {
                    parse_detok(arr.via.array.ptr[j], out);
                }
                break;
            }
        }
    }
    return out;
}

// ============================================================================
// 场景 1：单条 UserMsg → 一条 reply
// ============================================================================
void test_single_user_msg() {
    Fixture f(1);
    send_raw(f.peer->push, build_user_msg(42, {10, 20, 30, 40}));
    auto out = f.backend->run_once();
    CHECK(!out.exit, "run_once should not exit on UserMsg");
    CHECK(out.did_work, "run_once should report did_work after prefill");

    auto replies = recv_replies(f.peer->pull, 500);
    CHECK(replies.size() == 1, "expected 1 reply for single UserMsg");
    CHECK(replies[0].uid == 42, "reply uid mismatch");
    CHECK(replies[0].next_token >= 1 && replies[0].next_token <= 99,
          "next_token should be in DummyEngine's [1,99]");
    CHECK(!replies[0].finished, "should not finish after 1 step (max_output=1024)");

    send_raw(f.peer->push, build_exit_msg());
    CHECK(f.backend->run_once().exit, "run_once should return exit=true after ExitMsg");
}

// ============================================================================
// 场景 2：BatchBackendMsg 携带 2 条 → 2 条 reply（BatchTokenizerMsg 包裹）
// ============================================================================
void test_batch_user_msgs() {
    Fixture f(2);
    std::vector<msgpack::sbuffer> msgs;
    msgs.push_back(build_user_msg(100, {1, 2, 3}));
    msgs.push_back(build_user_msg(200, {4, 5, 6, 7}));
    send_raw(f.peer->push, build_batch_backend_msg(msgs));

    auto out = f.backend->run_once();
    CHECK(!out.exit && out.did_work, "run_once should report did_work for batch");

    auto replies = recv_replies(f.peer->pull, 500);
    CHECK(replies.size() == 2, "expected 2 replies for batch of 2");
    bool has100 = false, has200 = false;
    for (const auto& r : replies) {
        if (r.uid == 100) has100 = true;
        if (r.uid == 200) has200 = true;
    }
    CHECK(has100 && has200, "expected both uids in batch reply");

    send_raw(f.peer->push, build_exit_msg());
    CHECK(f.backend->run_once().exit, "ExitMsg should set exit=true");
}

// ============================================================================
// 场景 3：同一 req 连续两轮 decode，每轮各一条 reply，都不 finished
// ============================================================================
void test_multi_step_decode() {
    Fixture f(3);
    send_raw(f.peer->push, build_user_msg(7, {1, 2}));

    f.backend->run_once();  // prefill
    auto r1 = recv_replies(f.peer->pull, 500);
    CHECK(r1.size() == 1 && r1[0].uid == 7, "prefill reply missing");

    f.backend->run_once();  // decode step 1
    auto r2 = recv_replies(f.peer->pull, 500);
    CHECK(r2.size() == 1 && r2[0].uid == 7 && !r2[0].finished,
          "decode step 1 reply missing");

    f.backend->run_once();  // decode step 2
    auto r3 = recv_replies(f.peer->pull, 500);
    CHECK(r3.size() == 1 && r3[0].uid == 7 && !r3[0].finished,
          "decode step 2 reply missing");

    send_raw(f.peer->push, build_exit_msg());
    CHECK(f.backend->run_once().exit, "ExitMsg should set exit=true");
}

// ============================================================================
// 场景 4：AbortBackendMsg 中止一个 req——之后 run_once 不再为它产 reply
// ============================================================================
void test_abort() {
    Fixture f(4);
    send_raw(f.peer->push, build_user_msg(99, {1, 2, 3}));
    f.backend->run_once();  // prefill
    auto r1 = recv_replies(f.peer->pull, 500);
    CHECK(r1.size() == 1 && r1[0].uid == 99, "prefill reply missing");

    send_raw(f.peer->push, build_abort_msg(99));
    auto out = f.backend->run_once();  // 处理 abort
    CHECK(!out.exit, "Abort should not cause exit");

    auto r2 = recv_replies(f.peer->pull, 200);
    CHECK(r2.empty(), "should not receive any reply after abort");

    f.backend->run_once();
    auto r3 = recv_replies(f.peer->pull, 200);
    CHECK(r3.empty(), "aborted req should stay gone");

    send_raw(f.peer->push, build_exit_msg());
    CHECK(f.backend->run_once().exit, "ExitMsg should set exit=true");
}

// ============================================================================
// 场景 5：run() 空闲能干净退出（stop_flag 触发）
// ============================================================================
void test_run_idle_exits_on_flag() {
    Fixture f(5);
    std::atomic<bool> stop{false};
    std::thread stopper([&]() {
        std::this_thread::sleep_for(50ms);
        stop.store(true);
    });
    f.backend->run(stop);
    stopper.join();
    CHECK(stop.load(), "run() should observe stop_flag");

    auto out = f.backend->run_once();
    CHECK(!out.exit, "run_once after run() should still work");
}

// ============================================================================
// 场景 6：run() 收到 ExitMsg 后干净退出
// ============================================================================
void test_run_exits_on_exit_msg() {
    Fixture f(6);
    std::thread sender([&]() {
        std::this_thread::sleep_for(50ms);
        send_raw(f.peer->push, build_exit_msg());
    });

    std::atomic<bool> stop{false};
    f.backend->run(stop);
    sender.join();
    CHECK(true, "run() returned from ExitMsg without crashing");
}

// ============================================================================
// 场景 7：Overlap 模式——同一条 req 走完 prefill + 多步 decode + abort 收尾
// ============================================================================
// Overlap 与 sync 在驱动侧的可见差别：submit 与 wait 错位一步——首次 run_once 只
// submit prefill，prefill 的 reply 要等到第二次 run_once 的 phase 1 才回。
// 此处验证：连续 run_once 能在 Overlap 模式下稳定产出 reply；abort 后能干净收尾，
// req 被释放后再 tick 不会产 phantom reply。
void test_overlap_mode() {
    Fixture f(7, vt::Backend::Mode::Overlap);
    send_raw(f.peer->push, build_user_msg(5, {1, 2, 3}));

    // tick 1：submit prefill（async engine 下 GPU 开始算；DummyEngine 立即完成）
    auto o1 = f.backend->run_once();
    CHECK(!o1.exit && o1.did_work, "overlap tick 1 should submit prefill");

    // tick 2：phase 1 drain prefill → reply；phase 2 submit decode 1
    f.backend->run_once();
    auto r1 = recv_replies(f.peer->pull, 500);
    CHECK(r1.size() == 1 && r1[0].uid == 5 && !r1[0].finished,
          "overlap: prefill reply missing");

    // tick 3：drain decode 1 → reply；submit decode 2
    f.backend->run_once();
    auto r2 = recv_replies(f.peer->pull, 500);
    CHECK(r2.size() == 1 && r2[0].uid == 5, "overlap: decode 1 reply missing");

    // 此时 req 5 仍在 running（DummyEngine 不发 EOS，默认 max_output=1024 不会触顶）。
    // abort 它：tick 4 的 drain 阶段先 mark Finished（abort_req 走 inflight 分支），
    // 同一 tick 的 step_overlap phase 1 drain decode 2 时走 abort 分支
    // （process_inflight_results 见 state==Finished → 直接 free_finished，不 record）。
    send_raw(f.peer->push, build_abort_msg(5));
    f.backend->run_once();   // drain abort + drain inflight + free_finished
    auto r3 = recv_replies(f.peer->pull, 500);
    CHECK(r3.size() == 1 && r3[0].uid == 5,
          "overlap: should drain one final reply for aborted req");

    // 现在 pending/running/inflight 都空——真正 idle。再 tick 不应有 phantom reply。
    f.backend->run_once();
    auto r4 = recv_replies(f.peer->pull, 200);
    CHECK(r4.empty(), "overlap: should not produce reply when truly idle");

    send_raw(f.peer->push, build_exit_msg());
    CHECK(f.backend->run_once().exit, "ExitMsg should set exit=true");
}

} // anon namespace

int main() {
    using TestFn = void (*)();
    TestFn tests[] = {
        test_single_user_msg,
        test_batch_user_msgs,
        test_multi_step_decode,
        test_abort,
        test_run_idle_exits_on_flag,
        test_run_exits_on_exit_msg,
        test_overlap_mode,
    };

    for (TestFn t : tests) {
        t();
    }

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED (%zu)\n", sizeof(tests) / sizeof(tests[0]));
        return 0;
    }
    std::printf("%d FAILURE(S)\n", g_failures);
    return 1;
}
