#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <json.hpp>
#include <zmq.hpp>

#include <core/vt.hpp>
#include <core/vt_cuda.hpp>

#include "impl/qwen3.hpp"

namespace {
constexpr const char* REQ_SOCK_PATH = "ipc:///tmp/vtt_req.sock";
constexpr const char* OUT_SOCK_PATH = "ipc:///tmp/vtt_out.sock";

// SIGINT 仅置位 stop flag；真正清理交给 main 的栈析构。
std::atomic<bool> g_should_stop{false};
void on_sigint(int) { g_should_stop.store(true); }

// 与前端约定一致的 ZMQ 通道（同 backend_mock.cpp）：
//   req_sock : PULL，收前端发来的 generate / abort
//   out_sock : PUSH，向外流式发 output / error
// 按值返回给 main 持有；成员声明顺序保证 socket 先于 context 析构。
struct IpcChannels {
    zmq::context_t ctx;
    zmq::socket_t  req_sock;
    zmq::socket_t  out_sock;
};

// bind 失败（如已有实例占用 ipc 路径）经 zmq::error_t 转为 vt_panic 终止启动。
IpcChannels init_ipc() {
    IpcChannels ch;
    ch.req_sock = zmq::socket_t(ch.ctx, zmq::socket_type::pull);
    ch.out_sock = zmq::socket_t(ch.ctx, zmq::socket_type::push);

    auto bind_or_panic = [](zmq::socket_t& s, const char* path) {
        try {
            s.bind(path);
        } catch (const zmq::error_t& e) {
            vt_panic((std::string("init_ipc: bind ") + path + ": " + e.what()).c_str());
        }
    };
    bind_or_panic(ch.req_sock, REQ_SOCK_PATH);
    bind_or_panic(ch.out_sock, OUT_SOCK_PATH);

    std::printf("[backend] req=%s (PULL bind)\n[backend] out=%s (PUSH bind)\n",
                REQ_SOCK_PATH, OUT_SOCK_PATH);
    std::fflush(stdout);
    return ch;
}


// 前端未连接或缓冲已满时丢弃，避免无接收方时阻塞后端（同 backend_mock.cpp）。
void send_error(zmq::socket_t& out, int rid, const std::string& message) {
    nlohmann::json err;
    err["type"] = "error";
    err["rid"] = rid;
    err["message"] = message;
    try {
        out.send(zmq::buffer(err.dump()), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {
    }
}

// 正常输出：载荷与 qwen3::Response 一一对应，finish 为空串时序列化为 null。
// 丢弃策略同 send_error。
void send_output(zmq::socket_t& out, const qwen3::Response& r) {
    nlohmann::json msg;
    msg["type"] = "output";
    msg["rid"] = r.rid;
    msg["tokens"] = r.tokens;
    msg["finish"] = r.finish.empty() ? nlohmann::json() : nlohmann::json(r.finish);
    try {
        out.send(zmq::buffer(msg.dump()), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {
    }
}

std::optional<qwen3::Request> handle_request(zmq::socket_t& out, const std::string& raw) {
    nlohmann::json msg = nlohmann::json::parse(raw, nullptr, /*allow_exceptions=*/false);
    if (msg.is_discarded() || !msg.is_object()) {
        std::printf("[backend] 忽略非法消息: %.60s\n", raw.c_str());
        return std::nullopt;
    }

    try {
        const std::string type = msg.at("type").get<std::string>();
        const int rid = msg.at("rid").get<int>();

        if (type == "generate") {
            qwen3::Request r;
            r.rid = rid;
            r.is_abort = false;
            r.input_ids = msg.value("input_ids", nlohmann::json::array())
                              .get<std::vector<int32_t>>();
            const auto& sampling = msg.value("sampling", nlohmann::json::object());
            r.temperature    = sampling.value("temperature", 1.0f);
            r.top_p          = sampling.value("top_p", 1.0f);
            r.max_new_tokens = sampling.value("max_new_tokens", 128);
            for (const auto& t : sampling.value("stop_token_ids", nlohmann::json::array()))
                r.stop_token_ids.push_back(t.get<int32_t>());
            r.stream = msg.value("stream", true);

            // token_table 每行容量 kMaxSeqLen：空输入或超长输入无法服务，直接回 error。
            if (r.input_ids.empty() || static_cast<int>(r.input_ids.size()) >= qwen3::kMaxSeqLen) {
                send_error(out, rid,
                           "input_ids length " + std::to_string(r.input_ids.size()) +
                               " out of range [1, " + std::to_string(qwen3::kMaxSeqLen) + ")");
                return std::nullopt;
            }
            return r;
        } else if (type == "abort") {
            qwen3::Request r;
            r.rid = rid;
            r.is_abort = true;
            return r;
        } else {
            std::printf("[backend] 未知消息类型: %s\n", type.c_str());
        }
    } catch (const nlohmann::json::exception& e) {
        std::printf("[backend] 消息缺少必要字段或类型错误，已忽略: %s\n", e.what());
    }
    return std::nullopt;
}

// 核心处理逻辑定义
void run_event_loop(qwen3::Qwen3Engine& eng, IpcChannels& ipc, std::atomic<bool>& should_stop) {
    while (!should_stop.load()) {
        // 非阻塞查询：有消息就 drain，没有立刻返回，不在 token 之间引入固定等待。
        // 是否空闲由 process_last 的返回值表达：nullopt 表示本轮无事可做，调用方小睡避免忙等。
        std::vector<qwen3::Request> incoming;
        zmq::message_t msg;
        while (ipc.req_sock.recv(msg, zmq::recv_flags::dontwait).has_value()) {
            if (auto fresh = handle_request(ipc.out_sock, msg.to_string()))
                incoming.push_back(std::move(*fresh));
        }
        std::fflush(stdout);

        if (incoming.size() > 0) {
            const int admitted = eng.forward(incoming);
            // 槽位满：直接拒绝，不排队，由客户端看 error 消息决定是否重试。
            for (size_t i = static_cast<size_t>(admitted); i < incoming.size(); ++i) {
                send_error(ipc.out_sock, incoming[i].rid,
                           "all " + std::to_string(qwen3::kMaxRunningReqs) + " KV slots busy");
            }
        }
        if (auto resp = eng.process_last()) {
            send_output(ipc.out_sock, *resp);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);

    vt::Enviroment* env = vt::create_vt_cuda(0);
    {
        qwen3::Qwen3Engine eng(env);
        eng.init();

        IpcChannels ipc = init_ipc();
        run_event_loop(eng, ipc, g_should_stop);
    }

    delete env;
    return 0;
}
