// encoding: utf-8
//
// 推理后端 mock：实现与真实引擎一致的 ZMQ 消息协议，用于在没有
// GPU 与模型权重的情况下调试 http_main.py / cli_main.py 前端。
//
// 协议 (JSON over ipc://):
//   req 通道 (PULL bind)，前端 -> mock:
//     {"type":"generate","rid":N,"input_ids":[...],
//      "sampling":{"temperature":..,"top_p":..,"max_new_tokens":M,"stop_token_ids":[..]},
//      "stream":bool}
//     {"type":"abort","rid":N}
//   out 通道 (PUSH bind)，mock -> 前端:
//     {"type":"output","rid":N,"tokens":[..],"finish":"stop"|"length"|null}
//     {"type":"error","rid":N,"message":".."}
//
// 生成策略：逐 token 流式回放一段固定回复文本（含多字节字符与英文数字，
// 用于验证增量 detokenize 与流式输出）；达到 max_new_tokens 上限以
// finish=length 结束，命中 stop_token_ids 以 finish=stop 结束。
// 错误注入（固定行为，用于调试前端错误路径）: 前 2 个请求正常完成后，
// 第 3 个请求在输出 5 个 token 后以 error 消息结束，仅注入一次。
//
// 用法: ./mock [tick_ms]    每个 tick 向所有活跃请求各输出一个 token，默认 50ms

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <json.hpp>
#include <zmq.hpp>

namespace {

constexpr const char* REQ_SOCK_PATH = "ipc:///tmp/vtt_req.sock";
constexpr const char* OUT_SOCK_PATH = "ipc:///tmp/vtt_out.sock";
constexpr int DEFAULT_TICK_MS = 50;

// 固定回复文本经 Qwen3-0.6B tokenizer 编码的 token 序列，共 56 个，
// 文本: "你好！我是 mock 推理后端，正在逐 token 流式回放一段固定文本：The quick
// brown fox jumps over the lazy dog. 0123456789 用于验证增量解码与多字节字符补齐。"
constexpr int REPLY_TOKENS[] = {
    108386, 6313,  104198, 7860,  46602, 101,   21887, 33447, 78882, 3837,
    96555,  100062, 3950,   98313, 223,   28330, 18397, 53222, 104383, 101358,
    108704, 5122,  785,    3974,  13876, 38835, 34208, 916,   279,   15678,
    5562,   13,    220,    15,    16,    17,    18,    19,    20,    21,
    22,     23,    24,     220,   100751, 48927, 111134, 49238, 16476, 57218,
    42140,  18600, 55502,  48391, 117698, 1773,
};
constexpr int REPLY_TOTAL = static_cast<int>(sizeof(REPLY_TOKENS) / sizeof(REPLY_TOKENS[0]));

volatile sig_atomic_t g_running = 1;
void handle_sigint(int) { g_running = 0; }

// 固定错误注入策略: 前 NORMAL_REQUESTS_BEFORE_ERROR 个请求正常完成后，
// 下一个请求输出 ERROR_EMITTED_TOKENS 个 token 后以 error 消息结束（仅注入一次）
constexpr int NORMAL_REQUESTS_BEFORE_ERROR = 2;
constexpr int ERROR_EMITTED_TOKENS = 5;
int g_completed_requests = 0;  // 已正常结束（finish 而非 abort）的请求数
bool g_error_injected = false;

struct Request {
    int rid = 0;
    int emitted = 0;  // 已回放的 token 数
    int max_new_tokens = 128;
    std::vector<int> stop_ids;
    bool inject_error = false;  // 该请求在输出若干 token 后以 error 结束
};

bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// 发送该请求的下一条输出消息；返回 false 表示请求已结束
bool emit_step(zmq::socket_t& out, Request& r) {
    if (r.inject_error && r.emitted >= ERROR_EMITTED_TOKENS) {
        nlohmann::json err;
        err["type"] = "error";
        err["rid"] = r.rid;
        err["message"] = "mock 模拟错误: 输出 " + std::to_string(r.emitted) + " 个 token 后中断";
        try {
            out.send(zmq::buffer(err.dump()), zmq::send_flags::dontwait);
        } catch (const zmq::error_t&) {
        }
        std::printf("[mock] rid=%d error (after %d tokens)\n", r.rid, r.emitted);
        return false;
    }

    std::vector<int> batch;
    const char* finish = nullptr;

    if (r.emitted >= REPLY_TOTAL || r.emitted >= r.max_new_tokens) {
        // 没有可回放的 token（空输入语义或上限为 0），只发结束标记
        finish = (r.max_new_tokens < REPLY_TOTAL) ? "length" : "stop";
    } else {
        int tok = REPLY_TOKENS[r.emitted++];
        batch.push_back(tok);
        bool hit_stop = contains(r.stop_ids, tok);
        if (hit_stop || r.emitted >= REPLY_TOTAL || r.emitted >= r.max_new_tokens) {
            finish = (hit_stop || r.emitted >= REPLY_TOTAL) ? "stop" : "length";
        }
    }

    nlohmann::json msg;
    msg["type"] = "output";
    msg["rid"] = r.rid;
    msg["tokens"] = batch;
    msg["finish"] = finish != nullptr ? nlohmann::json(finish) : nlohmann::json();
    // 前端未连接或缓冲已满时丢弃，避免无接收方时阻塞 mock
    try {
        out.send(zmq::buffer(msg.dump()), zmq::send_flags::dontwait);
    } catch (const zmq::error_t&) {
    }

    if (finish != nullptr) {
        std::printf("[mock] rid=%d finish=%s emitted=%d\n", r.rid, finish, r.emitted);
        ++g_completed_requests;
        return false;
    }
    return true;
}

void handle_request(zmq::socket_t& out, const std::string& raw, std::vector<Request>& active) {
    nlohmann::json msg = nlohmann::json::parse(raw, nullptr, /*allow_exceptions=*/false);
    if (msg.is_discarded() || !msg.is_object()) {
        std::printf("[mock] 忽略非法消息: %.60s\n", raw.c_str());
        return;
    }

    try {
        const std::string type = msg.at("type").get<std::string>();
        const int rid = msg.at("rid").get<int>();

        if (type == "generate") {
            Request r;
            r.rid = rid;
            const auto& sampling = msg.value("sampling", nlohmann::json::object());
            r.max_new_tokens = sampling.value("max_new_tokens", 128);
            for (const auto& t : sampling.value("stop_token_ids", nlohmann::json::array()))
                r.stop_ids.push_back(t.get<int>());

            // 前固定数量的请求正常完成后，下一个请求注入模拟错误（仅一次）
            r.inject_error = (!g_error_injected &&
                              g_completed_requests == NORMAL_REQUESTS_BEFORE_ERROR);
            if (r.inject_error) g_error_injected = true;

            auto same_rid = [rid](const Request& x) { return x.rid == rid; };
            active.erase(std::remove_if(active.begin(), active.end(), same_rid), active.end());
            active.push_back(std::move(r));
            std::printf("[mock] generate rid=%d input_tokens=%zu max_new_tokens=%d%s\n", rid,
                        msg.value("input_ids", nlohmann::json::array()).size(),
                        active.back().max_new_tokens,
                        active.back().inject_error ? " (该请求将模拟错误)" : "");
        } else if (type == "abort") {
            auto same_rid = [rid](const Request& x) { return x.rid == rid; };
            active.erase(std::remove_if(active.begin(), active.end(), same_rid), active.end());
            std::printf("[mock] abort rid=%d\n", rid);
        } else {
            std::printf("[mock] 未知消息类型: %s\n", type.c_str());
        }
    } catch (const nlohmann::json::exception& e) {
        std::printf("[mock] 消息缺少必要字段或类型错误，已忽略: %s\n", e.what());
    }
}

}  // namespace

int main(int argc, char** argv) {
    int tick_ms = (argc > 1) ? std::max(std::atoi(argv[1]), 0) : DEFAULT_TICK_MS;

    zmq::context_t ctx;
    zmq::socket_t req_sock(ctx, zmq::socket_type::pull);
    req_sock.bind(REQ_SOCK_PATH);
    zmq::socket_t out_sock(ctx, zmq::socket_type::push);
    out_sock.bind(OUT_SOCK_PATH);

    std::signal(SIGINT, handle_sigint);
    std::printf("[mock] req=%s (PULL bind)\n[mock] out=%s (PUSH bind)\n[mock] tick=%dms, Ctrl+C 退出\n",
                REQ_SOCK_PATH, OUT_SOCK_PATH, tick_ms);
    std::printf("[mock] 第 %d 个请求正常结束后，下一请求输出 %d 个 token 后以 error 结束（仅一次）\n",
                NORMAL_REQUESTS_BEFORE_ERROR, ERROR_EMITTED_TOKENS);
    std::fflush(stdout);

    std::vector<Request> active;
    zmq::pollitem_t items[] = {{req_sock, 0, ZMQ_POLLIN, 0}};

    while (g_running) {
        // 以 poll 超时作为 tick：新请求最多等待一个 tick 后开始输出
        try {
            zmq::poll(items, 1, std::chrono::milliseconds(tick_ms));
        } catch (const zmq::error_t&) {
            if (g_running) continue;  // SIGINT 打断等非致命错误，重试
            break;
        }

        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t msg;
            while (req_sock.recv(msg, zmq::recv_flags::dontwait).has_value())
                handle_request(out_sock, msg.to_string(), active);
        }

        for (auto it = active.begin(); it != active.end();) {
            if (emit_step(out_sock, *it))
                ++it;
            else
                it = active.erase(it);
        }
        std::fflush(stdout);
    }

    std::printf("\n[mock] 退出, 未完成请求: %zu\n", active.size());
    return 0;
}
