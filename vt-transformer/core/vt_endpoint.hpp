#ifndef _VT_ENDPOINT_HPP_
#define _VT_ENDPOINT_HPP_

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "opt/msgpack.hpp"
#include "opt/zmq.hpp"

#include "vt.hpp"
#include "vt_request.hpp"

namespace vt {

// Endpoint: vt 这一侧与 sgl-frontend 之间的双向 ZMQ 端点。
//
//   in_  : PULL socket，从 frontend / tokenizer 收请求与控制信号
//   out_ : PUSH socket，把每步 decode 结果回送到 tokenizer / detokenizer
//
// 上行协议（in_ 收到的 msgpack 字典，按 "__type__" 判别）：
//   "UserMsg"          : { uid, input_ids: {__type__:"Tensor", buffer, dtype},
//                          sampling_params: {...} }
//   "BatchBackendMsg"  : { data: [<UserMsg>, ...] }
//   "AbortBackendMsg"  : { uid }
//   "ExitMsg"          : {}
//
// 下行协议（out_ 发出的 msgpack 字典）：
//   "DetokenizeMsg"     : { uid, next_token, finished }     — 单条结果
//   "BatchTokenizerMsg" : { data: [<DetokenizeMsg>, ...] }  — 多条结果
// （对齐 mini-sglang scheduler._reply_tokenizer_rank0 的派发逻辑。）
//
// 非 ExitMsg 帧的"sampling_params"字段当前不进入 vt::Request（Request 暂未承载
// 采样参数）；待 Request 扩展后再回填。
//
// 非线程安全：调用方串行化访问，与 mini-sglang scheduler 单线程模型一致。
template <typename TokenT = int32_t, typename IndexT = int32_t>
class Endpoint {
public:
    using Req   = Request<TokenT, IndexT>;
    using Token = typename Req::Token;

    // try_recv / blocking_recv 的产出。一次 ZMQ 帧至多对应一个 Event。
    struct NewRequests { std::vector<Req> requests; };  // 来自 UserMsg / BatchBackendMsg
    struct Abort       { uint64_t uid; };               // 来自 AbortBackendMsg
    struct Exit        {};                              // 来自 ExitMsg
    using Event = std::variant<NewRequests, Abort, Exit>;

    // send 的输入：一步 decode 后要回送的内容。
    struct Result {
        uint64_t uid;        // 对应 Request::id
        Token    next_token; // 本步新采样的 token
        bool     finished;   // 是否已命中 EOS / 达到上限
    };

    // recv_addr / send_addr 各自独立绑定或连接：
    //   - scheduler 默认拓扑（对齐 mini-sglang）：
    //       PULL bind recv_addr（frontend connect 进来）
    //       PUSH connect send_addr（detokenizer 那头 bind）
    //     → bind_recv=true, bind_send=false
    //   - 单元测试常需两端都 bind 或都 connect，靠这两个 flag 切换。
    Endpoint(zmq::context_t& ctx,
             const std::string& recv_addr,
             const std::string& send_addr,
             bool               bind_recv = true,
             bool               bind_send = false)
        : in_(ctx, zmq::socket_type::pull),
          out_(ctx, zmq::socket_type::push) {
        if (bind_recv) in_.bind(recv_addr);
        else           in_.connect(recv_addr);
        if (bind_send) out_.bind(send_addr);
        else           out_.connect(send_addr);
    }

    // ---- 收 ----

    // 非阻塞读取。无消息返回 nullopt；解析失败抛 std::runtime_error。
    std::optional<Event> try_recv() {
        zmq::message_t frame;
        auto got = in_.recv(frame, zmq::recv_flags::dontwait);
        if (!got.has_value()) return std::nullopt;
        return decode(frame);
    }

    // 阻塞读取一帧。解析失败抛 std::runtime_error。
    Event blocking_recv() {
        zmq::message_t frame;
        const auto n = in_.recv(frame, zmq::recv_flags::none);
        vt_assert(n.has_value(), "Endpoint::blocking_recv: zmq recv failed");
        return decode(frame);
    }

    // ---- 发 ----

    // 把一步 decode 的结果批量回送。results 为空时 no-op。
    // 单条 → DetokenizeMsg；多条 → BatchTokenizerMsg([DetokenizeMsg,...])。
    // 默认阻塞（HWM 满会等），与 mini-sglang ZmqPushQueue.put 行为一致。
    void send(const std::vector<Result>& results) {
        if (results.empty()) return;
        if (results.size() == 1) {
            msgpack::sbuffer sbuf;
            msgpack::packer<msgpack::sbuffer> pk(&sbuf);
            pack_detok(pk, results[0]);
            ship(sbuf);
            return;
        }
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);
        pk.pack_map(2);
        pk_str(pk, "__type__"); pk_str(pk, "BatchTokenizerMsg");
        pk_str(pk, "data");     pk.pack_array(results.size());
        for (const auto& r : results) pack_detok(pk, r);
        ship(sbuf);
    }

private:
    zmq::socket_t in_;
    zmq::socket_t out_;

    // ---- msgpack 解码（oh 的 zone 持有所有 string/bin 的生命周期）----
    static Event decode(const zmq::message_t& frame) {
        msgpack::object_handle oh = msgpack::unpack(
            static_cast<const char*>(frame.data()), frame.size());
        const msgpack::object& obj = oh.get();

        const std::string type_name = map_get_str(obj, "__type__");

        if (type_name == "UserMsg") {
            return NewRequests{ { parse_user_msg(obj) } };
        }
        if (type_name == "BatchBackendMsg") {
            return NewRequests{ parse_batch(obj) };
        }
        if (type_name == "AbortBackendMsg") {
            return Abort{ map_get_u64(obj, "uid") };
        }
        if (type_name == "ExitMsg") {
            return Exit{};
        }
        throw std::runtime_error("Endpoint: unknown __type__ = " + type_name);
    }

    static Req parse_user_msg(const msgpack::object& obj) {
        Req r;
        r.id    = map_get_u64(obj, "uid");
        r.input = parse_int32_tensor(map_get(obj, "input_ids"));
        return r;
    }

    static std::vector<Req> parse_batch(const msgpack::object& obj) {
        const msgpack::object& data = map_get(obj, "data");
        if (data.type != msgpack::type::ARRAY) {
            throw std::runtime_error("Endpoint: BatchBackendMsg.data must be array");
        }
        std::vector<Req> out;
        out.reserve(data.via.array.size);
        for (std::size_t i = 0; i < data.via.array.size; ++i) {
            out.push_back(parse_user_msg(data.via.array.ptr[i]));
        }
        return out;
    }

    // input_ids wire 形态：{__type__:"Tensor", buffer:bytes, dtype:"torch.int32"}。
    // mini-sglang 约定 1D int32 CPU tensor，这里按 sizeof(Token) 校验 dtype。
    static std::vector<Token> parse_int32_tensor(const msgpack::object& obj) {
        const std::string tname = map_get_str(obj, "__type__");
        if (tname != "Tensor") {
            throw std::runtime_error("Endpoint: input_ids must be Tensor, got " + tname);
        }
        const std::string dtype = map_get_str(obj, "dtype");
        const std::string want  = "torch.int32";
        if (dtype != want) {
            throw std::runtime_error("Endpoint: input_ids dtype must be " + want + ", got " + dtype);
        }
        const msgpack::object& buf = map_get(obj, "buffer");
        if (buf.type != msgpack::type::BIN) {
            throw std::runtime_error("Endpoint: tensor.buffer must be BIN");
        }
        if ((buf.via.bin.size % sizeof(Token)) != 0) {
            throw std::runtime_error("Endpoint: tensor.buffer size not aligned to Token");
        }
        std::vector<Token> out(buf.via.bin.size / sizeof(Token));
        if (!out.empty()) std::memcpy(out.data(), buf.via.bin.ptr, buf.via.bin.size);
        return out;
    }

    // ---- map 访问辅助（产生可读错误，不静默跳过缺失字段）----
    static const msgpack::object& map_get(const msgpack::object& obj, std::string_view key) {
        if (obj.type != msgpack::type::MAP) {
            throw std::runtime_error("Endpoint: expected map");
        }
        for (std::size_t i = 0; i < obj.via.map.size; ++i) {
            const auto& k = obj.via.map.ptr[i].key;
            if (k.type == msgpack::type::STR
                && std::string_view(k.via.str.ptr, k.via.str.size) == key) {
                return obj.via.map.ptr[i].val;
            }
        }
        throw std::runtime_error("Endpoint: missing key '" + std::string(key) + "'");
    }

    static std::string map_get_str(const msgpack::object& obj, std::string_view key) {
        const msgpack::object& v = map_get(obj, key);
        if (v.type != msgpack::type::STR) {
            throw std::runtime_error("Endpoint: '" + std::string(key) + "' must be str");
        }
        return std::string(v.via.str.ptr, v.via.str.size);
    }

    static uint64_t map_get_u64(const msgpack::object& obj, std::string_view key) {
        const msgpack::object& v = map_get(obj, key);
        if (v.type == msgpack::type::POSITIVE_INTEGER) return v.via.u64;
        if (v.type == msgpack::type::NEGATIVE_INTEGER) return static_cast<uint64_t>(v.via.i64);
        throw std::runtime_error("Endpoint: '" + std::string(key) + "' must be int");
    }

    // ---- msgpack 编码辅助 ----
    template <typename P>
    static void pk_str(P& pk, std::string_view s) {
        pk.pack_str(static_cast<uint32_t>(s.size()));
        pk.pack_str_body(s.data(), s.size());
    }

    template <typename P>
    static void pack_detok(P& pk, const Result& r) {
        pk.pack_map(4);  // __type__, uid, next_token, finished
        pk_str(pk, "__type__");   pk_str(pk, "DetokenizeMsg");
        pk_str(pk, "uid");        pk.pack_uint64(r.uid);
        pk_str(pk, "next_token"); pk.pack_int64(static_cast<int64_t>(r.next_token));
        pk_str(pk, "finished");
        if (r.finished) pk.pack_true();
        else            pk.pack_false();
    }

    void ship(const msgpack::sbuffer& sbuf) {
        zmq::message_t m(sbuf.data(), sbuf.size());
        out_.send(m, zmq::send_flags::none);
    }
};

} // namespace vt

#endif
