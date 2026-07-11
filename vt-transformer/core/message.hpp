#ifndef MINISGL_MESSAGE_HPP
#define MINISGL_MESSAGE_HPP

// Backend-side adapter for the sglfront MQ protocol.
//
// Spec: sglfront/docs/protocol.md
//
// Two ZMQ links, both PUSH/PULL:
//   * zmq_backend_addr     — tokenizer (PUSH) -> us (PULL, bind)
//                            carries: UserMsg, AbortBackendMsg, ExitMsg, BatchBackendMsg
//   * zmq_detokenizer_addr — us (PUSH, connect) -> tokenizer (PULL)
//                            carries: DetokenizeMsg
//
// Each ZMQ frame is a single msgpack object. The encoded form of a message
// is a flat dict: each dataclass field becomes a top-level key, plus a
// "__type__" key naming the class (e.g. "UserMsg"). Nested dataclasses and
// lists are encoded recursively. 1D int32 tensors (UserMsg.input_ids only)
// are encoded as {"__type__":"Tensor", "buffer":<raw bytes>, "dtype":"torch.int32"}.
//
// External dependencies (not in opt/):
//   * libzmq + its C header <zmq.h>  — cppzmq is a wrapper, not a standalone impl.
//     Install libzmq-dev (or point the include path at a zmq.h) and link -lzmq.
// Uses only the two headers shipped under opt/:
//   * opt/zmq.hpp     (cppzmq >= 4.x)
//   * opt/msgpack.hpp (msgpack-c 8.x, header-only, no Boost)

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "opt/msgpack.hpp"
#include "opt/zmq.hpp"

namespace minisgl {

// ---------------------------------------------------------------------------
// SamplingParams — mirrors sglfront.core.SamplingParams
// ---------------------------------------------------------------------------
struct SamplingParams {
    double temperature = 0.0;     // <= 0 means greedy
    int64_t top_k = -1;           // -1 disables
    double top_p = 1.0;           // 1.0 disables
    bool ignore_eos = false;      // if true, generate past EOS up to max_tokens
    int64_t max_tokens = 1024;

    bool is_greedy() const noexcept {
        return (temperature <= 0.0 || top_k == 1) && top_p == 1.0;
    }

    static SamplingParams from_object(const msgpack::object& obj) {
        if (obj.type != msgpack::type::MAP) {
            throw std::runtime_error("SamplingParams: expected msgpack MAP");
        }
        SamplingParams sp;
        for (uint32_t i = 0; i < obj.via.map.size; ++i) {
            const auto& kv = obj.via.map.ptr[i];
            const std::string key = kv.key.as<std::string>();
            if (key == "temperature") sp.temperature = kv.val.as<double>();
            else if (key == "top_k") sp.top_k = kv.val.as<int64_t>();
            else if (key == "top_p") sp.top_p = kv.val.as<double>();
            else if (key == "ignore_eos") sp.ignore_eos = kv.val.as<bool>();
            else if (key == "max_tokens") sp.max_tokens = kv.val.as<int64_t>();
        }
        return sp;
    }
};

// ---------------------------------------------------------------------------
// Incoming message types (from tokenizer via zmq_backend_addr)
// ---------------------------------------------------------------------------
struct UserMsg {
    int64_t uid = 0;
    std::vector<int32_t> input_ids;   // decoded from the Tensor frame
    SamplingParams sampling_params;
};

struct AbortBackendMsg {
    int64_t uid = 0;
};

struct ExitMsg {};

struct BatchBackendMsg;

// A batch item can be any non-batch backend message (batches do not nest).
using BackendMsgItem = std::variant<UserMsg, AbortBackendMsg, ExitMsg>;

struct BatchBackendMsg {
    std::vector<BackendMsgItem> data;
};

// Top-level message received on zmq_backend_addr.
using BackendMsg = std::variant<UserMsg, AbortBackendMsg, ExitMsg, BatchBackendMsg>;

// ---------------------------------------------------------------------------
// Outgoing message types (to tokenizer via zmq_detokenizer_addr)
// ---------------------------------------------------------------------------
struct DetokenizeMsg {
    int64_t uid = 0;
    int64_t next_token = 0;
    bool finished = false;
};

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------
namespace detail {

inline std::string read_type_tag(const msgpack::object& obj) {
    if (obj.type != msgpack::type::MAP) return {};
    for (uint32_t i = 0; i < obj.via.map.size; ++i) {
        const auto& kv = obj.via.map.ptr[i];
        if (kv.key.as<std::string>() == "__type__") {
            return kv.val.as<std::string>();
        }
    }
    return {};
}

// Decode the Tensor dict {"__type__":"Tensor","buffer":<bytes>,"dtype":"torch.int32"}
// into a 1D int32 vector. This is the only tensor shape the protocol uses.
inline std::vector<int32_t> decode_int32_tensor(const msgpack::object& obj) {
    if (obj.type != msgpack::type::MAP) {
        throw std::runtime_error("Tensor: expected msgpack MAP");
    }
    const char* ptr = nullptr;
    uint32_t size = 0;
    bool found = false;
    for (uint32_t i = 0; i < obj.via.map.size; ++i) {
        const auto& kv = obj.via.map.ptr[i];
        const std::string key = kv.key.as<std::string>();
        if (key == "buffer") {
            if (kv.val.type != msgpack::type::BIN) {
                throw std::runtime_error("Tensor.buffer: expected msgpack BIN");
            }
            ptr = kv.val.via.bin.ptr;
            size = kv.val.via.bin.size;
            found = true;
        }
    }
    if (!found) throw std::runtime_error("Tensor: missing 'buffer' field");
    if (size % sizeof(int32_t) != 0) {
        throw std::runtime_error("Tensor.buffer size not multiple of 4 bytes");
    }
    std::vector<int32_t> result(size / sizeof(int32_t));
    if (!result.empty()) std::memcpy(result.data(), ptr, size);
    return result;
}

inline BackendMsgItem decode_item(const msgpack::object& obj) {
    const std::string tag = read_type_tag(obj);
    if (tag == "UserMsg") {
        UserMsg m;
        for (uint32_t i = 0; i < obj.via.map.size; ++i) {
            const auto& kv = obj.via.map.ptr[i];
            const std::string key = kv.key.as<std::string>();
            if (key == "uid") m.uid = kv.val.as<int64_t>();
            else if (key == "input_ids") m.input_ids = decode_int32_tensor(kv.val);
            else if (key == "sampling_params") m.sampling_params = SamplingParams::from_object(kv.val);
        }
        return m;
    }
    if (tag == "AbortBackendMsg") {
        AbortBackendMsg m;
        for (uint32_t i = 0; i < obj.via.map.size; ++i) {
            const auto& kv = obj.via.map.ptr[i];
            if (kv.key.as<std::string>() == "uid") m.uid = kv.val.as<int64_t>();
        }
        return m;
    }
    if (tag == "ExitMsg") {
        return ExitMsg{};
    }
    throw std::runtime_error("Unknown backend message type: " + tag);
}

}  // namespace detail

// Decode one top-level backend message from a raw msgpack frame.
inline BackendMsg decode_backend_msg(const char* data, size_t size) {
    msgpack::object_handle oh = msgpack::unpack(data, size);
    const msgpack::object& obj = oh.get();
    const std::string tag = detail::read_type_tag(obj);

    if (tag == "BatchBackendMsg") {
        BatchBackendMsg m;
        for (uint32_t i = 0; i < obj.via.map.size; ++i) {
            const auto& kv = obj.via.map.ptr[i];
            if (kv.key.as<std::string>() == "data") {
                if (kv.val.type != msgpack::type::ARRAY) {
                    throw std::runtime_error("BatchBackendMsg.data: expected ARRAY");
                }
                m.data.reserve(kv.val.via.array.size);
                for (uint32_t j = 0; j < kv.val.via.array.size; ++j) {
                    m.data.push_back(detail::decode_item(kv.val.via.array.ptr[j]));
                }
            }
        }
        return m;
    }
    // decode_item returns the 3-element BackendMsgItem; widen to BackendMsg
    // (4-element variant that also holds BatchBackendMsg). std::visit with a
    // uniform return type forces the implicit conversion per alternative.
    return std::visit(
        [](auto&& item) -> BackendMsg { return std::forward<decltype(item)>(item); },
        detail::decode_item(obj));
}

inline BackendMsg decode_backend_msg(const zmq::message_t& msg) {
    return decode_backend_msg(static_cast<const char*>(msg.data()), msg.size());
}

// ---------------------------------------------------------------------------
// Encoding (DetokenizeMsg only — that's the only message the backend sends)
// ---------------------------------------------------------------------------
inline std::string encode_detokenize_msg(const DetokenizeMsg& msg) {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_map(4);
    pk.pack_str(8);  pk.pack_str_body("__type__", 8);
    pk.pack_str(13); pk.pack_str_body("DetokenizeMsg", 13);
    pk.pack_str(3);  pk.pack_str_body("uid", 3);        pk.pack_int64(msg.uid);
    pk.pack_str(10); pk.pack_str_body("next_token", 10); pk.pack_int64(msg.next_token);
    pk.pack_str(8);  pk.pack_str_body("finished", 8);   pk.pack(msg.finished);
    return std::string(buffer.data(), buffer.size());
}

// ---------------------------------------------------------------------------
// BackendMQ — owns the two sockets the backend needs
// ---------------------------------------------------------------------------
class BackendMQ {
public:
    // bind_pull_addr   — zmq_backend_addr     (we PULL, bind)
    // connect_push_addr— zmq_detokenizer_addr (we PUSH, connect)
    BackendMQ(const std::string& bind_pull_addr,
              const std::string& connect_push_addr,
              int io_threads = 1)
        : ctx_(io_threads),
          pull_(ctx_, zmq::socket_type::pull),
          push_(ctx_, zmq::socket_type::push) {
        // Drop pending messages on close so ~context_t() never blocks on a
        // stuck PUSH queue. Trade-off: in-flight DetokenizeMsgs may be lost
        // on shutdown, which the protocol allows.
        pull_.set(zmq::sockopt::linger, 0);
        push_.set(zmq::sockopt::linger, 0);
        pull_.bind(bind_pull_addr);
        push_.connect(connect_push_addr);
    }

    ~BackendMQ() {
        // Member order matters: pull_/push_ declared after ctx_, so their
        // destructors run first and close the sockets before ctx_ tears down.
    }

    BackendMQ(const BackendMQ&) = delete;
    BackendMQ& operator=(const BackendMQ&) = delete;

    // Block until a message arrives, then decode and dispatch on __type__.
    BackendMsg recv() {
        zmq::message_t msg;
        auto res = pull_.recv(msg, zmq::recv_flags::none);
        if (!res.has_value()) {
            throw std::runtime_error("BackendMQ::recv: unexpected EAGAIN on blocking socket");
        }
        return decode_backend_msg(msg);
    }

    // Non-blocking; returns std::nullopt when no message is queued.
    std::optional<BackendMsg> try_recv() {
        zmq::message_t msg;
        auto res = pull_.recv(msg, zmq::recv_flags::dontwait);
        if (!res.has_value()) return std::nullopt;
        return decode_backend_msg(msg);
    }

    // Wait up to timeout_ms for the next message; return std::nullopt on
    // timeout. The backend main loop can use this to time-slice between MQ
    // work and model steps: poll(0) == try_recv(), poll(-1) == recv().
    std::optional<BackendMsg> poll(int timeout_ms) {
        zmq::pollitem_t item{};
        item.socket = pull_.handle();
        item.events = ZMQ_POLLIN;
        zmq::poll(&item, 1, std::chrono::milliseconds(timeout_ms));
        if (!(item.revents & ZMQ_POLLIN)) return std::nullopt;
        zmq::message_t msg;
        auto res = pull_.recv(msg, zmq::recv_flags::none);
        if (!res.has_value()) return std::nullopt;
        return decode_backend_msg(msg);
    }

    // Send a DetokenizeMsg to the tokenizer. Called once per generated token.
    void send(const DetokenizeMsg& msg) {
        std::string packed = encode_detokenize_msg(msg);
        push_.send(zmq::buffer(packed.data(), packed.size()),
                   zmq::send_flags::none);
    }

    // Convenience: build + send a DetokenizeMsg in one call.
    void send_token(int64_t uid, int64_t next_token, bool finished) {
        send(DetokenizeMsg{uid, next_token, finished});
    }

    zmq::socket_t& pull_socket() { return pull_; }
    zmq::socket_t& push_socket() { return push_; }
    zmq::context_t& context() { return ctx_; }

private:
    zmq::context_t ctx_;
    zmq::socket_t pull_;
    zmq::socket_t push_;
};

// ---------------------------------------------------------------------------
// Convenience visitor helpers for dispatching on a received BackendMsg
// ---------------------------------------------------------------------------
// Example:
//   auto msg = mq.recv();
//   std::visit(overloaded{
//       [](const UserMsg& m)           { /* run prefill/decode */ },
//       [](const AbortBackendMsg& m)   { /* abort request */ },
//       [](const ExitMsg&)             { /* shutdown */ },
//       [](const BatchBackendMsg& b)   { for (auto& item : b.data) ... },
//   }, msg);
template <class... Fs>
struct overloaded : Fs... { using Fs::operator()...; };

template <class... Fs>
overloaded(Fs...) -> overloaded<Fs...>;

}  // namespace minisgl

#endif  // MINISGL_MESSAGE_HPP
