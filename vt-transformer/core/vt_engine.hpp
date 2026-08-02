#ifndef _VT_ENGINE_HPP_
#define _VT_ENGINE_HPP_

#include <cstdint>
#include <vector>

#include "vt.hpp"
#include "vt_request.hpp"

namespace vt {

// 一次 forward 提交给 Engine 的内容（对齐 mini-sglang core.py:71-94）。
//
// reqs：非拥有指针，所有权属于 FullScheduler 的 pending_ / running_。
// phase：Prefill / Decode 二选一——本实现沿用 mini-sglang 的"prefill OR decode"
//        单 batch 策略（scheduler.py:219-225），不在同一 batch 内混跑。
template <typename Token = int32_t, typename Index = int32_t>
struct Batch {
    using Req = Request<Token, Index>;

    enum Phase { Prefill, Decode };

    std::vector<Req*> reqs;
    Phase             phase = Prefill;

    bool is_prefill() const { return phase == Prefill; }
    bool is_decode()  const { return phase == Decode;  }
    int  size()       const { return (int)reqs.size(); }
};

// 一次 forward 的产出（对齐 mini-sglang engine.py:23-27 的 ForwardOutput）。
// next_tokens 长度必须 == batch.size，与 batch.reqs 同序。
template <typename Token = int32_t>
struct ForwardOutput {
    std::vector<Token> next_tokens;
};

// forward_async 的回执。默认 sync 路径直接把 Output 装在 handle 里；
// 真实 async engine 子类化 EngineBase 时可以重写 forward_async/wait，
// 用自己的 CUDA event / stream 状态替换 sync_output 字段（直接忽略它）。
template <typename Token = int32_t>
struct ForwardHandle {
    bool                valid = false;
    ForwardOutput<Token> sync_output;   // sync 路径专用
};

// Engine 抽象基类——FullScheduler 把 batch 喂进来，子类返回新 token。
// 真实实现（CudaEngine / MockEngine 等）子类化后注入 FullScheduler 构造函数。
// 对齐 mini-sglang engine.py:29 的 Engine.forward_batch。
template <typename Token = int32_t, typename Index = int32_t>
class EngineBase {
public:
    using BatchT         = Batch<Token, Index>;
    using ForwardOutputT = ForwardOutput<Token>;
    using ForwardHandleT = ForwardHandle<Token>;

    virtual ~EngineBase() = default;

    // 同步 forward：读取 batch 里每个 req 的 [cached_len, total_len) 段作为输入，
    // 把新 token 写到 KV（位置由 PageTable 决定），返回每个 req 一个新 token。
    // sync 路径（FullScheduler::step）走这条。
    virtual ForwardOutputT forward(const BatchT& batch) = 0;

    // 异步 forward：立即返回 handle，不阻塞；调用方在需要结果时再调 wait。
    // 默认实现就是包一层同步调用——sync engine 不需要重写。
    // 真实 async engine 重写 forward_async（提交到自己的 CUDA stream 立刻返回）
    // 和 wait（在调用方需要结果时同步 stream / event）。
    virtual ForwardHandleT forward_async(const BatchT& batch) {
        return ForwardHandleT{/*valid=*/true, forward(batch)};
    }
    virtual ForwardOutputT wait(ForwardHandleT&& h) {
        vt_assert(h.valid, "EngineBase::wait: handle not valid");
        return std::move(h.sync_output);
    }

    // EOS token id；decode 出该 id 视为完成。默认 -1 表示"不靠 EOS 终止"，
    // 仅靠 max_output_len 触发 Finished。
    virtual Token eos_token_id() const { return Token(-1); }
};

} // namespace vt

#endif
