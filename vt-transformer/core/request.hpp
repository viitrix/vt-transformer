#ifndef _VT_REQUEST_HPP_
#define _VT_REQUEST_HPP_

// 前后端分离推理引擎的后端核心数据结构。
// 后端的全部工作, 就是维护这张表里的每一个请求状态机，更新 output_ids 。

#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"

namespace vt {

// ---- 单个请求的状态机 ----
enum class RequestState {
    // 暂时状态，未开始处理，等待调度器唤醒
    WAITTING,   

    // 处理状态
    Chunking,
    Prefilling,
    Decoding,
    
    // 终态: 正常结束, 原因见 stop_reason | 中止, 原因文本见 abort_message (纯文本, 不带编号)。
    Stop,
    Aborted,
};

// Stop 状态的结束原因
enum class StopReason {
    Eos,          // 采样出结束符
    StopWord,     // 命中采样参数指定的停止词
    LengthLimit,  // 产出 token 数达到 max_new_tokens
    KvCacheFull,  // KVCache 空间不足, 被迫结束
};

// 日志用状态名
inline const char* request_state_name(RequestState s) {
    switch (s) {
        case RequestState::WAITTING:    return "Waiting";
        case RequestState::Chunking:     return "Chunking";
        case RequestState::Prefilling:  return "Prefilling";
        case RequestState::Decoding:    return "Decoding";
        case RequestState::Stop:        return "Stop";
        case RequestState::Aborted:     return "Aborted";
    }
    return "?";
}

// 采样参数, 由前端随请求下发
struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 1.0f;
    int   top_k = -1;      // -1 表示不截断
    int   max_new_tokens = 256;
    std::vector<int32_t> stop_token_ids;  // 结束符之外的停止词
};

// ---- 请求: 表中的每一项 ----
template <typename Token = int32_t>
struct Request {
    // ---- 身份 ----
    // rid 由前端在构造时分配一次, 之后只读; 私有存储且不提供 setter, 禁止修改
    explicit Request(uint64_t rid = 0) : rid_(rid) {}
    uint64_t rid() const { return rid_; }

    // ---- 状态机 ----
    RequestState state = RequestState::Chunking;
    StopReason stop_reason = StopReason::LengthLimit;  // state == Stop 时有效
    std::string abort_message;                         // state == Aborted 时有效

    // ---- 输入 (到达后不再变化) ----
    std::vector<Token> input_ids;  // prompt token 序列
    SamplingParams params;

    // ---- 输出 ----
    std::vector<Token> output_ids;

    // ---- 查询 ----
    bool is_finished() const {
        return state == RequestState::Stop || state == RequestState::Aborted;
    }
    bool is_active() const {
        if (state == RequestState::Chunking || state == RequestState::Prefilling || state == RequestState::Decoding) {
            return true;
        }
        return false;
    }

    // ---- 状态机转换 ----
    void to_stop(StopReason reason) {     
        // 任意非终态 -> Stop
        vt_assert(!is_finished(), "to_stop: request is already finished");
        state = RequestState::Stop;
        stop_reason = reason;
    }
    void to_abort(std::string message) {
        // 任意非终态 -> Aborted
        vt_assert(!is_finished(), "to_abort: request is already finished");
        state = RequestState::Aborted;
        abort_message = std::move(message);
    }  

private:
    uint64_t rid_ = 0;   // 前端分配的请求号, 全局唯一, 构造后不可变
};

} // end of namespace
#endif
