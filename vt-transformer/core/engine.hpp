#ifndef _VT_ENGINE_HPP_
#define _VT_ENGINE_HPP_

#include <cstdint>

#include "common.hpp"

namespace vt {

// 前置声明, 避免在 engine.hpp 中引入庞大的 vt.hpp
struct Enviroment;

// Batch 中的一个请求槽: 请求状态机 + KV 槽位 + 本次调度的计算进度。
// 该数据结构是提供 Kernel 使用
//
// slot_flags 布局 (每个请求 kFlagsSize 个 int32_t, host/device 各一份, 见 KVSlotTable):
//   [0] input_len : host 写入, 本次 forward 的输入长度。
//        - Prefill (首步): 输入为 tok_slot[0, input_len) 内的完整 prompt;
//        - Decode (其后)  : input_len == 1, 输入为 tok_slot[seq_len-1], 即上一步采样出的 token。
//   [1] seq_len   : device 维护, forward 结束时更新为追加新 token 后的序列总长;
//        host 通过 D2H 回读, 最新采样 token 位于 tok_slot[seq_len-1]。
//   [2..] 保留。
template <typename Token = int32_t, typename Index = int32_t>
struct BatchEntry {
    uint64_t rid;
    Token* tok_slot;      // 设备端指针
    Index* idx_slot;      // 设备端指针
    int32_t* slot_flags;  // 设备端指针
};

template <int kMaxReq, typename Token = int32_t, typename Index = int32_t>
struct BatchTable {
    int32_t batch_size;
    BatchEntry<Token,Index>  batch[ kMaxReq ];
};

template <std::size_t kMaxSeqLen, int kMaxReq, int kPageSize = 4, typename Token = int32_t, typename Index = int32_t>
struct EngineBase {
    // env 由外部创建并注入, 本类不持有/不释放
    explicit EngineBase(Enviroment* env) : env_(env) {
    }
    virtual ~EngineBase() = default;
    EngineBase(const EngineBase&)            = delete;
    EngineBase& operator=(const EngineBase&) = delete;

    // forward 为纯虚函数，由派生类实现具体的推理逻辑
    virtual int forward(BatchTable<kMaxReq, Token, Index>* host_batch, BatchTable<kMaxReq, Token, Index>* device_batch) {
        (void)host_batch; (void)device_batch;
        return 0;
    }

    // ---- Overlap Scheduler 流水线钩子 (默认空实现, 派生类按需覆盖) ----
    // 调用时 Enviroment ctx 的 current stream 已切换到对应拷贝流:
    virtual void upload(BatchTable<kMaxReq, Token, Index>* host_batch,
                        BatchTable<kMaxReq, Token, Index>* device_batch) {
        (void)host_batch; (void)device_batch;
    }
    virtual void download(BatchTable<kMaxReq, Token, Index>* device_batch,
                          BatchTable<kMaxReq, Token, Index>* host_batch) {
        (void)device_batch; (void)host_batch;
    }

protected:
    Enviroment* env_;   // 运行环境, 供派生类访问 VT 虚拟机能力
};

} // end of namespace
#endif
