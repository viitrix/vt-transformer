// qwen3.hpp — Qwen3-0.6B 权重全局视图（FP16，device memory）
//
// 数据来源：models/qwen3-0.6b/scripts/dump_weight_fp16.cpp
//   把 /home/teaonly/workspace/qwen3-0.6b/model.safetensors 里每个 tensor
//   转成 FP16 后写到 models/qwen3-0.6b/weights/<name>.fp16。
//
// 本头文件只描述「CUDA Kernel 怎么访问这些权重」：
//   - 每个权重 = __half* 设备指针 + rows/cols。
//   - 28 层权重用 Layer 结构体数组组织（layers[0..27]）。
//   - 顶层有 embed_tokens / norm / lm_head 三个非 layer 权重。
//
// 维度常量来自 config.json（Qwen3-0.6B）：
//   hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
//   num_attention_heads=16, num_key_value_heads=8, head_dim=128,
//   vocab_size=151936, tie_word_embeddings=true。
//
// 使用约定：
//   - 启动时某处把 .fp16 文件读出、cudaMalloc + cudaMemcpy 到设备，
//     把 ptr/rows/cols 填进 g_weights 的对应字段。
//   - 之后任何 .cu / __global__ kernel 直接读 qwen3::g_weights 即可。
//   - 全部权重按行主序（row-major）存储，与 safetensors / PyTorch 一致。

#ifndef _QWEN3_HPP_
#define _QWEN3_HPP_



#include <cstdint>
#include <string>

#include <core/vt.hpp>
#include <core/vt_engine.hpp>
#include <core/vt_pages.hpp>

#include "kernels/common.cuh"

namespace qwen3 {

// 调试断点：同步 current stream 后阻塞等 Enter。
// 用途：launch_prefill 之后插一个手动 pause，便于 dump GPU 状态 / 上 nsight / 接 gdb。
// tag 是日志前缀，多次调用时区分（如 "prefill" / "decode"）。
// 不需要调试时把调用点注释掉即可，函数本身留着不增加运行时开销（inline + 无外部符号）。
inline void debug_breakpoint(const char* tag) {
    std::printf("[debug] %s: stream synced, press Enter to continue... ", tag);
    std::fflush(stdout);
    int c;
    while ((c = std::getchar()) != EOF && c != '\n') { /* drain until newline */ }
}

// Qwen3Engine —— 把 vt::EngineBase 接到 Qwen3-0.6B 
//
// 当前只搭骨架：
//   - forward(batch)     同步跑一次 forward（待实现，区分 prefill/decode）。
//   - eos_token_id()     Qwen3 默认 EOS（<|im_end|> = 151645）。
//   - forward_async/wait 沿用 EngineBase 默认 sync 实现，先不重写。
class Qwen3Engine : public vt::EngineBase<int32_t, int32_t> {
public:
    using Base = vt::EngineBase<int32_t, int32_t>;
    using Token = int32_t;  // 对齐 Base 的模板参数，本类内直接用 Token 名字

    // env 由外部（main 等）管理生命周期，本类只持有非拥有指针：
    // 它的 ctx() 是 CUDA device / stream 等的统一入口。
    explicit Qwen3Engine(vt::Enviroment& env);
    ~Qwen3Engine() override;

    // 禁拷贝：env_ 是非拥有引用，拷贝会产生两个指向同一 env 的实例。
    Qwen3Engine(const Qwen3Engine&)            = delete;
    Qwen3Engine& operator=(const Qwen3Engine&) = delete;

    // ---- EngineBase 接口 ----
    // 同步 forward：读 batch 里每个 req 的 [cached_len, total_len) 段作为输入，
    // 写 KV 到 page_table 决定的 slot，返回每个 req 一个新 token。
    // TODO: 区分 batch.is_prefill() 走 prefill / decode 两条 kernel。
    ForwardOutputT forward(const BatchT& batch, PageTableT& page_table) override;

    // Qwen3 默认用 <|im_end|> 作为 assistant turn 终止符。
    // 若上层想用 <|endoftext|>(151643) 改这个常量即可。
    Token eos_token_id() const override { return kEosTokenId; }

    // 桥接层（kernels/prefill.h::prefill_forward）通过这两个 accessor 取
    // env（拿 hash / execute / ctx）和 init() 灌好的权重指针表 comm_。
    vt::Enviroment&    env()  { return *env_; }
    const CommonArgs&  comm() const { return comm_; }

    // 初始化
    void init();

private:
    bool run_dag(const char* fileName);

private:
    vt::Enviroment* env_ = nullptr;  // 非拥有：CUDA device / stream / DAG 都从它取
    CommonArgs comm_;
    static constexpr Token kEosTokenId = 151645;  // <|im_end|>
};



} // namespace qwen3

#endif // _QWEN3_HPP_
