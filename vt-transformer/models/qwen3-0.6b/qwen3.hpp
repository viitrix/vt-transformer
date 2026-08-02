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

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>

#include <core/vt.hpp>
#include <core/vt_engine.hpp>

namespace qwen3 {

// ---- 模型维度（来自 config.json）----
constexpr int kHiddenSize       = 1024;   // hidden_size / d_model
constexpr int kIntermediateSize = 3072;   // MLP 中间维
constexpr int kNumLayers        = 28;     // num_hidden_layers
constexpr int kNumHeads         = 16;     // num_attention_heads
constexpr int kNumKVHeads       = 8;      // num_key_value_heads (GQA)
constexpr int kHeadDim          = 128;    // head_dim
constexpr int kVocabSize        = 151936; // vocab_size

// 推导维度，便于 kernel 写 stride/grid
constexpr int kQDim = kNumHeads   * kHeadDim;  // 2048
constexpr int kKVDim = kNumKVHeads * kHeadDim; // 1024

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
    // 写 KV 到 PageTable 决定的 slot，返回每个 req 一个新 token。
    // TODO: 区分 batch.is_prefill() 走 prefill / decode 两条 kernel。
    ForwardOutputT forward(const BatchT& batch) override;

    // Qwen3 默认用 <|im_end|> 作为 assistant turn 终止符。
    // 若上层想用 <|endoftext|>(151643) 改这个常量即可。
    Token eos_token_id() const override { return kEosTokenId; }

    // 独立接口，外部调用使用
    bool run_dag(const char* fileName);

    // register_all — 外部注册 DAG word 的统一入口（main.cpp 通过 qwen3::register_all(eng) 调起）。
    void register_all();

private:
    vt::Enviroment* env_ = nullptr;  // 非拥有：CUDA device / stream / DAG 都从它取
    static constexpr Token kEosTokenId = 151645;  // <|im_end|>

};



} // namespace qwen3

#endif // _QWEN3_HPP_
