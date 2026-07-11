// weights.hpp — Qwen3-0.6B 权重全局视图（FP16，device memory）
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

#ifndef _QWEN3_WEIGHTS_HPP_
#define _QWEN3_WEIGHTS_HPP_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>

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

// ---- 单个权重：设备指针 + 二维形状 ----
// 1D 权重（RMSNorm 的 scale）按 rows=dim, cols=1 处理；
// 2D Linear 权重按 PyTorch 行主序：rows=out_features, cols=in_features。
struct Weight {
    __half* ptr   = nullptr;
    int     rows  = 0;
    int     cols  = 0;

    __device__ __host__ int numel() const { return rows * cols; }
};

// ---- Q/K 的 per-head RMSNorm（Qwen3 特有）----
struct QKNorm {
    Weight q_norm;   // [kHeadDim]
    Weight k_norm;   // [kHeadDim]
};

// ---- Self-Attention ----
// 行主序：q_proj.ptr[m * cols + k]，cols = in_features（hidden_size）。
struct SelfAttention {
    Weight q_proj;   // [kQDim,  kHiddenSize]
    Weight k_proj;   // [kKVDim, kHiddenSize]
    Weight v_proj;   // [kKVDim, kHiddenSize]
    Weight o_proj;   // [kHiddenSize, kQDim]
    QKNorm norm;
};

// ---- MLP (SwiGLU: down(silu(gate(x)) * up(x))) ----
struct MLP {
    Weight gate_proj;   // [kIntermediateSize, kHiddenSize]
    Weight up_proj;     // [kIntermediateSize, kHiddenSize]
    Weight down_proj;   // [kHiddenSize, kIntermediateSize]
};

// ---- 一个 Transformer block（pre-norm）----
struct Layer {
    SelfAttention self_attn;
    MLP           mlp;
    Weight        input_layernorm;          // [kHiddenSize]
    Weight        post_attention_layernorm; // [kHiddenSize]
};

// ---- 全部权重 ----
// lm_head 与 embed_tokens 在 Qwen3-0.6B 中是 tied（同一份权重），
// 这里仍保留两个指针，启动时让它们指向同一块设备内存即可。
struct ModelWeights {
    Weight embed_tokens;  // [kVocabSize, kHiddenSize]
    Weight norm;          // [kHiddenSize]   最终 RMSNorm
    Weight lm_head;       // [kVocabSize, kHiddenSize]
    Layer  layers[kNumLayers];
};

// 全局访问点：启动时填充，所有 CUDA kernel 共享只读。
extern ModelWeights g_weights;

// ---- 加载 / 释放 ----
//
// load_weights：
//   1. 枚举所有期望的权重名（28 层 × 11 + 顶层 3 = 311 个），
//      stat 每个 <dir>/<name>.fp16，校验字节数 == rows*cols*2；
//   2. 顺序累加每个文件的字节数得到 total —— 就是 `du -sb <dir>` 的值；
//   3. 一次 cudaMalloc(total) 拿到一片大内存；
//   4. 顺序把每个文件读到 host 暂存，再一次 cudaMemcpy 整片灌进 device；
//   5. 回填 g_weights 各字段的 ptr = base + offset。
//   由于每个文件都是 256 的整数倍，紧挨着打包即让每个权重起始 256-byte 对齐。
//
// free_weights：释放大内存并把 g_weights 清零。进程退出时可不调用。
//
// 重复调用 load_weights 是 no-op（已加载就直接返回 true）。
bool load_weights(const std::string& dir);
void free_weights();

// 取大内存的设备基址 / 字节数；未加载时基址为 nullptr。
// kernel 一般不需要用——直接读 g_weights 里的 ptr 即可。这两个函数留给调试。
const void*    weights_device_base();
uint64_t       weights_device_bytes();

} // namespace qwen3

#endif // _QWEN3_WEIGHTS_HPP_
