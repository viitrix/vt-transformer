// constants.cuh — Qwen3-0.6B 模型常量（编译期固定）。
//
// 仅由 kernels/*.cu 包含：把 num_layers / head_dim / block_size 这类 kernel
// 内部循环边界从 PrefillArgs 里挪出来，让 PrefillArgs 只承载真正随 batch 变化
// 的字段（指针 + per-req 描述）。host 端 buffer 分配仍由 dag/init.vt 决定，
// 因此本文件的数字必须与 init.vt 同名常量一致——两侧脱钩会出现 "buffer 够大
// 但 kernel 循环边界错" 这类静默错误，改任一侧都要同步另一侧。

#ifndef _QWEN3_CONSTANTS_CUH_
#define _QWEN3_CONSTANTS_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qwen3 {

// 模型结构（对齐 dag/init.vt 顶层常量）
constexpr int kHiddenSize       = 1024;
constexpr int kIntermediateSize = 3072;
constexpr int kNumLayers        = 28;
constexpr int kNumHeads         = 16;
constexpr int kNumKVHeads       = 8;
constexpr int kHeadDim          = 128;
constexpr int kVocabSize        = 151936;

// Page / KV pool / PageTable 形状
constexpr int kBlockSize        = 4;
constexpr int kMaxRunningReqs   = 4;
constexpr int kMaxSeqLen        = 256;

// 推导维度，便于 kernel 写 stride/grid
constexpr int kQDim = kNumHeads   * kHeadDim;  // 2048
constexpr int kKVDim = kNumKVHeads * kHeadDim; // 1024

// Prefill kernel 的 block size —— launcher (prefill.cu::launch_prefill) 和
// 子阶段 kernel (embedding.cu / rmsnorm.cu) 必须看到同一个值：
//   blockDim.x = PREFILL_MK_BLOCK_SIZE，
//   子阶段内部用它当 per-token stride 跨线程切 kHiddenSize 维度。
// 脱钩会出现 "stride 与实际 block size 不一致" 这种静默错误。
// Grid 大小不在此固定 —— launch_prefill 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor
// 在运行时按 kernel 资源占用推满，避免硬编码与 SM 数脱节。
constexpr int PREFILL_MK_BLOCK_SIZE = 96;

// Warp / reduce 相关常量（rmsnorm.cu 的 warp reduce + shared mem 跨 warp 归约用）。
//   PREFILL_MK_NUM_WARPS 必须 = PREFILL_MK_BLOCK_SIZE / WARP_SIZE —— rmsnorm 里
//   smem_reduce[PREFILL_MK_NUM_WARPS] 是按 warp 写、按 warp 读，size 不匹配会越界。
constexpr int WARP_SIZE             = 32;
constexpr int PREFILL_MK_NUM_WARPS  = PREFILL_MK_BLOCK_SIZE / WARP_SIZE;  // 3

// RMSNorm 的 eps —— 对齐 Qwen3-0.6B config.json 里的 rms_norm_eps = 1e-6。
constexpr float PREFILL_MK_RMS_EPS  = 1e-6f;

// KV pool 寻址常量（对齐 init.vt 同名变量；kv_cache 是 __half*，按 element 寻址）：
//   kKVPerLayerElems = 2 (K+V) × kKVDim               = 2048
//   kKVPerTokenElems = kNumLayers × kKVPerLayerElems  = 57344
// 写 KV 时：K 在 [slot*per_token + layer*per_layer, ...]
//           V 紧挨 K 后：[..., + kKVDim]
constexpr int kKVPerLayerElems = 2 * kKVDim;
constexpr int kKVPerTokenElems = kNumLayers * kKVPerLayerElems;

// Warp 级 sum 归约（__shfl_down_sync 树形）—— rmsnorm / qkv_proj 共用。
// 放头文件里保证 inline，跨 TU 调用既丢 inline 又要走 -rdc 链接，得不偿失。
// 注意：common.cuh 也被 main.cpp / qwen3.cpp（g++ 编译）包含，__shfl_down_sync
// 是 device-only intrinsic，必须用 __CUDACC__ guard 隔离（__CUDACC__ 由 nvcc
// 定义、g++ 不定义；__CUDA_ARCH__ 只在 device 函数体里有效，namespace scope 没用）。
#ifdef __CUDACC__
static __device__ __forceinline__ float prefill_mk_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
#endif

struct CommonArgs {
    // ---- KV pool ----
    __half*       kv_cache;        // device ptr，布局见 init.vt 注释
    __half*       embed_tokens;
    __half*       norm;
    __half*       lm_head;

    // ---- per-layer 权重（顺序对齐 dag/init.vt 的 %for 循环）----
    // self_attn：QKV/O 投影 + QK 的 RMSNorm（Qwen3 在投影后、RoPE 前对 Q/K 做 norm）
    __half*       self_attn_q_proj[kNumLayers];
    __half*       self_attn_k_proj[kNumLayers];
    __half*       self_attn_v_proj[kNumLayers];
    __half*       self_attn_o_proj[kNumLayers];
    __half*       self_attn_q_norm[kNumLayers];
    __half*       self_attn_k_norm[kNumLayers];
    // mlp：SwiGLU 三路投影（gate/up 同维，element-wise mul 后再过 down_proj）
    __half*       mlp_gate_proj[kNumLayers];
    __half*       mlp_up_proj[kNumLayers];
    __half*       mlp_down_proj[kNumLayers];
    // 两路 RMSNorm：attention 前 / MLP 前
    __half*       input_layernorm[kNumLayers];
    __half*       post_attention_layernorm[kNumLayers];

    // ---- 计算工作区（init.vt 分配，28 层共用，形状 [kMaxTokens × dim]）----
    //   xhidden / xout : kHiddenSize       （隐藏态 / 子层输出，便于 residual add）
    //   xnorm          : kHiddenSize       （RMSNorm 输出；独立于 xhidden 保护 residual stream）
    //   xq             : kQDim             （q_proj 输出；K/V 直接落 kv_cache，无独立 buffer）
    //   xinter         : kIntermediateSize （SwiGLU gate*up 中间结果）
    __half*       xhidden;
    __half*       xnorm;
    __half*       xq;
    __half*       xout;
    __half*       xinter;
};

// Prefill kernel 的纯 CUDA 输入。
//
// 入口语义（与 CPU scheduler 字段一一对应，对齐 vt_request.hpp）：
//   batch_idx[i]   = req.table_idx       // row 号 = PageTable 行号
//   cached_lens[i] = req.cached_len()    // 行内已落 KV 的 token 数（待处理段起点）
//   input_lens[i]  = req.input.size()    // 行内 token 总数（待处理段终点）
//   待处理段 = [cached_lens[i], input_lens[i])
//
struct PrefillArgs {
    // ---- device 端两块镜像表（init.vt 分配，scheduler 已 H2D 写好对应 row）----
    int32_t*      token_table;     // [kMaxRunningReqs × kMaxSeqLen]，按 row 存 token id
    int32_t*      slot_table;      // [kMaxRunningReqs × kMaxSeqLen]，按 row 存 page_id

    // ---- batch 描述（per req 三个 int，host 端小数组 H2D 到 device）----
    int            batch_size;
    const int32_t* batch_idx;     // [batch_size]，每个 req 的 row 号（= table_idx）
    const int32_t* cached_lens;   // [batch_size]，行内待处理段起点
    const int32_t* input_lens;    // [batch_size]，行内待处理段终点

    // ---- 输出 ----
    int32_t* next_tokens;           // [batch_size]，每个 req 一个新 token（device buffer）
};

} // namespace qwen3

#endif // _QWEN3_CONSTANTS_CUH_
