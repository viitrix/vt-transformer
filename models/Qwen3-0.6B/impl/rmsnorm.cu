// rmsnorm.cu — Prefill Phase 2：Batched RMSNorm（Input LayerNorm，纯 CUDA）。

#include "prefill.h"
#include "common.cuh"

#include <cooperative_groups.h>

namespace qwen3 {

namespace cg = cooperative_groups;

// Batch 级 RMSNorm：
//   对 batch 内每个 req i 处理 [cached_lens[i], input_lens[i]) 段的新 token：
//   把所有 req 的 extend token 拍平成 1D（同 embedding 的 s_req_offsets 方案），
//   再按 token 切给各 block。每个 token 在 xhidden 里的行地址由 (row, pos) 反查：
//     row = batch_idx[req_i]
//     pos = cached_lens[req_i] + local_pos
//   batch 间天然隔离（不同 row 段在 xhidden 里不重叠），cached_lens 前缀已落 KV
//   段直接跳过。
__device__ void prefill_mk_rmsnorm(
    const CommonArgs& comm,
    const PrefillArgs& args,
    int layer
) {
    const int batch_size = args.batch_size;
    const int num_blocks = gridDim.x;
    const int block_id   = blockIdx.x;
    const int warp_id    = threadIdx.x / WARP_SIZE;
    const int lane_id    = threadIdx.x % WARP_SIZE;

    __shared__ int   s_req_offsets[kMaxRunningReqs + 1];
    __shared__ float smem_reduce[PREFILL_MK_NUM_WARPS];

    if (threadIdx.x == 0) {
        int acc = 0;
        for (int i = 0; i < batch_size; ++i) {
            s_req_offsets[i] = acc;
            acc += args.input_lens[i] - args.cached_lens[i];
        }
        s_req_offsets[batch_size] = acc;
    }
    __syncthreads();

    const int total_tokens = s_req_offsets[batch_size];
    if (total_tokens == 0) {
        return;
    }

    const int tokens_per_block = (total_tokens + num_blocks - 1) / num_blocks;
    const int token_start = block_id * tokens_per_block;
    const int token_end   = min(token_start + tokens_per_block, total_tokens);

    const __half* weight = comm.input_layernorm[layer];

    for (int token_flat = token_start; token_flat < token_end; ++token_flat) {
        // 反查 req_i —— batch_size ≤ kMaxRunningReqs 较小，线性扫即可。
        int req_i = 0;
        while (req_i + 1 < batch_size && s_req_offsets[req_i + 1] <= token_flat) {
            ++req_i;
        }
        const int local_pos = token_flat - s_req_offsets[req_i];
        const int row       = args.batch_idx[req_i];
        const int pos       = args.cached_lens[req_i] + local_pos;

        // 输入：xhidden（residual stream，RMSNorm 不破坏它，后续 residual add 还要读）
        // 输出：xnorm（独立的 normalized buffer，喂给下游 QKV proj）
        const __half* xhidden_row = comm.xhidden + (row * kMaxSeqLen + pos) * kHiddenSize;
        __half*       xnorm_row   = comm.xnorm   + (row * kMaxSeqLen + pos) * kHiddenSize;

        // Compute sum of squares
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < kHiddenSize; i += PREFILL_MK_BLOCK_SIZE) {
            float v = __half2float(xhidden_row[i]);
            local_sum_sq += v * v;
        }

        local_sum_sq = prefill_mk_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < PREFILL_MK_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = prefill_mk_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(kHiddenSize) + PREFILL_MK_RMS_EPS);
            }
        }
        __syncthreads();

        const float rstd = smem_reduce[0];

        // Apply normalization：读 xhidden，写 xnorm；xhidden 保持原值留给 residual stream。
        // 舍入顺序与 HF Qwen3RMSNorm 对齐：(x * rstd) -> fp16 -> * weight(fp16)，
        // 否则 FP32 一次乘到底再转 FP16 会少一次中间舍入，跟 ref 对不上。
        for (int i = threadIdx.x; i < kHiddenSize; i += PREFILL_MK_BLOCK_SIZE) {
            float v = __half2float(xhidden_row[i]);
            __half normalized = __float2half(v * rstd);
            __half w = __ldg(weight + i);
            xnorm_row[i] = __hmul(normalized, w);
        }
        __syncthreads();
    }
}

} // namespace qwen3
