// qkv_proj.cu — Prefill 阶段 QKV 投影（Batched GEMM-like，纯 CUDA）。
//
// 输入：comm.xnorm（RMSNorm 输出，行存；xhidden 留给 residual stream）
// 权重：comm.self_attn_{q,k,v}_proj[layer]
// 输出目的地（关键）：
//   Q  → comm.xq         [row * kMaxSeqLen + pos, *kQDim]   独立 buffer
//   K  → comm.kv_cache   按 (slot, layer) 直接落 KV pool —— 无独立 K buffer
//   V  → comm.kv_cache   紧挨 K 之后，同 slot+layer 段
// init.vt 注释明确：K/V 跳过中间 buffer，避免多一次 round-trip。

#include "prefill.h"
#include "common.cuh"

#include <cooperative_groups.h>

namespace qwen3 {

namespace cg = cooperative_groups;

// Batch 级 QKV 投影：
//   总并行度 = total_tokens × (kQDim + 2*kKVDim) = total_tokens × 4096
//   每个 warp 算一个输出元素：对 kHiddenSize=1024 做 dot product。
//   block 内 PREFILL_MK_NUM_WARPS 个 warp grid-stride，每次拿一个输出。
__device__ void prefill_mk_qkv_projection(
    const CommonArgs& comm,
    const PrefillArgs& args,
    int layer
) {
    const int batch_size = args.batch_size;
    const int num_blocks = gridDim.x;
    const int block_id   = blockIdx.x;
    const int warp_id    = threadIdx.x / WARP_SIZE;
    const int lane_id    = threadIdx.x % WARP_SIZE;

    // batch 拍平：所有 req 的 [cached_lens[i], input_lens[i]) 段拼成 1D token 序列。
    __shared__ int s_req_offsets[kMaxRunningReqs + 1];
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
    if (total_tokens == 0) return;

    // 每 token 输出 kQDim + 2*kKVDim = 2048 + 1024 + 1024 = 4096 个元素
    constexpr int TOTAL_PROJ_SIZE = kQDim + 2 * kKVDim;  // 4096
    const int total_outputs = total_tokens * TOTAL_PROJ_SIZE;

    // 输出元素按 block 切片
    const int outputs_per_block = (total_outputs + num_blocks - 1) / num_blocks;
    const int output_start = block_id * outputs_per_block;
    const int output_end   = min(output_start + outputs_per_block, total_outputs);

    const __half* q_weight = comm.self_attn_q_proj[layer];
    const __half* k_weight = comm.self_attn_k_proj[layer];
    const __half* v_weight = comm.self_attn_v_proj[layer];

    // grid-stride：每 warp 一输出，block 内同时跑 PREFILL_MK_NUM_WARPS 个输出
    for (int out_base = output_start; out_base < output_end; out_base += PREFILL_MK_NUM_WARPS) {
        const int out_idx = out_base + warp_id;
        if (out_idx >= output_end) break;

        // out_idx → (token_flat, proj_idx)
        const int token_flat = out_idx / TOTAL_PROJ_SIZE;
        const int proj_idx   = out_idx % TOTAL_PROJ_SIZE;

        // token_flat → (req_i, local_pos, row, pos, page_id)
        int req_i = 0;
        while (req_i + 1 < batch_size && s_req_offsets[req_i + 1] <= token_flat) {
            ++req_i;
        }
        const int local_pos = token_flat - s_req_offsets[req_i];
        const int row       = args.batch_idx[req_i];
        const int pos       = args.cached_lens[req_i] + local_pos;

        // slot_table 存的是 page_id（对齐 PageTable.pages_），需翻译成 physical_slot：
        //   physical_slot = page_id + (pos % kBlockSize)
        // 同 page 内 kBlockSize=4 个 token 共享 page_id，物理 slot 各不同 —— 不翻译的话
        // 这 4 个 token 会全写同一 kv_cache 位置。
        const int page_id   = args.slot_table[row * kMaxSeqLen + pos];
        const int phys_slot = page_id + (pos % kBlockSize);

        // 输入行（RMSNorm 输出，从 xnorm 读 —— xhidden 是 residual stream 不能动）
        const __half* input_row = comm.xnorm + (row * kMaxSeqLen + pos) * kHiddenSize;

        // proj_idx → (weight_row, output_ptr)
        const __half* weight_row;
        __half* output_ptr;

        if (proj_idx < kQDim) {
            // Q → xq 独立 buffer
            const int q_idx = proj_idx;
            weight_row = q_weight + q_idx * kHiddenSize;
            output_ptr = comm.xq + (row * kMaxSeqLen + pos) * kQDim + q_idx;
        } else if (proj_idx < kQDim + kKVDim) {
            // K → kv_cache[phys_slot, layer] 段头部
            const int k_idx = proj_idx - kQDim;
            weight_row = k_weight + k_idx * kHiddenSize;
            output_ptr = comm.kv_cache + phys_slot * kKVPerTokenElems
                                       + layer * kKVPerLayerElems
                                       + k_idx;
        } else {
            // V → kv_cache[phys_slot, layer] 段中 K 之后（偏移 kKVDim）
            const int v_idx = proj_idx - kQDim - kKVDim;
            weight_row = v_weight + v_idx * kHiddenSize;
            output_ptr = comm.kv_cache + phys_slot * kKVPerTokenElems
                                       + layer * kKVPerLayerElems
                                       + kKVDim + v_idx;
        }

        // dot product（1024 维 → 每 lane 处理 32 元素，warp 末尾 __shfl_down 归约）
        // vec4 load：uint2 = 8 bytes = 4 个 __half；weight 和 input 都只读，全走 __ldg。
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < kHiddenSize; k += WARP_SIZE * 4) {
            uint2 w_u2  = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
            uint2 in_u2 = __ldg(reinterpret_cast<const uint2*>(input_row + k));
            const __half* w_p  = reinterpret_cast<const __half*>(&w_u2);
            const __half* in_p = reinterpret_cast<const __half*>(&in_u2);

            sum += __half2float(w_p[0]) * __half2float(in_p[0]) +
                   __half2float(w_p[1]) * __half2float(in_p[1]) +
                   __half2float(w_p[2]) * __half2float(in_p[2]) +
                   __half2float(w_p[3]) * __half2float(in_p[3]);
        }

        sum = prefill_mk_warp_reduce_sum(sum);
        if (lane_id == 0) {
            *output_ptr = __float2half(sum);
        }
    }
}

} // namespace qwen3
