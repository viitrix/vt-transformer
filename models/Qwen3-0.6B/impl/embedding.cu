// embedding.cu — Prefill Phase 1：Batch 级 embedding 查表（纯 CUDA，不依赖 vt）。

#include "prefill.h"
#include "common.cuh"

namespace qwen3 {

// Batch 级 embedding 查表：
//   对 batch 内每个 req i，把 token_table[row_i, cached_lens[i]..input_lens[i])
//   段的 token id 查 embed_tokens，铺到 xhidden 工作区。
//
// xhidden 布局（与 token_table / slot_table 同构）：
//   xhidden[(row * kMaxSeqLen + pos) * kHiddenSize + dim_idx]
//   row = batch_idx[i] 给 PageTable 行号；每个 req 占独立的一段 [row × kMaxSeqLen × kHiddenSize, ...)，
//   batch 之间天然隔离不相互覆盖。已落 KV 的 [0, cached_lens[i]) 段在 xhidden 里留空，
//   后续子阶段按 (req_i, [cached_lens[i], input_lens[i])) 显式跳过。
//
// 工作切分：把每个 req 的 extend_len × kHiddenSize 工作量拼成一段连续 1D 空间，
//   s_req_offsets[i] = sum_{j<i}(input_lens[j] - cached_lens[j])，
//   按 grid-stride 切给所有 thread；thread 拿 token_flat 反查 req_i 用线性扫即可。
//   batch_size ≤ kMaxRunningReqs 较小，每个 block 各自算一遍 prefix sum 写进 shared
//   mem，比跨 block 做 grid.sync 还便宜。
__device__ void prefill_mk_embedding(const CommonArgs& comm, const PrefillArgs& args) {
    const int batch_size = args.batch_size;

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

    const int total_elements = s_req_offsets[batch_size] * kHiddenSize;
    const int num_threads    = gridDim.x * blockDim.x;
    const int tid            = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < total_elements; idx += num_threads) {
        const int token_flat = idx / kHiddenSize;
        const int dim_idx    = idx % kHiddenSize;

        int req_i = 0;
        while (req_i + 1 < batch_size && s_req_offsets[req_i + 1] <= token_flat) {
            ++req_i;
        }

        const int local_pos = token_flat - s_req_offsets[req_i];
        const int row       = args.batch_idx[req_i];
        const int pos       = args.cached_lens[req_i] + local_pos;
        const int token_id  = args.token_table[row * kMaxSeqLen + pos];

        const int out_idx = (row * kMaxSeqLen + pos) * kHiddenSize + dim_idx;
        comm.xhidden[out_idx] = __ldg(comm.embed_tokens + token_id * kHiddenSize + dim_idx);
    }
}

} // namespace qwen3
