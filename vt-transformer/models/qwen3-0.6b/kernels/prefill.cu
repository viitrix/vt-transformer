// prefill.cu — Prefill 主 kernel：按层迭代调用各子阶段。
//
// 子阶段实现已拆到独立文件，本文件只负责拼装：
//   - embedding.cu : prefill_mk_embedding（Phase 1，token id → xhidden）
//   - rmsnorm.cu   : prefill_mk_rmsnorm  （Phase 2，xhidden RMSNorm）
// 后续 QKV / attention / MLP 等子阶段同理各自独立成文件，prefill.cu 只保留串接逻辑。

#include "prefill.h"
#include "common.cuh"

#include <cooperative_groups.h>

namespace qwen3 {

namespace cg = cooperative_groups;

// 子阶段签名 —— 定义在各自的 .cu 里，此处仅前向声明让 prefill_kernel 能调用。
// 子阶段内部不做 grid.sync()，由 prefill_kernel 在每个阶段之间统一 sync。
__device__ void prefill_mk_embedding(const CommonArgs& comm, const PrefillArgs& args);
__device__ void prefill_mk_rmsnorm(const CommonArgs& comm,
                                   const PrefillArgs& args,
                                   int layer);
__device__ void prefill_mk_qkv_projection(const CommonArgs& comm,
                                          const PrefillArgs& args,
                                          int layer);

// 真实 prefill kernel：按层迭代 RMSNorm/GEMM/attention/写 KV/MLP/lm_head。
// 框架阶段先把 embedding 这一步接上，后续子阶段在 grid.sync() 之后接力。
__global__ void prefill_kernel(CommonArgs comm, PrefillArgs args) {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: Embedding lookup
    prefill_mk_embedding(comm, args);
    grid.sync();

    for (int layer = 0; layer < 1; layer++) {
        // Phase 2: Input LayerNorm
        prefill_mk_rmsnorm(comm, args, layer);
        grid.sync();

        // Phase 3: QKV Projection（Q→xq，K/V 直接落 kv_cache）
        prefill_mk_qkv_projection(comm, args, layer);
        grid.sync();

    }
    
    // TODO: RMSNorm → QKV proj → qk_norm → RoPE → 写 KV → attn → o_proj → MLP → lm_head
}

// Mega kernel 用 cooperative launch + grid.sync()，要求所有 block 同时驻留 SM。
// 能 launch 的最大 grid = num_SMs × max_blocks_per_SM，后者由 kernel 的
// block size / 寄存器占用 / shared mem 占用共同决定，必须 runtime 查询。
//
// 查询时机：mega kernel 代码改动后（加阶段、寄存器涨、smem 涨）这个值会变，
// 所以每次 launch 都重新查一次 —— cudaOccupancyMaxActiveBlocksPerMultiprocessor
// 本身很轻量（微秒级），相对 kernel 执行可以忽略。
struct CooperativeGridInfo {
    int grid_size;
    int num_sm;
    int max_blocks_per_sm;
};

static CooperativeGridInfo compute_max_cooperative_blocks(const void* kernel_ptr,
                                                          int block_size,
                                                          size_t dynamic_smem) {
    int dev = 0;
    cudaGetDevice(&dev);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);

    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel_ptr, block_size, dynamic_smem);

    return {num_sm * max_blocks_per_sm, num_sm, max_blocks_per_sm};
}

void launch_prefill(cudaStream_t stream, const CommonArgs& comm, const PrefillArgs& args) {
    // 用 cudaLaunchCooperativeKernel 替换 <<<>>>：参数以 void** 数组传，
    // kernel 入口签名不变（仍按值收 CommonArgs / PrefillArgs）。
    void* kernel_args[] = {
        (void*)&comm,
        (void*)&args,
    };

    const auto grid = compute_max_cooperative_blocks(
        (const void*)prefill_kernel, PREFILL_MK_BLOCK_SIZE, /*dynamic_smem=*/0);
    fprintf(stderr, "[prefill] grid_size=%d (sm=%d, blocks/sm=%d)\n",
            grid.grid_size, grid.num_sm, grid.max_blocks_per_sm);

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (const void*)prefill_kernel,
        dim3(grid.grid_size), dim3(PREFILL_MK_BLOCK_SIZE),
        kernel_args,
        0u,
        stream
    ));
}

} // namespace qwen3
