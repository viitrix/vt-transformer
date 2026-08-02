// prefill.cu — Prefill 阶段 CUDA kernel 框架实现（纯 CUDA，不依赖 vt）。

#include "prefill.h"

namespace qwen3 {

// 真实 prefill kernel（stub）：本应做 RMSNorm/GEMM/attention/写 KV 等。
// 框架先放一个空 kernel 占位，仅用来撑 launch 语法。
__global__ void prefill_kernel(PrefillArgs args) {
    // TODO: per-block 处理 args 里 [req_token_offsets[i], req_token_offsets[i+1])
    //       段的 token，按层迭代写 KV，最后采样 next_tokens[i]。
    (void)args;
}

void launch_prefill(cudaStream_t stream, const PrefillArgs& args) {
    // 框架占位：grid/block 配置随便给一个能跑通的；真实实现按 total_tokens /
    // batch_size / SM 数量切分。
    prefill_kernel<<<1, 32, 0, stream>>>(args);
}

} // namespace qwen3
