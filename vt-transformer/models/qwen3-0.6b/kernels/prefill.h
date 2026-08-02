// prefill.h — Prefill 阶段入口（CUDA kernel 接口 + VT 集成桥接）。
//
// 文件分两段：
//   1) PrefillArgs / launch_prefill：纯 CUDA 部分，prefill.cu 实现真正的 kernel。
//      kernel 只看 raw ptr + 维度，不知道 PageTable / Request / Enviroment 存在。
//   2) prefill_forward（inline）：把 vt::Enviroment + vt::Batch 桥接成 PrefillArgs 再 launch。
//      写在本头文件里让 qwen3.cpp 不必夹带具体桥接逻辑；qwen3.cpp 只负责把
//      "qwen3.prefill" DAG word 接到 prefill_forward。
//
// 真实 prefill kernel 实现（按层迭代）：
//   1. embed_tokens 查表
//   2. 每层：RMSNorm → QKV GEMM + qk_norm → RoPE → 写 KV → flash attn → o_proj → MLP
//   3. 最终 norm + lm_head → logits → 采样 → next_tokens
// 当前框架只占位把 launch 语法跑通。

#ifndef _QWEN3_PREFILL_H_
#define _QWEN3_PREFILL_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

#include <core/vt.hpp>
#include <core/vt_engine.hpp>
#include <core/vt_request.hpp>
#include <core/vt_pages.hpp>
#include <core/vt_cuda.hpp>

namespace qwen3 {

// Prefill kernel 的纯 CUDA 输入。
// prefill_forward 负责把 vt::Batch<int32_t, int32_t> 拍平成下面这些 device 数组。
//
// 当前是骨架字段——真实实现按需补全（权重指针、scale、rope freqs 等）。
struct PrefillArgs {
    // ---- KV pool ----
    __half*       kv_cache;        // device ptr，布局见 init.vt 注释
    int           num_layers;
    int           num_kv_heads;
    int           head_dim;
    int           block_size;      // page_size（每 page 含多少 token）

    // ---- batch 输入（拍平后）----
    // 把 batch.reqs[i] 的 [cached_len, total_len) 段按顺序拼接成一段连续 token 流。
    // token_ids[k]         ：第 k 个待处理 token 的 id
    // page_ids[k]          ：第 k 个待处理 token 落进 KV pool 的物理 slot 索引
    //                        （已由上游从 PageTable 翻译好，含 page 内偏移）
    // req_token_offsets[i] ：第 i 个 req 的 token 在 token_ids 里的起始下标（i ∈ [0, batch_size]）
    //                        末尾元素 == total_tokens，方便 kernel 取每个 req 的段
    const int32_t* token_ids;         // [total_tokens]
    const int32_t* page_ids;          // [total_tokens]
    const int32_t* req_token_offsets; // [batch_size + 1]，host-side 常量
    int            batch_size;
    int            total_tokens;

    // ---- 输出 ----
    int32_t*       next_tokens;     // [batch_size]，每个 req 一个新 token
};

// 单纯的 kernel launcher：只取 stream + args，无 vt 依赖。
// 调用方负责 args 里所有 device 指针的生命周期与可见性。
void launch_prefill(cudaStream_t stream, const PrefillArgs& args);

// prefill_forward — 把 vt::Batch / env.hash 桥接到 PrefillArgs，跑一遍 launch_prefill。
//
// 框架阶段：batch 拍平用的 device buffer 还没接，目前用 host 暂存打 stub 占位。
// 后续真实实现要：
//   1) 把每个 req 的 [cached_len, total_len) 段 token 拼到 device token_ids
//   2) 从 PageTable 行 + block_size 算出每个 token 的物理 slot，写 device page_ids
//   3) 累加 req_token_offsets（host 常量即可）
//   4) 给 next_tokens 分配 device buffer，kernel 写完后 D2H 拷回
inline void prefill_forward(vt::Enviroment& env,
                            const vt::Batch<int32_t, int32_t>& batch,
                            vt::ForwardOutput<int32_t>& out) {
    vt_assert(batch.is_prefill(), "prefill_forward: batch not in Prefill phase");
    vt_assert(batch.size() > 0,    "prefill_forward: empty batch");

    // (1) 取 KV pool tensor → device ptr
    auto& h = env.hash();
    auto kv = h.find_tensor("kv_cache");
    auto* kv_tensor = dynamic_cast<vt::CudaTensor*>(kv.get());
    vt_assert(kv_tensor && kv_tensor->is_device(),
              "prefill_forward: kv_cache missing or not device tensor");
    __half* kv_ptr = static_cast<__half*>(kv_tensor->data());

    // (2) 从 env.hash 取 KV 维度常量（init.vt 写入）
    const int num_layers   = static_cast<int>(h.find_number("kNumLayers"));
    const int num_kv_heads = static_cast<int>(h.find_number("kNumKVHeads"));
    const int head_dim     = static_cast<int>(h.find_number("kHeadDim"));
    const int block_size   = static_cast<int>(h.find_number("kBlockSize"));

    // (3) 取 current stream
    auto* ctx = dynamic_cast<vt::CudaContext*>(env.ctx());
    vt_assert(ctx, "prefill_forward: env ctx is not CudaContext");
    cudaStream_t stream = ctx->current_stream();

    // (4) 统计 batch 待处理 token 数
    int total_tokens = 0;
    const auto kPrefillState = vt::Request<int32_t, int32_t>::Prefill;
    for (auto* req : batch.reqs) {
        vt_assert(req->state == kPrefillState,
                  "prefill_forward: req state != Prefill");
        total_tokens += req->extend_len();
    }
    vt_assert(total_tokens > 0, "prefill_forward: total_tokens == 0");

    // (5) 构造 PrefillArgs（device buffer 暂留 nullptr）+ launch
    PrefillArgs args{};
    args.kv_cache          = kv_ptr;
    args.num_layers        = num_layers;
    args.num_kv_heads      = num_kv_heads;
    args.head_dim          = head_dim;
    args.block_size        = block_size;
    args.token_ids         = nullptr;        // TODO: 拼到 device
    args.page_ids          = nullptr;        // TODO: 从 PageTable 翻译
    args.req_token_offsets = nullptr;        // TODO: host 常量
    args.batch_size        = batch.size();
    args.total_tokens      = total_tokens;
    args.next_tokens       = nullptr;        // TODO: device buffer + D2H

    launch_prefill(stream, args);

    // (6) stub：next_tokens 用 0 占位（kernel 没真跑）
    out.next_tokens.assign((size_t)batch.size(), 0);
}

} // namespace qwen3

#endif // _QWEN3_PREFILL_H_
