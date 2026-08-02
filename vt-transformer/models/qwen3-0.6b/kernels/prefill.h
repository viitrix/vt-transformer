// prefill.h — Prefill 阶段入口（CUDA kernel 接口 + VT 集成桥接）。
//
// 文件分两段：
//   1) PrefillArgs / launch_prefill：纯 CUDA 部分，prefill.cu 实现真正的 kernel。
//      kernel 只看 raw ptr + 维度，不知道 PageTable / Request / Enviroment 存在。
//   2) prefill_forward（inline）：把 vt::Enviroment + vt::Batch 桥接成 PrefillArgs 再 launch。
//      写在本头文件里让 qwen3.cpp 不必夹带具体桥接逻辑；qwen3.cpp 只负责把
//      "qwen3.prefill" DAG word 接到 prefill_forward。
//
// 数据流（row-based，对齐 init.vt 里分配的 device 端两块镜像表）：
//   调用 prefill_forward 之前，scheduler 已对每个 req 完成两段 H2D：
//     req.input             → token_table[table_idx, 0 .. input.size())
//     该 req 的 page_id 段 → slot_table[table_idx, ...]
//   prefill_forward 不再 H2D 这两块大表，只负责：
//     (a) 拼 per-req 描述小数组（row / cached_len / total_len）并 H2D 到 device
//     (b) launch kernel —— kernel 直接按 row × max_seq_len + pos 寻址两块 device 表
//     (c) D2H next_tokens 回 host 写到 out.next_tokens
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
#include <cstdio>
#include <vector>

#include <core/vt.hpp>
#include <core/vt_engine.hpp>
#include <core/vt_request.hpp>
#include <core/vt_pages.hpp>
#include <core/vt_cuda.hpp>

#include "common.cuh"
#include "../qwen3.hpp"

namespace qwen3 {



// 单纯的 kernel launcher：只取 stream + args，无 vt 依赖。
// 调用方负责 args 里所有 device 指针的生命周期与可见性。
void launch_prefill(cudaStream_t stream, const CommonArgs& comm, const PrefillArgs& args);

// 同步 forward：当前 kernel 还未真正实现，但入口处先把每个 req 的 token 序列
// 和 KV slot 锚点 dump 出来——下面几个字段是 CUDA kernel 切 token / 落 KV slot 的依据。
//
// CUDA kernel 视角（每个 req 独立处理，batch 间只是把 token 段拍平到一起）：
//   prefill_pos : input[0 .. prefill_pos) 已落 KV —— kernel 跳过这段，
//                 从 input[prefill_pos] 起处理到 input[input_len) 为止。
//   decode_pos  : output[0 .. decode_pos) 已落 KV —— Decode 阶段 kernel 只处理
//                 output[decode_pos]（永远 1 个 token，即上一步刚预测还没落 KV 的那个）。
//   cached_len  = prefill_pos + decode_pos
//                 KV pool 行里已写的 slot 数 —— 本步新写的 slot 从 row[cached_len] 开始。
//   input_len   : input.size()，prompt 总长（不变量）—— kernel 拼 req_token_offsets
//                 时用，与 prefill_pos 无关。
//
// table_idx 给 PageTable 行号；行内 slot → kv_cache 物理 offset 的翻译由 PageTable
// 配合 block_size（page_size）完成，kernel 入口拿到的 page_ids 已是绝对 slot 索引。
// radix_hit_len 与 kernel 无关——只用于 scheduler / CacheManager 决定哪些段进 radix 树。
inline void prefill_forward(Qwen3Engine* eng,
                            const vt::PageTable<int32_t>& page_table,
                            const vt::Batch<int32_t, int32_t>& batch,
                            vt::ForwardOutput<int32_t>& out) {
    auto& env = eng->env();
    auto& h = env.hash();
    const int max_running_reqs = static_cast<int>(h.find_number("kMaxRunningReqs"));
    const int max_seq_len      = static_cast<int>(h.find_number("kMaxSeqLen"));
    vt_assert(batch.size() <= max_running_reqs,
              "prefill_forward: batch.size > kMaxRunningReqs");
    vt_assert(page_table.max_seq_len() == max_seq_len,
              "prefill_forward: PageTable max_seq_len != init.vt kMaxSeqLen");

    // 取 init.vt 用 cuda.create "host" 分配的 pinned 镜像 buffer。
    auto get_cpu = [&h](const char* name) -> int32_t* {
        auto t = h.find_tensor(name);
        auto* ct = dynamic_cast<vt::CudaTensor*>(t.get());
        vt_assert(ct != nullptr && ct->is_host(),
                  "prefill_forward: host mirror buffer missing or wrong type");
        return static_cast<int32_t*>(ct->data());
    };
    int32_t* token_table_cpu = get_cpu("token_table_cpu");
    int32_t* slot_table_cpu  = get_cpu("slot_table_cpu");
    int32_t* batch_idx_cpu   = get_cpu("batch_idx_cpu");
    int32_t* cached_lens_cpu = get_cpu("cached_lens_cpu");
    int32_t* input_lens_cpu = get_cpu("input_lens_cpu");

    const int n = batch.size();
    for (int i = 0; i < n; ++i) {
        const auto* req = batch.reqs[i];
        const int input_len = (int)req->input.size();
        const int32_t row = req->table_idx;

        batch_idx_cpu[i] = row;
        cached_lens_cpu[i] = req->cached_len();
        input_lens_cpu[i]  = input_len;

        int32_t* token_row = token_table_cpu + static_cast<size_t>(row) * max_seq_len;
        int32_t* slot_row     = slot_table_cpu + static_cast<size_t>(row) * max_seq_len;
        const int32_t* pt_row = page_table.page_row(row);
        for (int j = 0; j < input_len; ++j) {
            token_row[j] = req->input[j];
            slot_row[j] = pt_row[j];
        }
    }

    // CPU参数复制到GPU
    env.execute("prefill_request_to_cuda");

    // 取 init.vt 用 cuda.create "cuda" 分配的 device 端镜像表 / 输出 buffer。
    auto get_dev = [&h](const char* name) -> int32_t* {
        auto  t  = h.find_tensor(name);
        auto* ct = dynamic_cast<vt::CudaTensor*>(t.get());
        vt_assert(ct != nullptr && ct->is_device(),
                  "prefill_forward: device buffer missing or wrong type");
        return static_cast<int32_t*>(ct->data());
    };

    // 构造 PrefillArgs：device 指针全部来自 init.vt 镜像表，
    // host 端 *_cpu 已填好并通过 prefill_request_to_cuda H2D 到 device。
    PrefillArgs args{};
    args.token_table = get_dev("token_table");
    args.slot_table  = get_dev("slot_table");
    args.batch_idx   = get_dev("batch_idx");
    args.cached_lens = get_dev("cached_lens");
    args.input_lens  = get_dev("input_lens");
    args.next_tokens = get_dev("next_tokens");
    args.batch_size  = n;

    // launch：comm 来自 Qwen3Engine::init() 灌好的权重表；
    // stream 从 CudaContext::current_stream() 取（与 init.vt 里 cuda.load / H2D 同路）。
    auto* ctx = dynamic_cast<vt::CudaContext*>(env.ctx());
    vt_assert(ctx != nullptr, "prefill_forward: ctx is not CudaContext");
    launch_prefill(ctx->current_stream(), eng->comm(), args);

    env.execute("dump_xhidden");

    // 调试用：launch 后 sync + 阻塞等 Enter。不需要时把这一行注释掉。
    debug_breakpoint("prefill");

    // 处理返回
    (void)out;
}

} // namespace qwen3

#endif // _QWEN3_PREFILL_H_
