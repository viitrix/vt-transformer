// qwen3.cpp — 全局权重实例 + 加载器实现。
//
// 配套 qwen3.hpp：
//   - g_weights 是所有 CUDA kernel 共享的全局只读视图（__half* ptr + rows/cols）。
//   - load_weights(dir) 把 <dir>/<name>.fp16 灌进一片大 device 内存，回填 g_weights。
//
// 加载流程：
//   1. 枚举 311 个权重名（顶层 3 + 28 层 × 11），stat 校验每个文件大小。
//   2. 累加得到 total = sum(file sizes) ≈ `du -sb <dir>` 的字节值。
//   3. cudaMalloc(total) 拿到大内存 dev。
//   4. host 暂存（优先 pinned，失败回退普通 malloc）顺序读所有文件。
//   5. 一次 cudaMemcpy(host→dev)。
//   6. 回填 g_weights.{embed_tokens, norm, lm_head, layers[i].*} 的 ptr = dev + offset。
//
// 对齐：所有文件大小都是 256 的整数倍（最小的 q_norm/k_norm 就是 256 B），
// 紧挨着打包自然让每个权重起始偏移 256-byte 对齐，无需 padding，total 精确等于 du。

#include "qwen3.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <core/vt.hpp>
#include <core/vt_engine.hpp>
#include <core/vt_request.hpp>
#include <core/vt_pages.hpp>
#include <core/vt_cuda.hpp>

#include "kernels/prefill.h"

namespace qwen3 {


// 构造即加载并执行 dag/init.vt：env_->execute 跑一遍 DAG，把权重 tensor 元信息
// 等落到 env 的 Stack / Hash 里。
Qwen3Engine::Qwen3Engine(vt::Enviroment& env) : env_(&env) {
}

Qwen3Engine::~Qwen3Engine() = default;



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
//   extend_len  = (input_len + output.size()) - cached_len
//                 本步 kernel 要处理的 token 数 = 待写 KV slot 数。
//                 Prefill：== input_len - prefill_pos（radix 命中后只剩 tail 段）；
//                 Decode ：== 1。
//   input_len   : input.size()，prompt 总长（不变量）—— kernel 拼 req_token_offsets
//                 时用，与 prefill_pos 无关。
//
// table_idx 给 PageTable 行号；行内 slot → kv_cache 物理 offset 的翻译由 PageTable
// 配合 block_size（page_size）完成，kernel 入口拿到的 page_ids 已是绝对 slot 索引。
// radix_hit_len 与 kernel 无关——只用于 scheduler / CacheManager 决定哪些段进 radix 树。
//
// input / output 只打前若干个 token，避免长序列刷屏。
Qwen3Engine::ForwardOutputT Qwen3Engine::forward(const BatchT& batch,
                                                  PageTableT& page_table) {
    const char* phase = batch.is_prefill() ? "prefill" : "decode";
    std::printf("[forward] phase=%s batch_size=%d\n", phase, batch.size());

    if (batch.is_prefill()) {
        ForwardOutputT out;
        prefill_forward(this, page_table, batch, out);
        return out;
    } else if (batch.is_decode()) {
        return ForwardOutputT{};
    }

    vt_panic("Can't be here！");
    return ForwardOutputT{};
}

void Qwen3Engine::init() {
    // 跑一遍 init.vt：在 env_->hash() 里建立权重 / kv_cache / device 镜像表等 tensor。
    // 文件打不开（路径错 / 缺文件）时 run_dag 返回 false，启动期直接 panic。
    static const char* fileName = "./dag/init.vt";
    if (!run_dag(fileName)) {
        vt_panic("Qwen3Engine::init: cannot open DAG file");
    }

    // 从 env_->hash() 把每个权重 tensor 的 device ptr 抽出来填进 comm_：
    // 之后 CUDA kernel 只读这份 ptr 表，不再回去翻 hash。
    auto& h = env_->hash();
    auto dev_half = [&h](const std::string& name) -> __half* {
        auto  t  = h.find_tensor(name);
        auto* ct = dynamic_cast<vt::CudaTensor*>(t.get());
        vt_assert(ct != nullptr && ct->is_device(),
                  "Qwen3Engine::init: weight tensor missing or not on device");
        return static_cast<__half*>(ct->data());
    };

    comm_.kv_cache     = dev_half("kv_cache");
    comm_.embed_tokens = dev_half("model.embed_tokens.weight");
    comm_.norm         = dev_half("model.norm.weight");
    comm_.lm_head      = dev_half("lm_head.weight");

    for (int i = 0; i < kNumLayers; ++i) {
        const auto base = "model.layers." + std::to_string(i) + ".";
        comm_.self_attn_q_proj[i]        = dev_half(base + "self_attn.q_proj.weight");
        comm_.self_attn_k_proj[i]        = dev_half(base + "self_attn.k_proj.weight");
        comm_.self_attn_v_proj[i]        = dev_half(base + "self_attn.v_proj.weight");
        comm_.self_attn_o_proj[i]        = dev_half(base + "self_attn.o_proj.weight");
        comm_.self_attn_q_norm[i]        = dev_half(base + "self_attn.q_norm.weight");
        comm_.self_attn_k_norm[i]        = dev_half(base + "self_attn.k_norm.weight");
        comm_.mlp_gate_proj[i]           = dev_half(base + "mlp.gate_proj.weight");
        comm_.mlp_up_proj[i]             = dev_half(base + "mlp.up_proj.weight");
        comm_.mlp_down_proj[i]           = dev_half(base + "mlp.down_proj.weight");
        comm_.input_layernorm[i]         = dev_half(base + "input_layernorm.weight");
        comm_.post_attention_layernorm[i] = dev_half(base + "post_attention_layernorm.weight");
    }

    // 计算工作区：名字与 init.vt 里 cuda.create 的字符串一致。
    comm_.xhidden = dev_half("xhidden");
    comm_.xnorm   = dev_half("xnorm");
    comm_.xq      = dev_half("xq");
    comm_.xout    = dev_half("xout");
    comm_.xinter  = dev_half("xinter");
}

// 读取 .vt 文件并交给 env_->execute 跑一遍。
// 文件打不开返回 false（让调用方决定是 panic 还是 retry）；其它错误由 execute 自身 panic。
bool Qwen3Engine::run_dag(const char* fileName) {
    std::ifstream f(fileName, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    env_->execute(ss.str());
    return true;
}

} // namespace qwen3
