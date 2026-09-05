#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <core/vt.hpp>
#include <core/vt_cuda.hpp>

#include "qwen3.hpp"

namespace qwen3 {


Qwen3Engine::Qwen3Engine(vt::Enviroment* env) : env_(env) {}

Qwen3Engine::~Qwen3Engine() {}

void Qwen3Engine::init() {
    init_comm();

    // create a empty cache table
    std::fill(comm_.token_table_cpu, comm_.token_table_cpu + kMaxRunningReqs * kMaxSeqLen, -1);
    std::fill(comm_.cached_lens_cpu, comm_.cached_lens_cpu + kMaxRunningReqs, 0);
    for (int i = 0; i < kMaxRunningReqs; ++i) {
        slots_[i].status = IDLE;
        slots_[i].rid    = -1;
    }
}

void Qwen3Engine::forward(std::vector<Request>& incoming) {
    for (const auto& req : incoming) {
        // abort 是控制消息不占槽位：按 rid 找到在途行置 ABORT，清理留给收尾流程；
        // rid 不在途（已结束 / 未受理过）时无事可做。计入 admitted 让它不进调用方等待队列。
        if (req.is_abort) {
            for (int i = 0; i < kMaxRunningReqs; ++i) {
                if (slots_[i].rid == req.rid
                    && (slots_[i].status == PREFILL || slots_[i].status == DECODE)) {
                    slots_[i].status = ABORT;
                    break;
                }
            }
            continue;
        }
        const auto [slot, prefix_len] = find_best_slot(req);
        if (slot < 0) break;  // 没有 IDLE 行：槽位满

        slots_[slot].status = PREFILL;
        slots_[slot].rid    = req.rid;
        int32_t* row = comm_.token_table_cpu + static_cast<size_t>(slot) * kMaxSeqLen;
        std::copy(req.input_ids.begin(), req.input_ids.end(), row);
        std::fill(row + req.input_ids.size(), row + kMaxSeqLen, -1);
        comm_.cached_lens_cpu[slot] = prefix_len;
        ++admitted;
    }
    return admitted;
}


std::optional<Response> Qwen3Engine::process_last() {
    // TODO
    return std::nullopt;
}



std::pair<int, int> Qwen3Engine::find_best_slot(const Request& req) const {
    int best = -1, best_len = 0;
    for (int i = 0; i < kMaxRunningReqs; ++i) {
        if (slots_[i].status != IDLE) continue;
        // 只扫 [0, cached_len)：前缀之外的行内容是上一任请求的残留，不作数。
        const int32_t* row    = comm_.token_table_cpu + static_cast<size_t>(i) * kMaxSeqLen;
        const int     cached  = comm_.cached_lens_cpu[i];
        int match = 0;
        while (match < static_cast<int>(req.input_ids.size()) && match < cached
               && row[match] == req.input_ids[match]) {
            ++match;
        }
        if (match > best_len) {
            best_len = match;
            best = i;
        }
    }
    return {best, best_len};
}

void Qwen3Engine::init_comm() {
    // 跑一遍 init.vt：在 env_->hash() 里建立权重 / kv_cache / device 镜像表等 tensor。
    // 文件打不开（路径错 / 缺文件）时 run_dag 返回 false，启动期直接 panic。
    static const char* fileName = "./dag/init.vt";
    if (!run_dag(fileName)) {
        vt_panic("Qwen3Engine::init: cannot open DAG file");
    }

    auto& h = env_->hash();

    // common.cuh 的编译期常量与 init.vt 顶层同名常量必须一致：
    // buffer 大小由 init.vt 决定，kernel 循环边界由 common.cuh 决定，
    // 两侧脱钩是静默错误（buffer 够大但边界错），启动期逐项核对。
    const struct { const char* name; int64_t expect; } mirror[] = {
        {"kHiddenSize",       kHiddenSize},
        {"kIntermediateSize", kIntermediateSize},
        {"kNumLayers",        kNumLayers},
        {"kNumHeads",         kNumHeads},
        {"kNumKVHeads",       kNumKVHeads},
        {"kHeadDim",          kHeadDim},
        {"kVocabSize",        kVocabSize},
        {"kQDim",             kQDim},
        {"kKVDim",            kKVDim},
        {"kMaxSeqLen",        kMaxSeqLen},
        {"kMaxRunningReqs",   kMaxRunningReqs},
    };
    for (const auto& m : mirror) {
        const auto actual = static_cast<int64_t>(h.find_number(m.name));
        if (actual != m.expect) {
            vt_panic((std::string("Qwen3Engine::init_comm: constant mismatch for ") + m.name
                      + ": common.cuh=" + std::to_string(m.expect)
                      + " init.vt=" + std::to_string(actual)).c_str());
        }
    }

    // 从 env_->hash() 把每个权重 tensor 的 device ptr 抽出来填进 comm_：
    // 之后 CUDA kernel 只读这份 ptr 表，不再回去翻 hash。
    auto dev_half = [&h](const std::string& name) -> __half* {
        // 名字不存在时 find_tensor 在 core 内部已 panic（打印该名字）。
        auto  t  = h.find_tensor(name);
        auto* ct = dynamic_cast<vt::CudaTensor*>(t.get());
        if (ct == nullptr || !ct->is_device() || ct->data() == nullptr) {
            vt_panic(("Qwen3Engine::init_comm: tensor missing/not device/empty: " + name).c_str());
        }
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

    // KV Cache 管理表：init.vt 里 "cuda" / "host" 成对分配，名字与 cuda.create 一致。
    //   token_table [kMaxRunningReqs × kMaxSeqLen] / cached_lens [kMaxRunningReqs]
    // host 侧是 pinned 镜像：prefill 填好后经 prefill_request_to_cuda（init.vt）H2D。
    auto dev_int = [&h](const std::string& name) -> int32_t* {
        auto  t  = h.find_tensor(name);
        auto* ct = dynamic_cast<vt::CudaTensor*>(t.get());
        if (ct == nullptr || !ct->is_device() || ct->data() == nullptr) {
            vt_panic(("Qwen3Engine::init_comm: tensor missing/not device/empty: " + name).c_str());
        }
        return static_cast<int32_t*>(ct->data());
    };
    auto host_int = [&h](const std::string& name) -> int32_t* {
        auto  t  = h.find_tensor(name);
        auto* ct = dynamic_cast<vt::CudaTensor*>(t.get());
        if (ct == nullptr || !ct->is_host() || ct->data() == nullptr) {
            vt_panic(("Qwen3Engine::init_comm: tensor missing/not host/empty: " + name).c_str());
        }
        return static_cast<int32_t*>(ct->data());
    };
    comm_.token_table      = dev_int("token_table");
    comm_.token_table_cpu  = host_int("token_table_cpu");
    comm_.cached_lens      = dev_int("cached_lens");
    comm_.cached_lens_cpu  = host_int("cached_lens_cpu");

    // 计算工作区：名字与 init.vt 里 cuda.create 的字符串一致。
    comm_.xhidden = dev_half("xhidden");
    comm_.xnorm   = dev_half("xnorm");
    comm_.xq      = dev_half("xq");
    comm_.xout    = dev_half("xout");
    comm_.xinter  = dev_half("xinter");

    // KV pool 容量检查：最坏情况下 kMaxRunningReqs 个 req 各写满 kMaxSeqLen，
    // 在途 token 总数达 kMaxTokens，每个 token 都要往 kv_cache 落一份 KV。
    // kNumCachedTokens <= kMaxTokens 时 prefill/decode 会越界写 pool，
    // 必须启动期 panic，不能留到运行期写脏显存。
    const auto num_cached_tokens = static_cast<int64_t>(h.find_number("kNumCachedTokens"));
    const auto max_tokens        = static_cast<int64_t>(h.find_number("kMaxTokens"));
    if (num_cached_tokens <= max_tokens) {
        vt_panic(("Qwen3Engine::init_comm: kNumCachedTokens (" + std::to_string(num_cached_tokens)
                  + ") must be > kMaxTokens (" + std::to_string(max_tokens) + ")").c_str());
    }
}

bool Qwen3Engine::run_dag(const char* fileName) {
    std::ifstream f(fileName, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    env_->execute(ss.str());
    return true;
}

} // namespace qwen3
