// config.hpp — SchedulerConfig
//
// 抽出到独立头文件,让 SchedulerConfig 的消费者(scheduler / backend / main)
// 共享一份参数定义,不必拉进 schedule.hpp。

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace minisgl {

struct SchedulerConfig {
    // ---- MQ ----
    // sglfront 协议里 backend 侧负责绑定的 PULL 地址(zmq_backend_addr)。
    std::string bind_pull_addr    = "ipc:///tmp/minisgl_0";
    // backend 侧负责 connect 的 PUSH 地址(zmq_detokenizer_addr)。
    std::string connect_push_addr = "ipc:///tmp/minisgl_1";

    // ---- 调度策略 ----
    // false: overlap_loop(CUDA stream overlap);true: normal_loop(同步)。
    bool   disable_overlap_scheduling = false;
    // TableManager 的 slot 池大小,即同时在跑(含 prefill/decode)的最大请求数。
    size_t max_running_reqs           = 1024;
    // 单次 prefill batch 的 token budget(对应 mini-sglang 的 max_extend_tokens)。
    size_t max_extend_tokens          = 8192;
    // 模型支持的最大序列长度(input + output)。对应 python engine.max_seq_len,
    // 通常来自 model_config.max_position_embeddings。PrefillManager 在 add_one_req
    // 时按它丢弃超长输入并 clamp sampling_params.max_tokens(端口 scheduler.py:177-189)。
    size_t max_seq_len                = 32768;

    // ---- KV cache ----
    size_t cache_page_size = 1;
    size_t cache_num_pages = 1024;

    // ---- EOS ----
    // Scheduler 权威 EOS,_process_last_data 判定 finished 用。
    // 默认 Qwen3 EOS;main.cpp 从 config.json 覆盖。
    int64_t eos_token_id = 151645;
};

}  // namespace minisgl
