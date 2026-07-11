// main.cpp — Qwen3-0.6B backend entry
//
// 起 minisgl::Scheduler,接 sglfront 发来的 UserMsg;forward 由 Qwen3CudaBackend
// 处理。启动时把全部权重灌进 device —— 后续 forward 直接读 qwen3::g_weights。
//
// 复用 config.json 只读 eos_token_id；其它字段等 forward 接进来再补。
//
// Compile:
//   see Makefile
// Run:
//   ./qwen3_infer [model_dir] [weights_dir]
//   默认 model_dir   = /home/teaonly/workspace/qwen3-0.6b
//   默认 weights_dir  = <repo>/models/qwen3-0.6b/weights

#include "schedule.hpp"
#include "qwen3_cuda_backend.hpp"
#include "weights.hpp"
#include "json.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using nlohmann::json;

static const char* kModelPath = "/home/teaonly/workspace/qwen3-0.6b";
static const char* kWeightPath = "/home/teaonly/workspace/vt-transformer/vt-transformer/models/qwen3-0.6b/weights";

// 这一版只从 config.json 拿 eos_token_id；其它字段等 forward 接进来再读。
static bool load_eos_token_id(const std::string& cfg_path, int64_t& out) {
    std::ifstream f(cfg_path);
    if (!f) return false;
    json j;
    try { j = json::parse(f); }
    catch (...) { return false; }
    if (!j.contains("eos_token_id")) return false;
    out = j["eos_token_id"].get<int64_t>();
    return true;
}

// 读 max_position_embeddings 作为 max_seq_len。失败返回 false(调用方用默认值)。
static bool load_max_seq_len(const std::string& cfg_path, size_t& out) {
    std::ifstream f(cfg_path);
    if (!f) return false;
    json j;
    try { j = json::parse(f); }
    catch (...) { return false; }
    if (!j.contains("max_position_embeddings")) return false;
    out = j["max_position_embeddings"].get<size_t>();
    return true;
}

int main(int argc, char** argv) {
    const std::string model_dir   = kModelPath;
    const std::string weights_dir = kWeightPath;
    const std::string cfg_path    = model_dir + "/config.json";

    minisgl::SchedulerConfig scfg;
    if (load_eos_token_id(cfg_path, scfg.eos_token_id)) {
        std::cerr << "[main] eos_token_id=" << scfg.eos_token_id
                  << " (from " << cfg_path << ")\n";
    } else {
        std::cerr << "[main] cannot read eos_token_id from " << cfg_path
                  << ", using default " << scfg.eos_token_id << "\n";
    }

    if (load_max_seq_len(cfg_path, scfg.max_seq_len)) {
        std::cerr << "[main] max_seq_len=" << scfg.max_seq_len
                  << " (from " << cfg_path << ")\n";
    } else {
        std::cerr << "[main] cannot read max_position_embeddings from " << cfg_path
                  << ", using default " << scfg.max_seq_len << "\n";
    }

    if (!qwen3::load_weights(weights_dir)) {
        std::cerr << "[main] load_weights failed: " << weights_dir << "\n";
        return 1;
    }
    std::cerr << "[main] weights loaded from " << weights_dir
              << " (" << qwen3::weights_device_bytes() << " B)\n";

    auto backend = std::make_unique<minisgl::Qwen3CudaBackend>(scfg);
    std::cerr << "[main] backend = Qwen3CudaBackend\n";
    minisgl::Scheduler sched(scfg, std::move(backend));
    sched.run_forever();

    qwen3::free_weights();
    return 0;
}
