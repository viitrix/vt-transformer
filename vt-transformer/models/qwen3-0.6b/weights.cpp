// weights.cpp — 全局权重实例 + 加载器实现。
//
// 配套 weights.hpp：
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

#include "weights.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace qwen3 {

// ---- 全局实例（zero-initialized：所有 ptr=nullptr, rows=0, cols=0）----
ModelWeights g_weights{};

namespace {

// 文件本地持有：唯一的大内存指针 + 字节数。
char*&    dev_buffer() { static char* p = nullptr;    return p; }
uint64_t& dev_bytes()  { static uint64_t b = 0;       return b; }

// 每个 Weight 字段对应的源文件 + 目标形状 + 大内存里的 offset。
struct LoadTask {
    std::string name;     // e.g. "model.layers.0.self_attn.q_proj.weight"
    int         rows;
    int         cols;
    uint64_t    bytes;    // rows * cols * sizeof(__half)
    uint64_t    offset;   // 第二遍填
};

bool file_size(const std::string& path, uint64_t& out) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    if (!S_ISREG(st.st_mode))           return false;
    out = uint64_t(st.st_size);
    return true;
}

bool read_into(const std::string& path, char* dst, uint64_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(dst, static_cast<std::streamsize>(bytes));
    return uint64_t(f.gcount()) == bytes;
}

// 把一个 (name, rows, cols) 推进 tasks；同时校验对应 .fp16 文件存在且大小匹配。
bool add_task(std::vector<LoadTask>& tasks,
              const std::string&     dir,
              const std::string&     name,
              int                    rows,
              int                    cols) {
    LoadTask t;
    t.name  = name;
    t.rows  = rows;
    t.cols  = cols;
    t.bytes = uint64_t(rows) * uint64_t(cols) * sizeof(__half);

    const std::string path = dir + "/" + name + ".fp16";
    uint64_t actual = 0;
    if (!file_size(path, actual)) {
        std::cerr << "[load_weights] missing file: " << path << "\n";
        return false;
    }
    if (actual != t.bytes) {
        std::cerr << "[load_weights] size mismatch " << path
                  << ": file=" << actual << " expect=" << t.bytes << "\n";
        return false;
    }
    tasks.push_back(std::move(t));
    return true;
}

void cleanup_host(char* host, bool pinned) {
    if (!host) return;
    if (pinned) cudaFreeHost(host);
    else        std::free(host);
}

} // namespace

bool load_weights(const std::string& dir) {
    if (dev_buffer() != nullptr) {
        std::cerr << "[load_weights] already loaded (" << dev_bytes() << " B)\n";
        return true;
    }

    // ---- pass 1：枚举所有期望的权重 ----
    std::vector<LoadTask> tasks;
    tasks.reserve(size_t(3) + size_t(kNumLayers) * 11);

    if (!add_task(tasks, dir, "model.embed_tokens.weight", kVocabSize,   kHiddenSize))        return false;
    if (!add_task(tasks, dir, "model.norm.weight",         kHiddenSize,  1))                  return false;
    if (!add_task(tasks, dir, "lm_head.weight",            kVocabSize,   kHiddenSize))        return false;

    for (int i = 0; i < kNumLayers; ++i) {
        const std::string L = "model.layers." + std::to_string(i) + ".";
        if (!add_task(tasks, dir, L + "self_attn.q_proj.weight",         kQDim,             kHiddenSize))       return false;
        if (!add_task(tasks, dir, L + "self_attn.k_proj.weight",         kKVDim,            kHiddenSize))       return false;
        if (!add_task(tasks, dir, L + "self_attn.v_proj.weight",         kKVDim,            kHiddenSize))       return false;
        if (!add_task(tasks, dir, L + "self_attn.o_proj.weight",         kHiddenSize,       kQDim))             return false;
        if (!add_task(tasks, dir, L + "self_attn.q_norm.weight",         kHeadDim,          1))                 return false;
        if (!add_task(tasks, dir, L + "self_attn.k_norm.weight",         kHeadDim,          1))                 return false;
        if (!add_task(tasks, dir, L + "mlp.gate_proj.weight",            kIntermediateSize, kHiddenSize))       return false;
        if (!add_task(tasks, dir, L + "mlp.up_proj.weight",              kIntermediateSize, kHiddenSize))       return false;
        if (!add_task(tasks, dir, L + "mlp.down_proj.weight",            kHiddenSize,       kIntermediateSize)) return false;
        if (!add_task(tasks, dir, L + "input_layernorm.weight",          kHiddenSize,       1))                 return false;
        if (!add_task(tasks, dir, L + "post_attention_layernorm.weight", kHiddenSize,       1))                 return false;
    }

    // ---- pass 2：算 layout（顺序紧挨，无 padding）----
    uint64_t total = 0;
    for (auto& t : tasks) {
        t.offset = total;
        total   += t.bytes;
    }
    std::cerr << "[load_weights] " << tasks.size() << " tensors, "
              << total << " bytes ("
              << (total / (1024.0 * 1024.0)) << " MiB)\n";

    // ---- pass 3：host 暂存（pinned 优先）----
    char* host = nullptr;
    cudaError_t e = cudaMallocHost(&host, total);
    const bool pinned = (e == cudaSuccess);
    if (!pinned) {
        std::cerr << "[load_weights] cudaMallocHost failed ("
                  << cudaGetErrorString(e) << "), fallback to malloc\n";
        host = static_cast<char*>(std::malloc(total));
        if (!host) {
            std::cerr << "[load_weights] malloc host staging failed\n";
            return false;
        }
    }

    // ---- pass 4：device 大内存 ----
    char* dev = nullptr;
    e = cudaMalloc(&dev, total);
    if (e != cudaSuccess) {
        std::cerr << "[load_weights] cudaMalloc(" << total << ") failed: "
                  << cudaGetErrorString(e) << "\n";
        cleanup_host(host, pinned);
        return false;
    }

    // ---- pass 5：顺序读文件到 host+offset ----
    for (const auto& t : tasks) {
        const std::string path = dir + "/" + t.name + ".fp16";
        if (!read_into(path, host + t.offset, t.bytes)) {
            std::cerr << "[load_weights] read failed: " << path << "\n";
            cudaFree(dev);
            cleanup_host(host, pinned);
            return false;
        }
    }

    // ---- pass 6：一次 H2D ----
    e = cudaMemcpy(dev, host, total, cudaMemcpyHostToDevice);
    cleanup_host(host, pinned);
    if (e != cudaSuccess) {
        std::cerr << "[load_weights] cudaMemcpy failed: "
                  << cudaGetErrorString(e) << "\n";
        cudaFree(dev);
        return false;
    }

    // ---- pass 7：回填 g_weights ----
    auto W = [&](const LoadTask& t) -> Weight {
        return Weight{ reinterpret_cast<__half*>(dev + t.offset), t.rows, t.cols };
    };

    g_weights.embed_tokens = W(tasks[0]);
    g_weights.norm         = W(tasks[1]);
    g_weights.lm_head      = W(tasks[2]);

    for (int i = 0; i < kNumLayers; ++i) {
        const size_t base = size_t(3) + size_t(i) * 11;
        Layer& L = g_weights.layers[i];
        L.self_attn.q_proj      = W(tasks[base + 0]);
        L.self_attn.k_proj      = W(tasks[base + 1]);
        L.self_attn.v_proj      = W(tasks[base + 2]);
        L.self_attn.o_proj      = W(tasks[base + 3]);
        L.self_attn.norm.q_norm = W(tasks[base + 4]);
        L.self_attn.norm.k_norm = W(tasks[base + 5]);
        L.mlp.gate_proj                = W(tasks[base + 6]);
        L.mlp.up_proj                  = W(tasks[base + 7]);
        L.mlp.down_proj                = W(tasks[base + 8]);
        L.input_layernorm              = W(tasks[base + 9]);
        L.post_attention_layernorm     = W(tasks[base + 10]);
    }

    dev_buffer() = dev;
    dev_bytes()  = total;
    return true;
}

void free_weights() {
    if (dev_buffer()) {
        cudaFree(dev_buffer());
        dev_buffer() = nullptr;
        dev_bytes()  = 0;
    }
    g_weights = ModelWeights{};   // 清掉所有 Weight::ptr，避免悬空
}

const void* weights_device_base() {
    return dev_buffer();
}

uint64_t weights_device_bytes() {
    return dev_bytes();
}

} // namespace qwen3
