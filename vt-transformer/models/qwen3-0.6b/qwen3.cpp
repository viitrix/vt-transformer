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

#include <cstdint>
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

// stub：vtable 需要一个定义撑着链接；真正 prefill/decode kernel 实现后替换。
Qwen3Engine::ForwardOutputT Qwen3Engine::forward(const BatchT& /*batch*/) {
    return ForwardOutputT{};
}

// ---- DAG 入口 "qwen3.prefill" ----
namespace op {

// 栈效果： ( batch_ptr out_ptr -- )
// 调用方需把 Batch*/ForwardOutput* 以 number 形式压栈（reinterpret_cast<uintptr_t>）。
struct Prefill : public vt::NativeWord {
    vt::Enviroment* env_ = nullptr;

    explicit Prefill(vt::Enviroment& env) : env_(&env) {}

    void run(vt::Stack& stack) override {
        // 栈顶先出：先 pop out_ptr，再 pop batch_ptr
        auto out_ptr   = (uintptr_t)stack.pop_number();
        auto batch_ptr = (uintptr_t)stack.pop_number();

        auto* batch = reinterpret_cast<const vt::Batch<int32_t, int32_t>*>(batch_ptr);
        auto* out   = reinterpret_cast<vt::ForwardOutput<int32_t>*>(out_ptr);
        vt_assert(batch && out, "qwen3.prefill: null pointer on stack");

        // prefill_forward 是 free function，env_ 在构造时拿到，无需反查 engine
        prefill_forward(*env_, *batch, *out);
    }

    static vt::NativeWord* creator(vt::Enviroment& env) {
        return new Prefill(env);
    }
};

} // namespace op

void Qwen3Engine::register_all() {
    env_->insert_native_word("qwen3.prefill", op::Prefill::creator);
}

} // namespace qwen3
