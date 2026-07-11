// safetensors_test.cpp — 解析 safetensors header 的测试
//
// 用 /home/teaonly/workspace/qwen3-0.6b/model.safetensors 做真实文件验证。
// 没有该文件时所有 test 都会跳过（不算失败）。
//
// Compile:
//   g++ -std=c++17 -O2 -Wall -I.. safetensors_test.cpp -o /tmp/safetensors_test
// Run:
//   /tmp/safetensors_test

#include <core/safetensors.hpp>

#include <cstdio>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using vt::SafeTensorFile;
using vt::SafeTensorInfo;
using vt::safetensors_dtype_bytes;

static int g_failures  = 0;
static int g_skips     = 0;

// 调整路径以匹配你的环境；用绝对路径方便测试
static const char* kModelPath = "/home/teaonly/workspace/qwen3-0.6b/model.safetensors";

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAIL [" << __func__ << ":" << __LINE__ << "] "    \
                      << msg << std::endl;                                  \
            ++g_failures;                                                   \
            return;                                                         \
        }                                                                   \
    } while (0)

#define SKIP(msg)                                                           \
    do {                                                                    \
        std::cerr << "SKIP [" << __func__ << "] " << msg << std::endl;      \
        ++g_skips;                                                          \
        return;                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

void test_open_single() {
    SafeTensorFile f;
    bool ok = f.open(kModelPath);
    if (!ok) SKIP("cannot open " << kModelPath << " (file missing?)");
    CHECK(f.num_tensors() > 0, "expected non-empty tensor map");
    CHECK(!f.is_sharded(),     "single file should not be sharded");
    CHECK(f.num_shards() == 0, "single file should report 0 shards");
    std::cout << "  num_tensors = " << f.num_tensors() << "\n";
}

void test_known_tensors_exist() {
    SafeTensorFile f;
    if (!f.open(kModelPath)) SKIP("cannot open file");

    // Qwen3-0.6B 应该有这些 tensor（标准 transformer 命名）
    const char* expected[] = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    };
    for (const char* name : expected) {
        const auto* info = f.find(name);
        CHECK(info != nullptr, std::string("missing tensor: ") + name);
        std::cout << "  " << name
                  << "  dtype=" << info->dtype
                  << "  shape=[";
        for (size_t i = 0; i < info->shape.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << info->shape[i];
        }
        std::cout << "]  bytes=" << info->bytes << "\n";
    }
}

void test_all_tensors_verify_size() {
    SafeTensorFile f;
    if (!f.open(kModelPath)) SKIP("cannot open file");

    size_t checked = 0;
    size_t failed  = 0;
    for (const auto& [name, info] : f.tensors()) {
        if (!info.verify_size()) {
            std::cerr << "  size mismatch: " << name
                      << " dtype=" << info.dtype
                      << " bytes=" << info.bytes
                      << " expected=" << (info.numel() * safetensors_dtype_bytes(info.dtype))
                      << "\n";
            ++failed;
        }
        ++checked;
    }
    CHECK(failed == 0, "some tensors failed verify_size()");
    std::cout << "  checked " << checked << " tensors, all OK\n";
}

void test_total_bytes_reasonable() {
    SafeTensorFile f;
    if (!f.open(kModelPath)) SKIP("cannot open file");

    uint64_t total = 0;
    for (const auto& [_, info] : f.tensors()) {
        total += info.bytes;
    }
    // Qwen3-0.6B 实际 ~1.2GB（BF16）。我们只校验量级合理。
    CHECK(total > (1ULL << 29), "total bytes too small, expected > 0.5GB, got " << total);
    CHECK(total < (1ULL << 33), "total bytes too large, expected < 8GB,  got " << total);
    std::cout << "  total weight bytes = " << total
              << " (" << (total / (1024.0 * 1024 * 1024)) << " GB)\n";
}

void test_dtype_distribution() {
    SafeTensorFile f;
    if (!f.open(kModelPath)) SKIP("cannot open file");

    std::unordered_map<std::string, size_t> by_dtype;
    for (const auto& [_, info] : f.tensors()) {
        by_dtype[info.dtype]++;
    }
    std::cout << "  dtype distribution:\n";
    for (const auto& [dtype, count] : by_dtype) {
        std::cout << "    " << dtype << ": " << count << " tensors\n";
    }
    CHECK(!by_dtype.empty(), "dtype map should not be empty");
}

// 把前 10 个 tensor 打印出来，方便人工 inspection
void test_print_sample() {
    SafeTensorFile f;
    if (!f.open(kModelPath)) SKIP("cannot open file");

    size_t n = 0;
    std::cout << "  first 10 tensors:\n";
    for (const auto& [name, info] : f.tensors()) {
        if (n++ >= 10) break;
        std::cout << "    " << name
                  << "  dtype=" << info.dtype
                  << "  bytes=" << info.bytes
                  << "  dims=" << info.shape.size() << "\n";
    }
}

// ---------------------------------------------------------------------------
// runner
// ---------------------------------------------------------------------------

using TestFn = void (*)();

int main() {
    TestFn tests[] = {
        test_open_single,
        test_known_tensors_exist,
        test_all_tensors_verify_size,
        test_total_bytes_reasonable,
        test_dtype_distribution,
        test_print_sample,
    };

    for (TestFn t : tests) {
        std::cout << "[" << (sizeof(tests)/sizeof(tests[0]) - g_skips) << "] running...\n";
        t();
    }

    std::cout << "========================================\n";
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED";
        if (g_skips) std::cout << " (" << g_skips << " skipped)";
        std::cout << "\n";
        return 0;
    }
    std::cout << g_failures << " FAILURE(S)";
    if (g_skips) std::cout << ", " << g_skips << " skipped";
    std::cout << "\n";
    return 1;
}
