#ifndef _VT_TEST_MOCK_BACKEND_HPP_
#define _VT_TEST_MOCK_BACKEND_HPP_

// 测试脚手架: Mock 后端实现, 仅供 test/ 下的测试使用, 不属于 core 产品代码。

#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "backend.hpp"

namespace vt {

// Mock 后端: 模拟从前端拉取请求。
// 行为:
//   - 内部维护一个调用计数, 每 10 次 "到达检查" 到达一批请求, 每批随机产出 1 个或 2 个;
//   - 请求带有 mock 模式输入: 长度随机 (8~32), TokenID 在 [1024, 2048] 内随机;
//   - 超时语义见 BackendBase 注释。
template <typename Token = int32_t>
struct MockBackend : public BackendBase<Token> {
    void tryFetchRequest(std::vector<Request<Token>>& newRequest, int timeOut) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(timeOut < 0 ? 0 : timeOut);

        while (true) {
            // 模拟到达事件: 每 10 次检查到达一批, 随机产出 1 个或 2 个请求
            tick_++;
            if (tick_ % 10 == 0) {
                int n = 1 + (int)(gen_() & 1);       // 1 或 2, 各一半概率
                for (int i = 0; i < n; i++) {
                    Request<Token>& req = newRequest.emplace_back(gen_());
                    fill_mock_input(req);
                }
                return;
            }

            if (timeOut == 0) {
                return;  // 非阻塞: 一次检查后立即返回
            }
            if (timeOut > 0 && std::chrono::steady_clock::now() >= deadline) {
                return;  // 超时: 请求未到, 空手而归
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    // 填充 mock 模式输入: 长度随机 (8~32), TokenID 在 [1024, 2048] 内均匀随机
    void fill_mock_input(Request<Token>& req) {
        static constexpr int kMinTok = 1024, kMaxTok = 2048;
        static constexpr int kMinLen = 8,     kMaxLen = 32;

        int len = kMinLen + (int)(gen_() % (uint64_t)(kMaxLen - kMinLen + 1));
        req.input_ids.reserve(len);
        for (int i = 0; i < len; i++) {
            req.input_ids.push_back((Token)(kMinTok + gen_() % (uint64_t)(kMaxTok - kMinTok + 1)));
        }
    }

    std::mutex mu_;
    int tick_ = 0;                       // 到达检查计数
    std::mt19937_64 gen_{12345};         // mock 随机源: rid / 个数 / 长度 / TokenID 共用
};

} // end of namespace
#endif
