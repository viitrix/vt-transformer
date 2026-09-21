#ifndef _VT_TEST_MOCK_ENGINE_HPP_
#define _VT_TEST_MOCK_ENGINE_HPP_

// 测试脚手架: Mock 推理引擎实现, 仅供 test/ 下的测试使用, 不属于 core 产品代码。

#include <cstdint>

#include "engine.hpp"

namespace vt {

// Mock 引擎: 模拟一次 forward 推理。
// 测试使用的固定配置: kMaxSeqLen = 1024, kMaxReq = 4, kPageSize = 4
struct MockEngine : public EngineBase<1024, 4, 4> {
    explicit MockEngine(Enviroment* env) : EngineBase(env) {}

    int forward(BatchTable<4>* host_batch, BatchTable<4>* device_batch) override {
        (void)host_batch; (void)device_batch;   // mock: 不做真实计算, 不产出新 token
        return 0;
    }
};

} // end of namespace
#endif
