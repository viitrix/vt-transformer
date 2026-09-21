// CUDA 环境下实现的 Overlap 调度
// 基于各种 CudaStrem/CudaEvent 机制实现，GPU 侧无气泡的运行

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <chrono>
#include <string>

#include "vt_cuda.hpp"
#include "scheduler.hpp"

namespace vt {



template <std::size_t kMaxSeqLen, int kMaxReq, int kPageSize, typename Token, typename Index>
int Scheduler<kMaxSeqLen, kMaxReq, kPageSize, Token, Index>::main_overlap_loop() {
    Enviroment* env = engine_->env_;
    CudaContext* ctx = (CudaContext *)env->ctx();
    
    // stream 0 : CPU->GPU 用; stream 1 : GPU 用; stream 2 : GPU->CPU 用
    // event j       : buffer j 的输入已就绪   (stream 0 记录)
    // event j + 2   : buffer j 的 GPU 输出已就绪 (stream 1 记录)
    // event j + 4   : buffer j 的输出已拷回 host (stream 2 记录)

    auto* batch_fifo = (vt::BatchTable<kMaxReq, Token, Index> *)env->hash().find_tensor("batch_table_host")->data();

    // 预热: 预记录 buffer 1 的三个事件, 保证 i=0 时 wait/sync 不被阻塞
    env->execute(R"(
        0 cuda.set_stream 1 cuda.record_event
        1 cuda.set_stream 3 cuda.record_event
        2 cuda.set_stream 5 cuda.record_event)");

    
    for(int i = 0; i < 10; i++) {
        const int cur = i % 2;
        const int prev = (i + 1) % 2;

        // 0. 从 backend 端获取请求, 写入 host 缓冲 cur, 并记录 batch_size
        

        // 1. 构建当前请求的输入 (写 host 缓冲 cur, 此时 GPU 正在算 prev, 无干扰),
        //    在 stream 0 上发起 H2D 拷贝并记录 event cur: 输入已就绪
        batch_fifo[cur].batch_size = 0;
        env->execute("0 cuda.set_stream \"batch_table_host\" @ \"batch_table\" @ cuda.to_device");
        env->execute(std::to_string(cur) + " cuda.record_event");

        // 2. 启动当前的批处理: stream 1 先等 cur 的输入就绪(跨流依赖, 关键!),
        //    forward 结束记录 event cur+2: 输出已就绪
        env->execute("1 cuda.set_stream " + std::to_string(cur) + " cuda.wait_event");
        engine_->forward(batch_ + cur, _batch_ + cur);
        std::cout << "I got forward!" << std::endl;
        env->execute("1 cuda.set_stream " + std::to_string(cur + 2) + " cuda.record_event");

        // 3. 回传: stream 2 等 cur 的 GPU 输出就绪, 发起 D2H 并记录 event cur+4
        env->execute("2 cuda.set_stream " + std::to_string(cur + 2) + " cuda.wait_event");
        // TODO: env->execute(... cuda.to_host ...)     // D2H: device cur -> host cur
        env->execute("2 cuda.set_stream " + std::to_string(cur + 4) + " cuda.record_event");

        // 4. 消费 prev 的输出: 等 prev 的 D2H 拷贝完成 (而不是 GPU 算完),
        //    host 侧读取/调度 prev 的结果; 同时这也是复用缓冲的最小同步门槛
        env->execute( std::to_string(prev + 4) + " cuda.sync_event");
        std::cout << "I got synced!" << std::endl;
    }

    return 0;
}

} // namespace vt

// 显式实例化: main_overlap_loop 的实现独立于头文件, 按测试使用的配置实例化
template struct vt::Scheduler<1024, 4, 4, int32_t, int32_t>;
