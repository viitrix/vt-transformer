// 定制的 CUDA kernel 代码: kernel 定义与对应的 host 侧启动接口。
// 本文件必须以 .cu 后缀参与编译 (nvcc device 编译);
// 对外的函数声明放在 vt_cuda.hpp, host 代码只 include 头文件。

#include "vt_cuda.hpp"

namespace vt {

// 延迟 kernel: 在 GPU 上空转约 delay_ms 毫秒。
// clock64() 频率通常约 1GHz, 按 1e6 cycles/ms 估算。
__global__ void spin_delay_kernel(long long delay_ms) {
    long long start = clock64();
    long long cycles = delay_ms * 1000000LL;
    while (clock64() - start < cycles) {
    }
}

void launch_delay_kernel(cudaStream_t stream, long long delay_ms) {
    spin_delay_kernel<<<1, 1, 0, stream>>>(delay_ms);
}

} // namespace vt
