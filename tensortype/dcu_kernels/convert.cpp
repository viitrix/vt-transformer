#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "dcu_kernels.hpp"

namespace vt { namespace dcu {

__global__ void float_to_fp16(const float *in, __half* out, const int items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    out[i] = __half2float(in[i]);
}

template<>
int kr_convert<float, __half>(const float* in, __half* out, const int items, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);

    float_to_fp16 <<< num_of_blocks, block_size, 0, stream >>> (in, out, items);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch float_to_fp16 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
