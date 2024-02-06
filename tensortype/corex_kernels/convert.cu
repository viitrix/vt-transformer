#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "corex_kernels.hpp"

namespace vt { namespace corex {

__global__ void float_to_fp16(const float *in, __half* out, const size_t items) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    out[i] = __half2float(in[i]);
}

template<>
int kr_convert<float, __half>(const float* in, __half* out, const size_t items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);

    float_to_fp16 <<< num_of_blocks, block_size, 0, stream >>> (in, out, items);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch float_to_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
