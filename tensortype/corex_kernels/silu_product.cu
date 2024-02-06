#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "corex_kernels.hpp"
namespace vt { namespace corex {

template<typename T>
__global__ void silu_product(const T* in_act, const T* in, T* out,  int items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    float act = in_act[i];
    float in_ = in[i];
    out[i] = (T)(act / (1.f + exp(-act)) * in_);
}

template <>
int kr_silu_product<float>(const float* in_act, const float* in, float* out, int nElementNumber, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    silu_product<float> <<< num_of_blocks, block_size, 0, stream >>> (in_act, in, out, nElementNumber);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch silu kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_silu_product<__half>(const __half* in_act, const __half* in, __half* out, int nElementNumber, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    silu_product<__half> <<< num_of_blocks, block_size, 0, stream >>> (in_act, in, out, nElementNumber);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch silu kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
