#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_kernels.hpp"

namespace vt { namespace cuda {

template<typename TF, typename TT>
__global__ void convert( const TF* src, TT* dst, const size_t items) {
    size_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= items ) {
        return;
    }
    
    dst[e] = (TT)src[e]; 
}

template <>
int kr_convert<float, __half>(const float* src, __half* dst, size_t items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);

    convert<float, __half> <<< num_of_blocks, block_size, 0, stream >>> (src, dst, items);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch convert kernel (from float to half) (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_convert<__half, float>(const __half* src, float* dst, size_t items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);

    convert<__half, float> <<< num_of_blocks, block_size, 0, stream >>> (src, dst, items);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch convert kernel (from float to half) (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
