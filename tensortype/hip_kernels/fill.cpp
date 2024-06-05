#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_kernels.hpp"

namespace vt { namespace hip {

template<typename T>
__global__ void fill(T* target, float value, size_t nElementNumber) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nElementNumber) {
        target[i] = (T)value;
    }
}

template<>
int kr_fill<float>(float* target, float value, size_t nElementNumber, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    fill<float> <<< num_of_blocks, block_size, 0, stream >>> (target, value, nElementNumber);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch fill kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_fill<int>(int* target, float value, size_t nElementNumber, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    fill<int> <<< num_of_blocks, block_size, 0, stream >>> (target, value, nElementNumber);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch fill kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_fill<__half>(__half* target, float value, size_t nElementNumber, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    fill<__half> <<< num_of_blocks, block_size, 0, stream >>> (target, value, nElementNumber);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch fill kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}
