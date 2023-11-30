#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "dcu_kernels.hpp"

namespace vt { namespace dcu {

//===========================================
// scale : A = B * alpha + beta
template<typename T>
__global__ void scale(const T* src, T* target, const float alpha, const float beta, const int n ) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
        target[i] = (float)src[i] * alpha + beta;
    }
}

template<>
int kr_scale<float>(const float* src, float* target, const float alpha, const float beta, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    scale<float> <<< num_of_blocks, block_size, 0, stream >>> (src, target, alpha, beta, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch scale kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_scale<__half>(const __half* src, __half* target, const float alpha, const float beta, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    scale<__half> <<< num_of_blocks, block_size, 0, stream >>> (src, target, alpha, beta, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch scale kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//===========================================
// add : C = A + B
template<typename T, typename TT>
__global__ void add(const T* A, const TT* B, T* C, const int n ) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
        C[i] = (T)((float)A[i] + (float)B[i]);
    }
}

template<>
int kr_add<float, float>(const float* A, const float* B, float* C, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    add<float, float> <<< num_of_blocks, block_size, 0, stream >>> (A, B, C, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch add kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_add<__half,__half>(const __half* A, const __half* B, __half* C, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    add<__half, __half> <<< num_of_blocks, block_size, 0, stream >>> (A, B, C, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch add kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_add<__half, float>(const __half* A, const float* B, __half* C, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    add<__half, float> <<< num_of_blocks, block_size, 0, stream >>> (A, B, C, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch add kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//===========================================
// mul : C = A .* B
template<typename T, typename TT>
__global__ void mul(const T* A, const TT* B, T* C, const int n ) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
        C[i] = (T)((float)A[i] * (float)B[i]);
    }
}

template<>
int kr_mul<float, float>(const float* A, const float* B, float* C, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    mul<float, float> <<< num_of_blocks, block_size, 0, stream >>> (A, B, C, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch mul kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_mul<__half,__half>(const __half* A, const __half* B, __half* C, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    mul<__half, __half> <<< num_of_blocks, block_size, 0, stream >>> (A, B, C, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch mul kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_mul<__half, float>(const __half* A, const float* B, __half* C, int n, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((n + block_size.x - 1) / block_size.x);

    mul<__half, float> <<< num_of_blocks, block_size, 0, stream >>> (A, B, C, n);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch mul kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}
