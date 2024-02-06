#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "corex_kernels.hpp"
namespace vt { namespace corex {

template<typename T>
__global__ void add_bias(const T* in, const T* bias, T* out, int length, int feature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= length * feature ) {
        return;
    }

    float a = in[i];
    float b = bias[i % feature];
    out[i] = a + b;
}

template <>
int kr_add_bias<float>(const float* in, const float* bias, float* out, int length, int feature, cudaStream_t stream) {
    int nElementNumber = length * feature;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    add_bias<float> <<< num_of_blocks, block_size, 0, stream >>> (in, bias, out, length, feature);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch add_bias kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_add_bias<__half>(const __half* in, const __half* bias, __half* out, int length, int feature, cudaStream_t stream) {
    int nElementNumber = length * feature;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    add_bias<__half> <<< num_of_blocks, block_size, 0, stream >>> (in, bias, out, length, feature);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch add_bias kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

// ===============================================
template<typename T>
__global__ void add_broadcast(const T* in, const T* bias, T* out, int length, int inter, int feature) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= length * feature * inter) {
        return;
    }

    int l = i / ( feature * inter );
    int f = i % feature;

    float a = in[i];
    float b = bias[l * feature + f];
    out[i] = a + b;
}

template <>
int kr_add_broadcast<float>(const float* in, const float* bias, float* out, int length, size_t inter, size_t feature, cudaStream_t stream) {
    size_t nElementNumber = length * feature * inter;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    add_broadcast<float> <<< num_of_blocks, block_size, 0, stream >>> (in, bias, out, length, inter, feature);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch add_bias kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_add_broadcast<__half>(const __half* in, const __half* bias, __half* out, int length, size_t inter, size_t feature, cudaStream_t stream) {
    size_t nElementNumber = length * feature * inter;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    add_broadcast<__half> <<< num_of_blocks, block_size, 0, stream >>> (in, bias, out, length, inter, feature);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch add_bias kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
