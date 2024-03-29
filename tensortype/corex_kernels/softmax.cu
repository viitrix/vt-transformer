#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "corex_kernels.hpp"
#include "block_reduce.hpp"

namespace vt { namespace corex {

const int BUFSIZE = 32;
__global__ void softmax_float(const float *in, float *out, int length, int hidden_size) {
    float buf[BUFSIZE];
    int i;

    // step 0. compute local max
    float l_max = -FLT_MAX;
    const float *inp = in + blockIdx.x * hidden_size;

    i = 0;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x, i++) {
        buf[i] = inp[idx];
        l_max = max(buf[i], l_max);
    }

    // step 1. compute reduce max
    blockReduceMax<float>(&l_max);

    // setp 2. compute exp
    i = 0;
    float l_sum = 0.0;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x, i++) {
        buf[i] = exp(buf[i] - l_max);
        l_sum += buf[i];
    }
    blockReduceSum<float>(&l_sum);

    // step 3. softmax result
    float *output = out + blockIdx.x * hidden_size;
    i = 0;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x, i++) {
        output[idx] = buf[i] / l_sum;
    }
}

__global__ void softmax_half(const __half *in, __half *out, int length, int hidden_size) {
    float4 buf[BUFSIZE];
    int i;

    // step 0. compute local max
    float l_max = -FLT_MAX;
    const float4 *inp_f4 =
        reinterpret_cast<const float4 *>(in) + blockIdx.x * hidden_size;

    i = 0;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x, i++) {
        buf[i] = inp_f4[idx];
        __half2 *val_h2 = (__half2 *)(&buf[i]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 val_f2 = __half22float2(val_h2[j]);
            float v = max(val_f2.x, val_f2.y);
            l_max = max(v, l_max);
        }
    }

    // step 1. compute reduce max
    blockReduceMax<float>(&l_max);

    // setp 2. compute exp
    i = 0;
    float l_sum = 0.0;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x, i++) {
        __half2 *val_h2 = (__half2 *)(&buf[i]);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 val_f2 = __half22float2(val_h2[j]);
            val_f2.x = exp(val_f2.x - l_max);
            val_f2.y = exp(val_f2.y - l_max);
            l_sum = val_f2.x + val_f2.y;
            val_h2[j] = __float22half2_rn(val_f2);
        }
    }
    blockReduceSum<float>(&l_sum);

    // step 3. softmax result
    float4 *output_f4 =
        reinterpret_cast<float4 *>(out) + blockIdx.x * hidden_size;

    i = 0;
    for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x, i++) {
        __half2 *val_h2 = (__half2 *)(&buf[i]);

 #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 val_f2 = __half22float2(val_h2[j]);
            val_f2.x = val_f2.x / l_sum;
            val_f2.y = val_f2.y / l_sum;
            val_h2[j] = __float22half2_rn(val_f2);
        }
        output_f4[idx] = buf[i];
    }


}


template <>
int kr_softmax<float>(const float *in, float *out, int length, int hidden_dim, cudaStream_t stream) {
    const int nThreads = 256;
    if ( hidden_dim / 256 >= BUFSIZE ) {
        fprintf(stderr, "hidden_dim is too big");
        exit(-1); 
    }

    dim3 grid_dim(length);
    dim3 block_dim(nThreads);

    softmax_float<<<grid_dim, block_dim, 0, stream>>>( in, out, length, hidden_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch softmax kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_softmax<__half>(const __half *in, __half *out, int length, int hidden_dim, cudaStream_t stream) {
    const int nThreads = 256;
    if (hidden_dim % 8 != 0) {
        fprintf(stderr, "violate hidden_dim aligen 8");
        exit(-1); 
    }
    hidden_dim >>= 3;

    if ( hidden_dim / 256 >= BUFSIZE ) {
        fprintf(stderr, "hidden_dim is too big");
        exit(-1); 
    }

    dim3 grid_dim(length);
    dim3 block_dim(nThreads);

    softmax_half<<<grid_dim, block_dim, 0, stream>>>( in, out, length, hidden_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch softmax kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}
