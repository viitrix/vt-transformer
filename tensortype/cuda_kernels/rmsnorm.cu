#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_kernels.hpp"

namespace vt { namespace cuda {
    template<typename T>
__global__ void rms_norm_kernel(const T *feature, const T *w,
                                T *out, T *norm2,
                                const int batch, const int dim, const float eps) {
    const int THREAD_NUMBER = 256;
    const int BLOCK_DIM = dim / THREAD_NUMBER;
    __shared__ float _all_value_[THREAD_NUMBER];

    int index = blockIdx.x * dim + threadIdx.x * BLOCK_DIM;
    
    float sum = 0.0;
    for (int i = 0; i < BLOCK_DIM; i++) {
        float s = (float) feature[index + i];
        sum = sum + s * s;
    }
    _all_value_[threadIdx.x] = sum;
    __syncthreads();

    if ( threadIdx.x == 0 ) {
        sum = 0.0;
        for ( int i = 0; i < THREAD_NUMBER; i++) {
            sum = sum + _all_value_[i];
        }
        sum = sum / (float)dim;
        sum = rsqrt( sum + eps);
        
        norm2[blockIdx.x] = (T)sum;
        _all_value_[0] = sum;    
    }
    __syncthreads();

    sum = _all_value_[0];
    int ii = threadIdx.x * BLOCK_DIM; 
    for (int i = 0; i < BLOCK_DIM; i++) {
        float fv = (float) feature[index + i];
        float wv = (float) w[ii + i];
        float ov = wv * fv * sum;
        out[index + i] = (T) ov;
    }
}
 
template<>
int kr_rmsnorm<__half>(const __half* feature, const __half *w, __half *out, __half *norm2, 
                     const int batch, const int dim, const float eps, cudaStream_t stream) {
    dim3 num_of_blocks(batch);
    dim3 block_size(256);
    
    if ( dim % 256 != 0) {
        fprintf(stderr, "rms_norm kernel only support dim mod 256 == 0!\n");
        exit(-1);
    }

    rms_norm_kernel<__half> <<< num_of_blocks, block_size, 0, stream >>> (feature, w, out, norm2, batch, dim, eps);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rms_norm kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_rmsnorm<float>(const float* feature, const float *w, float *out, float *norm2,
                     const int batch, const int dim, const float eps, cudaStream_t stream) {
    dim3 num_of_blocks(batch);
    dim3 block_size(256);

    if ( dim % 256 != 0) {
        fprintf(stderr, "rms_norm kernel only support dim mod 256 == 0!\n");
        exit(-1);
    }

    rms_norm_kernel<float> <<< num_of_blocks, block_size, 0, stream >>> (feature, w, out, norm2, batch, dim, eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rms_norm kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}