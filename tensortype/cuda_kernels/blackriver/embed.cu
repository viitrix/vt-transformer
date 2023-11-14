#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdio>

namespace vt { namespace cuda {

// just declare 
template <typename T>
int embed_forward(const int *ids, const T *table, T *out,
                  const int len,
                  const int hidden_size,
                  cudaStream_t stream);

__global__ void embed_float( const int *ids, const float *table, float *out,
                        const int len,
                        const int hidden_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= len ) {
        return;
    }

    int id = ids[i];
    const float* src = table + id * hidden_size;
    float* dst = out + i * hidden_size;
    for (int i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}

__global__ void embed_fp16( const int *ids, const __half *table, __half *out,
                        const int len,
                        const int hidden_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= len ) {
        return;
    }

    int id = ids[i];
    const __half* src = table + id * hidden_size;
    __half* dst = out + i * hidden_size;
    for (int i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}

// implement for float and __half
template <>
int embed_forward<float>(const int *ids, const float *table, float *out,
                        const int len,
                        const int hidden_size,
                        cudaStream_t stream) {

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    embed_float <<< num_of_blocks, block_size, 0, stream >>> (ids, table, out, len, hidden_size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
   
    return 0;
}

template <>
int embed_forward<__half>(const int *ids, const __half* table, __half* out,
                        const int len,
                        const int hidden_size,
                        cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    embed_fp16 <<< num_of_blocks, block_size, 0, stream >>> (ids, table, out, len, hidden_size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
   
    return 0;
}


}}
