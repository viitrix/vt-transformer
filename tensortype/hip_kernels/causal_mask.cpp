#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_kernels.hpp"

namespace vt { namespace hip {

__global__ void mask_float( const int *mask, float *out, const float minv,
                            const size_t batch, const size_t new_tokens, const size_t full_tokens) {

    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= batch * new_tokens ) {
        return;
    }

    int b = e / new_tokens;
    int nt = e % new_tokens;
    int nt_end = full_tokens - new_tokens + nt;

    const int* m = &mask[ b * full_tokens];
    float* o = &out[b * new_tokens * full_tokens + nt * full_tokens];

    for ( int i = 0;  i < full_tokens; i++) {
        o[i] = minv;
    }

    for ( int i = 0; i <= nt_end; i++) {
        if ( m[i] != 0 ) {
            o[i] = 0.0;
        }
    }
}

template <>
int kr_causal_mask<float>(const int *mask, float *out,
                          const size_t batch, const size_t new_tokens, const size_t full_tokens,
                          hipStream_t stream) {

    int len = batch * new_tokens;

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    float minv = -1.0 * std::numeric_limits<float>::max();

    mask_float <<< num_of_blocks, block_size, 0, stream >>> (mask, out, minv, batch, new_tokens, full_tokens);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch mask_float kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }

    return 0;
}

__global__ void mask_fp16( const int *mask, __half *out,
                        const int batch, const int new_tokens, const int full_tokens) {

    const __half minv = -65504;

    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= batch * new_tokens ) {
        return;
    }

    int b = e / new_tokens;
    int nt = e % new_tokens;
    int nt_end = full_tokens - new_tokens + nt;

    const int* m = &mask[ b * full_tokens];
    __half* o = &out[b * new_tokens * full_tokens + nt * full_tokens];

    for ( int i = 0;  i < full_tokens; i++) {
        o[i] = minv;
    }

    for ( int i = 0; i <= nt_end; i++) {
        if ( m[i] != 0 ) {
            o[i] = 0.0;
        }
    }
}

template <>
int kr_causal_mask<__half>(const int *mask, __half *out,
                           const size_t batch, const size_t new_tokens, const size_t full_tokens,
                           hipStream_t stream) {

    int len = batch * new_tokens;

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    mask_fp16 <<< num_of_blocks, block_size, 0, stream >>> (mask, out, batch, new_tokens, full_tokens);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch mask_float kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }

    return 0;
}


}}
