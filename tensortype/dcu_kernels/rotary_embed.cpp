#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "dcu_kernels.hpp"

namespace vt { namespace dcu {
template<typename T>
 __global__ void rotary_embed(const T *in, const float *cos_sin, const int* pos, T *out,
                                   const int bs, const int hnum, const int len, const int dims) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= bs * len * hnum ) {
        return;
    }
    in = in + e * dims;
    out = out + e * dims;

    int b = e / (len * hnum);
    int l = (e - b * len * hnum) / hnum + pos[b];
    cos_sin = cos_sin + l * dims * 2;

    for (int i = 0; i < dims / 2; i++) {
        int ii = i + dims/2;
        float x = in[i];
        float y = in[ii];
        out[i] = (T)(cos_sin[i*2] * x - cos_sin[i*2+1] * y);
        out[ii] = (T)(cos_sin[ii*2] * y + cos_sin[ii*2+1] * x);
    }
}

template<>
int kr_rotary_embed<float>( const float *in, const float *cos_sin, const int* pos, float *out,
                         const int bs, const int hnum, const int len, const int dims, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks( (bs*hnum*len + block_size.x - 1) / block_size.x);

    rotary_embed<float> <<< num_of_blocks, block_size, 0, stream >>> (in, cos_sin, pos, out, bs, hnum, len, dims);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch  rotary_embed_float kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<>
int kr_rotary_embed<__half>( const __half *in, const float *cos_sin, const int* pos,  __half *out,
                         const int bs, const int hnum, const int len, const int dims, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks( (bs*hnum*len + block_size.x - 1) / block_size.x);

    rotary_embed<__half> <<< num_of_blocks, block_size, 0, stream >>> (in, cos_sin, pos, out, bs, hnum, len, dims);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch  rotary_embed_fp16 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}
