#include <float.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "dcu_kernels.hpp"
#include "quantize.hpp"

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace vt { namespace dcu {

template <typename T>
__global__ void dequantize_pq(const __half *tab_, const uint8_t *idx, T *out, const size_t items, int S) {
    __shared__ __half tab[128];

    const int SS = blockIdx.x;
    const int DIM = blockDim.x;
    const int INDEX = threadIdx.x;

    int offset = SS * 128;
    for (int i = INDEX; i < 128; i += DIM ) {
        tab[i] = (__half)tab_[offset + i];
    }
    __syncthreads();

    const int gsize = items / S;
    int *src;
    int *dst;
    for (int i = gsize * SS + INDEX*3; i < gsize * SS + gsize; i += DIM*3) {
        uint8_t a = idx[i + 0];
        uint8_t b = idx[i + 1];
        uint8_t c = idx[i + 2];

        uint8_t i0 = a >> 2;
        uint8_t i1 = ( (a & 0x3) << 4)  + (b >> 4);
        uint8_t i2 = ( (b & 0x0F) << 2) + (c >> 6);
        uint8_t i3 = c & 0x3F;

        int ii = i / 3 * 8;
        src = (int *)&tab[i0 * 2];
        dst = (int *)&out[ii + 0 ];
        *dst = *src;

        src = (int *)&tab[i1 * 2];
        dst = (int *)&out[ii + 2];
        *dst = *src;

        src = (int *)&tab[i2 * 2];
        dst = (int *)&out[ii + 4];
        *dst = *src;

        src = (int *)&tab[i3 * 2];
        dst = (int *)&out[ii + 6];
        *dst = *src;
    }
}

template <>
int kr_dequantize_pq<__half>(const void *tab, const uint8_t *idx, __half *out, const size_t items_, int S, hipStream_t stream) {
    dim3 block_size(256);
    dim3 num_of_blocks(S);

    size_t items = items_ * 3 / 8;
    dequantize_pq<__half> <<< num_of_blocks, block_size, 0, stream>>> ((__half *)tab, idx, out, items, S);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch dequantize_pq kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }

    return 0;
}
// ===================================================

template<typename T>
__global__ void quantize_q4_kernel(const T *in, void *out, const size_t items) {
    size_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if ( blk * Q4_BLOCK_SIZE >= items ) {
        return;
    }

    q4_block_t* target = (q4_block_t *)out;
    target = &target[blk];
    const T* src = &in[blk * Q4_BLOCK_SIZE];

    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int i = 1; i < Q4_BLOCK_SIZE; i++) {
        const float v = src[i];
        if ( v < min ) min = v;
        if ( v > max ) max = v;
    }

    const float d  = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;
    target->m = min;
    target->d = d;

    for (int i = 0; i < Q4_BLOCK_SIZE / 2; i++) {
        const float x0 = ((float )src[i*2 + 0] - min) * id;
        const float x1 = ((float )src[i*2 + 1] - min) * id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        target->q[i]  = xi0;
        target->q[i] |= xi1 << 4;
    }
}

template <>
int kr_quantize_q4<float>(const float *in, void *out, const size_t items, hipStream_t stream) {
    int blk_num = items / Q4_BLOCK_SIZE;

    dim3 block_size(256);
	dim3 num_of_blocks((blk_num + block_size.x - 1) / block_size.x);

    quantize_q4_kernel<float> <<< num_of_blocks, block_size, 0, stream>>> (in, out, items);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch quantize_float_q4 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_quantize_q4<__half>(const __half *in, void *out, const size_t items, hipStream_t stream) {
    int blk_num = items / Q4_BLOCK_SIZE;

    dim3 block_size(256);
	dim3 num_of_blocks((blk_num + block_size.x - 1) / block_size.x);

    quantize_q4_kernel<__half> <<< num_of_blocks, block_size, 0, stream>>> (in, out, items);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch quantize_half_q4 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template<typename T>
__global__ void dequantize_q4_kernel(const void *in, T *out, const size_t items) {
    size_t blk = blockIdx.x;
    int idx = threadIdx.x;
    if ( blk * Q4_BLOCK_SIZE >= items ) {
        return;
    }

    const q4_block_t* target = (q4_block_t *)in;
    target = &target[blk];

    __shared__ float d;
    __shared__ float m;
    if ( idx == 0 ) {
        d = target->d;
        m = target->m;
    }
    __syncthreads();

    const uint8_t vu = target->q[ idx / 2];

    uint8_t qv;
    if ( idx % 2) {
        qv = vu >> 4;
    } else {
        qv = vu & 0xF;
    }

    out[blk * Q4_BLOCK_SIZE + idx] = qv * d + m;
}

template <>
int kr_dequantize_q4<__half>(const void *in, __half *out, const size_t items, hipStream_t stream) {
    dim3 block_size(Q4_BLOCK_SIZE);
    dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);

    dequantize_q4_kernel<__half> <<< num_of_blocks, block_size, 0, stream>>> (in, out, items);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch dequantize_half_q4 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

// ===================================================

template<typename T>
__global__ void quantize_q8_kernel(const T *in, void *out, int feature_num, int feature_size) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if ( l >= feature_num ) {
        return;
    }

    int stride_out = feature_size + sizeof(float) * 2;
    q8_head_t* target = (q8_head_t *)( (char *)out + stride_out * l );
    in = &in[ l * feature_size ];

    float min = FLT_MAX;
    float max = -FLT_MAX;
    for(int i = 0; i < feature_size; i++) {
        float v = in[i];
        if ( v < min ) min = v;
        if ( v > max ) max = v;
    }

    const float d  = (max - min) / ((1 << 8) - 1);
    const float id = d ? 1.0f/d : 0.0f;
    target->m = min;
    target->d = d;

    for (int i = 0; i < feature_size; i++) {
        float v = ( (float)in[i] - min ) * id;
        target->q[i] = (uint8_t)(v + 0.5);
    }
}

template <>
int kr_quantize_q8<__half>(const __half *input, void *out, int feature_num, int feature_size, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((feature_num + block_size.x - 1) / block_size.x);

    quantize_q8_kernel<__half> <<< num_of_blocks, block_size, 0, stream>>> (input, out, feature_num, feature_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch quantize_half_q8 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }

    return 0;
}

template <>
int kr_quantize_q8<float>(const float *input, void *out, int feature_num, int feature_size, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((feature_num + block_size.x - 1) / block_size.x);

    quantize_q8_kernel<float> <<< num_of_blocks, block_size, 0, stream>>> (input, out, feature_num, feature_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch quantize_float_q8 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }

    return 0;
}

template<typename T>
__global__ void dequantize_q8_kernel(const uint8_t *in, T *out, int feature_num, int feature_size) {
    const int l = blockIdx.x;
    const int SUB = feature_size / blockDim.x;

    q8_head_t* q8 = (q8_head_t *) ( in + ( feature_size + 2 * sizeof(float) ) * l);
    float m = q8->m;
    float d = q8->d;

    out = out + feature_size * l;
    for(int i = threadIdx.x; i < feature_size; i += SUB) {
        out[i] = (T)(q8->q[i] * d + m);
    }
}

template <>
int kr_dequantize_q8<__half>(const void *input, __half *out, int feature_num, int feature_size, hipStream_t stream) {
    dim3 block_size(128);
    dim3 num_of_blocks(feature_num);

    dequantize_q8_kernel<__half> <<< num_of_blocks, block_size, 0, stream>>> ((const uint8_t*)input, out, feature_num, feature_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch quantize_float_q8 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
