#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdio>

namespace vt { namespace cuda {

// just declare
template <typename T>
int silu_product(const T *in_act, const T* in, T* out, const int items,
            cudaStream_t stream);

template <typename T>
int rotary_embed(const T *in, const float *cos_sin,const int* pos, T* out, 
            const int bs, const int hnum, const int len, const int dims,
            cudaStream_t stream);

template <typename T>
int rsqrt(const T *in, T *out,
            const int len, float eps,
            cudaStream_t stream);

template <typename T>
int causal_mask(const int *mask, T *out,
                  const int batch,
                  const int new_tokens,
                  const int full_tokens,
                  cudaStream_t stream);

template <typename T>
int float_to_half(const float* in, T* out, const int size, cudaStream_t stream);

// --------------------------
__global__ void mask_float( const int *mask, float *out, const float minv, 
                        const int batch, const int new_tokens, const int full_tokens) {

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
int causal_mask<float>(const int *mask, float *out,
                        const int batch,
                        const int new_tokens,
                        const int full_tokens,
                        cudaStream_t stream) {

    int len = batch * new_tokens;

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    float minv = -1.0 * std::numeric_limits<float>::max();

    mask_float <<< num_of_blocks, block_size, 0, stream >>> (mask, out, minv, batch, new_tokens, full_tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch mask_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
   
    return 0;
}

__global__ void mask_fp16( const int *mask, __half *out, 
                        const int batch, const int new_tokens, const int full_tokens) {
    
    const __half minv = __ushort_as_half((unsigned short)0xFC00U);

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
int causal_mask<__half>(const int *mask, __half *out,
                        const int batch,
                        const int new_tokens,
                        const int full_tokens,
                        cudaStream_t stream) {

    int len = batch * new_tokens;

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    mask_fp16 <<< num_of_blocks, block_size, 0, stream >>> (mask, out, batch, new_tokens, full_tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch mask_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
   
    return 0;
}


//----------------
__global__ void rsqrt_float(const float *in, float *out, const int len, float eps) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= len ) {
        return;
    }
    out[e] = sqrt(1.0 / (in[e]*in[e] + eps));
}

template<>
int rsqrt<float>(const float *in, float *out, const int len, float eps, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);
   
    rsqrt_float <<< num_of_blocks, block_size, 0, stream >>> (in, out, len, eps);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch reverse_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

__global__ void rsqrt_fp16(const __half *in, __half *out, const int len, float eps) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= len ) {
        return;
    }

    float ine = __half2float(in[e]);

    out[e] = __float2half( sqrt(1.0 / (ine*ine + eps)) );
}

template<>
int rsqrt<__half>(const __half *in, __half *out, const int len, float eps, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);
  
    rsqrt_fp16 <<< num_of_blocks, block_size, 0, stream >>> (in, out, len, eps);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch reverse_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//----------------
__global__ void rotary_embed_float(const float *in, const float *cos_sin, const int* pos, float *out, 
                                   const int bs, const int hnum, const int len, const int dims) {
    size_t e = blockIdx.x * blockDim.x + threadIdx.x;
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
        out[i] = cos_sin[i*2] * x - cos_sin[i*2+1] * y;
        out[ii] = cos_sin[ii*2] * y + cos_sin[ii*2+1] * x;
    }
}

template<>
int rotary_embed<float>( const float *in, const float *cos_sin, const int* pos, float *out, 
                         const int bs, const int hnum, const int len, const int dims, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks( (bs*hnum*len + block_size.x - 1) / block_size.x);
   
    rotary_embed_float <<< num_of_blocks, block_size, 0, stream >>> (in, cos_sin, pos, out, bs, hnum, len, dims);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch  rotary_embed_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

__global__ void rotary_embed_fp16(const __half *in, const float *cos_sin, const int* pos,  __half *out,  
                                   const int bs, const int hnum, const int len, const int dims) {
    size_t e = blockIdx.x * blockDim.x + threadIdx.x;
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
        float x = __half2float(in[i]);
        float y = __half2float(in[ii]);

        out[i] = __float2half ( cos_sin[i*2] * x - cos_sin[i*2+1] * y );
        out[ii] = __float2half ( cos_sin[ii*2] * y + cos_sin[ii*2+1] * x);
    }
}

template<>
int rotary_embed<__half>( const __half *in, const float *cos_sin, const int* pos,  __half *out,  
                         const int bs, const int hnum, const int len, const int dims, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks( (bs*hnum*len + block_size.x - 1) / block_size.x);
   
    rotary_embed_fp16 <<< num_of_blocks, block_size, 0, stream >>> (in, cos_sin, pos, out, bs, hnum, len, dims);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch  rotary_embed_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//----------------
__global__ void silu_product_float(const float *in_act, const float *in, float *out, const int items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    float act = in_act[i];
    float in_ = in[i];
    out[i] = act / (1.f + __expf(-act)) * in_;
}

template<>
int silu_product<float>( const float *in_act, const float *in, float *out, 
                         const int items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);
   
    silu_product_float <<< num_of_blocks, block_size, 0, stream >>> (in_act, in, out, items);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch silu_product_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

__global__ void silu_product_fp16(const __half *in_act, const half *in, half *out, const int items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    float act = in_act[i];
    float in_ = in[i];
    out[i] = act / (1.f + __expf(-act)) * in_;
}

template<>
int silu_product<__half>( const half *in_act, const half *in, half *out, 
                         const int items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);
   
    silu_product_fp16 <<< num_of_blocks, block_size, 0, stream >>> (in_act, in, out, items);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch silu_product_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


//----------------
__global__ void float_to_fp16(const float *in, __half* out, const int items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    out[i] = __half2float(in[i]);
}

template<>
int float_to_half<__half>(const float* in, __half* out, const int items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);
   
    float_to_fp16 <<< num_of_blocks, block_size, 0, stream >>> (in, out, items);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch float_to_fp16 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
