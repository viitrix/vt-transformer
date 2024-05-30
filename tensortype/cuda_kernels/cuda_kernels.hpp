#ifndef _CUDA_KERNELS_HPP_
#define _CUDA_KERNELS_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>

namespace vt { namespace cuda {

void lt_sgemm(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha, /* host pointer */
             const void *A, cudaDataType_t at,
             int lda,
             const void *B, cudaDataType_t bt,
             int ldb,
             const float *beta, /* host pointer */
             void *C, cudaDataType_t ct,
             int ldc,
             void *workspace,
             size_t workspaceSize);

void lt_sgemm_batched(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha, /* host pointer */
             const void *A, cudaDataType_t at,
             int lda,
             const void *B, cudaDataType_t bt,
             int ldb,
             const float *beta, /* host pointer */
             void *C, cudaDataType_t ct,
             int ldc,
             int batchCount,
             void *workspace,
             size_t workspaceSize);


template <typename T>
int kr_causal_mask(const int *mask, T *out, const size_t batch, const size_t new_tokens, const size_t full_tokens, cudaStream_t stream);

template <typename TF, typename TT>
int kr_convert(const TF* src, TT* out, size_t items, cudaStream_t stream);

template <typename T>
int kr_layernorm(T *ln_res, T *vars, T *means,
                 const T *inp, const T *scale,
                 const T *bias, size_t batch_size, size_t hidden_dim, float eps,
                 cudaStream_t stream);

template <typename T>
int kr_rmsnorm(const T *feature, const T *w,
             T *out, T *norm2,
             const size_t batch,
             const size_t dim,
             const float eps,
             cudaStream_t stream);

template <typename T>
int kr_rotary_embed(const T *in, const float *cos_sin,const int* pos, T* out, const size_t bs, const size_t hnum, const size_t len, const size_t dims, cudaStream_t stream);

template <typename T>
int kr_transpose_0213(const T* src, T* target, size_t a, size_t b, size_t c ,size_t d, cudaStream_t stream);

template <typename T>
int kr_transpose_0213_repeated(const T* src, T* target, size_t a, size_t b, size_t cf , size_t ct, size_t d, cudaStream_t stream);

template <typename T>
int kr_gelu(const T* src, T* target, size_t items, cudaStream_t stream);

}}
#endif
