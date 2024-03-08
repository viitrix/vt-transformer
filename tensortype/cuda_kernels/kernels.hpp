#ifndef _ALL_KERNELS_HPP_
#define _ALL_KERNELS_HPP_

#include <cublasLt.h>
#include "lightseq/kernels.h"

namespace vt { namespace cuda {

template <typename T>
int linear2d_q8(const T *in, const void* w, T *out, int M, int N, int K, cudaStream_t stream);

template <typename T>
int linear2d_q4(const T *in, const void* w, T *out, int M, int N, int K, cudaStream_t stream);

template <typename T>
int quantize_q8(const T *input, void *out, int feature_num, int feature_size, cudaStream_t stream);

template <typename T>
int quantize_q4(const T *input, void *out, int items, cudaStream_t stream);

template <typename T>
int dequantize_q8(const void *input, T *out, int feature_num, int feature_size, cudaStream_t stream);

template <typename T>
int dequantize_q4(const void *input, T *out, int items, cudaStream_t stream);

template <typename T>
int dequantize_pq(const void *tab, const uint8_t *idx, T *out, int items, int S, cudaStream_t stream);

template <typename T>
int rms_norm(const T *feature, const T *w,
             T *out, T *norm2,
             const int batch,
             const int dim,
             const float eps,
             cudaStream_t stream);

template <typename T>
int easy_top1(const T *logits, int *out,
              const int batch,
              const int vocab_size,
              cudaStream_t stream);

template <typename T>
int easy_top3(const T *logits, int *out,
              const int batch,
              const int vocab_size,
              const float temperature,
              const float randx,
              cudaStream_t stream);

template <typename T>
int float_to_half(const float* in, T* out, const int size, cudaStream_t stream);

template <typename T>
int silu_product(const T *in_act, const T* in, T* out, const int items,
            cudaStream_t stream);

template <typename T>
int rotary_embed(const T *in, const float *cos_sin, const int* pos, T* out,
            const int bs, const int hnum, const int len, const int dims,
            cudaStream_t stream);

template <typename T>
int rsqrt(const T *in, T *out,
            const int len,
            float eps,
            cudaStream_t stream);

template <typename T>
int embed_forward(const int *ids, const T *table, T *out,
                  const int len,
                  const int hidden_size,
                  cudaStream_t stream);

template <typename T>
int causal_mask(const int *mask, T *out,
                  const int batch,
                  const int new_tokens,
                  const int full_tokens,
                  cudaStream_t stream);

template <typename T>
int gelu_forward(const T* src, T* target, int nElementNumber, cudaStream_t stream);

int gelu_backward(const float* out_g, const float* xi, float* x_g, int nElementNumber, cudaStream_t stream);
int nllloss_forward(const int* ids, const float* logsoftmax, float *output, float *dout, int n, int vocab, float loss_scale, cudaStream_t stream);
void LtSgemm(cublasLtHandle_t ltHandle,
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

void LtSgemmBatched(cublasLtHandle_t ltHandle,
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

}}

#endif
