#ifndef _COREX_KERNELS_HPP_
#define _COREX_KERNELS_HPP_

namespace vt { namespace corex {

template<typename T>
int kr_fill(T* target, float value, int nElementNumber, cudaStream_t stream);

template <typename T>
int kr_causal_mask(const int *mask, T *out, const int batch, const int new_tokens, const int full_tokens, cudaStream_t stream);

template <typename T>
int kr_quantize_q8(const T *input, void *out, int feature_num, int feature_size, cudaStream_t stream);

template <typename T>
int kr_dequantize_q8(const void *input, T *out, int feature_num, int feature_size, cudaStream_t stream);

template <typename T>
int kr_quantize_q4(const T *input, void *out, const size_t items, cudaStream_t stream);

template <typename T>
int kr_dequantize_q4(const void *input, T *out, const size_t items, cudaStream_t stream);

template <typename T>
int kr_dequantize_pq(const void *tab, const uint8_t *idx, T *out, const size_t items, int S, cudaStream_t stream);

template <typename T>
int kr_embed(const int *ids, const T *table, T *out, const int len, const int hidden_size, cudaStream_t stream);

template <typename T, typename TT>
int kr_convert(const T* in, TT* out, const size_t items, cudaStream_t stream);

template <typename T>
int kr_scale(const T* in, T* out, const float alpha, const float beta, const int n, cudaStream_t stream);

template <typename T, typename TT>
int kr_add(const T* A, const TT* B, T* C, const int n, cudaStream_t stream);

template <typename T, typename TT>
int kr_mul(const T* A, const TT* B, T* C, const int n, cudaStream_t stream);

template <typename T>
int kr_rmsnorm(const T *feature, const T *w, T *out, T *norm2, const int batch, const int dim, const float eps, cudaStream_t stream);

template <typename T>
int kr_rotary_embed(const T *in, const float *cos_sin,const int* pos, T* out, const int bs, const int hnum, const int len, const int dims, cudaStream_t stream);

template <typename T>
int kr_transpose_0213(const T* src, T* target, int a, int b, int c ,int d, cudaStream_t stream);

template <typename T>
int kr_gelu(const T* src, T* target, int nElementNumber, cudaStream_t stream);

template <typename T>
int kr_silu_product(const T *in_act, const T* in, T* out, const int items, cudaStream_t stream);

template <typename T>
int kr_add_bias(const T *in, const T* bias, T* out, const int length, const int feature, cudaStream_t stream);

template <typename T>
int kr_add_broadcast(const T *in, const T* bias, T* out, const int length, const size_t inter, const size_t feature, cudaStream_t stream);

template <typename T>
int kr_easy_top1(const T *logits, int *out, const int batch, const int vocab_size, cudaStream_t stream);

template <typename T>
int kr_easy_top3(const T *logits, int *out, const int batch, const int vocab_size, const float temperature, const float randx, cudaStream_t stream);

template <typename T>
int kr_layer_norm(T *ln_res, T *vars, T *means, const T *inp, const T *scale, const T *bias, int batch_size, int hidden_dim, float eps,  cudaStream_t stream);

template <typename T>
int kr_softmax(const T *in, T *out, int length, int feature, cudaStream_t stream);

}}

#endif
