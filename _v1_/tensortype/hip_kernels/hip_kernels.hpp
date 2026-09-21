#ifndef _HIP_KERNELS_HPP_
#define _HIP_KERNELS_HPP_

namespace vt { namespace hip {

template<typename T>
int kr_fill(T* target, float value, const size_t nElementNumber, hipStream_t stream);

template <typename T>
int kr_causal_mask(const int *mask, T *out, const size_t batch, const size_t new_tokens, const size_t full_tokens, hipStream_t stream);

/*
template <typename T>
int kr_quantize_q8(const T *input, void *out, const size_t feature_num, const size_t feature_size, hipStream_t stream);

template <typename T>
int kr_dequantize_q8(const void *input, T *out, const size_t feature_num, const size_t feature_size, hipStream_t stream);

template <typename T>
int kr_quantize_q4(const T *input, void *out, const size_t items, hipStream_t stream);

template <typename T>
int kr_dequantize_q4(const void *input, T *out, const size_t items, hipStream_t stream);

template <typename T>
int kr_dequantize_pq(const void *tab, const uint8_t *idx, T *out, const size_t items, int S, hipStream_t stream);
*/

template <typename T>
int kr_embed(const int *ids, const T *table, T *out, const size_t len, const size_t hidden_size, hipStream_t stream);

template <typename T, typename TT>
int kr_convert(const T* in, TT* out, const size_t items, hipStream_t stream);

template <typename T>
int kr_scale(const T* in, T* out, const float alpha, const float beta, const size_t n, hipStream_t stream);

template <typename T, typename TT>
int kr_add(const T* A, const TT* B, T* C, const size_t n, hipStream_t stream);

template <typename T, typename TT>
int kr_mul(const T* A, const TT* B, T* C, const size_t n, hipStream_t stream);

template <typename T>
int kr_rmsnorm(const T *feature, const T *w, T *out, T *norm2, const size_t batch, const size_t dim, const float eps, hipStream_t stream);

template <typename T>
int kr_rotary_embed(const T *in, const float *cos_sin, const int* pos, T* out, const size_t bs, const size_t hnum, const size_t len, const size_t dims, hipStream_t stream);

template <typename T>
int kr_transpose_0213(const T* src, T* target, const size_t a, const size_t b, const size_t c , const size_t d, hipStream_t stream);

template <typename T>
int kr_transpose_0213_repeated(const T* src, T* target, const size_t a, const size_t b,const size_t cf, const size_t ct, const size_t d, hipStream_t stream);

template <typename T>
int kr_gelu(const T* src, T* target, const size_t nElementNumber, hipStream_t stream);

template <typename T>
int kr_silu_product(const T *in_act, const T* in, T* out, const size_t items, hipStream_t stream);

template <typename T>
int kr_add_bias(const T *in, const T* bias, T* out, const size_t length, const size_t feature, hipStream_t stream);

template <typename T>
int kr_add_broadcast(const T *in, const T* bias, T* out, const size_t length, const size_t inter, const size_t feature, hipStream_t stream);

template <typename T>
int kr_sampling_top1(const T *logits, int *out, const size_t batch, const size_t vocab_size, hipStream_t stream);

template <typename T>
int kr_sampling_top3(const T *logits, int *out, const size_t batch, const size_t vocab_size, const float temperature, const float randx, hipStream_t stream);

template <typename T>
int kr_layer_norm(T *ln_res, T *vars, T *means, const T *inp, const T *scale, const T *bias, const size_t batch_size, const size_t hidden_dim, float eps,  hipStream_t stream);

template <typename T>
int kr_softmax(const T *in, T *out, const size_t length, const size_t feature, hipStream_t stream);

}}

#endif
