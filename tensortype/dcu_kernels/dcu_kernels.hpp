#ifndef _DCU_KERNELS_HPP_
#define _DCU_KERNELS_HPP_

namespace vt { namespace dcu {

template<typename T>
int kr_fill(T* target, float value, int nElementNumber, hipStream_t stream);

template <typename T>
int kr_causal_mask(const int *mask, T *out, const int batch, const int new_tokens, const int full_tokens, hipStream_t stream);

template <typename T>
int kr_quantize_q8(const T *input, void *out, int feature_num, int feature_size, hipStream_t stream);

template <typename T>
int kr_dequantize_q8(const void *input, T *out, int feature_num, int feature_size, hipStream_t stream);

template <typename T>
int kr_quantize_q4(const T *input, void *out, int items, hipStream_t stream);

template <typename T>
int kr_dequantize_q4(const void *input, T *out, int items, hipStream_t stream);

template <typename T>
int kr_embed(const int *ids, const T *table, T *out, const int len, const int hidden_size, hipStream_t stream);

template <typename T, typename TT>
int kr_convert(const T* in, TT* out, const int size, hipStream_t stream);

template <typename T>
int kr_scale(const T* in, T* out, const float alpha, const float beta, const int n, hipStream_t stream);

template <typename T, typename TT>
int kr_add(const T* A, const TT* B, T* C, const int n, hipStream_t stream);

template <typename T, typename TT>
int kr_mul(const T* A, const TT* B, T* C, const int n, hipStream_t stream);

template <typename T>
int kr_rmsnorm(const T *feature, const T *w, T *out, T *norm2, const int batch, const int dim, const float eps, hipStream_t stream);

template <typename T>
int kr_rotary_embed(const T *in, const float *cos_sin,const int* pos, T* out, const int bs, const int hnum, const int len, const int dims, hipStream_t stream);

template <typename T>
int kr_transpose_0213(const T* src, T* target, int a, int b, int c ,int d, hipStream_t stream);

template <typename T>
int kr_gelu(const T* src, T* target, int nElementNumber, hipStream_t stream);

template <typename T>
int kr_silu_product(const T *in_act, const T* in, T* out, const int items, hipStream_t stream);

}}

#endif
