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


}}

#endif
