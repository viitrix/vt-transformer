#ifndef _CUDA_KERNELS_HPP_
#define _CUDA_KERNELS_HPP_

namespace vt { namespace cuda {

template <typename T>
int kr_causal_mask(const int *mask, T *out, const int batch, const int new_tokens, const int full_tokens, cudaStream_t stream);


}}
#endif
