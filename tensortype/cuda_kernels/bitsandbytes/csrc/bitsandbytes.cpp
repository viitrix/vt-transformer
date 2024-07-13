#if BUILD_CUDA
#include <ops.cuh>
#endif
#if BUILD_MPS
// #include <mps_ops.h>
#endif
#include <cpu_ops.h>

#include "bitsandbytes.hpp"

namespace vt { namespace bnb {

void quantizeBlockwise_fp16(void *A, void *amax, void* out, int blocksize, const int n) {
    ::quantizeBlockwise<__half, 0, NF4>(NULL, (__half *)A, (float *)amax, (unsigned char *)out, NULL, 0, blocksize, n);
}

void dequantizeBlockwise_fp16(void *A, void *amax, void* out, int blocksize, const int n) {
    ::dequantizeBlockwise<half, NF4>(NULL, (unsigned char*)A, (float *)amax, (__half *)out, blocksize, n);
}



}}
