#if BUILD_CUDA
#include <ops.cuh>
#endif
#if BUILD_MPS
// #include <mps_ops.h>
#endif
#include <cpu_ops.h>

#include "bitsandbytes.hpp"

namespace bnb {

void quantizeBlockwise_fp16(void *A, void *out, int blocksize, const int n) {
    float* absmax = (float *)((unsigned char *) A + n / 2);
    ::quantizeBlockwise<__half, 0, NF4>(NULL, (__half *)A, absmax, (unsigned char *)out, NULL, 0, blocksize, n);
}



}
