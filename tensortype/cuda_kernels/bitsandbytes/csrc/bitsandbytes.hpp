#ifndef _BITSANDBYTES_HPP_
#define _BITSANDBYTES_HPP_

#include <stdio.h>
#include <iostream>
#include <assert.h>


#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusparse.h>


namespace vt { namespace bnb {

void quantizeBlockwise_fp16(void *A, void *out, void *amax, int blocksize, const int n);

}}

#endif
