#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

namespace vt {
namespace cuda {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;
#define WARP_SIZE 32
#define MAX_THREADS 1024
const float CUDA_FLOAT_INF_NEG = -100000000.f;  // FIXME later
const float CUDA_FLOAT_INF_POS = 100000000.f;   // FIXME later
const int CUDA_INT_INF = 2147483647;

}}