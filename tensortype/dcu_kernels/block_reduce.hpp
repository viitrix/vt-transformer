#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vt { namespace dcu {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;

template<typename T>
__inline__ __device__ void warpSum2(T *v0, T* v1) {
  T val0_tmp, val1_tmp;
#define WarpReduceSumOneStep(a)                    \
  val0_tmp = __shfl_xor(WARP_REDUCE_MASK, *v0, a); \
  val1_tmp = __shfl_xor(WARP_REDUCE_MASK, *v1, a); \
  *v0 += val0_tmp;                                 \
  *v1 += val1_tmp

  WarpReduceSumOneStep(16);
  WarpReduceSumOneStep(8);
  WarpReduceSumOneStep(4);
  WarpReduceSumOneStep(2);
  WarpReduceSumOneStep(1);
#undef WarpReduceSumOneStep
}

template <typename T>
__inline__ __device__ void blockReduceSum2(T *v0, T* v1) {
  const int num = 2;
  static __shared__ float shared[num][32];
  int lane_id = threadIdx.x & 0x3f;
  int wid = threadIdx.x >> 6;

  warpSum2(v0, v1);

  if (lane_id == 0) {
    shared[0][wid] = *v0;
    shared[1][wid] = *v1;
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x >> 6)) {
    *v0 = shared[0][lane_id];
    *v1 = shared[1][lane_id];
  } else {
    *v0 = 0;
    *v1 = 0;
  }
  warpSum2(v0, v1);
}


}}