#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cuda_kernels.hpp"

namespace vt { namespace cuda {

/* Convert vector index to 4-dim tensor index */
__forceinline__ __host__ __device__ void decompose_4dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int *id0, int *id1,
                                                        int *id2, int *id3) {
  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert 4-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_4dim(int id1, int id2, int id3,
                                                  int id4, int dim2, int dim3,
                                                  int dim4) {
  // return id1*(dim2*dim3*dim4) + id2*(dim3*dim4) + id3*dim4 + id4;
  int res = id4;

  int ld = dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

template <typename T>
__global__ void transpose_0213(const T *input, T *output, int sz0, int sz1, int sz2, int sz3) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int num_all = sz0 * sz1 * sz2 * sz3;
  if (offset >= num_all) {
    return;
  }
  int id0, id1, id2, id3;
  decompose_4dim(offset, sz1, sz2, sz3, &id0, &id1, &id2, &id3);
  int trg_offset = flat_4dim(id0, id2, id1, id3, sz2, sz1, sz3);
  output[trg_offset] = input[offset];
}

template <>
int kr_transpose_0213<float>(const float* src, float* target, size_t a, size_t b, size_t c, size_t d, cudaStream_t stream) {
    int nElementNumber = a * b * c * d;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    transpose_0213<float> <<< num_of_blocks, block_size, 0, stream >>> (src, target, a, b, c, d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch transpose_0213 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_transpose_0213<__half>(const __half* src, __half* target, size_t a, size_t b, size_t c, size_t d, cudaStream_t stream) {
    int nElementNumber = a * b * c * d;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    transpose_0213<__half> <<< num_of_blocks, block_size, 0, stream >>> (src, target, a, b, c, d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch transpose_0213 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <typename T>
__global__ void kr_transpose_0213_repeated(const T *input, T *output, int sz0, int sz1, int sz2_f, int sz2_t, int sz3) {
  int repeate = sz2_t / sz2_f;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int num_all = sz0 * sz2_t * sz1 * sz3;
  if (offset >= num_all) {
    return;
  }

  int id0, id1, id2, id3;
  decompose_4dim(offset, sz2_t, sz1, sz3, &id0, &id1, &id2, &id3);
  id1 = id1 / repeate;

  int trg_offset = flat_4dim(id0, id2, id1, id3, sz1, sz2_f, sz3);
  output[offset] = input[trg_offset];
}

template <>
int kr_transpose_0213_repeated<float>(const float* src, float* target, size_t a, size_t b, size_t cf, size_t ct, size_t d, cudaStream_t stream) {
    int nElementNumber = a * b * ct * d;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    kr_transpose_0213_repeated<float> <<< num_of_blocks, block_size, 0, stream >>> (src, target, a, b, cf, ct, d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch transpose_0213 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

template <>
int kr_transpose_0213_repeated<__half>(const __half* src, __half* target, size_t a, size_t b, size_t cf, size_t ct, size_t d, cudaStream_t stream) {
    int nElementNumber = a * b * ct * d;

    dim3 block_size(256);
	dim3 num_of_blocks((nElementNumber + block_size.x - 1) / block_size.x);

    kr_transpose_0213_repeated<__half> <<< num_of_blocks, block_size, 0, stream >>> (src, target, a, b, cf, ct, d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch transpose_0213 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}




}}
