#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_kernels.hpp"
#include "block_reduce.hpp"

namespace vt { namespace hip {

__device__ void insert_topk(float* value, int* idx, int K, float newV, int newIdx) {
    if ( newV < value[K-1] ) {
        return;
    }

    for (int i = K - 2; i >= 0; i--) {
        if ( newV > value[i]) {
            value[i+1] = value[i];
            idx[i+1] = idx[i];
        } else {
            value[i+1] = newV;
            idx[i+1] = newIdx;
            return;
        }
    }
    value[0] = newV;
    idx[0] = newIdx;
}

template<typename T>
__global__ void easy_top3(const T *logits, int *out, const size_t batch, const size_t vocab_size, const float temp, const float randx) {

    const int K = 3;
    const int DIM = 256;
    assert(blockDim.x == DIM);

    const int left_logit_idx = blockIdx.x * vocab_size + threadIdx.x;
    const int right_logit_idx = (blockIdx.x + 1) * vocab_size;

    float top_value[K] = {-1000.0, -1000.0, -1000.0};
    int top_idx[K] = {-1, -1, -1};

    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
        //float v = __half2float( logits[idx] );
        float v = (float)(logits[idx]) / temp;
        int ii = idx - blockIdx.x * vocab_size;

        insert_topk( top_value, top_idx, K, v, ii);
    }

    __shared__ float _all_value_[DIM * K];
    __shared__ int _all_idx_[DIM * K];
    for(int i = 0; i < K; i++) {
        _all_value_[K * threadIdx.x + i] = top_value[i];
        _all_idx_[K * threadIdx.x + i] = top_idx[i];
    }
    __syncthreads();

    for (int half = DIM / 2; half >= 1; half = half / 2 ) {
        int half_offset = K * (threadIdx.x + half);

        if ( threadIdx.x < half ) {
            for (int i = 0; i < K; i++) {
                insert_topk(top_value, top_idx, K, _all_value_[half_offset + i], _all_idx_[half_offset + i]);
            }
        }
        __syncthreads();

        if ( threadIdx.x < half ) {
            for(int i = 0; i < K; i++) {
                _all_value_[K * threadIdx.x + i] = top_value[i];
                _all_idx_[K * threadIdx.x + i] = top_idx[i];
            }
        }
        __syncthreads();
    }

    if ( threadIdx.x == 0) {
        float sum = 1.0;
        for (int i = 1; i < K; i++) {
            _all_value_[i] = expf( _all_value_[i] - _all_value_[0] );
            sum = sum + _all_value_[i];
        }

        _all_value_[0] = 1.0 / sum;
        for (int i = 1; i < K; i++) {
            _all_value_[i] = _all_value_[i] / sum;
        }

        sum = 0.0;
        for (int i = K-1; i >= 0; i--) {
            sum += _all_value_[i];
            if ( sum >= randx ) {
                out[blockIdx.x] = _all_idx_[i];
                return;
            }
        }

        // this should not happen
        out[blockIdx.x] = _all_idx_[0];
    }
}

template<>
int kr_sampling_top3<__half>(const __half* logits, int *out,
                         const size_t batch, const size_t vocab_size, const float temperature, const float randx,
                         hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks(batch);

    easy_top3<__half> <<< num_of_blocks, block_size, 0, stream >>> (logits, out, batch, vocab_size, temperature, randx);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch easy_top3 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//=====================================================================

template<typename T>
__global__ void easy_top1(const T *logits, int *out, const size_t batch, const size_t vocab_size) {

#if 1
    const int DIM = 256;
    assert(blockDim.x == DIM);

    const int left_logit_idx = blockIdx.x * vocab_size + threadIdx.x;
    const int right_logit_idx = (blockIdx.x + 1) * vocab_size;

    float top_value = -FLT_MAX;
    int top_idx = -1;

    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
        float v = (float)logits[idx];
        int ii = idx - blockIdx.x * vocab_size;

        if ( v > top_value ) {
            top_value = v;
            top_idx = ii;
        }
    }

    blockReduceTop(&top_value, &top_idx);

    if ( threadIdx.x == 0) {
        out[blockIdx.x] = top_idx;
    }
#else

    const int K = 3;
    const int DIM = 256;
    assert(blockDim.x == DIM);

    const int left_logit_idx = blockIdx.x * vocab_size + threadIdx.x;
    const int right_logit_idx = (blockIdx.x + 1) * vocab_size;

    float top_value[K] = {-1000.0, -1000.0, -1000.0};
    int top_idx[K] = {-1, -1, -1};

    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
        //float v = __half2float( logits[idx] );
        float v = (float)(logits[idx]) ;
        int ii = idx - blockIdx.x * vocab_size;

        insert_topk( top_value, top_idx, K, v, ii);
    }

    __shared__ float _all_value_[DIM * K];
    __shared__ int _all_idx_[DIM * K];
    for(int i = 0; i < K; i++) {
        _all_value_[K * threadIdx.x + i] = top_value[i];
        _all_idx_[K * threadIdx.x + i] = top_idx[i];
    }
    __syncthreads();

    for (int half = DIM / 2; half >= 1; half = half / 2 ) {
        int half_offset = K * (threadIdx.x + half);

        if ( threadIdx.x < half ) {
            for (int i = 0; i < K; i++) {
                insert_topk(top_value, top_idx, K, _all_value_[half_offset + i], _all_idx_[half_offset + i]);
            }
        }
        __syncthreads();

        if ( threadIdx.x < half ) {
            for(int i = 0; i < K; i++) {
                _all_value_[K * threadIdx.x + i] = top_value[i];
                _all_idx_[K * threadIdx.x + i] = top_idx[i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = _all_idx_[0];
    }

#endif
}

template<>
int kr_sampling_top1<__half>(const __half* logits, int *out,
                         const size_t batch, const size_t vocab_size, hipStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks(batch);

    easy_top1<__half> <<< num_of_blocks, block_size, 0, stream >>> (logits, out, batch, vocab_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Failed to launch easy_top3 kernel (error code %s)!\n", hipGetErrorString(err));
        exit(-1);
    }
    return 0;
}

}}
