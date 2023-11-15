#ifndef _CONTEXT_HPP_
#define _CONTEXT_HPP_

#ifdef _USING_DEVICE_CUDA_
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <nccl.h>
#endif

#ifdef _USING_HPC_OPENMPI_
#include <mpi.h>
#endif

#include <random>
#include "vt.hpp"

#define COMPLAIN_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns error: %d.\n", __FILE__, __LINE__, \
                what, status); \
        exit(1); \
    } while (0)

#define CUDA_CHECK(f) \
    do { \
        cudaError_t  s_ = f; \
        if (s_ != cudaSuccess) { \
            fprintf(stderr, "Error string: %s)!\n", cudaGetErrorString(s_)); \
            COMPLAIN_ERROR_AND_EXIT(#f, s_); \
        } \
    } while (0)

#define CUBLAS_CHECK(f) \
    do { \
        cublasStatus_t  s_ = f; \
        if (s_ != CUBLAS_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define CUDNN_CHECK(f) \
    do { \
        cudnnStatus_t  s_ = f; \
        if (s_ != CUDNN_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define NCCL_CHECK(f) \
    do { \
        ncclResult_t  s_ = f; \
        if (s_ != ncclSuccess) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define MPI_CHECK(f) \
    do { \
        int  s_ = f; \
        if (s_ != MPI_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

namespace vt {

#ifdef _USING_DEVICE_CUDA_
#define ALL_CUDA_STREAMS 8
#define ALL_CUDA_EVENTS 8
#endif

struct ComputingContext {
#ifdef _USING_DEVICE_CUDA_
    static int cuda_device;
    static cudaStream_t cuda_stream;
    static cudaStream_t assist_streams[ ALL_CUDA_STREAMS ];
    static cudaEvent_t events[ ALL_CUDA_STREAMS ];
    static cublasHandle_t cublas_handle;
    static cublasLtHandle_t cublasLt_handle;
    static cudnnHandle_t cudnn_handle;
    static void* cuda_workspace;
#endif

    static void* host_workspace;
    static size_t workspace_size;
    static std::mt19937* rng;

    static void boot(int cud);
    static void shutdown();

#ifdef _USING_DEVICE_CUDA_
    static float cuda_event(int flag);
#endif
};

struct CollectiveContext {
    static int      current;

    static int      pipe_world;
    static int      pipe_rank;
    static int*     pipe_fds;
#ifdef _USING_HPC_OPENMPI_
    static int      mpi_world;
    static int      mpi_rank;
#endif
#ifdef _USING_DEVICE_CUDA_
    static ncclComm_t      nccl_comm;
    static ncclUniqueId    nccl_id;
    static int             nccl_rank;
    static int             nccl_world;
#endif

#ifdef _USING_HPC_OPENMPI_
    static void boot_mpi(int argc, char* argv[], int gpus);
#endif
    static void boot_pipe(int gpus);
    static void shutdown();

    static int pipe_write(const int n, const void *buf, size_t nbyte);
    static int pipe_read(void* buf, size_t nbyte);

    static int now();
};

struct MemoryContext {
    const static size_t aligen_size;
    static size_t  total_size;
    static size_t  currentp;

    static void* alloc(size_t blk_size);
    static void free(void* m, size_t s);
    static void boot(size_t total_bytes);
    static void shutdown();
};

using local_fp16_t = uint16_t;
float fp16_to_fp32(local_fp16_t value);
local_fp16_t fp32_to_fp16(float value);

const int Q4_BLOCK_SIZE = 16;
typedef struct {
    float   d;
    uint8_t q[8];
} q4_block_t;
static_assert(sizeof(q4_block_t) == sizeof(local_fp16_t) * 2  + 8, "wrong q4_block_t size/padding");

typedef struct {
    float   d;
    float   m;
    uint8_t q[0];
} q8_head_t;

inline float dequantize_q4(const q4_block_t* q4, const int i) {
    const uint8_t uv = q4->q[i>>1];
    int8_t v = 0;
    if ( i % 2 == 0 ) {
        v = uv & 0x0F;
    } else {
        v = uv >> 4;
    }
    return q4->d * (v - 8);;
}

inline float dequantize_q8(const q8_head_t* q8, const int i) {
    uint8_t qv = q8->q[i];
    return qv * q8->d + q8->m;
}

} // end of namespace br


#endif
