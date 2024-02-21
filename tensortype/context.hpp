#ifndef _CONTEXT_HPP_
#define _CONTEXT_HPP_

#ifdef _USING_DEVICE_DNNL_
#include <dnnl.hpp>
#endif

#ifdef _USING_DEVICE_CUDA_
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <nccl.h>
#endif

#ifdef _USING_DEVICE_DCU_
#define __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#endif

#ifdef _USING_DEVICE_COREX_
#define __ILUVATAR__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
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

#define HIP_CHECK(f) \
    do { \
        hipError_t  s_ = f; \
        if (s_ != hipSuccess) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define HIPBLAS_CHECK(f) \
    do { \
        hipblasStatus_t  s_ = f; \
        if (s_ != HIPBLAS_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define COREX_CHECK(f) \
    do { \
        cudaError_t  s_ = f; \
        if (s_ != cudaSuccess) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define CXBLAS_CHECK(f) \
    do { \
        cublasStatus_t  s_ = f; \
        if (s_ != CUBLAS_STATUS_SUCCESS) COMPLAIN_ERROR_AND_EXIT(#f, s_); \
    } while (0)

namespace vt {

#ifdef _USING_DEVICE_CUDA_
#define ALL_CUDA_STREAMS 8
#define ALL_CUDA_EVENTS 8
#endif

struct ComputingContext {
#ifdef _USING_DEVICE_DNNL_
    static dnnl::engine*    dnnl_engine;
    static dnnl::stream*    dnnl_stream;
#endif

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

#ifdef _USING_DEVICE_DCU_
    static int dcu_device;
    static hipStream_t dcu_stream;
    static hipblasHandle_t hipblas_handle;
    static void* dcu_workspace;
#endif

#ifdef _USING_DEVICE_COREX_
    static int corex_device;
    static cudaStream_t corex_stream;
    static cublasHandle_t cxblas_handle;
    static void* corex_workspace;
#endif

    static void* host_workspace;
    static size_t workspace_size;
    static std::mt19937* rng;

    static void boot_dnnl(int device);
    static void boot_cuda(int device);
    static void boot_corex(int device);
    static void boot_dcu(int device);
    static void shutdown();

#ifdef _USING_DEVICE_CUDA_
    static float cuda_event(int flag);
#endif

private:
    static void boot_host();
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

// some common function and strcut
using local_fp16_t = uint16_t;
float fp16_to_fp32(local_fp16_t value);
local_fp16_t fp32_to_fp16(float value);

const int Q4_BLOCK_SIZE = 128;
typedef struct {
    float   d;
    float   m;
    uint8_t q[64];
} q4_block_t;
static_assert(sizeof(q4_block_t) == sizeof(float) * 2  + 64, "wrong q4_block_t size/padding");

typedef struct {
    float   d;
    float   m;
    uint8_t q[0];
} q8_head_t;

inline float dequantize_q4(const q4_block_t* q4, const int i) {
    const uint8_t uv = q4->q[i>>1];
    uint8_t v = 0;
    if ( i % 2 == 0 ) {
        v = uv & 0x0F;
    } else {
        v = uv >> 4;
    }
    return q4->d * v + q4->m;
}

inline float dequantize_q8(const q8_head_t* q8, const int i) {
    uint8_t qv = q8->q[i];
    return qv * q8->d + q8->m;
}

template<typename T>
void fill_alibi(std::vector<T>&data, int heads, int tokens);

template<typename T>
void fill_rotary_cache(std::vector<T>&data, int len, int dims, float base);


} // end of namespace vt


#endif
