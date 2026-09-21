#ifndef _CONTEXT_HPP_
#define _CONTEXT_HPP_

#include <random>

#ifdef _USING_DEVICE_CUDA_
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#endif

#ifdef _USING_DEVICE_HIP_
#define __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#endif

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



namespace vt {

struct ComputingContext {
    // something like host:0;cuda:1
    ComputingContext();
    ~ComputingContext() {
        shutdown();
    }

    void* host_workspace;
    size_t workspace_size;
    std::mt19937* rng;

    int      pipe_world;
    int      pipe_rank;
    std::vector<int> pipe_fds;
    void boot_host(const int ranks);
    int pipe_read(void *buf, size_t nbyte);
    int pipe_write(const int n, const void *buf, size_t nbyte);

#ifdef _USING_DEVICE_CUDA_
    int cuda_device;
    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublasLt_handle;
    cudnnHandle_t cudnn_handle;
    void* cuda_workspace;
    void boot_cuda(const int dev);
#endif

#ifdef _USING_DEVICE_HIP_
    int hip_device;
    hipStream_t hip_stream;
    hipblasHandle_t hipblas_handle;
    void* hip_workspace;
    void boot_hip(const int dev);
#endif

private:
    void shutdown();
};

} // end of namespace vt
#endif
