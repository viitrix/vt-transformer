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

namespace vt {

struct ComputingContext {
    // something like host:0;cuda:1
    ComputingContext(const std::vector<std::string>&);
    ~ComputingContext();

    void* host_workspace;
    size_t workspace_size;
    std::mt19937* rng;
    void boot_host(const int id);
    void shutdown();

#ifdef _USING_DEVICE_CUDA_
    int cuda_device;
    cudaStream_t cuda_stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublasLt_handle;
    cudnnHandle_t cudnn_handle;
    void* cuda_workspace;
    void boot_cuda(const int dev);
#endif

};

} // end of namespace vt
#endif
