#include <time.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <stdio.h>
#include <string.h>

#include "vt.hpp"
#include "context.hpp"

namespace vt {

const size_t DEFAULT_WORKSPACE_SIZE = 1024 * 1024 * 16;

ComputingContext::ComputingContext() {
    // basic init
    workspace_size = DEFAULT_WORKSPACE_SIZE;
    host_workspace = nullptr;
    pipe_world = 0;
    pipe_rank = -1;

#ifdef _USING_DEVICE_CUDA_
    cuda_device = -1;
#endif
}

void ComputingContext::boot_host(int rks) {
    pipe_world = rks + 1;
    pipe_rank = 0;
    pipe_fds.resize((rks + 1) * 2, -1);
    for (int i = 0; i < rks + 1; i++) {
        int* fds = pipe_fds.data() + i * 2;
        vt_assert( pipe(fds) >= 0, "Can't create pipe between parent and child process!");
    }

    for (int i = 0; i < rks; i++) {
        int n = fork();
        if ( n == 0 ) {
            pipe_rank = i + 1;
            break;
        }
    }

    host_workspace = malloc( workspace_size );
    rng = new std::mt19937(1979);
}

#ifdef _USING_DEVICE_CUDA_
void ComputingContext::boot_cuda(int dev) {
    cuda_device = dev;
    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    CUBLAS_CHECK( cublasCreate_v2(&cublas_handle) );
    CUBLAS_CHECK( cublasLtCreate(&cublasLt_handle) );
    CUBLAS_CHECK( cublasSetStream(cublas_handle, cuda_stream) );

    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, cuda_stream));
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, cuda_stream));

    CUDA_CHECK( cudaMalloc(&cuda_workspace, workspace_size) );
}
#endif

#ifdef _USING_DEVICE_HIP_
void ComputingContext::boot_hip(const int cud) {
    hip_device = cud;

    HIP_CHECK( hipSetDevice(hip_device) );
    HIP_CHECK( hipStreamCreate(&hip_stream) );
    HIPBLAS_CHECK( hipblasCreate(&hipblas_handle) );
    HIPBLAS_CHECK( hipblasSetStream(hipblas_handle, hip_stream) );
    HIP_CHECK( hipMalloc(&hip_workspace, workspace_size) );
}
#endif

void ComputingContext::shutdown() {
    // device shutdown
#ifdef _USING_DEVICE_CUDA_
    if ( cuda_device >= 0) {
        CUDA_CHECK( cudaFree(cuda_workspace) );
        CUDNN_CHECK( cudnnDestroy(cudnn_handle) );
        CUBLAS_CHECK( cublasLtDestroy(cublasLt_handle) );
        CUBLAS_CHECK( cublasDestroy(cublas_handle) );
        CUDA_CHECK( cudaStreamDestroy(cuda_stream) );
    }
#endif

    // host shutdown
    for (size_t i = 0; i < pipe_fds.size(); i++) {
        close( pipe_fds[i]);
    }
    free(host_workspace);
    delete rng;
}

int ComputingContext::pipe_write(const int n, const void *buf, size_t nbyte) {
    if ( pipe_fds.size() == 0) {
        vt_panic("pipe_fds is note initialized!");
    }
    int fd = pipe_fds[n * 2 + 1];
    return write(fd, buf, nbyte);
}

int ComputingContext::pipe_read(void *buf, size_t nbyte) {
    if ( pipe_fds.size() == 0) {
        vt_panic("pipe_fds is note initialized!");
    }
    int fd = pipe_fds[pipe_rank * 2 + 0];
    return read(fd, buf, nbyte);
}

}

