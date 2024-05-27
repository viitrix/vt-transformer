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

ComputingContext::ComputingContext(const std::vector<std::string>& devstr) {
    vt_assert( (devstr.size() > 0) && (devstr.size() % 2 == 0) , "creating ComputingContext error!");

    workspace_size = DEFAULT_WORKSPACE_SIZE;
    host_workspace = nullptr;
#ifdef _USING_DEVICE_CUDA_
    cuda_device = -1;
#endif

    for (size_t i = 0; i < devstr.size(); i += 2) {
        if ( devstr[i] == "host" ) {
            int id = std::stoi( devstr[i+1] );
            boot_host(id); 
            continue;      
        }
        if ( devstr[i] == "cuda" ) {
            int d = std::stoi( devstr[i+1] );
#ifdef _USING_DEVICE_CUDA_
            boot_cuda(d);
#else
            vt_panic("creating ComputingContext don't support CUDA!");
#endif
            continue;
        }
        vt_panic("creating ComputingContext don't support device!");
    }

}

void ComputingContext::boot_host(int id) {
    
    host_workspace = malloc( workspace_size );
    rng = new std::mt19937(1979);
}

#ifdef _USING_DEVICE_CUDA_
void ComputingContext::boot_cuda(int dev) {
    cuda_device = dev;
    CUDA_CHECK( cudaSetDevice(cuda_device) );
    CUDA_CHECK( cudaStreamCreate(&cuda_stream) );

    /*
    assist_streams[0] = cuda_stream;
    for (int i = 1; i < ALL_CUDA_STREAMS; i++) {
        CUDA_CHECK( cudaStreamCreate(&assist_streams[i]) );
    }
    for (int i = 0; i < ALL_CUDA_EVENTS; i++) {
        CUDA_CHECK( cudaEventCreate(&events[i]) );
    }
    */

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



}

