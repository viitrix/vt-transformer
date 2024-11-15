cmake_minimum_required(VERSION 3.20)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()



if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)
endif()
 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON) 


project(cuda_kernels LANGUAGES CUDA CXX)

set(cuda_kernel_files
    lt_sgemm.cu
    causal_mask.cu
    convert.cu
    layernorm.cu
    rmsnorm.cu
    rotary_embed.cu
    transpose_0213.cu
    gelu.cu
    silu_product.cu
    easy_sampling.cu
    )


add_library(cuda_kernels STATIC ${cuda_kernel_files})

target_include_directories(cuda_kernels PUBLIC $ENV{CUDNN_DIR}/include)
target_link_libraries(cuda_kernels PUBLIC -lcublas)

install(TARGETS cuda_kernels DESTINATION lib)
install(PROGRAMS cuda_kernels.hpp DESTINATION include )
