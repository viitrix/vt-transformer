cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 )
endif()
 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON) 

project(cuda_kernels LANGUAGES CUDA CXX)

set(cuda_kernel_files
    lightseq/util.cc.cu
    lightseq/cross_entropy.cu
    lightseq/cublas_wrappers.cu
    lightseq/cuda_util.cu
    lightseq/dropout_kernels.cu
    lightseq/embedding_kernels.cu
    lightseq/embKernels.cc.cu
    # lightseq/fused_adam_kernel.cu
    lightseq/general_kernels.cu
    lightseq/gptKernels.cc.cu
    lightseq/llama_kernels.cu
    lightseq/normalize_kernels.cu
    lightseq/softmax_kernels.cu
    lightseq/softmax_kernels_new.cu
    lightseq/transform_kernels.cu
    lightseq/transform_kernels_new.cu
    lightseq/crf.cu
    lightseq/transformerKernels.cc.cu
    blackriver/LtSgemm.cu
    blackriver/LtSgemmBatched.cu
    blackriver/gelu.cu
    blackriver/embed.cu
    blackriver/nll_loss.cu
    blackriver/easy_sampling.cu
    blackriver/rms_norm.cu
    blackriver/quantize.cu
    blackriver/linear2d_q.cu
    blackriver/misc.cu)


add_library(cuda_kernels STATIC ${cuda_kernel_files})

include_directories(lightseq/includes)
target_link_libraries(cuda_kernels PUBLIC -lcublas)

install(TARGETS cuda_kernels DESTINATION lib)
install(PROGRAMS kernels.hpp DESTINATION include/kernels )
install(PROGRAMS lightseq/includes/kernels.h DESTINATION include/kernels/lightseq)
