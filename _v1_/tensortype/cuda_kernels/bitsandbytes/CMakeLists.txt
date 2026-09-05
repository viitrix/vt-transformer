# This CMake config hopefully makes it easier to compile.
# Ensure the CUDA Toolkit is available on your path. Then run:
#   For  GCC: `cmake -B build . && cmake --build build`
#   For MSVC: `cmake -B build . && cmake --build build --config Release`
# You can also use the following options and variables
#  - COMPUTE_BACKEND: Set to `cpu`, `cuda`, or `mps` to select the backend
#  - NO_CUBLASLT: Default OFF, will skip building/linking CUBLASLT support
#  - CUDA_VERSION: The expected CUDA version, for sanity checking. The actual version
#                  is whatever CMake finds on your path.
#  - COMPUTE_CAPABILITY: Which GPU Arch/Compute codes to provide to NVCC.
#                        Separate by semicolons, i.e. `-DCOMPUTE_CAPABILITY=89;90`
#                        Check your compute capability here: https://developer.nvidia.com/cuda-gpus
#  - PTXAS_VERBOSE: Pass the `-v` option to the PTX Assembler
cmake_minimum_required(VERSION 3.22.1)

project(bitsandbytes LANGUAGES CXX)

# If run without specifying a build type, default to using the Release configuration:
#    optimizing the generated binaries for performance and also adds the `-DNDEBUG` flag,
#    which turns off a bunch of asserts which seem to link to new symbols in libstdc++,
#    worsening our many_linux compliance..
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Define included source files
set(CPP_FILES csrc/common.cpp csrc/cpu_ops.cpp csrc/bitsandbytes.cpp)
set(CUDA_FILES csrc/ops.cu csrc/kernels.cu)
# C++ sources are always included
list(APPEND SRC_FILES ${CPP_FILES})

option(PTXAS_VERBOSE "Pass through -v flag to PTX Assembler" OFF)

option(NO_CUBLASLT "Disable CUBLAS" OFF)
set(BUILD_CUDA ON)
message(STATUS "NO_CUBLASLT := ${NO_CUBLASLT}")


if(BUILD_CUDA)
    enable_language(CUDA) # This will fail if CUDA is not found
    find_package(CUDAToolkit REQUIRED)

    # Convert the CUDA version from X.Y.z to XY. There's probably a shorter way of doing this
    string(REGEX MATCH "^[0-9]+.[0-9]+" _CUDA_VERSION_FIRST_TWO "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REPLACE "." "" CUDA_VERSION_SHORT "${_CUDA_VERSION_FIRST_TWO}")

    # Expose a cache variable that the user can set to ensure the correct version of CUDA is found
    set(CUDA_VERSION "${CUDA_VERSION_SHORT}" CACHE STRING "Expected CUDA Version Shortcode")

    message(STATUS "CUDA Version: ${CUDA_VERSION_SHORT} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

    # It should match the discovered version
    if(NOT CUDA_VERSION STREQUAL "${CUDA_VERSION_SHORT}")
        message(FATAL_ERROR "You've specified CUDA version ${CUDA_VERSION} however the CUDA compiler found is ${CUDA_VERSION_SHORT}."
            " Ensure the desired CUDA compiler is the first one available on your PATH."
        )
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR "CUDA Version < 11 is not supported")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        message(FATAL_ERROR "CUDA Version > 12 is not supported")
    endif()

     
    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")

    if(PTXAS_VERBOSE)
        # Verbose? Outputs register usage information, and other things...
        string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
    endif()

    foreach(capability ${CMAKE_CUDA_ARCHITECTURES_ALL})
        # Most of the items here are like: `xx-real`, so we just extract the `xx` portion
        string(REGEX MATCH "[0-9]+" capability_id "${capability}")
        if(capability_id GREATER 0)
            list(APPEND POSSIBLE_CAPABILITIES ${capability_id})
        endif()
    endforeach()

    # This can be changed via -D argument to CMake
    # By default all possible capabilities are compiled
    set(COMPUTE_CAPABILITY "${POSSIBLE_CAPABILITIES}" CACHE STRING "Compute Capabilities Targeted")

    message(STATUS "CUDA Capabilities Available: ${POSSIBLE_CAPABILITIES}")
    message(STATUS "CUDA Capabilities  Selected: ${COMPUTE_CAPABILITY}")

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 80 87)
    endif()

    message(STATUS "CUDA Targets: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA NVCC Flags: ${CMAKE_CUDA_FLAGS}")

    list(APPEND SRC_FILES ${CUDA_FILES})

    add_compile_definitions(BUILD_CUDA)
endif()

if(WIN32)
    # Export all symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Weird MSVC hacks
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
endif()

set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CXX)
add_library(bitsandbytes STATIC ${SRC_FILES})
target_compile_features(bitsandbytes PUBLIC cxx_std_14)
target_include_directories(bitsandbytes PUBLIC csrc include)

if(BUILD_CUDA)
    target_include_directories(bitsandbytes PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(bitsandbytes PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
    if(NO_CUBLASLT)
        target_compile_definitions(bitsandbytes PUBLIC NO_CUBLASLT)
    else()
        target_link_libraries(bitsandbytes PUBLIC CUDA::cublasLt)
    endif()

    set_target_properties(bitsandbytes PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                  CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                  CUDA_SEPARABLE_COMPILATION ON)

    install(TARGETS bitsandbytes DESTINATION lib)
    install(PROGRAMS csrc/bitsandbytes.hpp DESTINATION include )
endif()

