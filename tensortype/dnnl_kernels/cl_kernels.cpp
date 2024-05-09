
#include <dnnl_ocl.hpp>

#include "cl_kernels.hpp"

namespace vt { namespace dnnl_kernels {

const char* cl_kernels::source_  =
#include "code.cl"
;

cl_program cl_kernels::programe_ = nullptr;
cl_kernel cl_kernels::rmsnorm_kernel_fp16 = nullptr;
cl_kernel cl_kernels::linear_kernel_fp16 = nullptr;
cl_kernel cl_kernels::rotary_embed_kernel_fp16 = nullptr;

void print_olc_compile_error(cl_device_id did,  cl_program prog) {
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(prog, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(prog, did, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    std::cout << log << std::endl;
}

void cl_kernels::init() {
    int err;
    auto ctx = dnnl::ocl_interop::get_context( *ComputingContext::dnnl_gpu_engine);
    auto did = dnnl::ocl_interop::get_device(*ComputingContext::dnnl_gpu_engine);
    programe_ = clCreateProgramWithSource(ctx, 1, (const char **) & source_, NULL, &err); 

    err = clBuildProgram(programe_, 1, &did, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {        
        print_olc_compile_error(did, programe_);
    }
    OPENCL_CHECK(err);

    rmsnorm_kernel_fp16 = clCreateKernel(programe_, "rmsnorm_fp16", &err);
    OPENCL_CHECK(err);
    linear_kernel_fp16 = clCreateKernel(programe_, "linear_fp16", &err);
    OPENCL_CHECK(err);
    rotary_embed_kernel_fp16 = clCreateKernel(programe_, "rotary_embed_fp16", &err);
    OPENCL_CHECK(err);
}

void cl_kernels::release() {
    OPENCL_CHECK(clReleaseKernel(rmsnorm_kernel_fp16));
    OPENCL_CHECK(clReleaseKernel(linear_kernel_fp16));
    OPENCL_CHECK(clReleaseKernel(rotary_embed_kernel_fp16));
    OPENCL_CHECK(clReleaseProgram(programe_));
}

}}

