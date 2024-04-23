
#include <dnnl_ocl.hpp>

#include "cl_kernels.hpp"

namespace vt { namespace dnnl_kernels {

const char* cl_kernels::source_  =
#include "code.cl"
;
cl_program cl_kernels::programe_ = nullptr;
cl_kernel cl_kernels::kr_fill_causal_mask_ = nullptr;

void cl_kernels::init() {
    int err;
    auto ctx = dnnl::ocl_interop::get_context( *ComputingContext::dnnl_gpu_engine);
    auto did = dnnl::ocl_interop::get_device(*ComputingContext::dnnl_gpu_engine);
    programe_ = clCreateProgramWithSource(ctx, 1, (const char **) & source_, NULL, &err); 

    err = clBuildProgram(programe_, 1, &did, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {        
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(programe_, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(programe_, did, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    OPENCL_CHECK(err);

    //cl_kernel kernel = clCreateKernel(programe_, "vecAdd", &err);
}

void cl_kernels::release() {
    clReleaseProgram(programe_);
}

}}

