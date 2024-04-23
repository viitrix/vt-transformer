#ifndef _CL_KERNELS_HPP_
#define _CL_KERNELS_HPP_

#include <dnnl_ocl.hpp>
#include "context.hpp"

namespace vt { namespace dnnl_kernels {
struct cl_kernels {
    static void init();
    static void release();

    
    static const char* source_;
    static cl_program programe_;
    static cl_kernel kr_fill_causal_mask_;
};

}}
#endif