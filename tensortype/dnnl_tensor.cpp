#include "dnnl_tensor.hpp"
#include "host_tensor.hpp"

#include "dnnl_kernels/impl.hpp"

#ifdef _DNNL_GPU_
#include "dnnl_kernels/cl_kernels.hpp"
#endif

namespace vt {

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

template <DataType _DTYPE_>
DNNLTensor<_DTYPE_>::~DNNLTensor() {
    if ( owner_ ) {
#ifdef _DNNL_GPU_
        if ( gpu_ == true) {
            if ( _DTYPE_ == DataType::Q8) {
                clReleaseMemObject((cl_mem)scale_);
            }            
            cl_mem mem = (cl_mem)mem_;
            clReleaseMemObject(mem);
            return;
        }
#endif
        MemoryContext::free(mem_, size_);
    }
}

template <DataType _DTYPE_>
DNNLTensor<_DTYPE_>::DNNLTensor(const ShapeType& shape, bool isGPU) : owner_(true), gpu_(isGPU) {

    if ( _DTYPE_ == DataType::Float ) {
        size_ = shape.numel() * sizeof(float);
    } else if ( _DTYPE_ == DataType::Int ) {
        size_ = shape.numel() * sizeof(int);
    } else if ( _DTYPE_ == DataType::FP16 ) {
        size_ =  shape.numel() * sizeof(local_fp16_t);
    } else if ( _DTYPE_ == DataType::Q8 ) {
        size_ =  shape.numel() + (shape.numel() / shape.vec().back()) * sizeof(float);
    } else {
        vt_panic("Can't be here!");
    }

#ifdef _DNNL_GPU_
    if ( vt::dnnl_kernels::cl_kernels::programe_ == nullptr) {
        vt::dnnl_kernels::cl_kernels::init();
    }

    if ( isGPU ) {
        int err;
        auto ctx = dnnl::ocl_interop::get_context( *ComputingContext::dnnl_gpu_engine);
        mem_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size_, nullptr, &err);
        OPENCL_CHECK(err);
        from_ = mem_;
        offset_ = 0;

        if ( _DTYPE_ == DataType::Q8) {
            scale_ = sub_ocl_buffer( shape.numel(), (shape.numel() / shape.vec().back()) * sizeof(float));
        } else {
            scale_ = nullptr;
        }

        return;
    }
#endif
    mem_ = MemoryContext::alloc(size_);
}

template <DataType _DTYPE_>
DNNLTensor<_DTYPE_>::DNNLTensor(const ShapeType& shape,  void *mem, bool isGPU) : owner_(false), gpu_(isGPU) , mem_(mem) {
    if ( _DTYPE_ == DataType::Float ) {
        size_ = shape.numel() * sizeof(float);
    } else if ( _DTYPE_ == DataType::Int ) {
        size_ = shape.numel() * sizeof(int);
    } else if ( _DTYPE_ == DataType::FP16 ) {
        size_ =  shape.numel() * sizeof(local_fp16_t);
    } else {
        vt_panic("Can't be here!");
    }
#ifdef _DNNL_GPU_
    from_ = nullptr;
    offset_ = 0;
#endif
}

template <DataType _DTYPE_>
dnnl::memory::desc DNNLTensor<_DTYPE_>::build_memory_desc(const std::vector<size_t>& shape, DataType dt, dnnl::memory::format_tag tag) {
    dnnl::memory::dims dims;
    for(int i = 0; i < (int)shape.size(); i++) {
        dims.push_back(shape[i]);
    }
    if ( dt == DataType::Float ) {
        return dnnl::memory::desc(dims,  dnnl::memory::data_type::f32, tag);
    }
    if ( dt == DataType::FP16 ) {
        return dnnl::memory::desc(dims,  dnnl::memory::data_type::f16, tag);
    }
    if ( dt == DataType::Q8 ) {
        return dnnl::memory::desc(dims,  dnnl::memory::data_type::s8, tag);
    }

    vt_panic("Can't be here!");
    return dnnl::memory::desc();
}

template <DataType _DTYPE_>
dnnl::memory::desc DNNLTensor<_DTYPE_>::build_memory_desc(const std::vector<size_t>& shape, dnnl::memory::format_tag tag) {
    dnnl::memory::dims dims;
    for(int i = 0; i < (int)shape.size(); i++) {
        dims.push_back(shape[i]);
    }
    if ( _DTYPE_ == DataType::Float ) {
        return dnnl::memory::desc(dims,  dnnl::memory::data_type::f32, tag);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        return dnnl::memory::desc(dims,  dnnl::memory::data_type::f16, tag);
    }
    if ( _DTYPE_ == DataType::Q8 ) {
        return dnnl::memory::desc(dims,  dnnl::memory::data_type::s8, tag);
    }

    vt_panic("Can't be here!");
    return dnnl::memory::desc();
}

template <DataType _DTYPE_>
dnnl::memory DNNLTensor<_DTYPE_>::build_memory(const dnnl::memory::desc& desc) {
    if ( _DTYPE_ != DataType::Q8) {
        vt_assert( desc.get_size() == size_ , "dnnl memory's data must have same size with desc");
    }
    if ( gpu_ == false) {
        return dnnl::memory(desc, *ComputingContext::dnnl_engine, mem_);
    } else {
#ifdef _DNNL_GPU_
        auto m =  dnnl::memory(desc, *ComputingContext::dnnl_gpu_engine, (cl_mem)mem_);
        return m;
#else
        return dnnl::memory{};
#endif
    }
}

#ifdef _DNNL_GPU_
template <DataType _DTYPE_>
void* DNNLTensor<_DTYPE_>::scale_buffer() {
    if ( _DTYPE_ != DataType::Q8) {
        vt_panic("Can't be here!");
    }
    return scale_;
}
template <DataType _DTYPE_>
void* DNNLTensor<_DTYPE_>::sub_ocl_buffer(size_t offset, size_t size) {
    cl_buffer_region region;
    region.origin = offset + offset_;
    region.size = size;

    int ret;
    cl_mem newData = clCreateSubBuffer( (cl_mem)from_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
    OPENCL_CHECK(ret);

    return newData;
}
#endif

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_dump(tensor_t self) {
    size_t first8 = std::min( self->items() , (size_t)8);

#ifdef _DNNL_GPU_
    if ( is_gpu()) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* target = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);
        if ( _DTYPE_ == DataType::Float ) {
            float* d = (float *)target;
            std::cout << "First " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;
            d = (float *)target + self->items() - first8;
            std::cout << "Last " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;
        } else if ( _DTYPE_ == DataType::Int ) {
            int* d = (int *)target;
            std::cout << "First " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;
            d = (int *)target + self->items() - first8;
            std::cout << "Last " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;
        } else if ( _DTYPE_ == DataType::FP16 ) {
            local_fp16_t* d = (local_fp16_t *)target;
            std::cout << "First " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << fp16_to_fp32(d[i]) << " ";
            }
            std::cout << std::endl;
            d = (local_fp16_t *)target + self->items() - first8;
            std::cout << "Last " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << fp16_to_fp32(d[i]) << " ";
            }
            std::cout << std::endl;
        } else  if ( _DTYPE_ == DataType::Q8 ) {
            int8_t* d = (int8_t *)target;
            float* tab = (float *)((int8_t *)target + self->items());
            size_t tab_size = self->items() / self->shape().vec().back();
            
            std::cout << ">>>>>>>>>>>>>>>>>> " << tab[0] << std::endl;
            std::cout << "First " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] * tab[0] << " ";
            }
            std::cout << std::endl;

            d = (int8_t *)target + self->items() - first8;
            std::cout << "Last " << first8 << " : ";
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] * tab[tab_size-1] << " ";
            }
            std::cout << std::endl;
        } else {
            return OP_TODO_ERROR;
        }
        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
        return OP_OK;
    }
#endif

    if ( _DTYPE_ == DataType::Float ) {
        float* d = (float *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (float *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        int* d = (int *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (int *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        local_fp16_t* d = (local_fp16_t *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(d[i]) << " ";
        }
        std::cout << std::endl;
        d = (local_fp16_t *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(d[i]) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Q8 ) {
        int8_t* d = (int8_t *)data();
        float* tab = (float *)((int8_t *)data() + self->items());
        size_t tab_size = self->items() / self->shape().vec().back();

        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] * tab[0] << " ";
        }
        std::cout << std::endl;

        d = (int8_t *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] * tab[tab_size-1] << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> DNNLTensor<_DTYPE_>::op_sizeof(tensor_t self) {
    return size_;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_zero(tensor_t self) {

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* target = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);
        memset(target, 0, size_);
        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
        return OP_OK;
    }
#endif

    memset(mem_, 0, size_);
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* target = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);
        const char* d = (const char *)target;
        size_t len = std::get<1>(self->op_sizeof(self));
        wf.write(d, len);
        wf.close();

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
        return OP_OK;
    }
#endif

    const char* d = (const char *)data();
    size_t len = std::get<1>(self->op_sizeof(self));
    wf.write(d, len);
    wf.close();
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {

    std::ifstream inf(fileName, std::ios::binary);
    if ( ! inf.is_open() ) {
        std::cout << "Can't open " << fileName << std::endl;
        vt_panic("Can't open file");
    }

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* target = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        if (_DTYPE_ == DataType::Float) {
            size_t ret = inf.read( (char *)target, sizeof(float) * self->items() ).gcount();
            vt_assert(ret == sizeof(float) * self->items(), "file size dont't match tensor");
        } else  if (_DTYPE_ == DataType::Int) {
            size_t ret = inf.read( (char *)target, sizeof(int) * self->items() ).gcount();
            vt_assert(ret == sizeof(int) * self->items(), "file size dont't match tensor");
        } else if (_DTYPE_ == DataType::FP16) {
            size_t ret = inf.read( (char *)target, sizeof(local_fp16_t) * self->items() ).gcount();
            vt_assert(ret == sizeof(local_fp16_t) * self->items(), "file size dont't match tensor");
        } else if (_DTYPE_ == DataType::Q8) {            
            size_t ret = inf.read( (char *)target, size_ ).gcount();
            vt_assert(ret == size_, "file size dont't match tensor");
        } else {
            clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
            vt_panic("DataType don't support");
            return OP_TODO_ERROR;
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);

        inf.close();
        return OP_OK;
    }
#endif


    if (_DTYPE_ == DataType::Float) {
        size_t ret = inf.read( (char *)data(), sizeof(float) * self->items() ).gcount();
        vt_assert(ret == sizeof(float) * self->items(), "file size dont't match tensor");
    } else  if (_DTYPE_ == DataType::Int) {
        size_t ret = inf.read( (char *)data(), sizeof(int) * self->items() ).gcount();
        vt_assert(ret == sizeof(int) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::FP16) {
        size_t ret = inf.read( (char *)data(), sizeof(local_fp16_t) * self->items() ).gcount();
        vt_assert(ret == sizeof(local_fp16_t) * self->items(), "file size dont't match tensor");
    } else {
        vt_panic("DataType don't support");
    }

    inf.close();
    return OP_OK;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_fill(tensor_t self, float value) {
    size_t items = self->items();
#ifdef _DNNL_GPU_
    if ( is_gpu()) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void*  target = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        if ( DT == DataType::Float ) {
            OPENCL_CHECK(ret);
            float *dst = (float *)target;
            for (size_t i = 0; i < items; i++) {
                dst[i] = value;
            }
        } else if ( DT == DataType::Int ) {
            int *dst = (int *)target;
            int v = value;
            for (size_t i = 0; i < items/2; i++) {
                dst[i] = v;
            }
        } else if ( DT == DataType::FP16 ) {
            local_fp16_t *dst = (local_fp16_t *)target;
            local_fp16_t v = fp32_to_fp16(value);
            for (size_t i = 0; i < items; i++) {
                dst[i] = v;
            }
        } else {
            return OP_TODO_ERROR;
        }
        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
        return OP_OK;
    }
#endif

    if ( DT == DataType::Float ) {
        float *dst = (float *)mem_;
        for (size_t i = 0; i < items; i++) {
            dst[i] = value;
        }
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        local_fp16_t *dst = (local_fp16_t *)mem_;
        local_fp16_t v = fp32_to_fp16(value);
        for (size_t i = 0; i < items; i++) {
            dst[i] = v;
        }
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        int *dst = (int *)mem_;
        int v = value;
        for (size_t i = 0; i < items; i++) {
            dst[i] = v;
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_alibi(tensor_t self) {
    int heads = self->shape()[1];
    int tokens = self->shape()[3];
    auto s = std::get<1>(self->op_sizeof(self));

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* target = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        if ( DT == DataType::Float ) {
            std::vector<float> buffer;
            vt::fill_alibi<float>(buffer, heads, tokens);
            memcpy( target, buffer.data(), s);
            clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
            return OP_OK;
        } else {
            std::vector<local_fp16_t> buffer;
            vt::fill_alibi<local_fp16_t>(buffer, heads, tokens);
            memcpy( target, buffer.data(), s);
            clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
            return OP_OK;
        }
        return OP_TODO_ERROR;
    }
#endif

    if ( DT == DataType::Float ) {
        std::vector<float> buffer;
        vt::fill_alibi<float>(buffer, heads, tokens);

        memcpy( data(), buffer.data(), s);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<local_fp16_t> buffer;
        vt::fill_alibi<local_fp16_t>(buffer, heads, tokens);
        memcpy( data(), buffer.data(), s);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        int* mask = (int *)clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        size_t out_size = std::get<1>(out->op_sizeof(out));
        if ( out->dtype() == DataType::Float ) {

            float* out32 = (float *)clEnqueueMapBuffer(queue, (cl_mem)out->dnnl_float()->mem_,  CL_TRUE, CL_MAP_WRITE , 0, out_size, 0, nullptr, nullptr, &ret);
            OPENCL_CHECK(ret);

            for (int e = 0; e < batch * new_tokens; e++) {
                int b = e / new_tokens;
                int nt = e % new_tokens;
                int nt_end = full_tokens - new_tokens + nt;

                int* m = &mask[ b * full_tokens ];
                float* o = &out32[ b * new_tokens * full_tokens + nt * full_tokens ];
                float minv = std::numeric_limits<float>::lowest();
                dnnl_kernels::fill_causal_mask<float>(m, o, minv, full_tokens, nt_end);
            }
            clEnqueueUnmapMemObject(queue, (cl_mem)out->dnnl_float()->mem_, out32, 0, nullptr,  nullptr);
        } else if ( out->dtype() == DataType::FP16 ) {
            local_fp16_t* out16 = (local_fp16_t *)clEnqueueMapBuffer(queue, (cl_mem)out->dnnl_fp16()->mem_,  CL_TRUE, CL_MAP_WRITE , 0, out_size, 0, nullptr, nullptr, &ret);
            OPENCL_CHECK(ret);

            for (int e = 0; e < batch * new_tokens; e++) {
                int b = e / new_tokens;
                int nt = e % new_tokens;
                int nt_end = full_tokens - new_tokens + nt;

                int* m = &mask[ b * full_tokens ];
                local_fp16_t* o = &out16[ b * new_tokens * full_tokens + nt * full_tokens ];
                local_fp16_t minv = (unsigned short)0xFC00U;
                dnnl_kernels::fill_causal_mask<local_fp16_t>(m, o, minv, full_tokens, nt_end);
            }
            clEnqueueUnmapMemObject(queue, (cl_mem)out->dnnl_fp16()->mem_, out16, 0, nullptr,  nullptr);
        } else {
            clEnqueueUnmapMemObject(queue, (cl_mem)mem_, mask, 0, nullptr,  nullptr);
            return OP_TODO_ERROR;
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, mask, 0, nullptr,  nullptr);
        return OP_OK;
    }
#endif

    int* mask  = (int *)data();
    float*          out32 = nullptr;
    local_fp16_t*   out16 = nullptr;
    if ( out->dtype() == DataType::Float ) {
        out32 = (float *)out->dnnl_float()->data();
    } else if ( out->dtype() == DataType::FP16 ) {
        out16 = (local_fp16_t *)out->dnnl_fp16()->data();
    } else {
        return OP_TODO_ERROR;
    }

    for (int e = 0; e < batch * new_tokens; e++) {
        int b = e / new_tokens;
        int nt = e % new_tokens;
        int nt_end = full_tokens - new_tokens + nt;

        int* m = &mask[ b * full_tokens ];
        if ( out32 != nullptr) {
            float* o = &out32[ b * new_tokens * full_tokens + nt * full_tokens ];
            float minv = std::numeric_limits<float>::lowest();
            dnnl_kernels::fill_causal_mask<float>(m, o, minv, full_tokens, nt_end);
        }
        if ( out16 != nullptr ) {
            local_fp16_t* o = &out16[ b * new_tokens * full_tokens + nt * full_tokens ];
            local_fp16_t minv = (unsigned short)0xFC00U;
            dnnl_kernels::fill_causal_mask<local_fp16_t>(m, o, minv, full_tokens, nt_end);
        }
    }

    return OP_OK;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_rotary_cache(tensor_t self, float base) {

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        if ( DT != DataType::Float ) {
            return OP_OUTPUT_ERROR;
        }

        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* target = (int *)clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        memcpy( target, cos_sin.data(), self->items() * sizeof(float));
        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, target, 0, nullptr,  nullptr);
        return OP_OK;
    }
#endif

    if ( DT == DataType::Float ) {
        // building inv_freq
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        memcpy( data(), cos_sin.data(), self->items() * sizeof(float));
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_quantize(tensor_t self, tensor_t out) {
#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        return OP_TODO_ERROR;
    }
#endif

    if ( DT == DataType::FP16 && out->dtype() == DataType::Q8 ) {
        size_t channel_size = self->shape().vec().back();
        size_t channel_num = self->items() / channel_size;
        float* tab = (float *)((int8_t *) out->device_data() + self->items());

#pragma omp parallel for
        for(size_t c = 0; c < channel_num; c++) {
            local_fp16_t* src = (local_fp16_t *)data() + c * channel_size;
            int8_t* dst = (int8_t*) out->device_data() + c * channel_size;
            float amax = 0.0;
            for(size_t i = 0; i < channel_size; i++) {
                float d = abs(fp16_to_fp32(src[i]));
                if ( d > amax ) {
                    amax = d;
                }
            }

            float scale = amax / ((1 << 8) - 1);
            tab[c] = scale;

            const float id = scale ? 1.0 / scale : 0.0f;
            for(size_t i = 0; i < channel_size; i++) {
                float d = fp16_to_fp32(src[i]);
                float v = d * id;
                dst[i] = (int8_t)(v + 0.5);
            }
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_dequantize(tensor_t self, tensor_t out) {
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_copy(tensor_t self, tensor_t from) {
#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        if ( from->is_host() ) {
            clEnqueueWriteBuffer(queue, (cl_mem)mem_, CL_TRUE, 0, size_, from->device_data(), 0, nullptr, nullptr);
        } else if ( from->is_dnnl() && from->is_shared() ) {
            clEnqueueWriteBuffer(queue, (cl_mem)mem_, CL_TRUE, 0, size_, from->device_data(), 0, nullptr, nullptr);
        } else if (from->is_dnnl()) {
            clEnqueueCopyBuffer(queue, (cl_mem)from->device_data(), (cl_mem)mem_, 0, 0, size_, 0, nullptr, nullptr);
        } else {
            return OP_TODO_ERROR;
        }
        return OP_OK;
    } else {
        if ( from->is_host() ) {
            memcpy(data(), from->device_data(), size_);
        } else if ( from->is_dnnl() && from->is_shared() ) {
            memcpy(data(), from->device_data(), size_);
        } else if (from->is_dnnl()) {
            auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
            clEnqueueReadBuffer(queue, (cl_mem)from->device_data(), CL_TRUE, 0, size_, data(), 0, nullptr, nullptr);
        } else {
            return OP_TODO_ERROR;
        }
        return OP_OK;
    }
#endif

    auto s = std::get<1>(self->op_sizeof(self));
    if ( from->is_host() || from->is_dnnl() ) {
        void* x = from->device_data();
        void* y = data();
        memcpy(y, x, s);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_convert(tensor_t self, tensor_t from) {
    auto tag = dnnl::memory::format_tag::abcd;
    if ( self->shape().dim() == 3) {
        tag = dnnl::memory::format_tag::abc;
    }
    if ( self->shape().dim() == 2) {
        tag = dnnl::memory::format_tag::ab;
    }
    if ( self->shape().dim() == 1) {
        tag = dnnl::memory::format_tag::a;
    }

    if ( DT == DataType::FP16 && from->is_float() ) {
        auto dst_desc = build_memory_desc(self->shape().vec(), tag);
        auto src_desc = from->dnnl_float()->build_memory_desc(from->shape().vec(), DataType::Float, tag);
        auto dst_mem = build_memory(dst_desc);
        auto src_mem = from->dnnl_float()->build_memory(src_desc);
        auto prim = dnnl::reorder(src_mem, dst_mem);

#ifdef _DNNL_GPU_
        if ( is_gpu() ) {
            prim.execute( *ComputingContext::dnnl_gpu_stream , src_mem, dst_mem);
            return OP_OK;
        }
#endif
        prim.execute( *ComputingContext::dnnl_stream , src_mem, dst_mem);
        return OP_OK;
    }
    if ( DT == DataType::Float && from->is_fp16() ) {
        auto dst_desc = build_memory_desc(self->shape().vec(), tag);
        auto src_desc = from->dnnl_fp16()->build_memory_desc(from->shape().vec(), DataType::FP16, tag);
        auto dst_mem = build_memory(dst_desc);
        auto src_mem = from->dnnl_fp16()->build_memory(src_desc);
        auto prim = dnnl::reorder(src_mem, dst_mem);

#ifdef _DNNL_GPU_
        if ( is_gpu() ) {
            prim.execute( *ComputingContext::dnnl_gpu_stream , src_mem, dst_mem);
            return OP_OK;
        }
#endif
        prim.execute( *ComputingContext::dnnl_stream , src_mem, dst_mem);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> DNNLTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        ShapeType newShape(newShape_);
        cl_buffer_region region;
        int ret;
        if ( _DTYPE_ == DataType::Float ) {
            region.origin = offset * sizeof(float) + offset_;
            region.size = newShape.numel() * sizeof(float);
            cl_mem newData = clCreateSubBuffer( (cl_mem)from_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
            OPENCL_CHECK(ret);
            auto* newTensor = new DNNLTensor<DataType::Float>(newShape, newData, true);
            newTensor->setup_from( from_, region.origin);
            return std::make_shared<TensorType>(newTensor, newShape);
        }
        if ( _DTYPE_ == DataType::Int ) {
            region.origin = offset * sizeof(int) + offset_;
            region.size = newShape.numel() * sizeof(int);
            cl_mem newData = clCreateSubBuffer( (cl_mem)from_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
            OPENCL_CHECK(ret);
            auto* newTensor = new DNNLTensor<DataType::Int>(newShape, newData, true);
            newTensor->setup_from( from_, region.origin);
            return std::make_shared<TensorType>(newTensor, newShape);
        }
        if ( _DTYPE_ == DataType::FP16 ) {
            region.origin = offset * sizeof(local_fp16_t) + offset_;
            region.size = newShape.numel() * sizeof(local_fp16_t);
            cl_mem newData = clCreateSubBuffer( (cl_mem)from_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
            OPENCL_CHECK(ret);
            auto* newTensor = new DNNLTensor<DataType::FP16>(newShape, newData, true);
            newTensor->setup_from( from_, region.origin);
            return std::make_shared<TensorType>(newTensor, newShape);
        }
        return OP_TODO_ERROR;
    }
#endif

    if ( _DTYPE_ == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCpuTensor = new DNNLTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Int ) {
        ShapeType newShape(newShape_);
        int *newData = (int *)data() + offset;
        auto* newCpuTensor = new DNNLTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        ShapeType newShape(newShape_);
        local_fp16_t *newData = (local_fp16_t *)data() + offset;
        auto* newCpuTensor = new DNNLTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> DNNLTensor<_DT_>::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
    DataType DT = DataType_from(dtype);

    ShapeType newShape(newShape_);
#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        ShapeType newShape(newShape_);
        cl_buffer_region region;
        int ret;
        if ( _DT_ == DataType::Float ) {
            region.origin = offset * sizeof(float);
        } else if ( _DT_ == DataType::Int ) {
            region.origin = offset * sizeof(int);
        } else if ( _DT_ == DataType::FP16 ) {
            region.origin = offset * sizeof(local_fp16_t);
        } else {
            return OP_TODO_ERROR;
        }
        if ( DT == DataType::Float ) {
            region.size = newShape.numel() * sizeof(float);
        } else if ( DT == DataType::Int ) {
            region.size = newShape.numel() * sizeof(int);
        } else if ( DT == DataType::FP16 ) {
            region.size = newShape.numel() * sizeof(local_fp16_t);
        } else {
            return OP_TODO_ERROR;
        }

        region.origin = region.origin + offset_;
        cl_mem newData = clCreateSubBuffer( (cl_mem)from_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
        OPENCL_CHECK(ret);

        if ( DT == DataType::Float ) {
            auto* newTensor = new DNNLTensor<DataType::Float>(newShape, newData, true);
            newTensor->setup_from( from_, region.origin);
            return std::make_shared<TensorType>(newTensor, newShape);
        }
        if ( DT == DataType::Int ) {
            auto* newTensor = new DNNLTensor<DataType::Int>(newShape, newData, true);
            newTensor->setup_from( from_, region.origin);
            return std::make_shared<TensorType>(newTensor, newShape);
        }
        if ( DT == DataType::FP16 ) {
            auto* newTensor = new DNNLTensor<DataType::FP16>(newShape, newData, true);
            newTensor->setup_from( from_, region.origin);
            return std::make_shared<TensorType>(newTensor, newShape);
        }
        return OP_TODO_ERROR;
    }
#endif
    void *newData = nullptr;
    if ( _DT_ == DataType::Float ) {
        newData = (char *)data() + offset * sizeof(float);
    } else if ( _DT_ == DataType::Int ) {
        newData = (char *)data() + offset * sizeof(int);
    } else if ( _DT_ == DataType::FP16 ) {
        newData = (char *)data() + offset * sizeof(local_fp16_t);
    } else {
        return OP_TODO_ERROR;
    }

    if ( DT == DataType::Float ) {
        auto* newTensor = new DNNLTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        auto* newTensor = new DNNLTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        auto* newTensor = new DNNLTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( owner_ == true ) {
        return OP_INPUT_ERROR;
    }
    if ( newShape.numel() + offset > self->items()  ) {
        return OP_INPUT_ERROR;
    }

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        ShapeType newShape(newShape_);
        cl_buffer_region region;
        int ret;
        if ( DT == DataType::Float ) {
            region.origin = offset * sizeof(float);
            region.size = newShape.numel() * sizeof(float);
        } else if ( DT == DataType::Int ) {
            region.origin = offset * sizeof(int);
            region.size = newShape.numel() * sizeof(int);
        } else if ( DT == DataType::FP16 ) {
            region.origin = offset * sizeof(local_fp16_t);
            region.size = newShape.numel() * sizeof(local_fp16_t);
        } else {
            return OP_TODO_ERROR;
        }

        cl_mem newData = clCreateSubBuffer( (cl_mem)mem_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
        mem_ = newData;
        OPENCL_CHECK(ret);
    }
#endif

    if ( DT == DataType::Float ) {
        mem_  = (char *)data() + offset * sizeof(float);
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        mem_  = (char *)data() + offset * sizeof(int);
        return OP_OK;
    }

    if ( DT == DataType::FP16 ) {
        mem_  = (char *)data() + offset * sizeof(local_fp16_t);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_scale(tensor_t self, float scale) {
    if (   DT == DataType::Float) {
        dnnl_kernels::eltwise<DNNLTensor<DataType::Float>>(self->dnnl_float(), self->dnnl_float(), self->items(),
            dnnl::algorithm::eltwise_linear, scale, 0.0);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::eltwise<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(),  self->dnnl_fp16(), self->items(),
            dnnl::algorithm::eltwise_linear, scale, 0.0);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    if (   DT == DataType::Float) {
        dnnl_kernels::binary_float(self, b, c, dnnl::algorithm::binary_add);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::binary_fp16(self, b, c, dnnl::algorithm::binary_add);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    if (   DT == DataType::Float) {
        dnnl_kernels::binary_float(self, b, c, dnnl::algorithm::binary_mul);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::binary_fp16(self, b, c, dnnl::algorithm::binary_mul);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

#ifdef _DNNL_GPU_
void linear_kernel(DNNLTensor<DataType::FP16>* src, DNNLTensor<DataType::FP16>* weight, DNNLTensor<DataType::FP16>* bias, DNNLTensor<DataType::FP16>* dst, 
                  size_t batch, size_t outFeature, size_t inFeature ) {
    cl_kernel kernel = dnnl_kernels::cl_kernels::linear_kernel_fp16;
    {
        cl_mem buffer = (cl_mem)src->data();
        clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
        buffer = (cl_mem)weight->data();
        clSetKernelArg(kernel, 1, sizeof(buffer), &buffer);
        buffer = (cl_mem)dst->data();
        clSetKernelArg(kernel, 2, sizeof(buffer), &buffer);
        
        if ( bias != nullptr) {
            buffer = (cl_mem)bias->data();
        }
        clSetKernelArg(kernel, 3, sizeof(buffer), &buffer);

        int ivalue = batch;
        clSetKernelArg(kernel, 4, sizeof(ivalue), &ivalue);
        ivalue = outFeature;
        clSetKernelArg(kernel, 5, sizeof(ivalue), &ivalue);
        ivalue = inFeature;
        clSetKernelArg(kernel, 6, sizeof(ivalue), &ivalue);
        ivalue = 1;
        if ( bias == nullptr) {
            ivalue = 0;
        }
        clSetKernelArg(kernel, 7, sizeof(ivalue), &ivalue);
    }
    const size_t TS = 16;
    const size_t local[2] =  { 1, TS };
    const size_t global[2] = { batch, outFeature * TS};

    auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
    OPENCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
}
#endif

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_linear(tensor_t self, tensor_t w, tensor_t bias, tensor_t dst) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w->shape()[0];

    size_t num = batch * tokens;
#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        if ( DT == DataType::FP16 && w->is_q8() ) {
            dnnl_kernels::linear_w8(self->dnnl_fp16(), w->dnnl_q8(),
                bias == nullptr? nullptr : bias->dnnl_fp16(), dst->dnnl_fp16(), num, outSize, inSize);
            return OP_OK;
        }
        if ( DT == DataType::FP16 ) {
            linear_kernel(self->dnnl_fp16(), w->dnnl_fp16(),
                bias == nullptr? nullptr : bias->dnnl_fp16(), dst->dnnl_fp16(), num, outSize, inSize);
            return OP_OK;
        }
    }
#endif
    
    if (   DT == DataType::Float) {
        dnnl_kernels::linear<DNNLTensor<DataType::Float>>(self->dnnl_float(), w->dnnl_float(),
            bias == nullptr? nullptr : bias->dnnl_float(), dst->dnnl_float(), num, outSize, inSize);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::linear<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(), w->dnnl_fp16(),
            bias == nullptr? nullptr : bias->dnnl_fp16(), dst->dnnl_fp16(), num, outSize, inSize);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t feature = self->shape()[2];

    vt_assert(mean == nullptr, "Current's impl don't need mean and var!");
    vt_assert(var == nullptr, "Current's impl don't need mean and var!");

    size_t num = batch * tokens;
    if (   _DTYPE_ == DataType::Float) {
        dnnl_kernels::layernrom<DNNLTensor<DataType::Float>>(self->dnnl_float(), scale->dnnl_float(), bias->dnnl_float(), y->dnnl_float(),
            num, feature, eps);
        return OP_OK;
    }
    if (   _DTYPE_ == DataType::FP16) {
        dnnl_kernels::layernrom<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(), scale->dnnl_fp16(), bias->dnnl_fp16(), y->dnnl_fp16(),
            num, feature, eps);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t feature = self->shape()[2];
    size_t num = batch * tokens;

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
#if 1
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* src = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        void* dst = clEnqueueMapBuffer(queue, (cl_mem)y->device_data(),  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        void* s = clEnqueueMapBuffer(queue, (cl_mem)scale->device_data(),  CL_TRUE, CL_MAP_READ , 0, feature, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        ComputingReturn result = OP_TODO_ERROR;
        if (   _DTYPE_ == DataType::Float) {
            dnnl_kernels::rmsnorm<float, DataType::Float>((float *)src, (float *)s, (float *)dst, num, feature, eps);
            result = OP_OK;
        }
        if (   _DTYPE_ == DataType::FP16) {
            dnnl_kernels::rmsnorm<local_fp16_t, DataType::FP16>((local_fp16_t *)src, (local_fp16_t *)s, (local_fp16_t *)dst, num, feature, eps);
            result = OP_OK;
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, src, 0, nullptr,  nullptr);
        clEnqueueUnmapMemObject(queue, (cl_mem)y->device_data(), dst, 0, nullptr,  nullptr);
        clEnqueueUnmapMemObject(queue, (cl_mem)scale->device_data(), s, 0, nullptr,  nullptr);
        return result;
#else
        if ( _DTYPE_ != DataType::FP16) {
            return OP_TODO_ERROR;
        }
        cl_kernel rmsnorm_kernel = dnnl_kernels::cl_kernels::rmsnorm_kernel_fp16;
        {
            cl_mem buffer = (cl_mem)mem_;
            clSetKernelArg(rmsnorm_kernel, 0, sizeof(buffer), &buffer);
            buffer = (cl_mem)scale->dnnl_fp16()->data();
            clSetKernelArg(rmsnorm_kernel, 1, sizeof(buffer), &buffer);
            buffer = (cl_mem)y->dnnl_fp16()->data();
            clSetKernelArg(rmsnorm_kernel, 2, sizeof(buffer), &buffer);
            buffer = (cl_mem)norm2->dnnl_fp16()->data();
            clSetKernelArg(rmsnorm_kernel, 3, sizeof(buffer), &buffer);
            int ivalue = num;
            clSetKernelArg(rmsnorm_kernel, 4, sizeof(ivalue), &ivalue);
            ivalue = feature;
            clSetKernelArg(rmsnorm_kernel, 5, sizeof(ivalue), &ivalue);
            clSetKernelArg(rmsnorm_kernel, 6, sizeof(eps), &eps);
        }

        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);

        const size_t local[1] =  {16};
        const size_t global[1] = {num * 16};
        OPENCL_CHECK(clEnqueueNDRangeKernel(queue, rmsnorm_kernel, 1, nullptr, global, local, 0, nullptr, nullptr));
        return OP_OK;
    #endif
    }
#endif

    void* src = data();
    void* dst = y->device_data();
    void* s = scale->device_data();
    if (   _DTYPE_ == DataType::Float) {
        dnnl_kernels::rmsnorm<float, DataType::Float>((float *)src, (float *)s, (float *)dst, num, feature, eps);
        return OP_OK;
    }
    if (   _DTYPE_ == DataType::FP16) {
        dnnl_kernels::rmsnorm<local_fp16_t, DataType::FP16>((local_fp16_t *)src, (local_fp16_t *)s, (local_fp16_t *)dst, num, feature, eps);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType DT>
ComputingReturn DNNLTensor<DT>::op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos_, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

    vt_assert(hidden == cached->shape()[1], "heads number must be same with cache");

    void* in = data();
    void* cos_sin = cached->device_data();
    void* out = y->device_data();
    void* pos = (int*) pos_->device_data();

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        if ( DT != DataType::FP16) {
            return OP_TODO_ERROR;
        }

        cl_kernel kernel = dnnl_kernels::cl_kernels::rotary_embed_kernel_fp16;
        {
            cl_mem buffer = (cl_mem)mem_;
            clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
            buffer = (cl_mem)cached->dnnl_float()->data();
            clSetKernelArg(kernel, 1, sizeof(buffer), &buffer);
            buffer = (cl_mem)pos_->dnnl_int()->data();
            clSetKernelArg(kernel, 2, sizeof(buffer), &buffer);
            buffer = (cl_mem)y->dnnl_fp16()->data();
            clSetKernelArg(kernel, 3, sizeof(buffer), &buffer);
            int ivalue = batch;
            clSetKernelArg(kernel, 4, sizeof(ivalue), &ivalue);
            ivalue = heads;
            clSetKernelArg(kernel, 5, sizeof(ivalue), &ivalue);
            ivalue = tokens;
            clSetKernelArg(kernel, 6, sizeof(ivalue), &ivalue);
            ivalue = hidden;
            clSetKernelArg(kernel, 7, sizeof(ivalue), &ivalue);
        }

        const size_t local[1] =  { 1};
        const size_t global[1] = { batch * tokens * heads};

        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        OPENCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global, local, 0, nullptr, nullptr));
        return OP_OK;
    }
 #endif

    if ( DT == DataType::Float ) {
        dnnl_kernels::rotary_embed<float>((float *)in, (float *)cos_sin, (int *)pos, (float *)out, batch, heads, tokens, hidden);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        dnnl_kernels::rotary_embed<local_fp16_t>((local_fp16_t *)in, (float *)cos_sin, (int *)pos, (local_fp16_t *)out, batch, heads, tokens, hidden);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType DT>
ComputingReturn DNNLTensor<DT>::op_transpose_0213(tensor_t self, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* in = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        void* out = clEnqueueMapBuffer(queue, (cl_mem)y->device_data(),  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        ComputingReturn result = OP_TODO_ERROR;
        if (   DT == DataType::Float) {
            dnnl_kernels::transpose_0213<float>((float *)in, (float *)out, batch, heads, tokens, hidden);
            result = OP_OK;
        }
        if (   DT == DataType::FP16) {
            dnnl_kernels::transpose_0213<local_fp16_t>((local_fp16_t *)in, (local_fp16_t *)out, batch, heads, tokens, hidden);
            result = OP_OK;
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, in, 0, nullptr,  nullptr);
        clEnqueueUnmapMemObject(queue, (cl_mem)y->device_data(), out, 0, nullptr,  nullptr);
        return result;
    }
 #endif

    if ( DT == DataType::Float ) {
        float* in = (float *)data();
        float* out = (float *)y->dnnl_float()->data();
        dnnl_kernels::transpose_0213<float>(in, out, batch, heads, tokens, hidden);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        local_fp16_t* in = (local_fp16_t *)data();
        local_fp16_t* out = (local_fp16_t *)y->dnnl_fp16()->data();

        dnnl_kernels::transpose_0213<local_fp16_t>(in, out, batch, heads, tokens, hidden);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_qk(tensor_t self, tensor_t key, tensor_t qk) {
#if 1
    auto shape_ = self->shape().vec();

    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int hhidden = shape_[3];
    int ftokens = key->shape()[2];

    size_t num = batch * heads;
    if ( _DTYPE_ == DataType::Float) {
        dnnl_kernels::query_key<DNNLTensor<DataType::Float>>(self->dnnl_float(), key->dnnl_float(), qk->dnnl_float(), num, ntokens, ftokens, hhidden);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16) {
        dnnl_kernels::query_key<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(), key->dnnl_fp16(), qk->dnnl_fp16(), num, ntokens, ftokens, hhidden);
        return OP_OK;
    }
#else
    auto shape_ = self->shape().vec();

    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int hhidden = shape_[3];
    int ftokens = key->shape()[2];

    int m = ntokens;
    int n = ftokens;
    int k = hhidden;

    float alpha = 1.0 / sqrt(hhidden);
    float beta = 0.0;

    if ( _DTYPE_ == DataType::Float ) {
        int HnT = hhidden * ntokens ;
        int HfT = hhidden * ftokens ;
        int TT = ftokens * ntokens;
        for (int i = 0; i < batch * heads; i++) {
            float* A = (float *)data() + i * HnT;
            float* B = (float *)(key->dnnl_float()->data()) + i * HfT;
            float* C = (float *)(qk->dnnl_float()->data()) + i * TT;
            dnnl::sgemm('N', 'T',
                        m, n, k,
                        alpha, A, k,
                        B, k, beta,
                        C, n);
        }
        return OP_OK;
    }
#endif
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_softmax(tensor_t self, tensor_t dst) {
    auto shape_ = self->shape().vec();

    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int hhidden = shape_[3];

    size_t num = batch * heads * ntokens;
    if ( _DTYPE_ == DataType::Float) {
        dnnl_kernels::softmax<DNNLTensor<DataType::Float>>(self->dnnl_float(), dst->dnnl_float(), num, hhidden);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16) {
        dnnl_kernels::softmax<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(), dst->dnnl_fp16(), num, hhidden);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DTYPE_>
ComputingReturn  DNNLTensor<_DTYPE_>::op_attn(tensor_t self, tensor_t value, tensor_t out) {
    auto shape_ = self->shape().vec();
    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int ftokens = shape_[3];
    int hhidden = value->shape()[3];


    size_t num = batch * heads;
    if ( _DTYPE_ == DataType::Float) {
        dnnl_kernels::attn<DNNLTensor<DataType::Float>>(self->dnnl_float(), value->dnnl_float(), out->dnnl_float(), num, ntokens, ftokens, hhidden);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16) {
        dnnl_kernels::attn<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(), value->dnnl_fp16(), out->dnnl_fp16(), num, ntokens, ftokens, hhidden);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_gelu(tensor_t self, tensor_t dst) {
    size_t total = self->items();

    if ( _DTYPE_ == DataType::Float ) {
        float* in = (float *)data();
        float* out = (float *)dst->dnnl_float()->data();
        dnnl_kernels::gelu<float>(in, out, total);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        local_fp16_t* in = (local_fp16_t *)data();
        local_fp16_t* out = (local_fp16_t *)dst->dnnl_fp16()->data();

        dnnl_kernels::gelu<local_fp16_t>(in, out, total);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_silu_product(tensor_t self, tensor_t in, tensor_t dst) {
    size_t total = self->items();

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int ret = 0;
        void* a = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        void* b = clEnqueueMapBuffer(queue, (cl_mem)in->device_data(),  CL_TRUE, CL_MAP_READ | CL_MAP_WRITE  , 0, size_, 0, nullptr, nullptr, &ret);
        OPENCL_CHECK(ret);

        void* out = b;
        if ( dst->device_data() != in->device_data() ) {
            out = clEnqueueMapBuffer(queue, (cl_mem)dst->device_data(),  CL_TRUE, CL_MAP_WRITE , 0, size_, 0, nullptr, nullptr, &ret);
            OPENCL_CHECK(ret);
        }

        ComputingReturn result = OP_TODO_ERROR;
        if (   _DTYPE_ == DataType::Float) {
            dnnl_kernels::silu_product<float>((float *)a, (float *)b, (float *)out, total);
            result = OP_OK;
        }
        if (   _DTYPE_ == DataType::FP16) {
             dnnl_kernels::silu_product<local_fp16_t>((local_fp16_t *)a, (local_fp16_t *)b, (local_fp16_t *)out, total);
            result = OP_OK;
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, a, 0, nullptr,  nullptr);
        clEnqueueUnmapMemObject(queue, (cl_mem)in->device_data(), b, 0, nullptr,  nullptr);
        clEnqueueUnmapMemObject(queue, (cl_mem)dst->device_data(), out, 0, nullptr,  nullptr);
        return result;
    }
 #endif

    if ( _DTYPE_ == DataType::Float ) {
        float* a = (float *)data();
        float* b = (float *)in->dnnl_float()->data();
        float* out = (float *)dst->dnnl_float()->data();

        dnnl_kernels::silu_product<float>(a, b, out, total);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        local_fp16_t* a = (local_fp16_t *)data();
        local_fp16_t* b = (local_fp16_t *)in->dnnl_fp16()->data();
        local_fp16_t* out = (local_fp16_t *)dst->dnnl_fp16()->data();

        dnnl_kernels::silu_product<local_fp16_t>(a, b, out, total);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
std::variant<ComputingReturn,int> DNNLTensor<DT>::op_all_logits(tensor_t self, tensor_t mask_,  tensor_t lm_head, tensor_t output) {
    int batch = self->shape()[0];
    int new_tokens = self->shape()[1];
    int hidden_size = self->shape()[2];
    int full_tokens = mask_->shape()[1];

    int vocab_size = lm_head->shape()[0];

    int* mask = (int *)mask_->host_int()->data();
    int pred = 0;

    auto ddt = dnnl::memory::data_type::f16;
    if ( DT == DataType::Float ) {
        ddt = dnnl::memory::data_type::f32;
    }
    auto src_md = dnnl::memory::desc({1, 1, hidden_size}, ddt, dnnl::memory::format_tag::abc);
    auto w_md = dnnl::memory::desc({1, hidden_size, vocab_size}, ddt, dnnl::memory::format_tag::acb);
    auto dst_md = dnnl::memory::desc({1, 1, vocab_size}, ddt, dnnl::memory::format_tag::abc);

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        std::vector<cl_mem> temp_buffers;

        for (int b = 0;  b < batch; b++) {
            int* mk = &mask[b * full_tokens];
            for ( int i = 0; i < new_tokens ; i++) {
                int ii = full_tokens - new_tokens + i;
                if ( mk[ii] != 2 ) {
                    continue;
                }
                int target = i;
                cl_mem x;
                cl_mem w;
                cl_mem y;
                if ( DT == DataType::Float ) {
                    x = (cl_mem)sub_ocl_buffer( ( b * new_tokens * hidden_size + target * hidden_size) * sizeof(local_fp16_t), hidden_size * sizeof(local_fp16_t));
                    w = (cl_mem)lm_head->device_data();
                    y = (cl_mem)output->dnnl_float()->sub_ocl_buffer( pred * vocab_size * sizeof(local_fp16_t) , vocab_size * sizeof(local_fp16_t));
                    dnnl_kernels::simple_gpu_gemm(x, w, y, src_md, w_md, dst_md);
                } else if ( DT == DataType::FP16 ) {
                    x = (cl_mem)sub_ocl_buffer( ( b * new_tokens * hidden_size + target * hidden_size) * sizeof(local_fp16_t), hidden_size * sizeof(local_fp16_t));
                    w = (cl_mem)lm_head->device_data();
                    y = (cl_mem)output->dnnl_fp16()->sub_ocl_buffer( pred * vocab_size * sizeof(local_fp16_t) , vocab_size * sizeof(local_fp16_t));
                    dnnl_kernels::simple_gpu_gemm(x, w, y, src_md, w_md, dst_md);
                } else {
                    return OP_TODO_ERROR;
                }

                temp_buffers.push_back(x);
                temp_buffers.push_back(y);
                pred ++;
            }
        }

        for(int i = 0; i < (int)temp_buffers.size(); i++) {
            clReleaseMemObject(temp_buffers[i]);
        }

        return pred;
    }
#endif

    for (int b = 0;  b < batch; b++) {
        int* mk = &mask[b * full_tokens];
        for ( int i = 0; i < new_tokens ; i++) {
            int ii = full_tokens - new_tokens + i;
            if ( mk[ii] != 2 ) {
                continue;
            }
            int target = i;
            if ( DT == DataType::Float ) {
                float* dst = (float *)output->dnnl_float()->data() + pred * vocab_size;
                float* src = (float *)data() + b * new_tokens * hidden_size + target * hidden_size;
                float* w = (float *)lm_head->dnnl_float()->data();
                dnnl_kernels::simple_gemm(src, w, dst, src_md, w_md, dst_md);
            } else if ( DT == DataType::FP16 ) {
                auto* dst = (local_fp16_t *)output->dnnl_fp16()->data() + pred * vocab_size;
                auto* src = (local_fp16_t *)data() + b * new_tokens * hidden_size + target * hidden_size;
                auto* w = (local_fp16_t *)lm_head->dnnl_fp16()->data();
                dnnl_kernels::simple_gemm(src, w, dst, src_md, w_md, dst_md);
            } else {
                return OP_TODO_ERROR;
            }
            pred ++;
        }
    }

    return pred;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t>  DNNLTensor<DT>::op_sampling_top1(tensor_t self) {
    if ( DT != DataType::Float && DT != DataType::FP16 ) {
        return OP_INPUT_ERROR;
    }

    int batch = self->shape()[0];
    int vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_int( ret_shape );
    int* out = (int *)ret->device_data();

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int check = 0;
        void* logits_ = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &check);
        OPENCL_CHECK(check);

        if ( DT == DataType::FP16 ) {
            local_fp16_t* logits = (local_fp16_t *)logits_;
            dnnl_kernels::easy_top1<local_fp16_t>(logits, out, batch, vocab_size);
        } else {
            float* logits = (float *)logits_;
            dnnl_kernels::easy_top1<float>(logits, out, batch, vocab_size);
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, logits_, 0, nullptr,  nullptr);
        return ret;
    }

#endif

    if ( DT == DataType::FP16 ) {
        local_fp16_t* logits = (local_fp16_t *) self->device_data();
        dnnl_kernels::easy_top1<local_fp16_t>(logits, out, batch, vocab_size);
    } else {
        float* logits = (float *) self->device_data();
        dnnl_kernels::easy_top1<float>(logits, out, batch, vocab_size);
    }
    return ret;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t>  DNNLTensor<DT>::op_sampling_top3(tensor_t self, float temp) {
    if ( DT != DataType::Float && DT != DataType::FP16 ) {
        return OP_INPUT_ERROR;
    }

    int batch = self->shape()[0];
    int vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_int( ret_shape );
    int* out = (int *)ret->device_data();

    std::uniform_real_distribution<> dist(0.0, 1.0);
    float randx = dist( *ComputingContext::rng );

#ifdef _DNNL_GPU_
    if ( is_gpu() ) {
        auto queue = dnnl::ocl_interop::get_command_queue(*ComputingContext::dnnl_gpu_stream);
        int check = 0;
        void* logits_ = clEnqueueMapBuffer(queue, (cl_mem)mem_,  CL_TRUE, CL_MAP_READ , 0, size_, 0, nullptr, nullptr, &check);
        OPENCL_CHECK(check);

        if ( DT == DataType::FP16 ) {
            local_fp16_t* logits = (local_fp16_t *) logits_;
            dnnl_kernels::easy_top3<local_fp16_t>(logits, out, batch, vocab_size, temp, randx);
        } else {
            float* logits = (float *) logits_;
            dnnl_kernels::easy_top3<float>(logits, out, batch, vocab_size, temp, randx);
        }

        clEnqueueUnmapMemObject(queue, (cl_mem)mem_, logits_, 0, nullptr,  nullptr);
        return ret;
    }

#endif

    if ( DT == DataType::FP16 ) {
        local_fp16_t* logits = (local_fp16_t *) self->device_data();
        dnnl_kernels::easy_top3<local_fp16_t>(logits, out, batch, vocab_size, temp, randx);
    } else {
        float* logits = (float *) self->device_data();
        dnnl_kernels::easy_top3<float>(logits, out, batch, vocab_size, temp, randx);
    }
    return ret;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_conv2d(tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int _stride, int _padding) {
    dnnl::memory::dims strides{_stride, _stride};
    dnnl::memory::dims padding{_padding, _padding};

    if ( _DTYPE_ == DataType::Float  && weight->is_float() && dst->is_float() ) {
        auto xmem_desc = build_memory_desc(self->shape().vec(), tag::nchw);
        auto wmem_desc = build_memory_desc(weight->shape().vec(), tag::oihw);
        dnnl::memory::desc bmem_desc;
        if ( bias != nullptr) {
            bmem_desc = build_memory_desc(bias->shape().vec(), tag::x);
        }
        auto ymem_desc = build_memory_desc(dst->shape().vec(), tag::nchw);

        dnnl::convolution_forward::primitive_desc conv_prim_desc;
        if ( bias != nullptr) {
            conv_prim_desc = dnnl::convolution_forward::primitive_desc(
                    *ComputingContext::dnnl_engine,
                    dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                    xmem_desc, wmem_desc, bmem_desc, ymem_desc,
                    strides, padding, padding);
        } else {
            conv_prim_desc = dnnl::convolution_forward::primitive_desc(
                    *ComputingContext::dnnl_engine,
                    dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                    xmem_desc, wmem_desc, ymem_desc,
                    strides, padding, padding);
        }
        auto conv_prim = dnnl::convolution_forward(conv_prim_desc);

        auto xmem = build_memory(xmem_desc);
        auto wmem = weight->dnnl_float()->build_memory(wmem_desc);
        auto ymem = dst->dnnl_float()->build_memory(ymem_desc);
        auto bmem = dnnl::memory();
        if ( bias != nullptr) {
            bmem = bias->dnnl_float()->build_memory(bmem_desc);
        }
        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, xmem},
            {DNNL_ARG_WEIGHTS, wmem},
            {DNNL_ARG_DST, ymem}};
        if ( bias != nullptr) {
            args[DNNL_ARG_BIAS] = bmem;
        }
        conv_prim.execute(*ComputingContext::dnnl_stream, args);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

tensor_t create_dnnl_float(std::vector<size_t>& shape_, bool gpu) {
#ifndef _DNNL_GPU_
    if ( gpu ) {
        vt_panic("Can't be here");
    }
#endif
    ShapeType shape(shape_);
    DNNLTensor<DataType::Float>* tensor = new DNNLTensor<DataType::Float>(shape, gpu);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_fp16(std::vector<size_t>& shape_, bool gpu) {
#ifndef _DNNL_GPU_
    if ( gpu ) {
        vt_panic("Can't be here");
    }
#endif
    ShapeType shape(shape_);
    DNNLTensor<DataType::FP16>* tensor = new DNNLTensor<DataType::FP16>(shape, gpu);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_int(std::vector<size_t>& shape_, bool gpu) {
#ifndef _DNNL_GPU_
    if ( gpu ) {
        vt_panic("Can't be here");
    }
#endif
    ShapeType shape(shape_);
    DNNLTensor<DataType::Int>* tensor = new DNNLTensor<DataType::Int>(shape, gpu);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_q8(std::vector<size_t>& shape_, bool gpu) {
#ifndef _DNNL_GPU_
    if ( gpu ) {
        vt_panic("Can't be here");
    }
#endif
    ShapeType shape(shape_);
    DNNLTensor<DataType::Q8>* tensor = new DNNLTensor<DataType::Q8>(shape, gpu);
    return std::make_shared<TensorType>(tensor, shape);
}


}
