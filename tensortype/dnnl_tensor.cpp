#include "dnnl_tensor.hpp"
#include "dnnl_kernels/misc.hpp"

namespace vt {

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_dump(tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
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
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> DNNLTensor<_DTYPE_>::op_sizeof(tensor_t self) {
    if ( _DTYPE_ == DataType::Float ) {
        return (size_t) self->items() * sizeof(float);
    }
    if ( _DTYPE_ == DataType::Int ) {
        return (size_t) self->items() * sizeof(int);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        return (size_t) self->items() * sizeof(local_fp16_t);
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::op_zero(tensor_t self) {
    memset(mem_, 0, size_);
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn DNNLTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

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

    if (_DTYPE_ == DataType::Float) {
        size_t ret = inf.read( (char *)data(), sizeof(float) * self->items() ).gcount();
        vt_assert(ret == sizeof(float) * self->items(), "file size dont't match tensor");
    } else  if (_DTYPE_ == DataType::Int) {
        size_t ret = inf.read( (char *)data(), sizeof(int) * self->items() ).gcount();
        vt_assert(ret == sizeof(int) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::FP16) {
        size_t ret = inf.read( (char *)data(), sizeof(local_fp16_t) * self->items() ).gcount();
        vt_assert(ret == sizeof(local_fp16_t) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::Q8 ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)data(), s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::Q4 ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)data(), s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::PQ ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)mem_, s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else {
        vt_panic("DataType don't support");
    }

    inf.close();
    return OP_OK;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_fill(tensor_t self, float value) {
    size_t items = self->items();
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

template<typename T>
void fill_causal_mask(int* m, T* o, T minv, int full_tokens, int nt_end) {
    for ( int i = 0; i < full_tokens; i++) {
        o[i] = minv;
    }

    for ( int i = 0; i <= nt_end; i++) {
        if ( m[i] != 0 ) {
            o[i] = 0;
        }
    }
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];

    int* mask  = (int *)data();

    float*          out32 = nullptr;
    local_fp16_t*   out16 = nullptr;

    if ( out->dtype() == DataType::Float ) {
        out32 = (float *)out->dnnl_float()->data();
    }
    if ( out->dtype() == DataType::FP16 ) {
        out16 = (local_fp16_t *)out->dnnl_fp16()->data();
    }

    for (int e = 0; e < batch * new_tokens; e++) {
        int b = e / new_tokens;
        int nt = e % new_tokens;
        int nt_end = full_tokens - new_tokens + nt;

        int* m = &mask[ b * full_tokens ];
        if ( out32 != nullptr) {
            float* o = &out32[ b * new_tokens * full_tokens + nt * full_tokens ];
            float minv = std::numeric_limits<float>::lowest();
            fill_causal_mask<float>(m, o, minv, full_tokens, nt_end);
        }
        if ( out16 != nullptr ) {
            local_fp16_t* o = &out16[ b * new_tokens * full_tokens + nt * full_tokens ];
            local_fp16_t minv = (unsigned short)0xFC00U;
            fill_causal_mask<local_fp16_t>(m, o, minv, full_tokens, nt_end);
        }
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_rotary_cache(tensor_t self, float base) {
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
ComputingReturn DNNLTensor<DT>::op_scale(tensor_t self, float scale) {
    if (   DT == DataType::Float) {
        dnnl_kernels::eltwise_operate<DNNLTensor<DataType::Float>>(self->dnnl_float(), self->dnnl_float(), self->items(), 
            dnnl::algorithm::eltwise_linear, scale, 0.0);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::eltwise_operate<DNNLTensor<DataType::FP16>>(self->dnnl_fp16(),  self->dnnl_fp16(), self->items(), 
            dnnl::algorithm::eltwise_linear, scale, 0.0);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    if (   DT == DataType::Float) {
        dnnl_kernels::binary_operate_float(self, b, c, dnnl::algorithm::binary_add);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::binary_operate_fp16(self, b, c, dnnl::algorithm::binary_add);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    if (   DT == DataType::Float) {
        dnnl_kernels::binary_operate_float(self, b, c, dnnl::algorithm::binary_mul);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        dnnl_kernels::binary_operate_fp16(self, b, c, dnnl::algorithm::binary_mul);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DNNLTensor<DT>::op_convert(tensor_t self, tensor_t from) {
    vt_assert( self->shape().dim() == 4, "convert support 4D tensor only!");
    if ( DT == DataType::FP16 && from->is_float() ) {
        auto dst_desc = build_memory_desc(self->shape().vec(), tag::abcd);
        auto src_desc = from->dnnl_float()->build_memory_desc(from->shape().vec(), DataType::Float, tag::abcd);
        auto dst_mem = build_memory(dst_desc);
        auto src_mem = from->dnnl_float()->build_memory(src_desc);
        auto prim = dnnl::reorder(src_mem, dst_mem);

        prim.execute( *ComputingContext::dnnl_stream , src_mem, dst_mem);
        return OP_OK;
    }
    if ( DT == DataType::Float && from->is_fp16() ) {
        auto dst_desc = build_memory_desc(self->shape().vec(), tag::abcd);
        auto src_desc = from->dnnl_fp16()->build_memory_desc(from->shape().vec(), DataType::FP16, tag::abcd);
        auto dst_mem = build_memory(dst_desc);
        auto src_mem = from->dnnl_fp16()->build_memory(src_desc);
        auto prim = dnnl::reorder(src_mem, dst_mem);

        prim.execute( *ComputingContext::dnnl_stream , src_mem, dst_mem);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> DNNLTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
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

tensor_t create_dnnl_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DNNLTensor<DataType::Float>* tensor = new DNNLTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DNNLTensor<DataType::FP16>* tensor = new DNNLTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dnnl_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DNNLTensor<DataType::Int>* tensor = new DNNLTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
