#include <arm_compute/core/Types.h>
#include <arm_compute/core/utils/quantization/AsymmHelpers.h>
#include <arm_compute/core/WindowIterator.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>

#include "acl_tensor.hpp"
#include "acl_kernels/impl.hpp"
#include "host_tensor.hpp"

namespace vt { 

inline arm_compute::TensorShape buildShape(const ShapeType& s) {
    auto vs = s.vec();
    if ( vs.size() == 1 ) {
        return arm_compute::TensorShape( {vs[0]} );
    }
    if ( vs.size() == 2 ) {
        return arm_compute::TensorShape( {vs[0], 1, 1, vs[1]} );
    }
    if ( vs.size() == 3 ) {
        return arm_compute::TensorShape( {vs[0], 1, vs[1], vs[2]} );
    }
    if ( vs.size() == 4 ) {
        return arm_compute::TensorShape( {vs[0], vs[1], vs[2], vs[3]} );
    }

    return arm_compute::TensorShape();
}


template <DataType _DTYPE_> 
ACLTensor<_DTYPE_>::~ACLTensor()  {
    if ( owner_ ) {
        MemoryContext::free(mem_, size_);
    }
    delete t_;
}

template <DataType _DTYPE_> 
ACLTensor<_DTYPE_>::ACLTensor(const ShapeType& shape) : owner_(true) {
    t_ = new arm_compute::Tensor();
    arm_compute::TensorShape ts = buildShape(shape);

    if ( _DTYPE_ == DataType::Float ) {
        size_ = shape.numel() * sizeof(float);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
    } else if ( _DTYPE_ == DataType::Int ) {
        size_ = shape.numel() * sizeof(int);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        size_ =  shape.numel() * sizeof(local_fp16_t);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
    }
    mem_ = MemoryContext::alloc(size_);
    t_->allocator()->import_memory(mem_);
}

template <DataType _DTYPE_> 
ACLTensor<_DTYPE_>::ACLTensor(const ShapeType& shape,  void *mem) : owner_(false), mem_(mem) {
    t_ = new arm_compute::Tensor();
    arm_compute::TensorShape ts = buildShape(shape);
    size_ = 0;

    if ( _DTYPE_ == DataType::Float ) {
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
    } else if ( _DTYPE_ == DataType::Int ) {
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
    }
    
    mem_ = MemoryContext::alloc(size_);
    t_->allocator()->import_memory(mem_);
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::io_dump(tensor_t self) {
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
std::variant<ComputingReturn, size_t> ACLTensor<_DTYPE_>::op_sizeof(tensor_t self) {
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
ComputingReturn ACLTensor<_DTYPE_>::op_zero(tensor_t self) {
    memset(mem_, 0, size_);
    return OP_OK;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_alibi(tensor_t self) {
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

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];

    int* mask  = (int *)data();

    float*          out32 = nullptr;
    local_fp16_t*   out16 = nullptr;

    if ( out->dtype() == DataType::Float ) {
        out32 = (float *)out->acl_float()->data();
    } else if ( out->dtype() == DataType::FP16 ) {
        out16 = (local_fp16_t *)out->acl_fp16()->data();
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
            acl_kernels::fill_causal_mask<float>(m, o, minv, full_tokens, nt_end);
        }
        if ( out16 != nullptr ) {
            local_fp16_t* o = &out16[ b * new_tokens * full_tokens + nt * full_tokens ];
            local_fp16_t minv = (unsigned short)0xFC00U;
            acl_kernels::fill_causal_mask<local_fp16_t>(m, o, minv, full_tokens, nt_end);
        }
    }

    return OP_OK;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_rotary_cache(tensor_t self, float base) {
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

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

    const char* d = (const char *)data();
    size_t len = std::get<1>(self->op_sizeof(self));
    wf.write(d, len);
    wf.close();
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {
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
ComputingReturn ACLTensor<DT>::op_fill(tensor_t self, float value) {
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
ComputingReturn ACLTensor<DT>::op_copy(tensor_t self, tensor_t from) {
    arm_compute::NECast op;
    if ( DT == DataType::Float) {
        op.configure(  from->acl_float()->t_, t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if ( DT == DataType::FP16) {
        op.configure(  from->acl_fp16()->t_, t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if ( DT == DataType::Int) {
        op.configure(  from->acl_int()->t_, t_,  arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_convert(tensor_t self, tensor_t from) {
    vt_assert( self->shape().dim() == 4, "convert support 4D tensor only!");
    if ( DT == DataType::FP16 && from->is_float() ) {
        
        return OP_OK;
    }
    if ( DT == DataType::Float && from->is_fp16() ) {
        
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> ACLTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    if ( _DTYPE_ == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCpuTensor = new ACLTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Int ) {
        ShapeType newShape(newShape_);
        int *newData = (int *)data() + offset;
        auto* newCpuTensor = new ACLTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        ShapeType newShape(newShape_);
        local_fp16_t *newData = (local_fp16_t *)data() + offset;
        auto* newCpuTensor = new ACLTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> ACLTensor<_DT_>::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
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
        auto* newTensor = new ACLTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        auto* newTensor = new ACLTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        auto* newTensor = new ACLTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( owner_ == true ) {
        return OP_INPUT_ERROR;
    }

    if ( newShape.numel() + offset > self->items()  ) {
        return OP_INPUT_ERROR;
    }

    delete t_;
    t_ = new arm_compute::Tensor();
    arm_compute::TensorShape ts = buildShape(newShape);

    if ( DT == DataType::Float ) {
        mem_  = (char *)data() + offset * sizeof(float);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
        
    } else if ( DT == DataType::Int ) {
        mem_  = (char *)data() + offset * sizeof(int);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( DT == DataType::FP16 ) {
        mem_  = (char *)data() + offset * sizeof(local_fp16_t);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
        return OP_INPUT_ERROR;
    }

    mem_ = MemoryContext::alloc(size_);
    t_->allocator()->import_memory(mem_);    
    return OP_OK;
}


template<DataType DT>
ComputingReturn ACLTensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    arm_compute::NEArithmeticAddition op;
    if (   DT == DataType::Float) {
        op.configure(t_, b->acl_float()->t_, c->acl_float()->t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        op.configure(t_, b->acl_fp16()->t_, c->acl_fp16()->t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if (   DT == DataType::Int) {
        op.configure(t_, b->acl_int()->t_, c->acl_int()->t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

tensor_t create_acl_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    ACLTensor<DataType::Float>* tensor = new ACLTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_acl_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    ACLTensor<DataType::FP16>* tensor = new ACLTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_acl_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    ACLTensor<DataType::Int>* tensor = new ACLTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}