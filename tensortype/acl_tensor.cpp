#include <arm_compute/core/Types.h>
#include <arm_compute/core/utils/quantization/AsymmHelpers.h>
#include <arm_compute/core/WindowIterator.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>

#include "acl_tensor.hpp"
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
    arm_compute::NECopy op;
    if ( DT == DataType::FP16) {
        op.configure(  from->acl_fp16()->t_, t_);
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