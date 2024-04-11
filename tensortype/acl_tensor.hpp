#ifndef _ACL_IMPL_HPP_
#define _ACL_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct ACLTensor : public TransformerComputing {
    virtual ~ACLTensor() {
        if ( owner_ ) {
            MemoryContext::free(mem_, size_);
        }
    }
    ACLTensor(const ShapeType& shape) : owner_(true) {
        if ( _DTYPE_ == DataType::Float ) {
            size_ = shape.numel() * sizeof(float);
        } else if ( _DTYPE_ == DataType::Int ) {
            size_ = shape.numel() * sizeof(int);
        } else if ( _DTYPE_ == DataType::FP16 ) {
            size_ =  shape.numel() * sizeof(local_fp16_t);
        } else {
            vt_panic("Can't be here!");
        }

        mem_ = MemoryContext::alloc(size_);
    }
    ACLTensor(const ShapeType& shape,  void *mem) : owner_(false), mem_(mem) {
        if ( _DTYPE_ == DataType::Float ) {
            size_ = shape.numel() * sizeof(float);
        } else if ( _DTYPE_ == DataType::Int ) {
            size_ = shape.numel() * sizeof(int);
        } else if ( _DTYPE_ == DataType::FP16 ) {
            size_ =  shape.numel() * sizeof(local_fp16_t);
        } else {
            vt_panic("Can't be here!");
        }
    }
    void* data() {
        return mem_;
    }

protected:
    const bool owner_;
    void* mem_;
    size_t size_;

    friend struct ACLTensor<DataType::Float>;
    friend struct ACLTensor<DataType::Int>;
    friend struct ACLTensor<DataType::FP16>;
};


} // end of namespace
#endif
