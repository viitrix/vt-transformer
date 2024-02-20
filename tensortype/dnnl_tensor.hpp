#ifndef _DNNL_IMPL_HPP_
#define _DNNL_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct DNNLTensor : public TransformerComputing {
    virtual ~DNNLTensor() {
        if ( owner_ ) {
            MemoryContext::free(mem_, size_);
        }
    }
    DNNLTensor(const ShapeType& shape) : owner_(true) {
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
    DNNLTensor(const ShapeType& shape,  void *mem) : owner_(false), mem_(mem) {
        if ( _DTYPE_ != DataType::Float
             && _DTYPE_ != DataType::Int
             && _DTYPE_ != DataType::FP16 ) {
            vt_panic("Can't be here!");
        }
        size_ = 0;
    }
    void* data() {
        return mem_;
    }
public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;

protected:
    const bool owner_;
    void* mem_;
    size_t size_;

    friend struct DNNLTensor<DataType::Float>;
    friend struct DNNLTensor<DataType::Int>;
    friend struct DNNLTensor<DataType::FP16>;
};


} // end of namespace
#endif
