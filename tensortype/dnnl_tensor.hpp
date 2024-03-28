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

    dnnl::memory::desc build_memory_desc(const std::vector<size_t>& shape, DataType dt, dnnl::memory::format_tag tag) {
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

        vt_panic("Can't be here!");
        return dnnl::memory::desc();
    }
    dnnl::memory::desc build_memory_desc(const std::vector<size_t>& shape, dnnl::memory::format_tag tag) {
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

        vt_panic("Can't be here!");
        return dnnl::memory::desc();
    }

    dnnl::memory build_memory(const dnnl::memory::desc& desc) {
        vt_assert( desc.get_size() == size_ , "dnnl memory's data must have same size with desc");
        return dnnl::memory(desc, *ComputingContext::dnnl_engine, mem_);
    }

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;

    ComputingReturn op_convert(tensor_t self, tensor_t from) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;

    ComputingReturn op_scale(tensor_t self, float scale) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_conv2d(tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) override;

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
