#ifndef _CUDA_IMPL_HPP_
#define _CUDA_IMPL_HPP_

#include "vt.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {


template <DataType _DTYPE_>
struct CUDATensor : public TransformerComputing {
    ~CUDATensor();
    CUDATensor(const ShapeType& shape);
    CUDATensor(const ShapeType& shape, void * vdata);

public:
    ComputingReturn io_load(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    //ComputingReturn io_save(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, size_t> op_sizeof(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, void *> op_data(ComputingContext* ctx, tensor_t self) override;

    ComputingReturn op_zero(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn op_fill(ComputingContext* ctx, tensor_t self, float value) override;
    ComputingReturn op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) override;
    ComputingReturn op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t output) override;

    ComputingReturn op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) override;
    ComputingReturn op_copy_to(ComputingContext* ctx, tensor_t self, tensor_t dst) override;
    ComputingReturn op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) override;
    std::variant<ComputingReturn, tensor_t> op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;

protected:
    inline void* data() {
        return mem_;
    }
    
protected:
    const size_t size_;
    const bool owner_;
    void* mem_;

    friend struct CUDATensor<DataType::F32>;
    friend struct CUDATensor<DataType::I32>;
    friend struct CUDATensor<DataType::F16>;
    friend struct CUDATensor<DataType::BF16>;
    friend struct CUDATensor<DataType::Q8>;
    friend struct CUDATensor<DataType::Q4>;
    friend struct CUDATensor<DataType::PQ>;
};

} // end of namespace
#endif
