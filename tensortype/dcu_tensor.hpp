#ifndef _DCU_IMPL_HPP_
#define _DCU_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct DCUTensor : public TransformerComputing {
    virtual ~DCUTensor() {
        if (mem_ != nullptr && owner_) {
            HIP_CHECK(hipFree(mem_));
        }
    }
    DCUTensor(const ShapeType& shape);
    DCUTensor(ShapeType& shape, void *mem) : owner_(false), mem_(mem) {
        if ( _DTYPE_ == DataType::Q4 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");
        }
        if ( _DTYPE_ == DataType::Q8 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim > 128, "Q8 tensor last dim must > 128");
        }
    }
    void* data() {
        return mem_;
    }

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;

    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    ComputingReturn op_quantize(tensor_t self, tensor_t out) override;
    ComputingReturn op_dequantize(tensor_t self, tensor_t out) override;
    ComputingReturn op_embed(tensor_t self, tensor_t table, tensor_t out) override;
    ComputingReturn op_copy(tensor_t self, tensor_t src) override;
    ComputingReturn op_convert(tensor_t self, tensor_t src) override;
    ComputingReturn op_scale(tensor_t self, float scale) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) override;


protected:
    const bool owner_;
    void* mem_;
    size_t size_;

    friend struct DCUTensor<DataType::Float>;
    friend struct DCUTensor<DataType::BF16>;
    friend struct DCUTensor<DataType::Int>;
    friend struct DCUTensor<DataType::Q8>;
    friend struct DCUTensor<DataType::Q4>;
};

} // end of namespace
#endif
