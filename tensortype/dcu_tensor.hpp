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
    DCUTensor(ShapeType& shape, void *mem) : mem_(mem), owner_(false) {
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
