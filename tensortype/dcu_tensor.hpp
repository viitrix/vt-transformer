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
        /*
        if ( _DTYPE_ == DataType::Q4 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");
        }
        if ( _DTYPE_ == DataType::Q8 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim > 128, "Q8 tensor last dim must > 128k");
        }
        */
    }
    void* data() {
        return mem_;
    }

public:
    // Interfaces from TransformerComputing

protected:
    const bool owner_;
    void* mem_;
    size_t size_;

    friend struct DCUTensor<DataType::Float>;
    friend struct DCUTensor<DataType::BF16>;
    friend struct DCUTensor<DataType::Int>;
};

} // end of namespace
#endif
