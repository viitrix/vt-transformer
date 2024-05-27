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

public:
    ComputingReturn io_load(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    //ComputingReturn io_save(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, size_t> op_sizeof(ComputingContext* ctx, tensor_t self) override;

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
