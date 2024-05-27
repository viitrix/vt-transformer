#ifndef _CUDA_IMPL_HPP_
#define _CUDA_IMPL_HPP_

#include "vt.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {


template <DataType _DTYPE_>
struct CUDATensor : public TransformerComputing {
    CUDATensor(const ShapeType& shape);

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
