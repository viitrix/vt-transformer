#include "cuda_tensor.hpp"

namespace vt {

template<DataType _DT_>
CUDATensor<_DT_>::CUDATensor(const ShapeType& shape) {
    size_t asize = 0;
    size_t number = shape.numel();
    if ( _DT_ == DataType::F32 ) {
        asize = sizeof(float) * number;
    } else if ( _DT_ == DataType::I32 ) {
        asize = sizeof(int) * number;
    } else {
        vt_fatal_error();
    }
}

}
