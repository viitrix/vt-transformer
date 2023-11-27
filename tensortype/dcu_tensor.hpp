#ifndef _DCU_IMPL_HPP_
#define _DCU_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

#include <hip/hip_runtime.h>

namespace vt {

template <DataType _DTYPE_>
struct DCUTensor : public TransformerComputing {
    virtual ~DCUTensor() { }
    DCUTensor(const ShapeType& shape) { }

public:
    // Interfaces from TransformerComputing
    friend struct DCUTensor<DataType::Float>;
    friend struct DCUTensor<DataType::BF16>;
    friend struct DCUTensor<DataType::Int>;
};

} // end of namespace
#endif
