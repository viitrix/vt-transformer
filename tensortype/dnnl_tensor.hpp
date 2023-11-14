#ifndef _DNNL_IMPL_HPP_
#define _DNNL_IMPL_HPP_

#include "host_tensor.hpp"

namespace vt {
using local_fp16_t = uint16_t;

template <DataType _DTYPE_>
struct DNNLTensor : public HostTensor<_DTYPE_> {
    virtual ~DNNLTensor() { }
    DNNLTensor(const ShapeType& shape) : HostTensor<_DTYPE_>(shape) { }

public:
    // Interfaces from TransformerComputing
    friend struct DNNLTensor<DataType::Float>;
    friend struct DNNLTensor<DataType::BF16>;
    friend struct DNNLTensor<DataType::Int>;
};

} // end of namespace
#endif
