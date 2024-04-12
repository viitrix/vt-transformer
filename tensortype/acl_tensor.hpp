#ifndef _ACL_IMPL_HPP_
#define _ACL_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace arm_compute {
    class Tensor;
}

namespace vt {


template <DataType _DTYPE_>
struct ACLTensor : public TransformerComputing {
    virtual ~ACLTensor();
    ACLTensor(const ShapeType& shape);
    ACLTensor(const ShapeType& shape,  void *mem);
    void* data() {
        return mem_;
    }

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;

    ComputingReturn op_copy(tensor_t self, tensor_t from) override;
protected:
    const bool owner_;
    void* mem_;
    size_t size_;

    arm_compute::Tensor* t_;

    friend struct ACLTensor<DataType::Float>;
    friend struct ACLTensor<DataType::Int>;
    friend struct ACLTensor<DataType::FP16>;
};


} // end of namespace
#endif
