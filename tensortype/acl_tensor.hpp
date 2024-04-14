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

private:
    void buildTensorWithShape(arm_compute::Tensor& target, const std::vector<size_t> newShape);
    void buildTensorWithShape(arm_compute::Tensor& target, const std::vector<size_t> newShape, void *mem);

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;

    ComputingReturn op_copy(tensor_t self, tensor_t from) override;
    ComputingReturn op_convert(tensor_t self, tensor_t from) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;

    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) override;
    ComputingReturn op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) override;
    ComputingReturn op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;

    ComputingReturn op_transpose_0213(tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(tensor_t self, tensor_t out) override;
    ComputingReturn op_attn(tensor_t self, tensor_t value, tensor_t out) override;
    ComputingReturn op_silu_product(tensor_t self, tensor_t in, tensor_t dst) override;

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
