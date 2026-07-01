#ifndef _HIP_IMPL_HPP_
#define _HIP_IMPL_HPP_

#include "vt.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct HIPTensor : public TransformerComputing {
    virtual ~HIPTensor();
    HIPTensor(const ShapeType& shape);
    HIPTensor(const ShapeType& shape, void * vdata);

public:
    ComputingReturn io_load(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    //ComputingReturn io_save(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, size_t> op_sizeof(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, void *> op_data(ComputingContext* ctx, tensor_t self) override;

    ComputingReturn op_zero(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn op_fill(ComputingContext* ctx, tensor_t self, float value) override;
    ComputingReturn op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) override;
    ComputingReturn op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t output) override;

    ComputingReturn op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) override;
    ComputingReturn op_copy_to(ComputingContext* ctx, tensor_t self, tensor_t dst) override;
    ComputingReturn op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) override;
    std::variant<ComputingReturn, tensor_t> op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;

    ComputingReturn op_quantize(ComputingContext* ctx, tensor_t self, tensor_t out) override;
    ComputingReturn op_dequantize(ComputingContext* ctx, tensor_t self, tensor_t out) override;
    ComputingReturn op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t output) override;
    ComputingReturn op_add(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) override;

    ComputingReturn op_linear(ComputingContext* ctx, tensor_t self, tensor_t w, tensor_t bias, tensor_t y) override;
    ComputingReturn op_layernorm(ComputingContext* ctx, tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;
    ComputingReturn op_rmsnorm(ComputingContext* ctx, tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) override;
    ComputingReturn op_rotary_embed(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;
    ComputingReturn op_transpose_0213(ComputingContext* ctx, tensor_t self, tensor_t y) override;
    ComputingReturn op_transpose_0213_repeated(ComputingContext* ctx, tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(ComputingContext* ctx, tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(ComputingContext* ctx, tensor_t self, tensor_t out) override ;
    ComputingReturn op_attn(ComputingContext* ctx, tensor_t self, tensor_t v, tensor_t attn) override;
    ComputingReturn op_gelu(ComputingContext* ctx, tensor_t self, tensor_t dst) override ;
    ComputingReturn op_silu_product(ComputingContext* ctx, tensor_t self, tensor_t up, tensor_t dst) override;
    std::variant<ComputingReturn, int> op_all_logits(ComputingContext* ctx, tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output ) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top1(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top3(ComputingContext* ctx, tensor_t self, float temp) override;

    ComputingReturn op_conv2d(ComputingContext* ctx, tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) override;

protected:
    inline void* data() {
        return mem_;
    }

protected:
    const size_t size_;
    const bool owner_;
    void* mem_;

    friend struct HIPTensor<DataType::F32>;
    friend struct HIPTensor<DataType::I32>;
    friend struct HIPTensor<DataType::F16>;
    friend struct HIPTensor<DataType::BF16>;
    friend struct HIPTensor<DataType::Q8>;
    friend struct HIPTensor<DataType::Q4>;
    friend struct HIPTensor<DataType::PQ>;
};

} // end of namespace
#endif
