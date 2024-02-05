#ifndef _COREX_IMPL_HPP_
#define _COREX_IMPL_HPP_

#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct CXTensor : public TransformerComputing {
    virtual ~CXTensor() {
        if (mem_ != nullptr && owner_) {
            COREX_CHECK(cudaFree(mem_));
        }
    }
    CXTensor(const ShapeType& shape);
    CXTensor(const ShapeType& shape, int S);
    CXTensor(ShapeType& shape, void *mem) : owner_(false), mem_(mem), PQ_S_(0) {
        if ( _DTYPE_ == DataType::Q4 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");
        }
        if ( _DTYPE_ == DataType::Q8 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim > 128, "Q8 tensor last dim must > 128");
        }
        if ( _DTYPE_ == PQ ) {
            vt_panic("Can't CX PQ type from this constructor!");
        }
    }
    CXTensor(ShapeType& shape, void *tab, uint8_t* idx, int S) : owner_(false), mem_(nullptr), PQ_S_(S), PQ_tab_(tab), PQ_idx_(idx) {
        vt_assert(_DTYPE_ == DataType::PQ, "Constructor for PQ view only");
    }
    void* data() {
        if ( _DTYPE_ == PQ && owner_ == false) {
            vt_panic("Can't use CX mem_ from PQ view  ");
        }
        return mem_;
    }

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;

    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;

#if 0
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    ComputingReturn op_quantize(tensor_t self, tensor_t out) override;
    ComputingReturn op_dequantize(tensor_t self, tensor_t out) override;
    ComputingReturn op_embed(tensor_t self, tensor_t table, tensor_t out) override;
    ComputingReturn op_copy(tensor_t self, tensor_t src) override;
    ComputingReturn op_convert(tensor_t self, tensor_t src) override;
    ComputingReturn op_scale(tensor_t self, float scale) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) override;

    ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) override;
    ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;
    ComputingReturn op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) override;
    ComputingReturn op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;

    ComputingReturn op_transpose_0213(tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(tensor_t self, tensor_t out) override;
    ComputingReturn op_attn(tensor_t self, tensor_t value, tensor_t out) override;
    ComputingReturn op_gelu(tensor_t self, tensor_t dst) override;
    ComputingReturn op_silu_product(tensor_t self, tensor_t in, tensor_t dst) override;
    std::variant<ComputingReturn, int> op_all_logits(tensor_t self, tensor_t mask,  tensor_t lm_head, tensor_t output) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top1(tensor_t self) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top3(tensor_t self, float temp) override;
#endif

protected:
    const bool owner_;
    void* mem_;
    size_t size_;
    const int PQ_S_;
    void *PQ_tab_;
    uint8_t *PQ_idx_;

    friend struct CXTensor<DataType::Float>;
    friend struct CXTensor<DataType::BF16>;
    friend struct CXTensor<DataType::Int>;
    friend struct CXTensor<DataType::Q8>;
    friend struct CXTensor<DataType::Q4>;
};

} // end of namespace
#endif
