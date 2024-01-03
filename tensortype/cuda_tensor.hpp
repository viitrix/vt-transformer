#ifndef _CUDA_IMPL_HPP_
#define _CUDA_IMPL_HPP_

#include "tensortype.hpp"

namespace vt {

template <DataType _DTYPE_>
struct CUDATensor : public TransformerComputing {
    virtual ~CUDATensor();
    CUDATensor(const ShapeType& shape);
    CUDATensor(ShapeType& shape, void *mem) : mem_(mem), owner_(false), PQ_S_(0) {
        if ( _DTYPE_ == DataType::Q4 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");
        }
        if ( _DTYPE_ == DataType::Q8 ) {
            size_t last_dim = shape.vec().back();
            vt_assert( last_dim > 128, "Q8 tensor last dim must > 128k");
        }
        if ( _DTYPE_ == PQ ) {
            vt_panic("Can't CUDA PQ type from view");
        }
    }
    CUDATensor(const ShapeType& shape, int S);
    CUDATensor(ShapeType& shape, void *tab, uint8_t* idx, int S) : mem_(nullptr), owner_(false), PQ_S_(S), PQ_tab_(tab), PQ_idx_(idx) {
        vt_assert(_DTYPE_ == DataType::PQ, "Constructor for PQ view only");
    }
    void* data() {
        if ( _DTYPE_ == PQ && owner_ == false) {
            vt_panic("Can't CUDA PQ type from view");
        }
        return mem_;
    }

    cudnnTensorDescriptor_t create_cudnn_td_with(const std::vector<size_t> shape) {
        cudnnTensorFormat_t  format = CUDNN_TENSOR_NCHW;

        cudnnDataType_t dtype;
        cudnnTensorDescriptor_t desc;

        if ( _DTYPE_ == DataType::Float ) {
            dtype = CUDNN_DATA_FLOAT;
        } else if ( _DTYPE_ == DataType::FP16 ) {
            dtype = CUDNN_DATA_HALF;
        } else if ( _DTYPE_ == DataType::Int ) {
            dtype = CUDNN_DATA_INT32;
        } else {
            vt_panic("cudnn don't support!");
        }

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

        if (shape.size() == 4) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], shape[2], shape[3]));
        } else if (shape.size() == 3) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, shape[2]));
        } else if (shape.size() == 2) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, 1));
        } else if (shape.size() == 1) {
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, 1, shape[0], 1, 1));
        } else {
            vt_panic("cudnnSetTensor4dDescriptor: can't convert shape");
        }
        return desc;
    }

    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_nccl_send(tensor_t self, int dst) override;
    ComputingReturn io_nccl_recv(tensor_t self, int src) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;
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

    std::variant<ComputingReturn, float> op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) override;
    ComputingReturn op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) override;
    ComputingReturn op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) override;
    ComputingReturn op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) override;
    ComputingReturn op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) override;
    ComputingReturn op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) override;
    ComputingReturn op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) override;
    ComputingReturn op_qk_backward(tensor_t self, tensor_t q, tensor_t k, tensor_t q_g, tensor_t k_g) override;


private:
    void*                       mem_;
    const bool                  owner_;
    const int PQ_S_;
    void *PQ_tab_;
    uint8_t *PQ_idx_;

    friend struct CUDATensor<DataType::Float>;
    friend struct CUDATensor<DataType::Int>;
    friend struct CUDATensor<DataType::FP16>;
    friend struct CUDATensor<DataType::Q8>;
    friend struct CUDATensor<DataType::Q4>;
    friend struct CUDATensor<DataType::PQ>;
};


} // end of namespace
#endif
