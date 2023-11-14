#ifndef _COMPUTING_HPP_
#define _COMPUTING_HPP_

#include <variant>
#include <vector>
#include <memory>

#define op_check(ret, Msg)                     \
    if ( ret != vt::OP_OK ) {                  \
        vt::_M_Panic(__FILE__, __LINE__, Msg); \
    }                                          \
    return ret

namespace vt {
struct TensorType;
using tensor_t = std::shared_ptr<TensorType>;

// low level API for implementing Transformer
enum ComputingReturn {
    OP_OK = 0,
    OP_TODO_ERROR = -1,
    OP_INPUT_ERROR = -2,
    OP_OUTPUT_ERROR = -3,
    OP_ATTR_ERROR = -4,
};

struct TransformerComputing {
    //TransformerComputing() = delete;
    virtual ComputingReturn io_load(tensor_t self, const char* fileName) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_save(tensor_t self, const char* fileName) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_dump(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_mpi_bcast(tensor_t self, int root) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_mpi_recv(tensor_t self, int source) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_mpi_send(tensor_t self, int dst) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_nccl_recv(tensor_t self, int source) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_nccl_send(tensor_t self, int dst) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_pipe_read(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_pipe_write(tensor_t self, int n) {
        return OP_TODO_ERROR;
    }

    virtual std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_zero(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_fill(tensor_t self, float value) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_alibi(tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rotary_cache(tensor_t self, float base) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_causal_mask(tensor_t self, tensor_t output) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_copy(tensor_t self, tensor_t src) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_convert(tensor_t self, tensor_t src) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_quantize(tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_dequantize(tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_embed(tensor_t self, tensor_t table, tensor_t output) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_scale(tensor_t self, float scale) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t bias, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_transpos_0213(tensor_t self, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_softmax(tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_attn(tensor_t self, tensor_t v, tensor_t attn) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_xattn(tensor_t self, tensor_t k, tensor_t v, tensor_t qk, tensor_t attn) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_gelu(tensor_t self, tensor_t dst) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_silu_product(tensor_t self, tensor_t up, tensor_t dst) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, int> op_all_logits(tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output ) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_sampling_top3(tensor_t self, float temp) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, float> op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rmsnorm_backward(tensor_t self, tensor_t x, tensor_t scale, tensor_t norm2, tensor_t dscale, tensor_t dx, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_qk_backward(tensor_t self, tensor_t q, tensor_t k, tensor_t q_g, tensor_t k_g) {
        return OP_TODO_ERROR;
    }
};


} // endof namespace vt


#endif
