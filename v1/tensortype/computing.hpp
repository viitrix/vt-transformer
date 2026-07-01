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

const int Q4_BLOCK_SIZE = 64;
const int Q8_BLOCK_SIZE = 1024;

struct ComputingContext;
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
    virtual ComputingReturn io_load(ComputingContext* ctx, tensor_t self, const char* fileName) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_save(ComputingContext* ctx, tensor_t self, const char* fileName) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_dump(ComputingContext* ctx, tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_pipe_read(ComputingContext* ctx, tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn io_pipe_write(ComputingContext* ctx, tensor_t self, int dst) {
        return OP_TODO_ERROR;
    }

    virtual std::variant<ComputingReturn, size_t> op_sizeof(ComputingContext* ctx, tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, void *> op_data( ComputingContext* ctx, tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_zero(ComputingContext* ctx, tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_fill(ComputingContext* ctx, tensor_t self, float value) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t output) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_copy_to(ComputingContext* ctx, tensor_t self, tensor_t dst) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_quantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_dequantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t output) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_add(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_mul(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_linear(ComputingContext* ctx, tensor_t self, tensor_t w, tensor_t bias, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_layernorm(ComputingContext* ctx, tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rmsnorm(ComputingContext* ctx, tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_rotary_embed(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_transpose_0213(ComputingContext* ctx, tensor_t self, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_transpose_0213_rotary(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_transpose_0213_repeated(ComputingContext* ctx, tensor_t self, tensor_t y) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_qk(ComputingContext* ctx, tensor_t self, tensor_t k, tensor_t qk) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_softmax(ComputingContext* ctx, tensor_t self, tensor_t out) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_attn(ComputingContext* ctx, tensor_t self, tensor_t v, tensor_t attn) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_gelu(ComputingContext* ctx, tensor_t self, tensor_t dst) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_silu_product(ComputingContext* ctx, tensor_t self, tensor_t up, tensor_t dst) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, int> op_all_logits(ComputingContext* ctx, tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output ) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_sampling_top1(ComputingContext* ctx, tensor_t self) {
        return OP_TODO_ERROR;
    }
    virtual std::variant<ComputingReturn, tensor_t> op_sampling_top3(ComputingContext* ctx, tensor_t self, float temp) {
        return OP_TODO_ERROR;
    }
    virtual ComputingReturn op_conv2d(ComputingContext* ctx, tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) {
        return OP_TODO_ERROR;
    }

};


} // endof namespace vt


#endif
