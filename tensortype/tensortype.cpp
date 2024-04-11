/*
 *  Common arguments check and shared pre/post process for all device and data type.
 */

#include "vt.hpp"
#include "tensortype.hpp"
#include "host_tensor.hpp"
#ifdef _USING_DEVICE_DNNL_
#include "dnnl_tensor.hpp"
#endif
#ifdef _USING_DEVICE_ACL_
#include "acl_tensor.hpp"
#endif
#ifdef _USING_DEVICE_CUDA_
#include "cuda_tensor.hpp"
#endif
#ifdef _USING_DEVICE_DCU_
#include "dcu_tensor.hpp"
#endif
#ifdef _USING_DEVICE_COREX_
#include "corex_tensor.hpp"
#endif

namespace vt {

std::variant<ComputingReturn, size_t> TensorType::op_sizeof(tensor_t self) {
    vt_assert(self.get() == this, "can't be here!");

    auto result = impl()->op_sizeof(self);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view");
    }
    return result;
}

ComputingReturn TensorType::op_zero(tensor_t self) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_zero(self);
    op_check(ret, "zero");
}

ComputingReturn TensorType::op_fill(tensor_t self, float value) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_fill(self, value);
    op_check(ret, "fill");
}

ComputingReturn TensorType::op_alibi(tensor_t self) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(shape().dim() == 4, "alibi shape: [1, heads, 1, len], used for broadcast");
    vt_assert(shape()[0] == 1, "alibi shape: [1, heads, 1, len]");
    vt_assert(shape()[2] == 1, "alibi shape: [1, heads, 1, len]");
    auto ret = impl()->op_alibi(self);
    op_check(ret, "alibi");
}

ComputingReturn TensorType::op_rotary_cache(tensor_t self, float base) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(shape().dim() == 3, "rotary_cache shape: [lens, head_hiddens, 2]");
    vt_assert(shape()[2] == 2, "rotary_cache shape: [lens, head_hiddens, 2]");
    auto ret = impl()->op_rotary_cache(self, base);
    op_check(ret, "rotary_cache");
}

ComputingReturn TensorType::op_causal_mask(tensor_t self, tensor_t out) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(self->dtype() == DataType::Int, " mask must be int !");
    vt_assert(shape().dim() == 2, "op_causal_mask input mask shape: [batch, length]");
    vt_assert(out->shape().dim() == 4, "op_causal_mask input mask shape: [batch, 1, length, length]");
    vt_assert(out->shape()[1] == 1, "op_causal_mask input mask shape: [batch, 1, length, length]");
    auto ret = impl()->op_causal_mask(self, out);
    op_check(ret, "causal_mask");
}

ComputingReturn TensorType::op_copy(tensor_t self, tensor_t src) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(items() == src->items(), "copy must has same size");
    vt_assert(self->dtype() == src->dtype(), "Copy must has same data type");
    auto ret = impl()->op_copy(self, src);
    op_check(ret, "copy");
}

ComputingReturn TensorType::op_convert(tensor_t self, tensor_t src) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(items() == src->items(), "copy must has same size");
    vt_assert(self->dtype() != src->dtype(), "Convert must has diff data type");
    vt_assert(self->device_name() == src->device_name(), "Convert must has same device");
    auto ret = impl()->op_convert(self, src);
    op_check(ret, "copy");
}

std::variant<ComputingReturn, tensor_t> TensorType::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    vt_assert(self.get() == this, "can't be here!");
    int dynamic = -1;
    size_t sub_size = 1;
    for (size_t i = 0; i < newShape_.size(); i++) {
        if ( newShape_[i] == 0 ) {
            if ( dynamic == -1 ) {
                dynamic = i;
            } else {
                vt_panic("dynamic view must has one zero shape");
            }
        } else {
            sub_size = sub_size * newShape_[i];
        }
    }

    std::vector<size_t> newShape = newShape_;
    if ( dynamic >= 0 ) {
        size_t d = (self->items() - offset) / sub_size;
        newShape[dynamic] = d;
    }

    ShapeType s(newShape);
    vt_assert(offset + s.numel() <= items() , "view out of shape!");
    auto result = impl()->op_view(self, offset, newShape);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view");
    }
    return result;
}

std::variant<ComputingReturn, tensor_t> TensorType::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype_ ) {
    vt_assert(self.get() == this, "can't be here!");
    auto result = impl()->op_view_as(self, offset, newShape_, dtype_);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view_as");
    }
    return result;
}

ComputingReturn TensorType::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    vt_assert(self.get() == this, "can't be here!");

    ShapeType shape(newShape_);
    if (  shape.numel() + offset > items()  ) {
        vt_panic("op_reshape out of memory size!");
    }

    auto result = impl()->op_reshape(self, offset, newShape_);
    if ( result != ComputingReturn::OP_OK ) {
        return result;
    }
    self->shape_ = shape;
    return ComputingReturn::OP_OK;
}

ComputingReturn TensorType::op_quantize(tensor_t self, tensor_t out) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(self->shape() == out->shape(), "Input & output must has same size");
    vt_assert(out->is_quantized(), "Output must a quantized type");
    auto ret = impl()->op_quantize(self, out);
    op_check(ret, "quantize");
}

ComputingReturn TensorType::op_dequantize(tensor_t self, tensor_t out) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(self->shape() == out->shape(), "Input & output must has same size");
    vt_assert(self->is_quantized(), "Input must a quantized type");
    auto ret = impl()->op_dequantize(self, out);
    op_check(ret, "dequantize");
}

ComputingReturn TensorType::op_embed(tensor_t self, tensor_t table, tensor_t out) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(self->dtype() == DataType::Int, "token id must be Int");
    vt_assert(table->dtype() == out->dtype(), " output and table must have same DataType" );
    //vt_assert(table->impl_index() == out->impl_index(), "table and outspace must have same device");
    vt_assert(table->shape()[1] == out->shape()[2], "table and out must have same hidden size");
    auto ret = impl()->op_embed(self, table, out);
    op_check(ret, "embed");
}

ComputingReturn TensorType::op_scale(tensor_t self, float scale) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_scale(self, scale);
    op_check(ret, "scale");
}

ComputingReturn TensorType::op_add(tensor_t self, tensor_t b, tensor_t c) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(items() == c->items(), "add input and output must has same size");
    auto ret = impl()->op_add(self, b, c);
    op_check(ret, "add");
}

ComputingReturn TensorType::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(items() == c->items(), "add input and output must has same size");
    auto ret = impl()->op_mul(self, b, c);
    op_check(ret, "mul");
}

ComputingReturn TensorType::op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(shape().dim() == 3, " linear input shape: [batch, len, hidden] ");
    vt_assert(w->shape().dim() == 2, " linear weight shape: [outSize, inSize] ");
    vt_assert(w->shape()[1] == shape()[2], " linear input and weight must match" );
    auto ret = impl()->op_linear(self, w, b, y);
    op_check(ret, "linear");
}

ComputingReturn TensorType::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(mean->shape().dim() == 1, "layernorm size error!");
    vt_assert(var->shape().dim() == 1, "layernorm size error!");
    vt_assert(scale->shape().dim() == 1, "layernorm size error!");
    vt_assert(bias->shape().dim() == 1, "layernorm size error!");
    vt_assert(y->shape() == self->shape(), "layernorm size error!");
    vt_assert(self->shape()[-1] == mean->shape()[0] , "layernorm size error!");
    auto ret = impl()->op_layernorm(self, mean, var, scale, bias, y, eps);
    op_check(ret, "layernorm");
}

ComputingReturn TensorType::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    vt_assert(self.get() == this, "can't be here!");
    vt_assert(scale->shape().dim() == 3, "rmsnorm size error!");
    vt_assert(norm2->shape().dim() == 3, "rmsnorm size error!");
    vt_assert(self->shape().dim() == 3, "rmsnorm size error!");
    vt_assert(y->shape() == self->shape(), "rmsnorm size error!");
    vt_assert(self->shape()[-1] == scale->shape()[-1] , "rmsnorm size error!");
    auto ret = impl()->op_rmsnorm(self, scale, norm2, y, eps);
    op_check(ret, "rmsnorm");
}

ComputingReturn TensorType::op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_rotary_embed(self, cached, pos, y);
    op_check(ret, "rotary_embed");
}

ComputingReturn TensorType::op_transpose_0213(tensor_t self, tensor_t y) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_transpose_0213(self, y);
    op_check(ret, "transpose_0213");
}

ComputingReturn TensorType::op_qk(tensor_t self, tensor_t k, tensor_t qk) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_qk(self, k, qk);
    op_check(ret, "qk");
}

ComputingReturn TensorType::op_softmax(tensor_t self, tensor_t out) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax(self, out);
    op_check(ret, "softmax");
}

ComputingReturn TensorType::op_attn(tensor_t self, tensor_t v, tensor_t attn) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_attn(self, v, attn);
    op_check(ret, "attn");
}

ComputingReturn TensorType::op_xattn(tensor_t self, tensor_t k, tensor_t v, tensor_t qk, tensor_t attn) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_xattn(self, k, v, qk, attn);
    op_check(ret, "attn");
}

ComputingReturn TensorType::op_gelu(tensor_t self, tensor_t dst) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_gelu(self, dst);
    op_check(ret, "gelu");
}

ComputingReturn TensorType::op_silu_product(tensor_t self, tensor_t up, tensor_t dst) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_silu_product(self, up, dst);
    op_check(ret, "silu_product");
}

std::variant<ComputingReturn, int> TensorType::op_all_logits(tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_all_logits(self, mask, lm_head, output);
    if ( ret.index() == 0) {
        ComputingReturn r = std::get<0>(ret);
        op_check(r, "op_last_logits");
    }
    return ret;
}

std::variant<ComputingReturn, tensor_t> TensorType::op_sampling_top1(tensor_t self) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_sampling_top1(self);
    if ( ret.index() == 0) {
        ComputingReturn r = std::get<0>(ret);
        op_check(r, "op_sampling_top1");
    }
    return ret;
}

std::variant<ComputingReturn, tensor_t> TensorType::op_sampling_top3(tensor_t self, float temp) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_sampling_top3(self, temp);
    if ( ret.index() == 0) {
        ComputingReturn r = std::get<0>(ret);
        op_check(r, "op_sampling_top3");
    }
    return ret;
}

ComputingReturn TensorType::op_conv2d(tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) {
    vt_assert(self.get() == this, "can't be here!");
    // checking shape
    vt_assert(self->shape().dim() == 4, "conv2d accept NCHW only!");
    vt_assert(weight->shape().dim() == 4, "conv2d weight OIHW only!");
    vt_assert(self->shape().dims()[1] == weight->shape().dims()[1], "conv2d input channel must be matched with weight");
    auto ret = impl()->op_conv2d(self, weight, bias, dst, stride, padding);
    op_check(ret, "op_conv2d");
}

ComputingReturn TensorType::op_flash_attention(tensor_t query, tensor_t key, tensor_t value, tensor_t dst) {
    vt_assert(query.get() == this, "can't be here!");
    auto ret = impl()->op_flash_attention(query, key, value, dst);
    op_check(ret, "op_flash_attention");
}

std::variant<ComputingReturn, float> TensorType::op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) {
    vt_assert(self.get() == this, "can't be here!");

    auto result = impl()->op_loss_backward(self, ids, mask, lm_head, all_logits, x_g, lm_head_g);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "loss_backward");
    }
    return result;
}

ComputingReturn TensorType::op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_layernorm_backward(self, scale, bias, var, y, dscale, dbias, din, eps);
    op_check(ret, "layernorm_backward");
}

ComputingReturn TensorType::op_rmsnorm_backward(tensor_t self, tensor_t x, tensor_t scale, tensor_t norm2, tensor_t dscale, tensor_t dx, float eps) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_rmsnorm_backward(self, x, scale, norm2, dscale, dx, eps);
    op_check(ret, "rmsnorm_backward");
}

ComputingReturn TensorType::op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_linear_backward(self, x, weight, bias, x_g, weight_g, bias_g);
    op_check(ret, "linear_backward");
}

ComputingReturn TensorType::op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_gelu_backward(self, x, x_g);
    op_check(ret, "gelu_backward");
}

ComputingReturn TensorType::op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_attn_backward(self, attn, v, attn_g, v_g);
    op_check(ret, "gelu_attn_backward");
}

ComputingReturn TensorType::op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax_backward(self, out, x_g);
    op_check(ret, "gelu_softmax_backward");
}

ComputingReturn TensorType::op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax_attn_backward(self, attn, v, attn_g, v_g);
    op_check(ret, "gelu_softmax_attn_backward");
}

ComputingReturn TensorType::op_qk_backward(tensor_t self, tensor_t q, tensor_t k, tensor_t q_g, tensor_t k_g) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_qk_backward(self, q, k, q_g, k_g);
    op_check(ret, "gelu_qk_backward");
}

ComputingReturn TensorType::io_load(tensor_t self, const char* fileName) {
    vt_assert(this == self.get() , "can't be here!");
    auto ret = impl()->io_load(self, fileName);
    op_check(ret, "load");
}

ComputingReturn TensorType::io_save(tensor_t self, const char* fileName) {
    vt_assert(this == self.get() , "can't be here!");
    auto ret = impl()->io_save(self, fileName);
    op_check(ret, "save");
}

ComputingReturn TensorType::io_dump(tensor_t self) {
    vt_assert(self.get() == this, "can't be here!");
    std::cout << "--------------" << std::endl;
    std::cout << to_string() << std::endl;
    auto ret = impl()->io_dump(self);
    op_check(ret, "dump");
}

ComputingReturn TensorType::io_mpi_bcast(tensor_t self, int root) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_mpi_bcast(self, root);
    op_check(ret, "mpi_bcast");
}

ComputingReturn TensorType::io_mpi_recv(tensor_t self, int source) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_mpi_recv(self, source);
    op_check(ret, "mpi_recv");
}

ComputingReturn TensorType::io_mpi_send(tensor_t self, int dst) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_mpi_send(self, dst);
    op_check(ret, "mpi_send");
}

ComputingReturn TensorType::io_nccl_recv(tensor_t self, int source) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_nccl_recv(self, source);
    op_check(ret, "nccl_recv");
}

ComputingReturn TensorType::io_nccl_send(tensor_t self, int dst) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_nccl_send(self, dst);
    op_check(ret, "nccl_send");
}

ComputingReturn TensorType::io_pipe_read(tensor_t self) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_pipe_read(self);
    op_check(ret, "pipe_read");
}

ComputingReturn TensorType::io_pipe_write(tensor_t self, int n) {
    vt_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_pipe_write(self, n);
    op_check(ret, "pipe_write");
}

TensorType::~TensorType() {
    if ( impl_index() == ImplType::HOST_FLOAT ) {
        host_float_t* tensor = std::get<HOST_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::HOST_FP16 ) {
        host_fp16_t* tensor = std::get<HOST_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::HOST_INT ) {
        host_int_t* tensor = std::get<HOST_INT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::HOST_Q8 ) {
        host_q8_t* tensor = std::get<HOST_Q8>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::HOST_Q4 ) {
        host_q4_t* tensor = std::get<HOST_Q4>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::HOST_PQ ) {
        host_pq_t* tensor = std::get<HOST_PQ>(impl_);
        delete tensor;
    }
#ifdef _USING_DEVICE_CUDA_
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_FP16 ) {
        cuda_fp16_t* tensor = std::get<CUDA_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_INT ) {
        cuda_int_t* tensor = std::get<CUDA_INT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_Q8 ) {
        cuda_q8_t* tensor = std::get<CUDA_Q8>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_Q4 ) {
        cuda_q4_t* tensor = std::get<CUDA_Q4>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_PQ ) {
        cuda_pq_t* tensor = std::get<CUDA_PQ>(impl_);
        delete tensor;
    }
#endif

#ifdef _USING_DEVICE_DCU_
    if ( impl_index() == ImplType::DCU_FLOAT ) {
        dcu_float_t* tensor = std::get<DCU_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DCU_FP16 ) {
        dcu_fp16_t* tensor = std::get<DCU_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DCU_INT ) {
        dcu_int_t* tensor = std::get<DCU_INT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DCU_Q8 ) {
        dcu_q8_t* tensor = std::get<DCU_Q8>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DCU_Q4 ) {
        dcu_q4_t* tensor = std::get<DCU_Q4>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DCU_PQ ) {
        dcu_pq_t* tensor = std::get<DCU_PQ>(impl_);
        delete tensor;
    }
#endif

#ifdef _USING_DEVICE_COREX_
    if ( impl_index() == ImplType::CX_FLOAT ) {
        cx_float_t* tensor = std::get<CX_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CX_FP16 ) {
        cx_fp16_t* tensor = std::get<CX_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CX_INT ) {
        cx_int_t* tensor = std::get<CX_INT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CX_Q8 ) {
        cx_q8_t* tensor = std::get<CX_Q8>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CX_Q4 ) {
        cx_q4_t* tensor = std::get<CX_Q4>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CX_PQ ) {
        cx_pq_t* tensor = std::get<CX_PQ>(impl_);
        delete tensor;
    }
#endif

#ifdef _USING_DEVICE_ACL_
    if ( impl_index() == ImplType::ACL_FLOAT ) {
        acl_float_t* tensor = std::get<ACL_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::ACL_FP16 ) {
        acl_fp16_t* tensor = std::get<ACL_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::ACL_INT ) {
        acl_int_t* tensor = std::get<ACL_INT>(impl_);
        delete tensor;
    }
#endif

#ifdef _USING_DEVICE_DNNL_
    if ( impl_index() == ImplType::DNNL_FLOAT ) {
        dnnl_float_t* tensor = std::get<DNNL_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DNNL_FP16 ) {
        dnnl_fp16_t* tensor = std::get<DNNL_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::DNNL_INT ) {
        dnnl_int_t* tensor = std::get<DNNL_INT>(impl_);
        delete tensor;
    }
#endif

}

TransformerComputing* TensorType::impl() {
    if ( impl_index() == ImplType::HOST_FLOAT ) {
        host_float_t* tensor = std::get<HOST_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::HOST_INT ) {
        host_int_t* tensor = std::get<HOST_INT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::HOST_FP16 ) {
        host_fp16_t* tensor = std::get<HOST_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::HOST_Q8 ) {
        host_q8_t* tensor = std::get<HOST_Q8>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::HOST_Q4 ) {
        host_q4_t* tensor = std::get<HOST_Q4>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::HOST_PQ ) {
        host_pq_t* tensor = std::get<HOST_PQ>(impl_);
        return tensor;
    }

#ifdef _USING_DEVICE_CUDA_
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_INT ) {
        cuda_int_t* tensor = std::get<CUDA_INT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_FP16 ) {
        cuda_fp16_t* tensor = std::get<CUDA_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_Q8 ) {
        cuda_q8_t* tensor = std::get<CUDA_Q8>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_Q4 ) {
        cuda_q4_t* tensor = std::get<CUDA_Q4>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_PQ ) {
        cuda_pq_t* tensor = std::get<CUDA_PQ>(impl_);
        return tensor;
    }
#endif

#ifdef _USING_DEVICE_DCU_
    if ( impl_index() == ImplType::DCU_FLOAT ) {
        dcu_float_t* tensor = std::get<DCU_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DCU_INT ) {
        dcu_int_t* tensor = std::get<DCU_INT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DCU_FP16 ) {
        dcu_fp16_t* tensor = std::get<DCU_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DCU_Q8 ) {
        dcu_q8_t* tensor = std::get<DCU_Q8>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DCU_Q4 ) {
        dcu_q4_t* tensor = std::get<DCU_Q4>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DCU_PQ ) {
        dcu_pq_t* tensor = std::get<DCU_PQ>(impl_);
        return tensor;
    }
#endif

#ifdef _USING_DEVICE_COREX_
    if ( impl_index() == ImplType::CX_FLOAT ) {
        cx_float_t* tensor = std::get<CX_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CX_INT ) {
        cx_int_t* tensor = std::get<CX_INT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CX_FP16 ) {
        cx_fp16_t* tensor = std::get<CX_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CX_Q8 ) {
        cx_q8_t* tensor = std::get<CX_Q8>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CX_Q4 ) {
        cx_q4_t* tensor = std::get<CX_Q4>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CX_PQ ) {
        cx_pq_t* tensor = std::get<CX_PQ>(impl_);
        return tensor;
    }
#endif

#ifdef _USING_DEVICE_ACL_
    if ( impl_index() == ImplType::ACL_FLOAT ) {
        acl_float_t* tensor = std::get<ACL_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::ACL_FP16 ) {
        acl_fp16_t* tensor = std::get<ACL_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::ACL_INT ) {
        acl_int_t* tensor = std::get<ACL_INT>(impl_);
        return tensor;
    }
#endif

#ifdef _USING_DEVICE_DNNL_
    if ( impl_index() == ImplType::DNNL_FLOAT ) {
        dnnl_float_t* tensor = std::get<DNNL_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DNNL_FP16 ) {
        dnnl_fp16_t* tensor = std::get<DNNL_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::DNNL_INT ) {
        dnnl_int_t* tensor = std::get<DNNL_INT>(impl_);
        return tensor;
    }
#endif

    vt_panic("Can't be here!");
    return nullptr;
}

void* TensorType::device_data() {
    return device_data( impl_index() );
}

void* TensorType::device_data(size_t index) {
    if ( index == ImplType::HOST_FLOAT ) {
        host_float_t* tensor = std::get<HOST_FLOAT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::HOST_INT ) {
        host_int_t* tensor = std::get<HOST_INT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::HOST_FP16 ) {
        host_fp16_t* tensor = std::get<HOST_FP16>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::HOST_Q8 ) {
        host_q8_t* tensor = std::get<HOST_Q8>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::HOST_Q4 ) {
        host_q4_t* tensor = std::get<HOST_Q4>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::HOST_PQ ) {
        host_pq_t* tensor = std::get<HOST_PQ>(impl_);
        return tensor->data();
    }

#ifdef _USING_DEVICE_CUDA_
    if ( index == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CUDA_INT ) {
        cuda_int_t* tensor = std::get<CUDA_INT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CUDA_FP16 ) {
        cuda_fp16_t* tensor = std::get<CUDA_FP16>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CUDA_Q8 ) {
        cuda_q8_t* tensor = std::get<CUDA_Q8>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CUDA_Q4 ) {
        cuda_q4_t* tensor = std::get<CUDA_Q4>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CUDA_PQ ) {
        cuda_pq_t* tensor = std::get<CUDA_PQ>(impl_);
        return tensor->data();
    }
#endif

#ifdef _USING_DEVICE_DCU_
    if ( index == ImplType::DCU_FLOAT ) {
        dcu_float_t* tensor = std::get<DCU_FLOAT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DCU_INT ) {
        dcu_int_t* tensor = std::get<DCU_INT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DCU_FP16 ) {
        dcu_fp16_t* tensor = std::get<DCU_FP16>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DCU_Q8 ) {
        dcu_q8_t* tensor = std::get<DCU_Q8>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DCU_Q4 ) {
        dcu_q4_t* tensor = std::get<DCU_Q4>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DCU_PQ ) {
        dcu_pq_t* tensor = std::get<DCU_PQ>(impl_);
        return tensor->data();
    }
#endif

#ifdef _USING_DEVICE_COREX_
    if ( index == ImplType::CX_FLOAT ) {
        cx_float_t* tensor = std::get<CX_FLOAT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CX_INT ) {
        cx_int_t* tensor = std::get<CX_INT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CX_FP16 ) {
        cx_fp16_t* tensor = std::get<CX_FP16>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CX_Q8 ) {
        cx_q8_t* tensor = std::get<CX_Q8>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CX_Q4 ) {
        cx_q4_t* tensor = std::get<CX_Q4>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::CX_PQ ) {
        cx_pq_t* tensor = std::get<CX_PQ>(impl_);
        return tensor->data();
    }
#endif

#ifdef _USING_DEVICE_ACL_
    if ( index == ImplType::ACL_FLOAT ) {
        acl_float_t* tensor = std::get<ACL_FLOAT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::ACL_INT ) {
        acl_int_t* tensor = std::get<ACL_INT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::ACL_FP16 ) {
        acl_fp16_t* tensor = std::get<ACL_FP16>(impl_);
        return tensor->data();
    }
#endif

#ifdef _USING_DEVICE_DNNL_
    if ( index == ImplType::DNNL_FLOAT ) {
        dnnl_float_t* tensor = std::get<DNNL_FLOAT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DNNL_INT ) {
        dnnl_int_t* tensor = std::get<DNNL_INT>(impl_);
        return tensor->data();
    }
    if ( index == ImplType::DNNL_FP16 ) {
        dnnl_fp16_t* tensor = std::get<DNNL_FP16>(impl_);
        return tensor->data();
    }
#endif
    vt_panic("Can't be here!");
    return nullptr;
}

}

