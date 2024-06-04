#include "tensortype.hpp"

#include "host_tensor.hpp"
#ifdef _USING_DEVICE_CUDA_
#include "cuda_tensor.hpp"
#endif

#define CHECK_SELF() vt_assert(self.get() == this, "can't be here!")

namespace vt {

ComputingReturn TensorType::io_load(ComputingContext* ctx, tensor_t self, const char* fileName) {
    CHECK_SELF();
    auto ret = impl()->io_load(ctx, self, fileName);
    op_check(ret, "load");
}

ComputingReturn TensorType::io_save(ComputingContext* ctx, tensor_t self, const char* fileName) {
    CHECK_SELF();
    auto ret = impl()->io_save(ctx, self, fileName);
    op_check(ret, "save");
}

ComputingReturn TensorType::io_dump(ComputingContext* ctx, tensor_t self) {
    CHECK_SELF();
    std::cout << "--------------" << std::endl;
    std::cout << to_string() << std::endl;
    auto ret = impl()->io_dump(ctx, self);
    op_check(ret, "dump");
}

ComputingReturn TensorType::io_pipe_read(ComputingContext* ctx, tensor_t self) {
    CHECK_SELF();
    auto ret = impl()->io_pipe_read(ctx, self);
    op_check(ret, "pipe_read");
}

ComputingReturn TensorType::io_pipe_write(ComputingContext* ctx, tensor_t self, int n) {
    CHECK_SELF();
    auto ret = impl()->io_pipe_write(ctx, self, n);
    op_check(ret, "pipe_write");
}

std::variant<ComputingReturn, size_t> TensorType::op_sizeof(ComputingContext* ctx, tensor_t self) {
    CHECK_SELF();
    auto result = impl()->op_sizeof(ctx, self);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "sizeof");
    }
    return result;
}

std::variant<ComputingReturn, void *> TensorType::op_data(ComputingContext* ctx, tensor_t self) {
    CHECK_SELF();
    auto result = impl()->op_data(ctx, self);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "data");
    }
    return result;
}


ComputingReturn TensorType::op_zero(ComputingContext* ctx, tensor_t self) {
    CHECK_SELF();
    auto ret = impl()->op_zero(ctx, self);
    op_check(ret, "zero");
}

ComputingReturn TensorType::op_fill(ComputingContext* ctx, tensor_t self, float value) {
    CHECK_SELF();
    auto ret = impl()->op_fill(ctx, self, value);
    op_check(ret, "fill");
}

ComputingReturn TensorType::op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) {
    CHECK_SELF();
    vt_assert(shape().dim() == 3, "rotary_cache shape: [max_length, head_hiddens, 2]");
    vt_assert(shape()[2] == 2, "rotary_cache shape: [max_length, head_hiddens, 2]");
    auto ret = impl()->op_rotary_cache(ctx, self, base);
    op_check(ret, "rotary_cache");
}

ComputingReturn TensorType::op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t out) {
    CHECK_SELF();
    vt_assert(self->dtype() == DataType::I32, "mask must be int!");
    vt_assert(shape().dim() == 2, "op_causal_mask input mask shape: [batch, length]");
    vt_assert(out->shape().dim() == 4, "op_causal_mask input mask shape: [batch, 1, length, length]");
    vt_assert(out->shape()[1] == 1, "op_causal_mask input mask shape: [batch, 1, length, length]");
    auto ret = impl()->op_causal_mask(ctx, self, out);
    op_check(ret, "causal_mask");
}

ComputingReturn TensorType::op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) {
    CHECK_SELF();
    vt_assert(items() == src->items(), "copy_from must has same size");
    vt_assert(self->dtype() == src->dtype(), "copy_from must has same data type");
    auto ret = impl()->op_copy_from(ctx, self, src);
    op_check(ret, "copy_from");
}

ComputingReturn TensorType::op_copy_to(ComputingContext* ctx, tensor_t self, tensor_t dst) {
    CHECK_SELF();
    vt_assert(items() == dst->items(), "copy_to must has same size");
    vt_assert(self->dtype() == dst->dtype(), "copy_to must has same data type");
    auto ret = impl()->op_copy_to(ctx, self, dst);
    op_check(ret, "copy_to");
}

ComputingReturn TensorType::op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) {
    CHECK_SELF();
    vt_assert(items() == src->items(), "convert must has same size");
    vt_assert(self->dtype() != src->dtype(), "convert must has diff data type");
    vt_assert(self->device_name() == src->device_name(), "convert must has same device");
    auto ret = impl()->op_convert(ctx, self, src);
    op_check(ret, "convert");
}

std::variant<ComputingReturn, tensor_t> TensorType::op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    CHECK_SELF();
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
    vt_assert(offset + s.numel() <= items() , "view out of shape");
    auto result = impl()->op_view(ctx, self, offset, newShape);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view");
    }
    return result;
}

std::variant<ComputingReturn, tensor_t> TensorType::op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype_ ) {
    CHECK_SELF();
    auto result = impl()->op_view_as(ctx, self, offset, newShape_, dtype_);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view_as");
    }
    return result;
}

ComputingReturn TensorType::op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    CHECK_SELF();

    ShapeType shape(newShape_);
    if (  shape.numel() + offset > items()  ) {
        vt_panic("op_reshape out of memory size");
    }

    auto result = impl()->op_reshape(ctx, self, offset, newShape_);
    if ( result != ComputingReturn::OP_OK ) {
        return result;
    }
    *const_cast<ShapeType *>(&self->shape_) = shape;
    return ComputingReturn::OP_OK;
}

ComputingReturn TensorType::op_quantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
    CHECK_SELF();
    vt_assert(self->shape() == out->shape(), "input & output must has same size");
    vt_assert(out->is_quantized(), "output must a quantized type");
    auto ret = impl()->op_quantize(ctx, self, out);
    op_check(ret, "quantize");
}

ComputingReturn TensorType::op_dequantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
    CHECK_SELF();
    vt_assert(self->shape() == out->shape(), "input & output must has same size");
    vt_assert(self->is_quantized(), "input must a quantized type");
    auto ret = impl()->op_dequantize(ctx, self, out);
    op_check(ret, "dequantize");
}

ComputingReturn TensorType::op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t out) {
    CHECK_SELF();
    vt_assert(self->dtype() == DataType::I32, "token id must be int");
    vt_assert(table->dtype() == out->dtype(), "output and table must have same DataType" );
    //vt_assert(table->impl_index() == out->impl_index(), "table and outspace must have same device");
    vt_assert(table->shape()[1] == out->shape()[2], "table and out must have same hidden size");
    auto ret = impl()->op_embed(ctx, self, table, out);
    op_check(ret, "embed");
}

ComputingReturn TensorType::op_add(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
    CHECK_SELF();
    vt_assert(items() == c->items(), "add input and output must has same size");
    auto ret = impl()->op_add(ctx, self, b, c);
    op_check(ret, "add");
}

ComputingReturn TensorType::op_mul(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
    CHECK_SELF();
    vt_assert(items() == c->items(), "add input and output must has same size");
    auto ret = impl()->op_mul(ctx, self, b, c);
    op_check(ret, "mul");
}

ComputingReturn TensorType::op_linear(ComputingContext* ctx, tensor_t self, tensor_t w, tensor_t b, tensor_t y) {
    CHECK_SELF();
    vt_assert(shape().dim() == 3, "linear input shape: [batch, len, hidden] ");
    vt_assert(w->shape().dim() == 2, "linear weight shape: [outSize, inSize] ");
    vt_assert(w->shape()[1] == shape()[2], "linear input and weight must match" );
    auto ret = impl()->op_linear(ctx, self, w, b, y);
    op_check(ret, "linear");
}

ComputingReturn TensorType::op_layernorm(ComputingContext* ctx, tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    CHECK_SELF();
    vt_assert(mean->shape().dim() == 1, "layernorm size error!");
    vt_assert(var->shape().dim() == 1, "layernorm size error!");
    vt_assert(scale->shape().dim() == 1, "layernorm size error!");
    vt_assert(bias->shape().dim() == 1, "layernorm size error!");
    vt_assert(y->shape() == self->shape(), "layernorm size error!");
    vt_assert(self->shape()[-1] == mean->shape()[0] , "layernorm size error!");
    auto ret = impl()->op_layernorm(ctx, self, mean, var, scale, bias, y, eps);
    op_check(ret, "layernorm");
}

ComputingReturn TensorType::op_rmsnorm(ComputingContext* ctx, tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    CHECK_SELF();
    vt_assert(scale->shape().dim() == 3, "rmsnorm size error!");
    vt_assert(norm2->shape().dim() == 3, "rmsnorm size error!");
    vt_assert(self->shape().dim() == 3, "rmsnorm size error!");
    vt_assert(y->shape() == self->shape(), "rmsnorm size error!");
    vt_assert(self->shape()[-1] == scale->shape()[-1] , "rmsnorm size error!");
    auto ret = impl()->op_rmsnorm(ctx, self, scale, norm2, y, eps);
    op_check(ret, "rmsnorm");
}

ComputingReturn TensorType::op_rotary_embed(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) {
    CHECK_SELF();
    auto ret = impl()->op_rotary_embed(ctx, self, cached, pos, y);
    op_check(ret, "rotary_embed");
}

ComputingReturn TensorType::op_transpose_0213(ComputingContext* ctx, tensor_t self, tensor_t y) {
    CHECK_SELF();
    auto ret = impl()->op_transpose_0213(ctx, self, y);
    op_check(ret, "transpose_0213");
}

ComputingReturn TensorType::op_transpose_0213_rotary(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) {
    CHECK_SELF();
    auto ret = impl()->op_transpose_0213_rotary(ctx, self, cached, pos, y);
    op_check(ret, "transpose_0213_rotary");
}

ComputingReturn TensorType::op_transpose_0213_repeated(ComputingContext* ctx, tensor_t self, tensor_t y) {
    CHECK_SELF();
    auto ret = impl()->op_transpose_0213_repeated(ctx, self, y);
    op_check(ret, "transpose_0213_repeated");
}

ComputingReturn TensorType::op_qk(ComputingContext* ctx, tensor_t self, tensor_t k, tensor_t qk) {
    CHECK_SELF();
    auto ret = impl()->op_qk(ctx, self, k, qk);
    op_check(ret, "qk");
}

ComputingReturn TensorType::op_softmax(ComputingContext* ctx, tensor_t self, tensor_t out) {
    CHECK_SELF();
    auto ret = impl()->op_softmax(ctx, self, out);
    op_check(ret, "softmax");
}

ComputingReturn TensorType::op_attn(ComputingContext* ctx, tensor_t self, tensor_t v, tensor_t attn) {
    CHECK_SELF();
    auto ret = impl()->op_attn(ctx, self, v, attn);
    op_check(ret, "attn");
}

ComputingReturn TensorType::op_gelu(ComputingContext* ctx, tensor_t self, tensor_t dst) {
    CHECK_SELF();
    auto ret = impl()->op_gelu(ctx, self, dst);
    op_check(ret, "gelu");
}

ComputingReturn TensorType::op_silu_product(ComputingContext* ctx, tensor_t self, tensor_t up, tensor_t dst) {
    CHECK_SELF();
    auto ret = impl()->op_silu_product(ctx, self, up, dst);
    op_check(ret, "silu_product");
}

std::variant<ComputingReturn, int> TensorType::op_all_logits(ComputingContext* ctx, tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output) {
    CHECK_SELF();
    auto ret = impl()->op_all_logits(ctx, self, mask, lm_head, output);
    if ( ret.index() == 0) {
        ComputingReturn r = std::get<0>(ret);
        op_check(r, "op_last_logits");
    }
    return ret;
}

std::variant<ComputingReturn, tensor_t> TensorType::op_sampling_top1(ComputingContext* ctx, tensor_t self) {
    CHECK_SELF();
    auto ret = impl()->op_sampling_top1(ctx, self);
    if ( ret.index() == 0) {
        ComputingReturn r = std::get<0>(ret);
        op_check(r, "op_sampling_top1");
    }
    return ret;
}

std::variant<ComputingReturn, tensor_t> TensorType::op_sampling_top3(ComputingContext* ctx, tensor_t self, float temp) {
    CHECK_SELF();
    auto ret = impl()->op_sampling_top3(ctx, self, temp);
    if ( ret.index() == 0) {
        ComputingReturn r = std::get<0>(ret);
        op_check(r, "op_sampling_top3");
    }
    return ret;
}

ComputingReturn TensorType::op_conv2d(ComputingContext* ctx, tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) {
    CHECK_SELF();
    // checking shape
    vt_assert(self->shape().dim() == 4, "conv2d accept NCHW only!");
    vt_assert(weight->shape().dim() == 4, "conv2d weight OIHW only!");
    vt_assert(self->shape().dims()[1] == weight->shape().dims()[1], "conv2d input channel must be matched with weight");
    auto ret = impl()->op_conv2d(ctx, self, weight, bias, dst, stride, padding);
    op_check(ret, "op_conv2d");
}

//*************************************************************************
#define _DELETE_(TT) \
    if ( impl_.index() == ImplType::TT ) { \
        delete std::get<ImplType::TT>(impl_); \
        return;                               \
    }

#define DELETE_DEVICE(TT)\
    _DELETE_(TT ## _F32)\
    _DELETE_(TT ## _I32)\
    _DELETE_(TT ## _F16)\
    _DELETE_(TT ## _BF16)\
    _DELETE_(TT ## _Q8)\
    _DELETE_(TT ## _Q4)\
    _DELETE_(TT ## _PQ)

TensorType::~TensorType() {
    DELETE_DEVICE(HOST)

#ifdef _USING_DEVICE_CUDA_
    DELETE_DEVICE(CUDA)
#endif
}

#define LIST_DEVICE_CONSTRUCTOR_IMPL(T) \
TensorType::TensorType(T ## _f32_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::F32) {\
    impl_ = tensor;\
}\
TensorType::TensorType(T ## _i32_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::I32) {\
    impl_ = tensor;\
}\
TensorType::TensorType(T ## _f16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::F16) {\
    impl_ = tensor;\
}\
TensorType::TensorType(T ## _bf16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::BF16) {\
    impl_ = tensor;\
}\
TensorType::TensorType(T ## _q8_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q8) {\
    impl_ = tensor;\
}\
TensorType::TensorType(T ## _q4_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q4) {\
    impl_ = tensor;\
}\
TensorType::TensorType(T ## _pq_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::PQ) {\
    impl_ = tensor;\
}

LIST_DEVICE_CONSTRUCTOR_IMPL(host)
#ifdef _USING_DEVICE_CUDA_
LIST_DEVICE_CONSTRUCTOR_IMPL(cuda)
#endif


}
