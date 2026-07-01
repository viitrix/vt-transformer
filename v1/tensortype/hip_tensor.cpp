#include "common.hpp"
#include "context.hpp"
#include "hip_tensor.hpp"
#include "hip_kernels.hpp"

namespace vt {
using device_fp16_t = __half;

void copy_to_local(std::vector<unsigned char>& dst, void* src, size_t length, hipStream_t stream) {
    dst.clear();
    dst.resize( length);
    HIP_CHECK(hipMemcpyAsync(dst.data(), src, length, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

template<DataType _DT_>
HIPTensor<_DT_>::~HIPTensor() {
    if (mem_ != nullptr && owner_) {
        HIP_CHECK(hipFree(mem_));
    }
}

template<DataType _DT_>
HIPTensor<_DT_>::HIPTensor(const ShapeType& shape) : size_(0), owner_(true) {
    size_t asize = 0;
    size_t number = shape.numel();
    if ( _DT_ == DataType::F32 ) {
        asize = sizeof(float) * number;
    } else if ( _DT_ == DataType::I32 ) {
        asize = sizeof(int) * number;
    } else if ( _DT_ == DataType::F16 ) {
        asize = sizeof(unsigned short) * number;
    } else if ( _DT_ == DataType::BF16 ) {
        asize = sizeof(unsigned short) * number;
    } else {
        vt_fatal_error();
    }

    *const_cast<size_t *>(&size_) = asize;
    HIP_CHECK(hipMalloc(&mem_, asize));
}

template<DataType _DT_>
HIPTensor<_DT_>::HIPTensor(const ShapeType& shape, void* vdata) : size_(0), owner_(false), mem_(vdata) {
    size_t asize = 0;
    size_t number = shape.numel();
    if ( _DT_ == DataType::F32 ) {
        asize = sizeof(float) * number;
    } else if ( _DT_ == DataType::I32 ) {
        asize = sizeof(int) * number;
    } else if ( _DT_ == DataType::F16 ) {
        asize = sizeof(unsigned short) * number;
    } else if ( _DT_ == DataType::BF16 ) {
        asize = sizeof(unsigned short) * number;
    } else {
        vt_fatal_error();
    }
    *const_cast<size_t *>(&size_) = asize;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::io_load(ComputingContext* ctx, tensor_t self, const char* fileName) {
    std::vector<unsigned char> src;
    read_data(fileName, src);

    vt_assert(src.size() == size_, "loaded data must has same size");

    auto stream = ctx->hip_stream;
    HIP_CHECK(hipMemcpyAsync(data(), (void *)src.data(), size_, hipMemcpyHostToDevice, stream));
    return OP_OK;
}

template <DataType _DT_>
ComputingReturn HIPTensor<_DT_>::io_dump(ComputingContext* ctx, tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    std::vector<unsigned char> localData;
    if ( _DT_ == DataType::F32 ) {
        copy_to_local(localData, data(), first8 * sizeof(float), ctx->hip_stream);
        float* d = (float *)localData.data();
        SIMPLE_DUMP(d);

        d = (float *)data() + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(float), ctx->hip_stream);
        d = (float *)localData.data();
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::I32 ) {
        copy_to_local(localData, data(), first8 * sizeof(int), ctx->hip_stream);
        int* d = (int *)localData.data();
        SIMPLE_DUMP(d);
        d = (int *)data() + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(int), ctx->hip_stream);
        d = (int *)localData.data();
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        copy_to_local(localData, data(), first8 * sizeof(local_fp16_t), ctx->hip_stream);
        local_fp16_t* d = (local_fp16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        d = (local_fp16_t *)data() + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(local_fp16_t), ctx->hip_stream);
        d = (local_fp16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        return OP_OK;
    }
    if ( _DT_ == DataType::BF16 ) {
        copy_to_local(localData, mem_, first8 * sizeof(local_bf16_t), ctx->hip_stream);
        local_bf16_t* d = (local_bf16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, bf16_to_fp32);
        d = (local_bf16_t *)mem_ + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(local_bf16_t), ctx->hip_stream);
        d = (local_bf16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, bf16_to_fp32);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DT_>
std::variant<ComputingReturn, size_t> HIPTensor<_DT_>::op_sizeof(ComputingContext* ctx, tensor_t self) {
    return size_;
}

template <DataType _DT_>
std::variant<ComputingReturn, void *> HIPTensor<_DT_>::op_data(ComputingContext* ctx, tensor_t self) {
    return mem_;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_zero(ComputingContext* ctx, tensor_t self) {
    if ( (_DT_ == DataType::F32) || (_DT_ == DataType::I32) || (_DT_ == DataType::F16) || (_DT_ == DataType::BF16) ) {
        HIP_CHECK( hipMemset(data(), 0, size_) );
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_fill(ComputingContext* ctx, tensor_t self, float value) {
    auto stream = ctx->hip_stream;
    if ( (_DT_ == DataType::F32)) {
        float* target = (float *)data();
        hip::kr_fill<float>(target, value, self->items(), stream);
        return OP_OK;
    }
    if ( (_DT_ == DataType::I32)) {
        int* target = (int *)data();
        hip::kr_fill<int>(target, value, self->items(), stream);
        return OP_OK;
    }
    if ( (_DT_ == DataType::F16)) {
        device_fp16_t* target = (device_fp16_t *)data();
        hip::kr_fill<device_fp16_t>(target, value, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) {
    if ( (_DT_ == DataType::F32) ) {
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        auto stream = ctx->hip_stream;
        HIP_CHECK(hipMemcpyAsync( data(), cos_sin.data(), self->items() * sizeof(float), hipMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];
    auto stream = ctx->hip_stream;

    int* mask  = (int *)data();
    if ( out->dtype() == DataType::F32 ) {
        float* dst = (float *)out->hip_f32()->data();
        hip::kr_causal_mask<float>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }
    if ( out->dtype() == DataType::F16 ) {
        device_fp16_t* dst = (device_fp16_t *)out->hip_f16()->data();
        hip::kr_causal_mask<device_fp16_t>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) {
    if ( src->is_host() ) {
        auto stream = ctx->hip_stream;
        void* from = std::get<1>( src->op_data(ctx, src) );
        HIP_CHECK(hipMemcpyAsync(data(), from, size_, hipMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( src->is_hip() ) {
        auto stream = ctx->hip_stream;
        void* from = std::get<1>( src->op_data(ctx, src) );
        HIP_CHECK(hipMemcpyAsync(data(), from, size_, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}
template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_copy_to(ComputingContext* ctx, tensor_t self,  tensor_t dst) {
    if ( dst->is_host() ) {
        auto stream = ctx->hip_stream;
        void* to = std::get<1>( dst->op_data(ctx, dst) );
        HIP_CHECK(hipMemcpyAsync(to, data(), size_, hipMemcpyDeviceToHost, stream));
        return OP_OK;
    }
    if ( dst->is_hip() ) {
        auto stream = ctx->hip_stream;
        void* to = std::get<1>( dst->op_data(ctx, dst) );
        HIP_CHECK(hipMemcpyAsync(to, data(), size_, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) {
    if ( _DT_ == DataType::F16 && src->dtype() == DataType::F32) {
        auto stream = ctx->hip_stream;
        device_fp16_t* to = (device_fp16_t*)data();
        float* from = (float  *) std::get<1>(src->op_data(ctx, src));
        hip::kr_convert<float, device_fp16_t>(from, to, self->items(), stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F32 && src->dtype() == DataType::F16) {
        auto stream = ctx->hip_stream;
        float* to = (float *)data();
        device_fp16_t* from = (device_fp16_t  *) std::get<1>(src->op_data(ctx, src));
        hip::kr_convert<device_fp16_t, float>(from, to, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> HIPTensor<_DT_>::op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( _DT_ == DataType::F32 ) {
        void *newData = (char *)data() + offset * sizeof(float);
        auto* newHIPTensor = new HIPTensor<DataType::F32>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    if ( _DT_ == DataType::I32 ) {
        void *newData = (char *)data() + offset * sizeof(int);
        auto* newHIPTensor = new HIPTensor<DataType::I32>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    if ( _DT_ == DataType::F16 ) {
        void *newData = (char *)data() + offset * sizeof(local_fp16_t);
        auto* newHIPTensor = new HIPTensor<DataType::F16>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    if ( _DT_ == DataType::BF16 ) {
        void *newData = (char *)data() + offset * sizeof(local_bf16_t);
        auto* newHIPTensor = new HIPTensor<DataType::BF16>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> HIPTensor<_DT_>::op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
    DataType DT = DataType_from(dtype);

    ShapeType newShape(newShape_);

    void *newData = nullptr;
    if ( _DT_ == DataType::F32 ) {
        newData = (char *)data() + offset * sizeof(float);
    } else if ( _DT_ == DataType::I32 ) {
        newData = (char *)data() + offset * sizeof(int);
    } else if ( _DT_ == DataType::F16 ) {
        newData = (char *)data() + offset * sizeof(local_fp16_t);
    } else if ( _DT_ == DataType::BF16 ) {
        newData = (char *)data() + offset * sizeof(local_bf16_t);
    } else {
        return OP_TODO_ERROR;
    }

    if ( DT == DataType::F32 ) {
        auto* newHIPTensor = new HIPTensor<DataType::F32>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    if ( DT == DataType::I32 ) {
        auto* newHIPTensor = new HIPTensor<DataType::I32>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    if ( DT == DataType::F16 ) {
        auto* newHIPTensor = new HIPTensor<DataType::F16>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    if ( DT == DataType::BF16 ) {
        auto* newHIPTensor = new HIPTensor<DataType::BF16>(newShape, newData);
        return std::make_shared<TensorType>(newHIPTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( owner_ == true ) {
        return OP_INPUT_ERROR;
    }
    if ( newShape.numel() + offset > self->items()  ) {
        return OP_INPUT_ERROR;
    }

    if ( _DT_ == DataType::F32 ) {
        mem_  = (char *)data() + offset * sizeof(float);
        return OP_OK;
    }
    if ( _DT_ == DataType::I32 ) {
        mem_  = (char *)data() + offset * sizeof(int);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        mem_  = (char *)data() + offset * sizeof(local_fp16_t);
        return OP_OK;
    }
    if ( _DT_ == DataType::BF16 ) {
        mem_  = (char *)data() + offset * sizeof(local_bf16_t);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_quantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_dequantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t output) {
    size_t batch = self->shape()[0];
    size_t len = self->shape()[1];
    size_t hidden = table->shape()[1];

    auto stream = ctx->hip_stream;
    int* text = (int *)data();

    if ( table->dtype() == DataType::F32 ) {
        float* from = (float *)table->hip_f32()->data();
        float* out = (float *)output->hip_f32()->data();
        hip::kr_embed<float>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    if ( table->dtype() == DataType::F16 ) {
        device_fp16_t* from = (device_fp16_t *)table->hip_f16()->data();
        device_fp16_t* out = (device_fp16_t *)output->hip_f16()->data();
        hip::kr_embed<device_fp16_t>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_add(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
    auto stream = ctx->hip_stream;
    if ( self->items() == b->items() ) {
        if ( _DT_ == DataType::F32 && b->dtype() == DataType::F32 ) {
            float* A = (float *)data();
            float* B = (float *)b->hip_f32()->data();
            float* C = (float *)c->hip_f32()->data();

            hip::kr_add<float, float>(A, B, C, self->items(), stream);
            return OP_OK;
        }
        if ( _DT_ == DataType::F16 && b->dtype() == DataType::F16 ) {
            device_fp16_t* A = (device_fp16_t *)data();
            device_fp16_t* B = (device_fp16_t *)b->hip_f16()->data();
            device_fp16_t* C = (device_fp16_t *)c->hip_f16()->data();

            hip::kr_add<device_fp16_t, device_fp16_t>(A, B, C, self->items(), stream);

            return OP_OK;
        }
        if ( _DT_ == DataType::F16 && b->dtype() == DataType::F32 ) {
            device_fp16_t* A = (device_fp16_t *)data();
            float* B = (float *)b->hip_f32()->data();
            device_fp16_t* C = (device_fp16_t *)c->hip_f16()->data();

            hip::kr_add<device_fp16_t, float>(A, B, C, self->items(), stream);
            return OP_OK;
        }
    } else {
        auto ashape = self->shape().vec();
        auto bshape = b->shape().vec();

        if ( ashape.size() == 4 &&
             bshape.size() == 4 &&
             ashape[0] == bshape[0] &&
             ashape[2] == bshape[2] &&
             ashape[3] == bshape[3] &&
             bshape[1] == 1 ) {

            int length = ashape[0];
            int inter = ashape[1];
            int feature = ashape[2] * ashape[3];

            if ( _DT_ == DataType::F16 &&  b->dtype() == DataType::F16 ) {
                device_fp16_t* A = (device_fp16_t *)data();
                device_fp16_t* B = (device_fp16_t *)b->hip_f16()->data();
                device_fp16_t* C = (device_fp16_t *)c->hip_f16()->data();

                hip::kr_add_broadcast<device_fp16_t>(A, B, C, length, inter, feature, stream);
                return OP_OK;
            }
            if ( _DT_ == DataType::F32 && b->dtype() == DataType::F32 ) {
                float* A = (float *)data();
                float* B = (float *)b->hip_f32()->data();
                float* C = (float *)c->hip_f32()->data();

                hip::kr_add_broadcast<float>(A, B, C, length, inter, feature, stream);
                return OP_OK;
            }
        }
    }
    return OP_TODO_ERROR;
}


template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_mul(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
    vt_assert( self->items() == b->items() , "HIP's add don't support brodcast");
    auto stream = ctx->hip_stream;
    if ( _DT_ == DataType::F32 && b->dtype() == DataType::F32 ) {
        float* A = (float *)data();
        float* B = (float *)b->hip_f32()->data();
        float* C = (float *)c->hip_f32()->data();

        hip::kr_mul<float, float>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 && b->dtype() == DataType::F16 ) {
        device_fp16_t* A = (device_fp16_t *)data();
        device_fp16_t* B = (device_fp16_t *)b->hip_f16()->data();
        device_fp16_t* C = (device_fp16_t *)c->hip_f16()->data();

        hip::kr_mul<device_fp16_t, device_fp16_t>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 && b->dtype() == DataType::F32 ) {
        device_fp16_t* A = (device_fp16_t *)data();
        float* B = (float *)b->hip_f32()->data();
        device_fp16_t* C = (device_fp16_t *)c->hip_f16()->data();

        hip::kr_mul<device_fp16_t, float>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;

}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_linear(ComputingContext* ctx, tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    auto stream = ctx->hip_stream;
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w_->shape()[0];

    float alpha = 1.0;
    float beta = 0.0;

    if ( _DT_ == DataType::F32 && w_->dtype() == DataType::F32 ) {
        void* A = w_->hip_f32()->data();
        void* B = data();
        void* C = y_->hip_f32()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        HIPBLAS_CHECK( hipblasGemmEx(ctx->hipblas_handle,
                       HIPBLAS_OP_T, HIPBLAS_OP_N,
                       m, n, k,
                       &alpha, A, HIPBLAS_R_32F, k,
                       B, HIPBLAS_R_32F, k, &beta,
                       C, HIPBLAS_R_32F, m,
                       HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        if ( b_ != nullptr ) {
            hip::kr_add_bias<float>((float *)C, (float *)b_->hip_f32()->data(), (float *)C,  n, m, stream);
        }

        return OP_OK;
    }

    if ( _DT_ == DataType::F16 && w_->dtype() == DataType::F16 ) {
        void* A = w_->hip_f16()->data();
        void* B = data();
        void* C = y_->hip_f16()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        HIPBLAS_CHECK( hipblasGemmEx(ctx->hipblas_handle,
                       HIPBLAS_OP_T, HIPBLAS_OP_N,
                       m, n, k,
                       &alpha, A, HIPBLAS_R_16F, k,
                       B, HIPBLAS_R_16F, k, &beta,
                       C, HIPBLAS_R_16F, m,
                       HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        if ( b_ != nullptr) {
            hip::kr_add_bias<device_fp16_t>((device_fp16_t *)C, (device_fp16_t *)b_->hip_f16()->data(), (device_fp16_t *)C, n, m, stream);
        }

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_layernorm(ComputingContext* ctx, tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    auto stream = ctx->hip_stream;
    if ( _DT_ == DataType::F32 ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->hip_f32();
        auto v = var->hip_f32();
        auto s = scale->hip_f32();
        auto b = bias->hip_f32();
        auto out = y->hip_f32();

        hip::kr_layer_norm<float>((float *)out->data(), (float *)v->data(), (float *)m->data(),
                                 (float *)x->data(), (float *)s->data(), (float *)b->data(), batch, hidden, eps, stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->hip_f16();
        auto v = var->hip_f16();
        auto s = scale->hip_f16();
        auto b = bias->hip_f16();
        auto out = y->hip_f16();

        hip::kr_layer_norm<__half>((__half *)out->data(), (__half *)v->data(), (__half *)m->data(),
                                   (__half *)x->data(), (__half *)s->data(), (__half *)b->data(), batch, hidden, eps, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_rmsnorm(ComputingContext* ctx, tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t hidden = self->shape()[2];

    void* norm2_ = std::get<1>(norm2->op_data(ctx, norm2));
    void* feature = data();
    void* w = std::get<1>(scale->op_data(ctx, scale));
    void* out = std::get<1>(y->op_data(ctx, y));

    auto stream = ctx->hip_stream;
    if ( _DT_ == DataType::F32) {
        hip::kr_rmsnorm<float>((float *)feature, (float *)w, (float *)out, (float *)norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }

    if ( _DT_ == DataType::F16 ) {
        hip::kr_rmsnorm<device_fp16_t>((device_fp16_t *)feature, (device_fp16_t *)w, (device_fp16_t *)out, (device_fp16_t *)norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_rotary_embed(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos_, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

    vt_assert(hidden == cached->shape()[1], "heads number must be same with cache");

    int* pos = (int*) pos_->hip_i32()->data();
    float* cos_sin = (float *)cached->hip_f32()->data();
    auto stream = ctx->hip_stream;

    if ( _DT_ == DataType::F32 ) {
        float* in = (float *)data();
        float* out = (float *)y->hip_f32()->data();
        hip::kr_rotary_embed<float>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        device_fp16_t* in = (device_fp16_t *)data();
        device_fp16_t* out = (device_fp16_t *)y->hip_f16()->data();
        hip::kr_rotary_embed<device_fp16_t>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_transpose_0213(ComputingContext* ctx, tensor_t self, tensor_t y) {
    auto stream = ctx->hip_stream;

    int sz0 = self->shape()[0];
    int sz1 = self->shape()[1];
    int sz2 = self->shape()[2];
    int sz3 = self->shape()[3];

    void* x = data();
    void* out = std::get<1>( y->op_data(ctx, y) );
    if ( _DT_ == DataType::F32 ) {
        hip::kr_transpose_0213<float>((float *)x, (float *)out, sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        hip::kr_transpose_0213<device_fp16_t>((device_fp16_t *)x, (device_fp16_t *)out, sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_transpose_0213_repeated(ComputingContext* ctx, tensor_t self, tensor_t y) {
    auto stream = ctx->hip_stream;

    int sz0 = self->shape()[0];
    int sz1 = self->shape()[1];
    int sz2_f = self->shape()[2];
    int sz2_t = y->shape()[1];
    int sz3 = self->shape()[3];

    void* x = data();
    void* out = std::get<1>( y->op_data(ctx, y) );
    if ( _DT_ == DataType::F32 ) {
        hip::kr_transpose_0213_repeated<float>((float *)x, (float *)out, sz0, sz1, sz2_f, sz2_t, sz3, stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        hip::kr_transpose_0213_repeated<device_fp16_t>((device_fp16_t *)x, (device_fp16_t *)out, sz0, sz1, sz2_f, sz2_t, sz3, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_qk(ComputingContext* ctx, tensor_t self, tensor_t k_, tensor_t qk_) {
    auto shape_ = self->shape().vec();

    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int hhidden = shape_[3];
    int ftokens = k_->shape()[2];

    int m = ftokens;
    int n = ntokens;
    int k = hhidden;

    float alpha = 1.0 / sqrt(hhidden);
    float beta = 0.0;

    int HnT = hhidden * ntokens ;
    int HfT = hhidden * ftokens ;
    int TT = ftokens * ntokens;

    if ( _DT_ == DataType::F32 ) {
        void *A_ = k_->hip_f32()->data();
        void *B_ = data();
        void *C_ = qk_->hip_f32()->data();

        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ctx->hipblas_handle,
                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_32F, k, HnT,
                        B_, HIPBLAS_R_32F, k, HfT,
                        &beta, C_, HIPBLAS_R_32F, m, TT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( _DT_ == DataType::F16 && qk_->dtype() == DataType::F32 ) {
        std::vector<void*> As;
        std::vector<void*> Bs;
        std::vector<void*> Cs;
        for (int i = 0; i < batch * heads; i++) {
            device_fp16_t* B = (device_fp16_t *)data() + i * HnT;
            device_fp16_t* A = (device_fp16_t *)(k_->hip_f16()->data()) + i * HfT;
            float* C = (float *)(qk_->hip_f32()->data()) + i * TT;
            As.push_back(A);
            Bs.push_back(B);
            Cs.push_back(C);
        }

        size_t pointers_size = As.size() * sizeof(void *);
        void *A_ = ctx->hip_workspace;
        void *B_ = (char *)A_ + pointers_size;
        void *C_ = (char *)B_ + pointers_size;

        HIP_CHECK( hipMemcpyAsync(A_, As.data(), pointers_size, hipMemcpyHostToDevice));
        HIP_CHECK( hipMemcpyAsync(B_, Bs.data(), pointers_size, hipMemcpyHostToDevice));
        HIP_CHECK( hipMemcpyAsync(C_, Cs.data(), pointers_size, hipMemcpyHostToDevice));

        HIPBLAS_CHECK( hipblasGemmBatchedEx(
                        ctx->hipblas_handle,
                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, (const void **)A_, HIPBLAS_R_16F, k,
                        (const void **)B_, HIPBLAS_R_16F, k,
                        &beta,  (void **)C_, HIPBLAS_R_32F, m,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );
        return OP_OK;
    }

    if ( _DT_ == DataType::F16 && qk_->dtype() == DataType::F16 ) {
        std::vector<void*> As;
        std::vector<void*> Bs;
        std::vector<void*> Cs;
        for (int i = 0; i < batch * heads; i++) {
            device_fp16_t* B = (device_fp16_t *)data() + i * HnT;
            device_fp16_t* A = (device_fp16_t *)(k_->hip_f16()->data()) + i * HfT;
            device_fp16_t* C = (device_fp16_t *)(qk_->hip_f16()->data()) + i * TT;
            As.push_back(A);
            Bs.push_back(B);
            Cs.push_back(C);
        }

        size_t pointers_size = As.size() * sizeof(void *);
        void *A_ = ctx->hip_workspace;
        void *B_ = (char *)A_ + pointers_size;
        void *C_ = (char *)B_ + pointers_size;

        HIP_CHECK( hipMemcpyAsync(A_, As.data(), pointers_size, hipMemcpyHostToDevice));
        HIP_CHECK( hipMemcpyAsync(B_, Bs.data(), pointers_size, hipMemcpyHostToDevice));
        HIP_CHECK( hipMemcpyAsync(C_, Cs.data(), pointers_size, hipMemcpyHostToDevice));

        HIPBLAS_CHECK( hipblasGemmBatchedEx(
                        ctx->hipblas_handle,
                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, (const void **)A_, HIPBLAS_R_16F, k,
                        (const void **)B_, HIPBLAS_R_16F, k,
                        &beta,  (void **)C_, HIPBLAS_R_16F, m,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );
        return OP_OK;
    }
    return OP_TODO_ERROR;
}
template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_softmax(ComputingContext* ctx, tensor_t self, tensor_t out) {
    auto stream = ctx->hip_stream;
    size_t hidden = self->shape().vec().back();
    size_t length = self->items() / hidden;

    if ( _DT_ == DataType::F32 ) {
        hip::kr_softmax<float>((float *)data(), (float *)out->hip_f32()->data(), length, hidden, stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        hip::kr_softmax<device_fp16_t>((device_fp16_t *)data(), (device_fp16_t *)out->hip_f16()->data(), length, hidden, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_attn(ComputingContext* ctx, tensor_t self, tensor_t value_, tensor_t out_) {
    float alpha = 1.0;
    float beta = 0.0;

    auto shape_ = self->shape().vec();
    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int ftokens = shape_[3];
    int hhidden = value_->shape()[3];

    int m = hhidden;
    int n = ntokens;
    int k = ftokens;

    int HfT = hhidden * ftokens;
    int HnT = hhidden * ntokens;
    int TT = ftokens * ntokens;

    if ( value_->dtype() == DataType::F32 && _DT_ == DataType::F32 ) {
        void *A_ = value_->hip_f32()->data();
        void *B_ = data();
        void *C_ = out_->hip_f32()->data();
        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ctx->hipblas_handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_32F, m, HfT,
                        B_, HIPBLAS_R_32F, k, TT,
                        &beta, C_, HIPBLAS_R_32F, m, HnT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( value_->dtype() == DataType::F16 && _DT_ == DataType::F32 ) {
        void *A_ = value_->hip_f16()->data();
        void *B_ = data();
        void *C_ = out_->hip_f16()->data();
        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ctx->hipblas_handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_16F, m, HfT,
                        B_, HIPBLAS_R_32F, k, TT,
                        &beta, C_, HIPBLAS_R_16F, m, HnT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( value_->dtype() == DataType::F16 && _DT_ == DataType::F16  ) {
        std::vector<void*> As;
        std::vector<void*> Bs;
        std::vector<void*> Cs;
        for (int i = 0; i < batch*heads; i++) {
            void *A = (device_fp16_t*)value_->hip_f16()->data() + i * HfT;
            void *B = (device_fp16_t*)data() + i * TT;
            void *C = (device_fp16_t*)out_->hip_f16()->data() + i * HnT;
            As.push_back(A);
            Bs.push_back(B);
            Cs.push_back(C);
        }
        size_t pointers_size = As.size() * sizeof(void *);
        void *A_ = ctx->hip_workspace;
        void *B_ = (char *)A_ + pointers_size;
        void *C_ = (char *)B_ + pointers_size;

        HIP_CHECK( hipMemcpyAsync(A_, As.data(), pointers_size, hipMemcpyHostToDevice));
        HIP_CHECK( hipMemcpyAsync(B_, Bs.data(), pointers_size, hipMemcpyHostToDevice));
        HIP_CHECK( hipMemcpyAsync(C_, Cs.data(), pointers_size, hipMemcpyHostToDevice));

        HIPBLAS_CHECK( hipblasGemmBatchedEx(
                        ctx->hipblas_handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, (const void **)A_, HIPBLAS_R_16F, m,
                        (const void**)B_, HIPBLAS_R_16F, k,
                        &beta, (void **)C_, HIPBLAS_R_16F, m,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );
        return OP_OK;
    }
    return OP_TODO_ERROR;

}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_gelu(ComputingContext* ctx, tensor_t self, tensor_t out) {
    auto stream = ctx->hip_stream;
    void* src = data();
    void* dst = std::get<1>(out->op_data(ctx, out) );

    if ( _DT_ == DataType::F32 ) {
        hip::kr_gelu<float>((float *)src, (float *)dst, self->items(), stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        hip::kr_gelu<device_fp16_t>((device_fp16_t *)src, (device_fp16_t *)dst, self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_silu_product(ComputingContext* ctx, tensor_t self, tensor_t in, tensor_t out) {
    auto stream = ctx->hip_stream;
    if ( _DT_ == DataType::F32 ) {
        float* src = (float *)data();
        float* in_ = (float *)in->hip_f32()->data();
        float* dst = (float *)out->hip_f32()->data();

        hip::kr_silu_product(src, in_, dst, self->items(), stream);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        auto* src = (device_fp16_t *)data();
        auto* in_ = (device_fp16_t *)in->hip_f16()->data();
        auto* dst = (device_fp16_t *)out->hip_f16()->data();

        hip::kr_silu_product(src, in_, dst, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, int> HIPTensor<_DT_>::op_all_logits(ComputingContext* ctx, tensor_t self, tensor_t mask_, tensor_t lm_head, tensor_t output ) {
    int batch = self->shape()[0];
    int new_tokens = self->shape()[1];
    int hidden_size = self->shape()[2];
    int full_tokens = mask_->shape()[1];

    int vocab_size = lm_head->shape()[0];

    int m = vocab_size;
    int n = 1;
    int k = hidden_size;

    float alpha = 1.0;
    float beta = 0.0;

    int* mask = (int *)std::get<1>(mask_->op_data(ctx, mask_));
    int pred = 0;
    for (int b = 0;  b < batch; b++) {
        int* mk = &mask[b * full_tokens];
        for ( int i = 0; i < new_tokens ; i++) {
            int ii = full_tokens - new_tokens + i;
            if ( mk[ii] != 2 ) {
                continue;
            }
            int target = i;

            if ( _DT_ == DataType::F32 ) {
                float* dst = (float *)output->hip_f32()->data() + pred * vocab_size;
                float* x = (float *)data() + b * new_tokens * hidden_size + target * hidden_size;

                float* A = (float *)lm_head->hip_f32()->data();
                float* B = x;
                float* C = dst;

                HIPBLAS_CHECK( hipblasGemmEx(ctx->hipblas_handle,
                    HIPBLAS_OP_T, HIPBLAS_OP_N,
                    m, n, k,
                    &alpha, A, HIPBLAS_R_32F, k,
                    B, HIPBLAS_R_32F, k, &beta,
                    C, HIPBLAS_R_32F, m,
                    HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));

            } else if ( _DT_ == DataType::F16 ) {
                auto* dst = (device_fp16_t *)output->hip_f16()->data() + pred * vocab_size;
                auto* x = (device_fp16_t *)data() + b * new_tokens * hidden_size + target * hidden_size;

                auto* A = (device_fp16_t *)lm_head->hip_f16()->data();
                auto* B = x;
                auto* C = dst;

                HIPBLAS_CHECK( hipblasGemmEx(ctx->hipblas_handle,
                    HIPBLAS_OP_T, HIPBLAS_OP_N,
                    m, n, k,
                    &alpha, A, HIPBLAS_R_16F, k,
                    B, HIPBLAS_R_16F, k, &beta,
                    C, HIPBLAS_R_16F, m,
                    HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));

            } else {
                return OP_TODO_ERROR;
            }

            pred ++;
        }
     }

    return pred;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> HIPTensor<_DT_>::op_sampling_top1(ComputingContext* ctx, tensor_t self) {
    if ( _DT_ != DataType::F32 && _DT_ != DataType::F16 ) {
        return OP_INPUT_ERROR;
    }

    size_t batch = self->shape()[0];
    size_t vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_i32( ret_shape );

    auto stream = ctx->hip_stream;
    int* out = (int *)ctx->hip_workspace;

    if ( _DT_ == DataType::F16 ) {
        device_fp16_t* logits = (device_fp16_t *) data();
        hip::kr_sampling_top1<device_fp16_t>(logits, out, batch, vocab_size, stream);
    } else {
        float* logits = (float *) data();
        hip::kr_sampling_top1<float>(logits, out, batch, vocab_size, stream);
    }

    HIP_CHECK(hipMemcpyAsync(std::get<1>(ret->op_data(ctx, ret)), out, batch * sizeof(int), hipMemcpyDeviceToHost, stream));
    return ret;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> HIPTensor<_DT_>::op_sampling_top3(ComputingContext* ctx, tensor_t self, float temp) {
    if ( _DT_ != DataType::F32 && _DT_ != DataType::F16 ) {
        return OP_INPUT_ERROR;
    }
    size_t batch = self->shape()[0];
    size_t vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_i32( ret_shape );

    auto stream = ctx->hip_stream;
    int* out = (int *)ctx->hip_workspace;
    std::uniform_real_distribution<> dist(0.0, 1.0);
    float randx = dist( *ctx->rng );

    if ( _DT_ == DataType::F16 ) {
        device_fp16_t* logits = (device_fp16_t *) data();
        hip::kr_sampling_top3<device_fp16_t>(logits, out, batch, vocab_size, temp, randx, stream);
    } else {
        float* logits = (float *) data();
        hip::kr_sampling_top3<float>(logits, out, batch, vocab_size, temp, randx, stream);
    }

    HIP_CHECK(hipMemcpyAsync(std::get<1>(ret->op_data(ctx, ret)), out, batch * sizeof(int), hipMemcpyDeviceToHost, stream));
    return ret;
}

template<DataType _DT_>
ComputingReturn HIPTensor<_DT_>::op_conv2d(ComputingContext* ctx, tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) {
    return OP_TODO_ERROR;
}

tensor_t create_hip_f32(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::F32>* tensor = new HIPTensor<DataType::F32>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_hip_i32(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::I32>* tensor = new HIPTensor<DataType::I32>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_hip_f16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::F16>* tensor = new HIPTensor<DataType::F16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_hip_bf16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::BF16>* tensor = new HIPTensor<DataType::BF16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_hip_q8(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::Q8>* tensor = new HIPTensor<DataType::Q8>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_hip_q4(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::Q4>* tensor = new HIPTensor<DataType::Q4>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_hip_pq(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HIPTensor<DataType::PQ>* tensor = new HIPTensor<DataType::PQ>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}



}
