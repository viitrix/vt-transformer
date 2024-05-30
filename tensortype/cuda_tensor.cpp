#include "cuda_tensor.hpp"
#include "common.hpp"
#include "context.hpp"
#include "host_tensor.hpp"
#include "cuda_kernels.hpp"

namespace vt {

using device_fp16_t = __half;

void copy_to_local(std::vector<unsigned char> dst, void* src, size_t length, cudaStream_t stream) {
    dst.clear();
    dst.resize( length);
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, length, cudaMemcpyDeviceToHost, stream));
}

template<DataType _DT_>
cudnnTensorDescriptor_t create_cudnn_td_with(std::vector<size_t> shape) {
    cudnnTensorFormat_t  format = CUDNN_TENSOR_NCHW;
    cudnnDataType_t dtype;
    cudnnTensorDescriptor_t desc;

    if ( _DT_ == DataType::F32 ) {
        dtype = CUDNN_DATA_FLOAT;
    } else if ( _DT_ == DataType::F16 ) {
        dtype = CUDNN_DATA_HALF;
    } else if ( _DT_ == DataType::BF16 ) {
        dtype = CUDNN_DATA_BFLOAT16;
    } else if ( _DT_ == DataType::I32 ) {
        dtype = CUDNN_DATA_INT32;
    } else {
        vt_panic("cudnn don't support!");
    }

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    if (shape.size() == 4) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], shape[2], shape[3]));
    } else if (shape.size() == 3) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], 1, shape[1], shape[2]));
    } else if (shape.size() == 2) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, shape[0], shape[1], 1, 1));
    } else if (shape.size() == 1) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, format, dtype, 1, shape[0], 1, 1));
    } else {
        vt_panic("cudnnSetTensor4dDescriptor: can't convert shape");
    }
    return desc;
}

template<DataType _DT_>
CUDATensor<_DT_>::~CUDATensor() {
    if (mem_ != nullptr && owner_) {
        CUDA_CHECK(cudaFree(mem_));
    }
}

template<DataType _DT_>
CUDATensor<_DT_>::CUDATensor(const ShapeType& shape) : size_(0), owner_(true) {
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
    CUDA_CHECK(cudaMalloc(&mem_, asize));
}

template<DataType _DT_>
CUDATensor<_DT_>::CUDATensor(const ShapeType& shape, void* vdata) : size_(0), owner_(false), mem_(vdata) {
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
ComputingReturn CUDATensor<_DT_>::io_load(ComputingContext* ctx, tensor_t self, const char* fileName) {
    std::vector<unsigned char> src;
    read_data(fileName, src);

    vt_assert(src.size() == size_, "loaded data must has same size");

    auto stream = ctx->cuda_stream;
    CUDA_CHECK(cudaMemcpyAsync(data(), (void *)src.data(), size_, cudaMemcpyHostToDevice, stream));
    return OP_OK;
}

template <DataType _DT_>
ComputingReturn CUDATensor<_DT_>::io_dump(ComputingContext* ctx, tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    std::vector<unsigned char> localData;
    if ( _DT_ == DataType::F32 ) {
        copy_to_local(localData, data(), first8 * sizeof(float), ctx->cuda_stream);
        float* d = (float *)localData.data();
        SIMPLE_DUMP(d);
        d = (float *)data() + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(float), ctx->cuda_stream);
        d = (float *)localData.data();
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::I32 ) {
        copy_to_local(localData, data(), first8 * sizeof(int), ctx->cuda_stream);
        int* d = (int *)localData.data();
        SIMPLE_DUMP(d);
        d = (int *)data() + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(int), ctx->cuda_stream);
        d = (int *)localData.data();
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        copy_to_local(localData, data(), first8 * sizeof(local_fp16_t), ctx->cuda_stream);
        local_fp16_t* d = (local_fp16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        d = (local_fp16_t *)data() + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(local_fp16_t), ctx->cuda_stream);
        d = (local_fp16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        return OP_OK;
    }
    if ( _DT_ == DataType::BF16 ) {
        copy_to_local(localData, mem_, first8 * sizeof(local_bf16_t), ctx->cuda_stream);
        local_bf16_t* d = (local_bf16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, bf16_to_fp32);
        d = (local_bf16_t *)mem_ + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(local_bf16_t), ctx->cuda_stream);
        d = (local_bf16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, bf16_to_fp32);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DT_>
std::variant<ComputingReturn, size_t> CUDATensor<_DT_>::op_sizeof(ComputingContext* ctx, tensor_t self) {
    return size_;
}

template <DataType _DT_>
std::variant<ComputingReturn, void *> CUDATensor<_DT_>::op_data(ComputingContext* ctx, tensor_t self) {
    return mem_;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_zero(ComputingContext* ctx, tensor_t self) {
    if ( (_DT_ == DataType::F32) || (_DT_ == DataType::I32) || (_DT_ == DataType::F16) || (_DT_ == DataType::BF16) ) {
        CUDA_CHECK( cudaMemset(data(), 0, size_) );
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_fill(ComputingContext* ctx, tensor_t self, float value) {
    if ( (_DT_ == DataType::F32) || (_DT_ == DataType::I32) || (_DT_ == DataType::F16) || (_DT_ == DataType::BF16) ) {
        auto desc = create_cudnn_td_with<_DT_>( {self->items()} );
        CUDNN_CHECK( cudnnSetTensor( ctx->cudnn_handle, desc, data(), &value) );
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) {
    if ( (_DT_ == DataType::F32) ) {
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        auto stream = ctx->cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync( data(), cos_sin.data(), self->items() * sizeof(float), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];
    auto stream = ctx->cuda_stream;

    int* mask  = (int *)data();
    if ( out->dtype() == DataType::F32 ) {
        float* dst = (float *)out->cuda_f32()->data();
        cuda::kr_causal_mask<float>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }
    if ( out->dtype() == DataType::F16 ) {
        device_fp16_t* dst = (device_fp16_t *)out->cuda_f16()->data();
        cuda::kr_causal_mask<device_fp16_t>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) {
    if ( !src->is_host() ) {
        return OP_TODO_ERROR;
    }
    auto stream = ctx->cuda_stream;
    void* from = std::get<1>( src->op_data(ctx, src) );
    CUDA_CHECK(cudaMemcpyAsync(data(), from, size_, cudaMemcpyHostToDevice, stream));
    return OP_OK;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_copy_to(ComputingContext* ctx, tensor_t self,  tensor_t dst) {
    if ( !dst->is_host() ) {
        return OP_TODO_ERROR;
    }
    auto stream = ctx->cuda_stream;
    void* to = std::get<1>( dst->op_data(ctx, dst) );
    CUDA_CHECK(cudaMemcpyAsync(to, data(), size_, cudaMemcpyDeviceToHost, stream));
    return OP_OK;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) {
    if ( _DT_ == DataType::F16 && src->dtype() == DataType::F32) {
        auto stream = ctx->cuda_stream;
        device_fp16_t* to = (device_fp16_t*)data();
        float* from = (float  *) std::get<1>(src->op_data(ctx, src));
        cuda::kr_convert<float, device_fp16_t>(from, to, self->items(), stream);
    }
    if ( _DT_ == DataType::F32 && src->dtype() == DataType::F16) {
        auto stream = ctx->cuda_stream;
        float* to = (float *)data();
        device_fp16_t* from = (device_fp16_t  *) std::get<1>(src->op_data(ctx, src));
        cuda::kr_convert<device_fp16_t, float>(from, to, self->items(), stream);
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> CUDATensor<_DT_>::op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( _DT_ == DataType::F32 ) {
        void *newData = (char *)data() + offset * sizeof(float);
        auto* newCUDATensor = new CUDATensor<DataType::F32>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    if ( _DT_ == DataType::I32 ) {
        void *newData = (char *)data() + offset * sizeof(int);
        auto* newCUDATensor = new CUDATensor<DataType::I32>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    if ( _DT_ == DataType::F16 ) {
        void *newData = (char *)data() + offset * sizeof(local_fp16_t);
        auto* newCUDATensor = new CUDATensor<DataType::F16>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    if ( _DT_ == DataType::BF16 ) {
        void *newData = (char *)data() + offset * sizeof(local_bf16_t);
        auto* newCUDATensor = new CUDATensor<DataType::BF16>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> CUDATensor<_DT_>::op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
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
        auto* newCUDATensor = new CUDATensor<DataType::F32>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    if ( DT == DataType::I32 ) {
        auto* newCUDATensor = new CUDATensor<DataType::I32>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    if ( DT == DataType::F16 ) {
        auto* newCUDATensor = new CUDATensor<DataType::F16>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    if ( DT == DataType::BF16 ) {
        auto* newCUDATensor = new CUDATensor<DataType::BF16>(newShape, newData);
        return std::make_shared<TensorType>(newCUDATensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
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
ComputingReturn CUDATensor<_DT_>::op_quantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_dequantize(ComputingContext* ctx, tensor_t self, tensor_t out) {
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t output) {
    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_add(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
    float alpha = 1.0;
    float beta = 0.0;
    cudnnOpTensorDescriptor_t opTensorDesc;
    cudnnTensorDescriptor_t adesc, bdesc, cdesc;
    void* data_a = data();
    void* data_b = std::get<1>(b->op_data(ctx, b));
    void* data_c = std::get<1>(c->op_data(ctx, c));

    CUDNN_CHECK( cudnnCreateOpTensorDescriptor(&opTensorDesc) );
    if ( _DT_ == DataType::F32 ) {
        adesc = create_cudnn_td_with<DataType::F32>( self->shape().vec() );
        bdesc = create_cudnn_td_with<DataType::F32>( b->shape().vec() );
        cdesc = create_cudnn_td_with<DataType::F32>( c->shape().vec() );
    } else if ( _DT_ == DataType::F16 ) {
        adesc = create_cudnn_td_with<DataType::F16>( self->shape().vec() );
        bdesc = create_cudnn_td_with<DataType::F16>( b->shape().vec() );
        cdesc = create_cudnn_td_with<DataType::F16>( c->shape().vec() );
    } else {
        return OP_TODO_ERROR;
    }

    CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );
    CUDNN_CHECK( cudnnOpTensor(ctx->cudnn_handle,
                                opTensorDesc,
                                &alpha, adesc, data_a,
                                &alpha, bdesc, data_b,
                                &beta,  cdesc, data_c) );

    CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );
    return OP_OK;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_mul(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) {
    float alpha = 1.0;
    float beta = 0.0;
    cudnnOpTensorDescriptor_t opTensorDesc;
    cudnnTensorDescriptor_t adesc, bdesc, cdesc;
    void* data_a = data();
    void* data_b = std::get<1>(b->op_data(ctx, b));
    void* data_c = std::get<1>(c->op_data(ctx, c));

    CUDNN_CHECK( cudnnCreateOpTensorDescriptor(&opTensorDesc) );
    if ( _DT_ == DataType::F32 ) {
        adesc = create_cudnn_td_with<DataType::F32>( self->shape().vec() );
        bdesc = create_cudnn_td_with<DataType::F32>( b->shape().vec() );
        cdesc = create_cudnn_td_with<DataType::F32>( c->shape().vec() );
    } else if ( _DT_ == DataType::F16 ) {
        adesc = create_cudnn_td_with<DataType::F16>( self->shape().vec() );
        bdesc = create_cudnn_td_with<DataType::F16>( b->shape().vec() );
        cdesc = create_cudnn_td_with<DataType::F16>( c->shape().vec() );
    } else {
        return OP_TODO_ERROR;
    }

    CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );
    CUDNN_CHECK( cudnnOpTensor(ctx->cudnn_handle,
                                opTensorDesc,
                                &alpha, adesc, data_a,
                                &alpha, bdesc, data_b,
                                &beta,  cdesc, data_c) );

    CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );
    return OP_OK;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_linear(ComputingContext* ctx, tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w_->shape()[0];

    float alpha = 1.0;
    float beta = 0.0;
    int m = outSize;
    int n = batch * tokens;
    int k = inSize;

    void* A = std::get<1>(w_->op_data(ctx, w_));
    void* B = data();
    void* C = std::get<1>(y_->op_data(ctx, y_));

    if ( _DT_ == DataType::F32 && w_->dtype() == DataType::F32 ) {
        cuda::lt_sgemm(ctx->cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_32F, k,
                B, CUDA_R_32F, k, &beta,
                C, CUDA_R_32F, m,
                ctx->cuda_workspace,
                ctx->workspace_size);

        if ( b_ != nullptr ) {
            auto ydesc = create_cudnn_td_with<DataType::F32>({batch, 1, tokens, outSize});
            auto bdesc = create_cudnn_td_with<DataType::F32>({1, 1, 1, outSize});
            void* bias = std::get<1>(b_->op_data(ctx, b_));

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ctx->cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
    }
    if ( _DT_ == DataType::F16 && w_->dtype() == DataType::F16 ) {
        cuda::lt_sgemm(ctx->cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_16F, k,
                B, CUDA_R_16F, k, &beta,
                C, CUDA_R_16F, m,
                ctx->cuda_workspace,
                ctx->workspace_size);

        if ( b_ != nullptr ) {
            auto ydesc = create_cudnn_td_with<DataType::F16>({batch, 1, tokens, outSize});
            auto bdesc = create_cudnn_td_with<DataType::F16>({1, 1, 1, outSize});
            void* bias = std::get<1>(b_->op_data(ctx, b_));

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ctx->cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
    }
    return OP_OK;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_layernorm(ComputingContext* ctx, tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    size_t batch = self->shape()[0] * self->shape()[1];
    size_t hidden = self->shape()[2];

    auto x = data();
    auto m = std::get<1>(mean->op_data(ctx, mean));
    auto v = std::get<1>(var->op_data(ctx, var));
    auto s = std::get<1>(scale->op_data(ctx, scale));
    auto b = std::get<1>(bias->op_data(ctx, bias));
    auto out = std::get<1>(y->op_data(ctx, y));

    auto stream = ctx->cuda_stream;
    if ( _DT_ == DataType::F32 ) {
        cuda::kr_layernorm<float>((float *)out, (float *)v, (float *)m,
                              (float *)x, (float *)s, (float *)b, batch, hidden, eps, stream);

        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        cuda::kr_layernorm<device_fp16_t>((device_fp16_t *)out, (device_fp16_t *)v, (device_fp16_t *)m,
                              (device_fp16_t *)x, (device_fp16_t *)s, (device_fp16_t *)b, batch, hidden, eps, stream);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
ComputingReturn CUDATensor<_DT_>::op_rmsnorm(ComputingContext* ctx, tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t hidden = self->shape()[2];

    void* norm2_ = std::get<1>(norm2->op_data(ctx, norm2));
    void* feature = data();
    void* w = std::get<1>(scale->op_data(ctx, scale));
    void* out = std::get<1>(y->op_data(ctx, y));

    auto stream = ctx->cuda_stream;
    if ( _DT_ == DataType::F32) {
        cuda::kr_rmsnorm<float>((float *)feature, (float *)w, (float *)out, (float *)norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }

    if ( _DT_ == DataType::F16 ) {
        cuda::kr_rmsnorm<device_fp16_t>((device_fp16_t *)feature, (device_fp16_t *)w, (device_fp16_t *)out, (device_fp16_t *)norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

tensor_t create_cuda_f32(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::F32>* tensor = new CUDATensor<DataType::F32>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_i32(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::I32>* tensor = new CUDATensor<DataType::I32>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_f16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::F16>* tensor = new CUDATensor<DataType::F16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_bf16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::BF16>* tensor = new CUDATensor<DataType::BF16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_q8(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Q8>* tensor = new CUDATensor<DataType::Q8>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_q4(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Q4>* tensor = new CUDATensor<DataType::Q4>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_pq(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::PQ>* tensor = new CUDATensor<DataType::PQ>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}


}
