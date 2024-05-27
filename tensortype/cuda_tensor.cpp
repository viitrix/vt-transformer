#include "cuda_tensor.hpp"
#include "common.hpp"
#include "context.hpp"

namespace vt {

void copy_to_local(std::vector<unsigned char> dst, void* src, size_t length, cudaStream_t stream) {
    dst.clear();
    dst.resize( length);
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, length, cudaMemcpyDeviceToHost, stream));
}

template<DataType _DT_> 
cudnnTensorDescriptor_t create_cudnn_td_with(std::vector<size_t>& shape) {
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
CUDATensor<_DT_>::CUDATensor(const ShapeType& shape) : owner_(true) {
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

    const_cast<size_t>(size_) = asize;
    CUDA_CHECK(cudaMalloc(&mem_, asize));
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

        auto stream = ComputingContext::cuda_stream;
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
    auto stream = ComputingContext::cuda_stream;

    int* mask  = (int *)data();
    if ( out->dtype() == DataType::F32 ) {
        /*
        float* dst = (float *)out->cuda_f32()->data();
        cuda::causal_mask<float>(mask, dst, batch, new_tokens, full_tokens, stream);
        */
        return OP_OK;
    }
    if ( out->dtype() == DataType::F16 ) {
        /*
        device_fp16_t* dst = (device_fp16_t *)out->cuda_fp16()->data();
        cuda::causal_mask<device_fp16_t>(mask, dst, batch, new_tokens, full_tokens, stream);
        */
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

}
