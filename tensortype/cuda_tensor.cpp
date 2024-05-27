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
    CUDA_CHECK(cudaMemcpyAsync(mem_, (void *)src.data(), size_, cudaMemcpyHostToDevice, stream));
    return OP_OK;
}

template <DataType _DT_>
ComputingReturn CUDATensor<_DT_>::io_dump(ComputingContext* ctx, tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    std::vector<unsigned char> localData;
    if ( _DT_ == DataType::F32 ) {
        copy_to_local(localData, mem_, first8 * sizeof(float), ctx->cuda_stream);
        float* d = (float *)localData.data();
        SIMPLE_DUMP(d);
        d = (float *)mem_ + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(float), ctx->cuda_stream);
        d = (float *)localData.data();         
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::I32 ) {
        copy_to_local(localData, mem_, first8 * sizeof(int), ctx->cuda_stream);
        int* d = (int *)localData.data();
        SIMPLE_DUMP(d);
        d = (int *)mem_ + self->items() - first8;
        copy_to_local(localData, d, first8 * sizeof(int), ctx->cuda_stream);
        d = (int *)localData.data();         
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        copy_to_local(localData, mem_, first8 * sizeof(local_fp16_t), ctx->cuda_stream);
        local_fp16_t* d = (local_fp16_t *)localData.data();
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        d = (local_fp16_t *)mem_ + self->items() - first8;
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

}
