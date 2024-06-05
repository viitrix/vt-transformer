#include "hip_tensor.hpp"
#include "common.hpp"
#include "context.hpp"
#include "host_tensor.hpp"
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

}
