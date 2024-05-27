#include "host_tensor.hpp"
#include "common.hpp"

namespace vt {

template<DataType _DT_>
HostTensor<_DT_>::~HostTensor() {
    if (owner_) {
        free(mem_);
    }
}

template<DataType _DT_>
HostTensor<_DT_>::HostTensor(const ShapeType& shape) : owner_(true) {
    size_t asize = 0;
    size_t number = shape.numel();
    if ( _DT_ == DataType::F32 ) {
        asize = sizeof(float) * number;
    } else if ( _DT_ == DataType::I32 ) {
        asize = sizeof(int) * number;
    } else if ( _DT_ == DataType::F16 || _DT_ == DataType::BF16 ) {
        asize = sizeof(local_fp16_t) * number;
    } else {
        vt_fatal_error();
    }

    const_cast<size_t>(size_) = asize;
    mem_ = malloc(asize);
}

template<DataType _DT_>
ComputingReturn HostTensor<_DT_>::io_load(ComputingContext* ctx, tensor_t self, const char* fileName) {
    std::ifstream inf(fileName, std::ios::binary);
    if ( ! inf.is_open() ) {
        std::cout << "can't open " << fileName << std::endl;
        vt_panic("can't open file");
        return OP_INPUT_ERROR;
    }

    size_t ret = inf.read((char *)mem_, size_).gcount();
    vt_assert(ret == size_, "file size dont't match tensor");

    inf.close();
    return OP_OK;
}

template <DataType _DT_>
ComputingReturn HostTensor<_DT_>::io_save(ComputingContext* ctx, tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

    const char* d = (const char *)mem_;
    wf.write(d, size_);
    wf.close();
    return OP_OK;
}

template <DataType _DT_>
ComputingReturn HostTensor<_DT_>::io_dump(ComputingContext* ctx, tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    if ( _DT_ == DataType::F32 ) {
        float* d = (float *)mem_;
        SIMPLE_DUMP(d);
        d = (float *)mem_ + self->items() - first8;
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::I32 ) {
        int* d = (int *)mem_;
        SIMPLE_DUMP(d);
        d = (int *)mem_ + self->items() - first8;
        SIMPLE_DUMP(d);
        return OP_OK;
    }
    if ( _DT_ == DataType::F16 ) {
        local_fp16_t* d = (local_fp16_t *)mem_;
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        d = (local_fp16_t *)mem_ + self->items() - first8;
        SIMPLE_DUMP_WITH(d, fp16_to_fp32);
        return OP_OK;
    }
    if ( _DT_ == DataType::BF16 ) {
        local_fp16_t* d = (local_fp16_t *)mem_;
        SIMPLE_DUMP_WITH(d, bf16_to_fp32);
        d = (local_fp16_t *)mem_ + self->items() - first8;
        SIMPLE_DUMP_WITH(d, bf16_to_fp32);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DT_>
ComputingReturn HostTensor<_DT_>::io_pipe_read(ComputingContext* ctx, tensor_t self) {
    /*
    auto size = std::get<1>( self->op_sizeof(self) );
    int ret = CollectiveContext::pipe_read(data(), size);
    if ( ret < 0 ) {
        return OP_OUTPUT_ERROR;
    }
    */

    return OP_OK;
}

template <DataType _DT_>
ComputingReturn HostTensor<_DT_>::io_pipe_write(ComputingContext* ctx, tensor_t self, int n) {
    /*
    auto size = std::get<1>( self->op_sizeof(self) );
    int ret = CollectiveContext::pipe_write(n, data(), size);
    if ( ret < 0 ) {
        return OP_OUTPUT_ERROR;
    }
    */
    return OP_OK;
}

}
