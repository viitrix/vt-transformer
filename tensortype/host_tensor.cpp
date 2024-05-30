#include "host_tensor.hpp"
#include "context.hpp"
#include "common.hpp"


namespace vt {

template<DataType _DT_>
HostTensor<_DT_>::~HostTensor() {
    if (owner_) {
        free(mem_);
    }
}

template<DataType _DT_>
HostTensor<_DT_>::HostTensor(const ShapeType& shape) : size_(0), owner_(true) {
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

    *const_cast<size_t*>(&size_) = asize;
    mem_ = malloc(asize);
}

template<DataType _DT_>
HostTensor<_DT_>::HostTensor(const ShapeType& shape, void* mem) : size_(0), owner_(false), mem_(mem) {
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

    *const_cast<size_t*>(&size_) = asize;
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
    auto size = std::get<1>( self->op_sizeof(ctx, self) );
    int ret = ctx->pipe_read(mem_, size);
    if ( ret < 0 ) {
        return OP_OUTPUT_ERROR;
    }
    return OP_OK;
}

template <DataType _DT_>
ComputingReturn HostTensor<_DT_>::io_pipe_write(ComputingContext* ctx, tensor_t self, int n) {
    auto size = std::get<1>( self->op_sizeof(ctx, self) );
    int ret = ctx->pipe_write(n, mem_, size);
    if ( ret < 0 ) {
        return OP_OUTPUT_ERROR;
    }
    return OP_OK;
}

template <DataType _DT_>
std::variant<ComputingReturn, size_t> HostTensor<_DT_>::op_sizeof(ComputingContext* ctx, tensor_t self) {
    return size_;
}

template <DataType _DT_>
std::variant<ComputingReturn, void *> HostTensor<_DT_>::op_data(ComputingContext* ctx, tensor_t self) {
    return mem_;
}

template <DataType _DT_>
std::variant<ComputingReturn, tensor_t> HostTensor<_DT_>::op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( _DT_ == DataType::F32 ) {
        void *newData = (char *)mem_ + offset * sizeof(float);
        auto* newHostTensor = new HostTensor<DataType::F32>(newShape, newData);
        return std::make_shared<TensorType>(newHostTensor, newShape);
    }
    if ( _DT_ == DataType::I32 ) {
        void *newData = (char *)mem_ + offset * sizeof(int);
        auto* newHostTensor = new HostTensor<DataType::I32>(newShape, newData);
        return std::make_shared<TensorType>(newHostTensor, newShape);
    }
    if ( _DT_ == DataType::F16 ) {
        void *newData = (char *)mem_ + offset * sizeof(local_fp16_t);
        auto* newHostTensor = new HostTensor<DataType::F16>(newShape, newData);
        return std::make_shared<TensorType>(newHostTensor, newShape);
    }
    if ( _DT_ == DataType::BF16 ) {
        void *newData = (char *)mem_ + offset * sizeof(local_bf16_t);
        auto* newHostTensor = new HostTensor<DataType::BF16>(newShape, newData);
        return std::make_shared<TensorType>(newHostTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template <DataType _DT_>
ComputingReturn HostTensor<_DT_>::op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t output) {
    const size_t feature_size = table->shape().vec().back();
    const int items = self->items();
    int* tokens = (int * ) mem_;

    std::string odev = output->device_name();
    if(  odev != "host" ) {
        vt_panic("Can't do op_embed on output's device!");
    }

    if ( table->dtype() == DataType::F32 && output->dtype() == DataType::F32 ) {
        float* dst = (float *) std::get<1>( output->op_data(ctx, output) );
        float* src = (float *) std::get<1>( table->op_data(ctx, table) );
        for ( int i = 0; i < items; i++) {
            float* emb = &src[ tokens[i] * feature_size ];
            memcpy(dst, emb, sizeof(float)*feature_size);
            dst += feature_size;
        }
        return OP_OK;
    }
    if ( table->dtype() == DataType::F16 && output->dtype() == DataType::F16) {
        local_fp16_t* dst = (local_fp16_t *) std::get<1>( output->op_data(ctx, output) );
        local_fp16_t* src = (local_fp16_t *) std::get<1>( table->op_data(ctx, table) );
        for ( int i = 0; i < items; i++) {
            local_fp16_t* emb = &src[ tokens[i] * feature_size ];
            memcpy(dst, emb, sizeof(local_fp16_t)*feature_size);
            dst += feature_size;
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}


tensor_t create_host_f32(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::F32>* tensor = new HostTensor<DataType::F32>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_i32(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::I32>* tensor = new HostTensor<DataType::I32>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_f16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::F16>* tensor = new HostTensor<DataType::F16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_bf16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::BF16>* tensor = new HostTensor<DataType::BF16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}


tensor_t create_host_q8(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::Q8>* tensor = new HostTensor<DataType::Q8>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_q4(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::Q4>* tensor = new HostTensor<DataType::Q4>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_host_pq(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    HostTensor<DataType::PQ>* tensor = new HostTensor<DataType::PQ>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}


}
