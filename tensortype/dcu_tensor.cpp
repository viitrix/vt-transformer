#include "dcu_tensor.hpp"
#include <dcu_kernels.hpp>

namespace vt {

using device_fp16_t = __half;

template<DataType _DTYPE_>
DCUTensor<_DTYPE_>::DCUTensor(const ShapeType& shape) : owner_(true) {
    if ( _DTYPE_ == DataType::Float ) {
        HIP_CHECK(hipMalloc(&mem_, shape.numel() * sizeof(float)));
    } else if ( _DTYPE_ == DataType::Int ) {
        HIP_CHECK(hipMalloc(&mem_, shape.numel() * sizeof(int)));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        HIP_CHECK(hipMalloc(&mem_, shape.numel() * sizeof(device_fp16_t)));
    } else if ( _DTYPE_ == DataType::Q8 ) {
        size_t last_dim = shape.vec().back();
        size_t feature_num = shape.numel() / last_dim;
        last_dim += sizeof(float) * 2;
        HIP_CHECK(hipMalloc(&mem_, last_dim * feature_num));
    } else if ( _DTYPE_ == DataType::Q4 ) {
        size_t last_dim = shape.vec().back();
        vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");

        size_t numel = shape.numel();
        size_t blk_num = numel / Q4_BLOCK_SIZE;
        HIP_CHECK(hipMalloc(&mem_, blk_num * sizeof( q4_block_t )));
    } else {
        vt_panic("Don't support DataType for HIP");
    }
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::io_dump(tensor_t self) {
    auto stream = ComputingContext::dcu_stream;

    if ( DT == DataType::Float || DT == DataType::Int || DT == DataType::FP16 ) {
        size_t first8 = std::min(self->items(), (size_t)8);
        size_t datasize = 4 * first8;
        if ( DT == DataType::FP16 ) {
            datasize = 2 * first8;
        }
        std::vector<char> mem;
        mem.resize(datasize);

        void* dst = mem.data();
        char* src = (char *)data();
        HIP_CHECK(hipMemcpyAsync(mem.data(), src, datasize, hipMemcpyDeviceToHost, stream));

        std::cout << "First " << first8 << " : ";
        if ( DT == DataType::Float ) {
            const float* d = (float *)dst;
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
        }
        if ( DT == DataType::Int ) {
            const int* d = (int *)dst;
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
        }
        if ( DT == DataType::FP16 ) {
            const local_fp16_t* d = (local_fp16_t *)dst;
            for(size_t i = 0; i < first8; i++) {
                std::cout << fp16_to_fp32( d[i] ) << " ";
            }
        }
        std::cout << std::endl;

        size_t dataoffset = 4 * (self->items() - first8);
        if ( DT == DataType::FP16 ) {
            dataoffset = 2 * (self->items() - first8);
        }
        src = src + dataoffset;
        HIP_CHECK(hipMemcpyAsync(mem.data(), src, datasize, hipMemcpyDeviceToHost, stream));

        std::cout << "Last " << first8 << " : ";
        if ( DT == DataType::Float ) {
            const float* d = (float *)dst;
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
        }
        if ( DT == DataType::Int ) {
            const int* d = (int *)dst;
            for(size_t i = 0; i < first8; i++) {
                std::cout << d[i] << " ";
            }
        }
        if ( DT == DataType::FP16 ) {
            const local_fp16_t* d = (local_fp16_t *)dst;
            for(size_t i = 0; i < first8; i++) {
                std::cout << fp16_to_fp32( d[i] ) << " ";
            }
        }

        std::cout << std::endl;
        return OP_OK;
    }
    if ( DT == DataType::Q8 ) {
        size_t last_dim = self->shape().vec().back();
        const size_t feature_num = self->items() / last_dim;
        last_dim = last_dim + sizeof(float) * 2;

        std::vector<char> local_data;
        local_data.resize(last_dim);
        HIP_CHECK(hipMemcpyAsync(local_data.data(), data(), last_dim, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        const q8_head_t* target = (const q8_head_t *)local_data.data();
        std::cout << "First 8 : ";
        for (size_t i = 0; i < 8; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;

        HIP_CHECK(hipMemcpyAsync(local_data.data(), (char *) data() + (feature_num - 1) * last_dim, last_dim, hipMemcpyDeviceToHost, stream));
        last_dim = last_dim - sizeof(float) * 2;
        std::cout << "Last 8 : ";
        for (size_t i = last_dim - 8; i < last_dim; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::io_load(tensor_t self, const char* fileName) {
    std::vector<char> src;
    read_data(fileName, src);

    size_t len = std::get<1>(self->op_sizeof(self));
    vt_assert(src.size() == len , "loaded data must has same size");
    void* x = src.data();
    void* y = data();

    auto stream = ComputingContext::dcu_stream;
    HIP_CHECK(hipMemcpyAsync(y, x, src.size(), hipMemcpyHostToDevice, stream));
    return OP_OK;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> DCUTensor<_DTYPE_>::op_sizeof(tensor_t self) {
    if ( _DTYPE_ == DataType::Float ) {
        return (size_t) self->items() * sizeof(float);
    }
    if ( _DTYPE_ == DataType::Int ) {
        return (size_t) self->items() * sizeof(int);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        return (size_t) self->items() * sizeof(device_fp16_t);
    }
    if ( _DTYPE_ == DataType::Q8 ) {
        auto last_dim = self->shape().vec().back();
        size_t feature_num = self->items() / last_dim;
        return feature_num * ( last_dim + sizeof(float) * 2 );
    }
    if ( _DTYPE_ == DataType::Q4 ) {
        auto shape = self->shape();
        size_t numel = shape.numel();
        size_t blk_num = numel / Q4_BLOCK_SIZE;
        return blk_num * sizeof( q4_block_t );
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_zero(tensor_t self) {
    void *dst = data();
    auto s = std::get<1>(self->op_sizeof(self));
    HIP_CHECK( hipMemset(dst, 0, s) );
    return OP_OK;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_fill(tensor_t self, float value) {
    auto stream = ComputingContext::dcu_stream;
    if ( DT == DataType::Float ) {
        float* target = (float *)data();
        dcu::kr_fill<float>(target, value, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        int* target = (int *)data();
        dcu::kr_fill<int>(target, value, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t* target = (device_fp16_t *)data();
        dcu::kr_fill(target, value, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_alibi(tensor_t self) {
    int heads = self->shape()[1];
    int tokens = self->shape()[3];

    auto stream = ComputingContext::dcu_stream;
    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        std::vector<float> buffer;
        vt::fill_alibi<float>(buffer, heads, tokens);

        HIP_CHECK(hipMemcpyAsync(data(), buffer.data(), s,  hipMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<local_fp16_t> buffer;
        vt::fill_alibi<local_fp16_t>(buffer, heads, tokens);
        HIP_CHECK(hipMemcpyAsync(data(), buffer.data(), s,  hipMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template<DataType DT>
ComputingReturn DCUTensor<DT>::op_rotary_cache(tensor_t self, float base) {
    if ( DT == DataType::Float ) {
        // building inv_freq
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        auto stream = ComputingContext::dcu_stream;
        HIP_CHECK(hipMemcpyAsync( data(), cos_sin.data(), self->items() * sizeof(float), hipMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];

    int* mask  = (int *)data();

    auto stream = ComputingContext::dcu_stream;
    return OP_OUTPUT_ERROR;
}


template<DataType DT>
std::variant<ComputingReturn, tensor_t> DCUTensor<DT>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( DT == DataType::Float ) {
        void *newData = (char *)data() + offset * sizeof(float);
        auto* newDcuTensor = new DCUTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        void *newData = (char *)data() + offset * sizeof(int);
        auto* newDcuTensor = new DCUTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        void *newData = (char *)data() + offset * sizeof(device_fp16_t);
        auto* newDcuTensor = new DCUTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Q8 ) {
        auto last_dim = self->shape().vec().back();
        vt_assert(offset % last_dim == 0, "Q8's view must aligen with last dim");
        vt_assert(newShape_.back() == last_dim, "Q8's view must aligen with last dim");

        void *newData = (char *)data() + (offset / last_dim) * ( last_dim + sizeof(float) * 2 );
        auto* newDcuTensor = new DCUTensor<DataType::Q8>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Q4 ) {
        vt_assert(offset % Q4_BLOCK_SIZE == 0, "Q4's view must aligen with Q4_BLOCK_T");
        void *newData = (char *)data() + (offset / Q4_BLOCK_SIZE) * sizeof(q4_block_t);
        auto* newDcuTensor = new DCUTensor<DataType::Q4>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> DCUTensor<_DT_>::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
    DataType DT = DataType_from(dtype);

    ShapeType newShape(newShape_);

    void *newData = nullptr;
    if ( _DT_ == DataType::Float ) {
        newData = (char *)data() + offset * sizeof(float);
    } else if ( _DT_ == DataType::Int ) {
        newData = (char *)data() + offset * sizeof(int);
    } else if ( _DT_ == DataType::FP16 ) {
        newData = (char *)data() + offset * sizeof(device_fp16_t);
    } else {
        return OP_TODO_ERROR;
    }

    if ( DT == DataType::Float ) {
        auto* newDcuTensor = new DCUTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        auto* newDcuTensor = new DCUTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        auto* newDcuTensor = new DCUTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

// ============================================
tensor_t create_dcu_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::Float>* tensor = new DCUTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dcu_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::FP16>* tensor = new DCUTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dcu_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::Int>* tensor = new DCUTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dcu_q8(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::Q8>* tensor = new DCUTensor<DataType::Q8>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_dcu_q4(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    DCUTensor<DataType::Q4>* tensor = new DCUTensor<DataType::Q4>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
