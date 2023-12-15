#include "dcu_tensor.hpp"
#include "host_tensor.hpp"
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
    if ( out->dtype() == DataType::Float ) {
        float* dst = (float *)out->dcu_float()->data();
        dcu::kr_causal_mask<float>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }

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

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( owner_ == true ) {
        return OP_INPUT_ERROR;
    }

    if ( newShape.numel() + offset > self->items()  ) {
        return OP_INPUT_ERROR;
    }

    if ( DT == DataType::Float ) {
        mem_  = (char *)data() + offset * sizeof(float);
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        mem_  = (char *)data() + offset * sizeof(int);
        return OP_OK;
    }

    if ( DT == DataType::FP16 ) {
        mem_  = (char *)data() + offset * sizeof(device_fp16_t);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DCUTensor<_DTYPE_>::op_quantize(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::dcu_stream;
    if ( _DTYPE_ == DataType::Float && out->is_q4() ) {
        const float* src = (float *)data();
        void* dst = out->dcu_q4()->data();
        dcu::kr_quantize_q4<float>(src, dst, self->items(), stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::FP16 && out->is_q4() ) {
        const device_fp16_t* src = (device_fp16_t *)data();
        void* dst = out->dcu_q4()->data();
        dcu::kr_quantize_q4<device_fp16_t>(src, dst, self->items(), stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::Float && out->is_q8() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        const float* src = (float *)data();
        void* dst = out->dcu_q8()->data();
        dcu::kr_quantize_q8<float>(src, dst, feature_num, feature_size, stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::FP16 && out->is_q8() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        const device_fp16_t* src = (device_fp16_t *)data();
        void* dst = out->dcu_q8()->data();
        dcu::kr_quantize_q8<device_fp16_t>(src, dst, feature_num, feature_size, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DCUTensor<_DTYPE_>::op_dequantize(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::dcu_stream;
    if ( _DTYPE_ == DataType::Q4 && out->is_fp16() ) {
        void* src = data();
        device_fp16_t* dst =(device_fp16_t *) out->dcu_fp16()->data();
        dcu::kr_dequantize_q4<device_fp16_t>(src, dst, self->items(), stream);

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Q8 && out->is_fp16() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;
        void* src = data();
        device_fp16_t* dst =(device_fp16_t *) out->dcu_fp16()->data();
        dcu::kr_dequantize_q8<device_fp16_t>(src, dst, feature_num, feature_size, stream);

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn DCUTensor<_DTYPE_>::op_embed(tensor_t self, tensor_t table, tensor_t outspace) {
    size_t batch = self->shape()[0];
    size_t len = self->shape()[1];
    size_t hidden = table->shape()[1];

    auto stream = ComputingContext::dcu_stream;
    int* text = (int *)data();

    if ( table->dtype() == DataType::Float ) {
        float* from = (float *)table->dcu_float()->data();
        float* out = (float *)outspace->dcu_float()->data();
        dcu::kr_embed<float>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    if ( table->dtype() == DataType::FP16 ) {
        device_fp16_t* from = (device_fp16_t *)table->dcu_fp16()->data();
        device_fp16_t* out = (device_fp16_t *)outspace->dcu_fp16()->data();
        dcu::kr_embed<device_fp16_t>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_copy(tensor_t self, tensor_t src) {
    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        if ( src->is_host() ) {
            void* x = src->host_float()->data();
            void* y = data();

            auto stream = ComputingContext::dcu_stream;
            HIP_CHECK(hipMemcpyAsync(y, x, s, hipMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::dcu_stream;
        HIP_CHECK(hipMemcpyAsync(data(), src->dcu_float()->data(), s, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        if ( src->is_host() ) {
            void* x = src->host_int()->data();
            void* y = data();

            auto stream = ComputingContext::dcu_stream;
            HIP_CHECK(hipMemcpyAsync(y, x, s, hipMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::dcu_stream;
        HIP_CHECK(hipMemcpyAsync(data(), src->dcu_int()->data(), s, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        if ( src->is_host() ) {
            void* x = src->host_fp16()->data();
            void* y = data();

            auto stream = ComputingContext::dcu_stream;
            HIP_CHECK(hipMemcpyAsync(y, x, s, hipMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::dcu_stream;
        HIP_CHECK(hipMemcpyAsync(data(), src->dcu_fp16()->data(), s, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q8 ) {
        if ( src->is_host() ) {
            void* x = src->host_q8()->data();
            void* y = data();

            auto stream = ComputingContext::dcu_stream;
            HIP_CHECK(hipMemcpyAsync(y, x, s, hipMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::dcu_stream;
        HIP_CHECK(hipMemcpyAsync(data(), src->dcu_q8()->data(), s, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q4 ) {
        if ( src->is_host() ) {
            void* x = src->host_q4()->data();
            void* y = data();

            auto stream = ComputingContext::dcu_stream;
            HIP_CHECK(hipMemcpyAsync(y, x, s, hipMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::dcu_stream;
        HIP_CHECK(hipMemcpyAsync(data(), src->dcu_q4()->data(), s, hipMemcpyDeviceToDevice, stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_convert(tensor_t self, tensor_t src) {
    if ( DT == DataType::FP16 && src->is_float() ) {
        auto stream = ComputingContext::dcu_stream;
        dcu::kr_convert<float, device_fp16_t>((float *)src->dcu_float()->data(), (device_fp16_t *)data(), self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_scale(tensor_t self, float scale) {
    auto stream = ComputingContext::dcu_stream;
    size_t n = self->items();
    if (   DT == DataType::Float) {
        float* src = (float *)data();
        float* dst = (float *)data();
        dcu::kr_scale<float>(src, dst, scale, 0.0, n, stream);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        device_fp16_t* src = (device_fp16_t *)data();
        device_fp16_t* dst = (device_fp16_t *)data();
        dcu::kr_scale<device_fp16_t>(src, dst, scale, 0.0, n, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    auto stream = ComputingContext::dcu_stream;
    if ( self->items() == b->items() ) {
        if ( DT == DataType::Float && b->is_float() ) {
            float* A = (float *)data();
            float* B = (float *)b->dcu_float()->data();
            float* C = (float *)c->dcu_float()->data();

            dcu::kr_add<float, float>(A, B, C, self->items(), stream);
            return OP_OK;
        }
        if ( DT == DataType::FP16 && b->is_fp16() ) {
            device_fp16_t* A = (device_fp16_t *)data();
            device_fp16_t* B = (device_fp16_t *)b->dcu_fp16()->data();
            device_fp16_t* C = (device_fp16_t *)c->dcu_fp16()->data();

            dcu::kr_add<device_fp16_t, device_fp16_t>(A, B, C, self->items(), stream);

            return OP_OK;
        }
        if ( DT == DataType::FP16 && b->is_float() ) {
            device_fp16_t* A = (device_fp16_t *)data();
            float* B = (float *)b->dcu_fp16()->data();
            device_fp16_t* C = (device_fp16_t *)c->dcu_fp16()->data();

            dcu::kr_add<device_fp16_t, float>(A, B, C, self->items(), stream);
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

            if ( DT == DataType::FP16 && b->is_fp16() ) {
                device_fp16_t* A = (device_fp16_t *)data();
                device_fp16_t* B = (device_fp16_t *)b->dcu_fp16()->data();
                device_fp16_t* C = (device_fp16_t *)c->dcu_fp16()->data();

                dcu::kr_add_broadcast<device_fp16_t>(A, B, C, length, inter, feature, stream);
                return OP_OK;
            }
            if ( DT == DataType::Float && b->is_float() ) {
                float* A = (float *)data();
                float* B = (float *)b->dcu_float()->data();
                float* C = (float *)c->dcu_float()->data();

                dcu::kr_add_broadcast<float>(A, B, C, length, inter, feature, stream);
                return OP_OK;
            }
        }
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    vt_assert( self->items() == b->items() , " DCU's add don't support brodcast");
    auto stream = ComputingContext::dcu_stream;
    if ( DT == DataType::Float && b->is_float() ) {
        float* A = (float *)data();
        float* B = (float *)b->dcu_float()->data();
        float* C = (float *)c->dcu_float()->data();

        dcu::kr_mul<float, float>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 && b->is_fp16() ) {
        device_fp16_t* A = (device_fp16_t *)data();
        device_fp16_t* B = (device_fp16_t *)b->dcu_fp16()->data();
        device_fp16_t* C = (device_fp16_t *)c->dcu_fp16()->data();

        dcu::kr_mul<device_fp16_t, device_fp16_t>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 && b->is_float() ) {
        device_fp16_t* A = (device_fp16_t *)data();
        float* B = (float *)b->dcu_fp16()->data();
        device_fp16_t* C = (device_fp16_t *)c->dcu_fp16()->data();

        dcu::kr_mul<device_fp16_t, float>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_linear(tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    auto stream = ComputingContext::dcu_stream;
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w_->shape()[0];

    float alpha = 1.0;
    float beta = 0.0;

    if ( DT == DataType::Float && w_->is_float() ) {
        void* A = w_->dcu_float()->data();
        void* B = data();
        void* C = y_->dcu_float()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        HIPBLAS_CHECK( hipblasGemmEx(ComputingContext::hipblas_handle,
                       HIPBLAS_OP_T, HIPBLAS_OP_N,
                       m, n, k,
                       &alpha, A, HIPBLAS_R_32F, k,
                       B, HIPBLAS_R_32F, k, &beta,
                       C, HIPBLAS_R_32F, m,
                       HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        if ( b_ != nullptr ) {
            dcu::kr_add_bias<float>((float *)C, (float *)b_->dcu_float()->data(), (float *)C,  n, m, stream);
        }

        return OP_OK;
    }

    if ( DT == DataType::Float && w_->is_q8() ) {
        return OP_OK;
    }

    if ( DT == DataType::Float && w_->is_q4() ) {
        return OP_OK;
    }

    if ( DT == DataType::FP16 && w_->is_fp16() ) {
        void* A = w_->dcu_fp16()->data();
        void* B = data();
        void* C = y_->dcu_fp16()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        HIPBLAS_CHECK( hipblasGemmEx(ComputingContext::hipblas_handle,
                       HIPBLAS_OP_T, HIPBLAS_OP_N,
                       m, n, k,
                       &alpha, A, HIPBLAS_R_16F, k,
                       B, HIPBLAS_R_16F, k, &beta,
                       C, HIPBLAS_R_16F, m,
                       HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        if ( b_ != nullptr) {
            dcu::kr_add_bias<device_fp16_t>((device_fp16_t *)C, (device_fp16_t *)b_->dcu_fp16()->data(), (device_fp16_t *)C, n, m, stream);
        }

        return OP_OK;
    }

    if ( DT == DataType::FP16 && w_->is_q8() ) {
        return OP_OK;
    }

    if ( DT == DataType::FP16 && w_->is_q4() ) {
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    auto stream = ComputingContext::dcu_stream;

    if ( DT == DataType::Float ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->dcu_float();
        auto v = var->dcu_float();
        auto s = scale->dcu_float();
        auto b = bias->dcu_float();
        auto out = y->dcu_float();

        dcu::kr_layer_norm<float>((float *)out->data(), (float *)v->data(), (float *)m->data(),
                                 (float *)x->data(), (float *)s->data(), (float *)b->data(), batch, hidden, eps, stream);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->dcu_fp16();
        auto v = var->dcu_fp16();
        auto s = scale->dcu_fp16();
        auto b = bias->dcu_fp16();
        auto out = y->dcu_fp16();

        dcu::kr_layer_norm<__half>((__half *)out->data(), (__half *)v->data(), (__half *)m->data(),
                                 (__half *)x->data(), (__half *)s->data(), (__half *)b->data(), batch, hidden, eps, stream);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t hidden = self->shape()[2];

    auto stream = ComputingContext::dcu_stream;

    if ( DT == DataType::Float) {
        float* norm2_ = (float *)norm2->dcu_float()->data();
        float* feature = (float *)self->dcu_float()->data();
        float* w = (float *)scale->dcu_float()->data();
        float* out = (float *)y->dcu_float()->data();

        dcu::kr_rmsnorm<float>(feature, w, out, norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }


   if ( DT == DataType::FP16 ) {
        device_fp16_t* norm2_ = (device_fp16_t *)norm2->dcu_fp16()->data();
        device_fp16_t* feature = (device_fp16_t *)self->dcu_fp16()->data();
        device_fp16_t* w = (device_fp16_t *)scale->dcu_fp16()->data();
        device_fp16_t* out = (device_fp16_t *)y->dcu_fp16()->data();

        dcu::kr_rmsnorm<device_fp16_t>(feature, w, out, norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn DCUTensor<DT>::op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos_, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

    vt_assert(hidden == cached->shape()[1], "heads number must be same with cache");

    auto stream = ComputingContext::dcu_stream;
    int* pos = (int*) pos_->dcu_int()->data();
    if ( DT == DataType::Float ) {

        float* in = (float *)data();
        float* cos_sin = (float *)cached->dcu_float()->data();
        float* out = (float *)y->dcu_float()->data();

        dcu::kr_rotary_embed<float>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t* in = (device_fp16_t *)data();
        device_fp16_t* out = (device_fp16_t *)y->dcu_fp16()->data();
        float* cos_sin = (float *)cached->dcu_float()->data();

        dcu::kr_rotary_embed<device_fp16_t>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  DCUTensor<DT>::op_transpose_0213(tensor_t self, tensor_t y) {
    auto x = this;
    auto stream = ComputingContext::dcu_stream;

    int sz0 = self->shape()[0];
    int sz1 = self->shape()[1];
    int sz2 = self->shape()[2];
    int sz3 = self->shape()[3];

    if ( DT == DataType::Float ) {
        auto out = y->dcu_float();
        dcu::kr_transpose_0213<float>((float *)x->data(), (float *)out->data(), sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto out = y->dcu_fp16();
        dcu::kr_transpose_0213<device_fp16_t>((device_fp16_t *)x->data(), (device_fp16_t *)out->data(), sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  DCUTensor<DT>::op_qk(tensor_t self, tensor_t k_, tensor_t qk_) {
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

    if ( DT == DataType::Float ) {
        void *A_ = k_->dcu_float()->data();
        void *B_ = data();
        void *C_ = qk_->dcu_float()->data();

        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ComputingContext::hipblas_handle,
                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_32F, k, HnT,
                        B_, HIPBLAS_R_32F, k, HfT,
                        &beta, C_, HIPBLAS_R_32F, m, TT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
#if 1
        void *A_ = k_->dcu_fp16()->data();
        void *B_ = data();
        void *C_ = qk_->dcu_float()->data();

        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ComputingContext::hipblas_handle,
                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_16F, k, HnT,
                        B_, HIPBLAS_R_16F, k, HfT,
                        &beta, C_, HIPBLAS_R_32F, m, TT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );
#else
        // out loop and single hipblasGemmEx
        for (int i = 0; i < batch * heads; i++) {
            device_fp16_t* B = (device_fp16_t *)data() + i * HnT;
            device_fp16_t* A = (device_fp16_t *)(k_->dcu_fp16()->data()) + i * HfT;
            float* C = (float *)(qk_->dcu_float()->data()) + i * TT;

            HIPBLAS_CHECK( hipblasGemmEx(
                        ComputingContext::hipblas_handle,
                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A, HIPBLAS_R_16F, k,
                        B, HIPBLAS_R_16F, k, &beta,
                        C, HIPBLAS_R_32F, m,
                        HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );
        }
#endif
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  DCUTensor<DT>::op_softmax(tensor_t self, tensor_t y) {
    auto stream = ComputingContext::dcu_stream;
    size_t hidden = self->shape().vec().back();
    size_t length = self->items() / hidden;

    if ( DT == DataType::Float ) {
        dcu::kr_softmax<float>((float *)data(), (float *)y->dcu_float()->data(), length, hidden, stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        dcu::kr_softmax<device_fp16_t>((device_fp16_t *)data(), (device_fp16_t *)y->dcu_fp16()->data(), length, hidden, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template<DataType DT>
ComputingReturn  DCUTensor<DT>::op_attn(tensor_t self, tensor_t value_, tensor_t out_) {
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

    if ( value_->is_float() && self->is_float() ) {
        void *A_ = value_->dcu_float()->data();
        void *B_ = data();
        void *C_ = out_->dcu_float()->data();
        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ComputingContext::hipblas_handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_32F, m, HfT,
                        B_, HIPBLAS_R_32F, k, TT,
                        &beta, C_, HIPBLAS_R_32F, m, HnT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( value_->is_fp16() && self->is_float() ) {
        void *A_ = value_->dcu_fp16()->data();
        void *B_ = data();
        void *C_ = out_->dcu_fp16()->data();
        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ComputingContext::hipblas_handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_16F, m, HfT,
                        B_, HIPBLAS_R_32F, k, TT,
                        &beta, C_, HIPBLAS_R_16F, m, HnT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( value_->is_fp16() && self->is_fp16()  ) {
        void *A_ = value_->dcu_fp16()->data();
        void *B_ = data();
        void *C_ = out_->dcu_fp16()->data();
        HIPBLAS_CHECK( hipblasGemmStridedBatchedEx(
                        ComputingContext::hipblas_handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, HIPBLAS_R_16F, m, HfT,
                        B_, HIPBLAS_R_16F, k, TT,
                        &beta, C_, HIPBLAS_R_16F, m, HnT,
                        batch*heads, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT) );
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  DCUTensor<DT>::op_gelu(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::dcu_stream;

    if ( DT == DataType::Float ) {
        float* src = (float *)data();
        float* dst = (float *)out->dcu_float()->data();

        dcu::kr_gelu<float>(src, dst, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::Float ) {
        device_fp16_t* src = (device_fp16_t *)data();
        device_fp16_t* dst = (device_fp16_t *)out->dcu_fp16()->data();

        dcu::kr_gelu<device_fp16_t>(src, dst, self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  DCUTensor<DT>::op_silu_product(tensor_t self, tensor_t in, tensor_t out) {
    auto stream = ComputingContext::dcu_stream;

    if ( DT == DataType::Float ) {
        float* src = (float *)data();
        float* in_ = (float *)in->dcu_float()->data();
        float* dst = (float *)out->dcu_float()->data();

        dcu::kr_silu_product<float>(src, in_, dst, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto* src = (device_fp16_t *)data();
        auto* in_ = (device_fp16_t *)in->dcu_fp16()->data();
        auto* dst = (device_fp16_t *)out->dcu_fp16()->data();

        dcu::kr_silu_product<device_fp16_t>(src, in_, dst, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
std::variant<ComputingReturn,int> DCUTensor<DT>::op_all_logits(tensor_t self, tensor_t mask_,  tensor_t lm_head, tensor_t output) {
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

    int* mask = (int *)mask_->host_int()->data();
    int pred = 0;
    for (int b = 0;  b < batch; b++) {
        int* mk = &mask[b * full_tokens];
        for ( int i = 0; i < new_tokens ; i++) {
            int ii = full_tokens - new_tokens + i;
            if ( mk[ii] != 2 ) {
                continue;
            }
            int target = i;

            if ( DT == DataType::Float ) {
                float* dst = (float *)output->dcu_float()->data() + pred * vocab_size;
                float* x = (float *)data() + b * new_tokens * hidden_size + target * hidden_size;

                float* A = (float *)lm_head->dcu_float()->data();
                float* B = x;
                float* C = dst;

                HIPBLAS_CHECK( hipblasGemmEx(ComputingContext::hipblas_handle,
                    HIPBLAS_OP_T, HIPBLAS_OP_N,
                    m, n, k,
                    &alpha, A, HIPBLAS_R_32F, k,
                    B, HIPBLAS_R_32F, k, &beta,
                    C, HIPBLAS_R_32F, m,
                    HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT));

            } else if ( DT == DataType::FP16 ) {
                auto* dst = (device_fp16_t *)output->dcu_fp16()->data() + pred * vocab_size;
                auto* x = (device_fp16_t *)data() + b * new_tokens * hidden_size + target * hidden_size;

                auto* A = (device_fp16_t *)lm_head->dcu_fp16()->data();
                auto* B = x;
                auto* C = dst;

                HIPBLAS_CHECK( hipblasGemmEx(ComputingContext::hipblas_handle,
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

template<DataType DT>
std::variant<ComputingReturn, tensor_t>  DCUTensor<DT>::op_sampling_top3(tensor_t self, float temp) {
    if ( DT == DataType::FP16 ) {
        int batch = self->shape()[0];
        int vocab_size = self->shape()[1];

        std::vector<size_t> ret_shape{ (size_t)batch};
        tensor_t ret = vt::create_host_int( ret_shape );

        auto stream = ComputingContext::dcu_stream;
        int* out = (int *)ComputingContext::dcu_workspace;
        device_fp16_t* logits = (device_fp16_t *) self->device_data();

        std::uniform_real_distribution<> dist(0.0, 1.0);
        float randx = dist( *ComputingContext::rng );

        dcu::kr_easy_top3<device_fp16_t>(logits, out, batch, vocab_size, temp, randx, stream);

        HIP_CHECK(hipMemcpyAsync( ret->device_data(), out, batch * sizeof(int), hipMemcpyDeviceToHost, stream));

        return ret;
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
