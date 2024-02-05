#include "corex_tensor.hpp"
#include "host_tensor.hpp"
#include <corex_kernels.hpp>

namespace vt {

using device_fp16_t = __half;

template<DataType _DTYPE_>
CXTensor<_DTYPE_>::CXTensor(const ShapeType& shape) : owner_(true), PQ_S_(0) {
    if ( _DTYPE_ == DataType::Float ) {
        COREX_CHECK(cudaMalloc(&mem_, shape.numel() * sizeof(float)));
    } else if ( _DTYPE_ == DataType::Int ) {
        COREX_CHECK(cudaMalloc(&mem_, shape.numel() * sizeof(int)));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        COREX_CHECK(cudaMalloc(&mem_, shape.numel() * sizeof(device_fp16_t)));
    } else if ( _DTYPE_ == DataType::Q8 ) {
        size_t last_dim = shape.vec().back();
        size_t feature_num = shape.numel() / last_dim;
        last_dim += sizeof(float) * 2;
        COREX_CHECK(cudaMalloc(&mem_, last_dim * feature_num));
    } else if ( _DTYPE_ == DataType::Q4 ) {
        size_t last_dim = shape.vec().back();
        vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");

        size_t numel = shape.numel();
        size_t blk_num = numel / Q4_BLOCK_SIZE;
        COREX_CHECK(cudaMalloc(&mem_, blk_num * sizeof( q4_block_t )));
    } else if ( _DTYPE_ == DataType::PQ ) {
        vt_panic("Don't use this constructor for COREX PQ");
    } else {
        vt_panic("Don't support DataType for COREX");
    }
}

template<DataType _DTYPE_>
CXTensor<_DTYPE_>::CXTensor(const ShapeType& shape, int S) : owner_(true) ,  PQ_S_(S) {
    if ( _DTYPE_ != DataType::PQ ) {
        vt_panic("Can't be here!");
    }
    size_t items = shape.numel();
    vt_assert( items % (8 * PQ_S_) == 0, "PQ tensor must aligened with config");

    size_t size = sizeof(device_fp16_t) * 128 * PQ_S_ + items * 3 / 8;
    COREX_CHECK(cudaMalloc(&mem_, size));

    PQ_tab_ = (device_fp16_t *)mem_;
    PQ_idx_ = (uint8_t *)mem_ + 128 * PQ_S_ * sizeof(device_fp16_t);
}

template<DataType DT>
ComputingReturn CXTensor<DT>::io_dump(tensor_t self) {
    auto stream = ComputingContext::corex_stream;

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
        COREX_CHECK(cudaMemcpyAsync(mem.data(), src, datasize, cudaMemcpyDeviceToHost, stream));

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
        COREX_CHECK(cudaMemcpyAsync(mem.data(), src, datasize, cudaMemcpyDeviceToHost, stream));

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
        COREX_CHECK(cudaMemcpyAsync(local_data.data(), data(), last_dim, cudaMemcpyDeviceToHost, stream));
        COREX_CHECK(cudaStreamSynchronize(stream));

        const q8_head_t* target = (const q8_head_t *)local_data.data();
        std::cout << "First 8 : ";
        for (size_t i = 0; i < 8; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;

        COREX_CHECK(cudaMemcpyAsync(local_data.data(), (char *) data() + (feature_num - 1) * last_dim, last_dim, cudaMemcpyDeviceToHost, stream));
        last_dim = last_dim - sizeof(float) * 2;
        std::cout << "Last 8 : ";
        for (size_t i = last_dim - 8; i < last_dim; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    if ( DT == DataType::PQ ) {
        local_fp16_t tab[PQ_S_ * 128];
        COREX_CHECK(cudaMemcpyAsync(tab, PQ_tab_, sizeof(device_fp16_t) * PQ_S_ * 128, cudaMemcpyDeviceToHost, stream));

        const uint8_t*  pqidx = PQ_idx_;
        uint8_t v[4];
        COREX_CHECK(cudaMemcpyAsync(v, pqidx, 4, cudaMemcpyDeviceToHost, stream));

        std::cout << "First 8 : ";
        {
            uint8_t a = v[0];
            uint8_t b = v[1];
            uint8_t c = v[2];

            int i0 = a >> 2;
            int i1 = ( (a & 0x3) << 4)  + (b >> 4);
            int i2 = ( (b & 0x0F) << 2) + (c >> 6);
            int i3 = c & 0x3F;

            local_fp16_t * pqtab = &tab[0];
            std::cout << fp16_to_fp32(pqtab[i0*2 + 0]) << " " << fp16_to_fp32(pqtab[i0*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i1*2 + 0]) << " " << fp16_to_fp32(pqtab[i1*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i2*2 + 0]) << " " << fp16_to_fp32(pqtab[i2*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i3*2 + 0]) << " " << fp16_to_fp32(pqtab[i3*2 + 1]) << " ";
        }
        std::cout << std::endl;

        int offset = self->items() * 3 / 8;
        COREX_CHECK(cudaMemcpyAsync(v, &pqidx[offset-4], 4, cudaMemcpyDeviceToHost, stream));

        const size_t gsize = self->items() / PQ_S_;
        int s = (self->items() - 1) / gsize;
        std::cout << "Last 8 : ";
        {
            uint8_t a = v[1];
            uint8_t b = v[2];
            uint8_t c = v[3];

            int i0 = a >> 2;
            int i1 = ( (a & 0x3) << 4)  + (b >> 4);
            int i2 = ( (b & 0x0F) << 2) + (c >> 6);
            int i3 = c & 0x3F;

            local_fp16_t * pqtab = &tab[s * 128];
            std::cout << fp16_to_fp32(pqtab[i0*2 + 0]) << " " << fp16_to_fp32(pqtab[i0*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i1*2 + 0]) << " " << fp16_to_fp32(pqtab[i1*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i2*2 + 0]) << " " << fp16_to_fp32(pqtab[i2*2 + 1]) << " ";
            std::cout << fp16_to_fp32(pqtab[i3*2 + 0]) << " " << fp16_to_fp32(pqtab[i3*2 + 1]) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::io_load(tensor_t self, const char* fileName) {
    std::vector<char> src;
    read_data(fileName, src);

    size_t len = std::get<1>(self->op_sizeof(self));
    vt_assert(src.size() == len , "loaded data must has same size");
    void* x = src.data();
    void* y = data();

    auto stream = ComputingContext::corex_stream;
    COREX_CHECK(cudaMemcpyAsync(y, x, src.size(), cudaMemcpyHostToDevice, stream));
    return OP_OK;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> CXTensor<_DTYPE_>::op_sizeof(tensor_t self) {
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
    if ( _DTYPE_ == DataType::PQ ) {
        if ( owner_ == true ) {
            auto shape = self->shape();
            size_t numel = shape.numel();
            size_t size = sizeof(device_fp16_t) * 128 * PQ_S_  + numel * 3 / 8;
            return size;
        } else {
            return OP_INPUT_ERROR;
        }
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_zero(tensor_t self) {
    void *dst = data();
    auto s = std::get<1>(self->op_sizeof(self));
    COREX_CHECK( cudaMemset(dst, 0, s) );
    return OP_OK;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_fill(tensor_t self, float value) {
    auto stream = ComputingContext::corex_stream;
    if ( DT == DataType::Float ) {
        float* target = (float *)data();
        corex::kr_fill<float>(target, value, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        int* target = (int *)data();
        corex::kr_fill<int>(target, value, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t* target = (device_fp16_t *)data();
        corex::kr_fill(target, value, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_alibi(tensor_t self) {
    int heads = self->shape()[1];
    int tokens = self->shape()[3];

    auto stream = ComputingContext::corex_stream;
    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        std::vector<float> buffer;
        vt::fill_alibi<float>(buffer, heads, tokens);

        COREX_CHECK(cudaMemcpyAsync(data(), buffer.data(), s,  cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<local_fp16_t> buffer;
        vt::fill_alibi<local_fp16_t>(buffer, heads, tokens);
        COREX_CHECK(cudaMemcpyAsync(data(), buffer.data(), s,  cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template<DataType DT>
ComputingReturn CXTensor<DT>::op_rotary_cache(tensor_t self, float base) {
    if ( DT == DataType::Float ) {
        // building inv_freq
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        auto stream = ComputingContext::corex_stream;
        COREX_CHECK(cudaMemcpyAsync( data(), cos_sin.data(), self->items() * sizeof(float), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];

    int* mask  = (int *)data();
    auto stream = ComputingContext::corex_stream;
    if ( out->dtype() == DataType::Float ) {
        float* dst = (float *)out->cx_float()->data();
        corex::kr_causal_mask<float>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }

    return OP_OUTPUT_ERROR;
}

// ============================================
tensor_t create_cx_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CXTensor<DataType::Float>* tensor = new CXTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cx_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CXTensor<DataType::FP16>* tensor = new CXTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cx_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CXTensor<DataType::Int>* tensor = new CXTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cx_q8(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CXTensor<DataType::Q8>* tensor = new CXTensor<DataType::Q8>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cx_q4(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CXTensor<DataType::Q4>* tensor = new CXTensor<DataType::Q4>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cx_pq(std::vector<size_t>& shape_, int S) {
    ShapeType shape(shape_);
    CXTensor<DataType::PQ>* tensor = new CXTensor<DataType::PQ>(shape, S);
    return std::make_shared<TensorType>(tensor, shape);
}

}
