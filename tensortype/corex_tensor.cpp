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

template<DataType DT>
std::variant<ComputingReturn, tensor_t> CXTensor<DT>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( DT == DataType::Float ) {
        void *newData = (char *)data() + offset * sizeof(float);
        auto* newDcuTensor = new CXTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        void *newData = (char *)data() + offset * sizeof(int);
        auto* newDcuTensor = new CXTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        void *newData = (char *)data() + offset * sizeof(device_fp16_t);
        auto* newDcuTensor = new CXTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Q8 ) {
        auto last_dim = self->shape().vec().back();
        vt_assert(offset % last_dim == 0, "Q8's view must aligen with last dim");
        vt_assert(newShape_.back() == last_dim, "Q8's view must aligen with last dim");

        void *newData = (char *)data() + (offset / last_dim) * ( last_dim + sizeof(float) * 2 );
        auto* newDcuTensor = new CXTensor<DataType::Q8>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Q4 ) {
        vt_assert(offset % Q4_BLOCK_SIZE == 0, "Q4's view must aligen with Q4_BLOCK_T");
        void *newData = (char *)data() + (offset / Q4_BLOCK_SIZE) * sizeof(q4_block_t);
        auto* newDcuTensor = new CXTensor<DataType::Q4>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::PQ ) {
        size_t gsize = self->items() / PQ_S_;
        vt_assert(offset % gsize == 0, "PQ's view must aligen with PQ_S");
        void *newTab = (char *)PQ_tab_ + offset / gsize * 128 * sizeof(device_fp16_t);
        uint8_t *newIdx = PQ_idx_ + offset * 3 / 8;

        size_t newItems = newShape.numel();
        vt_assert(newItems % gsize == 0, "PQ's view must aligen with PQ_S");
        int newS = newItems  / gsize;

        auto* newDcuTensor = new CXTensor<DataType::PQ>(newShape, newTab, newIdx, newS);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }

    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> CXTensor<_DT_>::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
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
        auto* newDcuTensor = new CXTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        auto* newDcuTensor = new CXTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        auto* newDcuTensor = new CXTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newDcuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
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
ComputingReturn CXTensor<_DTYPE_>::op_quantize(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::corex_stream;
    if ( _DTYPE_ == DataType::Float && out->is_q4() ) {
        const float* src = (float *)data();
        void* dst = out->cx_q4()->data();
        corex::kr_quantize_q4<float>(src, dst, self->items(), stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::FP16 && out->is_q4() ) {
        const device_fp16_t* src = (device_fp16_t *)data();
        void* dst = out->cx_q4()->data();
        corex::kr_quantize_q4<device_fp16_t>(src, dst, self->items(), stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::Float && out->is_q8() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        const float* src = (float *)data();
        void* dst = out->cx_q8()->data();
        corex::kr_quantize_q8<float>(src, dst, feature_num, feature_size, stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::FP16 && out->is_q8() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        const device_fp16_t* src = (device_fp16_t *)data();
        void* dst = out->cx_q8()->data();
        corex::kr_quantize_q8<device_fp16_t>(src, dst, feature_num, feature_size, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CXTensor<_DTYPE_>::op_dequantize(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::corex_stream;
    if ( _DTYPE_ == DataType::Q4 && out->is_fp16() ) {
        void* src = data();
        device_fp16_t* dst =(device_fp16_t *) out->cx_fp16()->data();
        corex::kr_dequantize_q4<device_fp16_t>(src, dst, self->items(), stream);

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Q8 && out->is_fp16() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;
        void* src = data();
        device_fp16_t* dst =(device_fp16_t *) out->cx_fp16()->data();
        corex::kr_dequantize_q8<device_fp16_t>(src, dst, feature_num, feature_size, stream);

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::PQ && out->is_fp16() ) {
        device_fp16_t* dst =(device_fp16_t *) out->cx_fp16()->data();

        corex::kr_dequantize_pq<device_fp16_t>((device_fp16_t *)PQ_tab_, PQ_idx_, dst, self->items(), PQ_S_, stream);

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CXTensor<_DTYPE_>::op_embed(tensor_t self, tensor_t table, tensor_t outspace) {
    size_t batch = self->shape()[0];
    size_t len = self->shape()[1];
    size_t hidden = table->shape()[1];

    auto stream = ComputingContext::corex_stream;
    int* text = (int *)data();

    if ( table->dtype() == DataType::Float ) {
        float* from = (float *)table->cx_float()->data();
        float* out = (float *)outspace->cx_float()->data();
        corex::kr_embed<float>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    if ( table->dtype() == DataType::FP16 ) {
        device_fp16_t* from = (device_fp16_t *)table->cx_fp16()->data();
        device_fp16_t* out = (device_fp16_t *)outspace->cx_fp16()->data();
        corex::kr_embed<device_fp16_t>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_copy(tensor_t self, tensor_t src) {
    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        if ( src->is_host() ) {
            void* x = src->host_float()->data();
            void* y = data();

            auto stream = ComputingContext::corex_stream;
            COREX_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::corex_stream;
        COREX_CHECK(cudaMemcpyAsync(data(), src->cx_float()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        if ( src->is_host() ) {
            void* x = src->host_int()->data();
            void* y = data();

            auto stream = ComputingContext::corex_stream;
            COREX_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::corex_stream;
        COREX_CHECK(cudaMemcpyAsync(data(), src->cx_int()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        if ( src->is_host() ) {
            void* x = src->host_fp16()->data();
            void* y = data();

            auto stream = ComputingContext::corex_stream;
            COREX_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::corex_stream;
        COREX_CHECK(cudaMemcpyAsync(data(), src->cx_fp16()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q8 ) {
        if ( src->is_host() ) {
            void* x = src->host_q8()->data();
            void* y = data();

            auto stream = ComputingContext::corex_stream;
            COREX_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::corex_stream;
        COREX_CHECK(cudaMemcpyAsync(data(), src->cx_q8()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q4 ) {
        if ( src->is_host() ) {
            void* x = src->host_q4()->data();
            void* y = data();

            auto stream = ComputingContext::corex_stream;
            COREX_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::corex_stream;
        COREX_CHECK(cudaMemcpyAsync(data(), src->cx_q4()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_convert(tensor_t self, tensor_t src) {
    if ( DT == DataType::FP16 && src->is_float() ) {
        auto stream = ComputingContext::corex_stream;
        corex::kr_convert<float, device_fp16_t>((float *)src->cx_float()->data(), (device_fp16_t *)data(), self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_scale(tensor_t self, float scale) {
    auto stream = ComputingContext::corex_stream;
    size_t n = self->items();
    if (   DT == DataType::Float) {
        float* src = (float *)data();
        float* dst = (float *)data();
        corex::kr_scale<float>(src, dst, scale, 0.0, n, stream);
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        device_fp16_t* src = (device_fp16_t *)data();
        device_fp16_t* dst = (device_fp16_t *)data();
        corex::kr_scale<device_fp16_t>(src, dst, scale, 0.0, n, stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    auto stream = ComputingContext::corex_stream;
    if ( self->items() == b->items() ) {
        if ( DT == DataType::Float && b->is_float() ) {
            float* A = (float *)data();
            float* B = (float *)b->cx_float()->data();
            float* C = (float *)c->cx_float()->data();

            corex::kr_add<float, float>(A, B, C, self->items(), stream);
            return OP_OK;
        }
        if ( DT == DataType::FP16 && b->is_fp16() ) {
            device_fp16_t* A = (device_fp16_t *)data();
            device_fp16_t* B = (device_fp16_t *)b->cx_fp16()->data();
            device_fp16_t* C = (device_fp16_t *)c->cx_fp16()->data();

            corex::kr_add<device_fp16_t, device_fp16_t>(A, B, C, self->items(), stream);

            return OP_OK;
        }
        if ( DT == DataType::FP16 && b->is_float() ) {
            device_fp16_t* A = (device_fp16_t *)data();
            float* B = (float *)b->cx_fp16()->data();
            device_fp16_t* C = (device_fp16_t *)c->cx_fp16()->data();

            corex::kr_add<device_fp16_t, float>(A, B, C, self->items(), stream);
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
                device_fp16_t* B = (device_fp16_t *)b->cx_fp16()->data();
                device_fp16_t* C = (device_fp16_t *)c->cx_fp16()->data();

                corex::kr_add_broadcast<device_fp16_t>(A, B, C, length, inter, feature, stream);
                return OP_OK;
            }
            if ( DT == DataType::Float && b->is_float() ) {
                float* A = (float *)data();
                float* B = (float *)b->cx_float()->data();
                float* C = (float *)c->cx_float()->data();

                corex::kr_add_broadcast<float>(A, B, C, length, inter, feature, stream);
                return OP_OK;
            }
        }
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    vt_assert( self->items() == b->items() , " COREX's add don't support brodcast");
    auto stream = ComputingContext::corex_stream;
    if ( DT == DataType::Float && b->is_float() ) {
        float* A = (float *)data();
        float* B = (float *)b->cx_float()->data();
        float* C = (float *)c->cx_float()->data();

        corex::kr_mul<float, float>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 && b->is_fp16() ) {
        device_fp16_t* A = (device_fp16_t *)data();
        device_fp16_t* B = (device_fp16_t *)b->cx_fp16()->data();
        device_fp16_t* C = (device_fp16_t *)c->cx_fp16()->data();

        corex::kr_mul<device_fp16_t, device_fp16_t>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 && b->is_float() ) {
        device_fp16_t* A = (device_fp16_t *)data();
        float* B = (float *)b->cx_fp16()->data();
        device_fp16_t* C = (device_fp16_t *)c->cx_fp16()->data();

        corex::kr_mul<device_fp16_t, float>(A, B, C, self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_linear(tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    auto stream = ComputingContext::corex_stream;
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w_->shape()[0];

    float alpha = 1.0;
    float beta = 0.0;

    if ( DT == DataType::Float && w_->is_float() ) {
        void* A = w_->cx_float()->data();
        void* B = data();
        void* C = y_->cx_float()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        CXBLAS_CHECK( cublasGemmEx(ComputingContext::cxblas_handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       m, n, k,
                       &alpha, A, CUDA_R_32F, k,
                       B, CUDA_R_32F, k, &beta,
                       C, CUDA_R_32F, m,
                       CUDA_R_32F, CUBLAS_GEMM_DEFAULT) );

        if ( b_ != nullptr ) {
            corex::kr_add_bias<float>((float *)C, (float *)b_->cx_float()->data(), (float *)C,  n, m, stream);
        }

        return OP_OK;
    }

    if ( DT == DataType::Float && w_->is_q8() ) {
        return OP_TODO_ERROR;
    }

    if ( DT == DataType::Float && w_->is_q4() ) {
        return OP_TODO_ERROR;
    }

    if ( DT == DataType::FP16 && w_->is_fp16() ) {
        void* A = w_->cx_fp16()->data();
        void* B = data();
        void* C = y_->cx_fp16()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        CXBLAS_CHECK( cublasGemmEx(ComputingContext::cxblas_handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       m, n, k,
                       &alpha, A, CUDA_R_16F, k,
                       B, CUDA_R_16F, k, &beta,
                       C, CUDA_R_16F, m,
                       CUDA_R_32F, CUBLAS_GEMM_DEFAULT) );

        if ( b_ != nullptr) {
            corex::kr_add_bias<device_fp16_t>((device_fp16_t *)C, (device_fp16_t *)b_->cx_fp16()->data(), (device_fp16_t *)C, n, m, stream);
        }

        return OP_OK;
    }

    if ( DT == DataType::FP16 && w_->is_q8() ) {
        return OP_TODO_ERROR;
    }

    if ( DT == DataType::FP16 && w_->is_q4() ) {
        return OP_TODO_ERROR;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    auto stream = ComputingContext::corex_stream;

    if ( DT == DataType::Float ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->cx_float();
        auto v = var->cx_float();
        auto s = scale->cx_float();
        auto b = bias->cx_float();
        auto out = y->cx_float();

        corex::kr_layer_norm<float>((float *)out->data(), (float *)v->data(), (float *)m->data(),
                                 (float *)x->data(), (float *)s->data(), (float *)b->data(), batch, hidden, eps, stream);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->cx_fp16();
        auto v = var->cx_fp16();
        auto s = scale->cx_fp16();
        auto b = bias->cx_fp16();
        auto out = y->cx_fp16();

        corex::kr_layer_norm<__half>((__half *)out->data(), (__half *)v->data(), (__half *)m->data(),
                                 (__half *)x->data(), (__half *)s->data(), (__half *)b->data(), batch, hidden, eps, stream);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t hidden = self->shape()[2];

    auto stream = ComputingContext::corex_stream;

    if ( DT == DataType::Float) {
        float* norm2_ = (float *)norm2->cx_float()->data();
        float* feature = (float *)self->cx_float()->data();
        float* w = (float *)scale->cx_float()->data();
        float* out = (float *)y->cx_float()->data();

        corex::kr_rmsnorm<float>(feature, w, out, norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }


   if ( DT == DataType::FP16 ) {
        device_fp16_t* norm2_ = (device_fp16_t *)norm2->cx_fp16()->data();
        device_fp16_t* feature = (device_fp16_t *)self->cx_fp16()->data();
        device_fp16_t* w = (device_fp16_t *)scale->cx_fp16()->data();
        device_fp16_t* out = (device_fp16_t *)y->cx_fp16()->data();

        corex::kr_rmsnorm<device_fp16_t>(feature, w, out, norm2_, batch * tokens, hidden, eps, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CXTensor<DT>::op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos_, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

    vt_assert(hidden == cached->shape()[1], "heads number must be same with cache");

    auto stream = ComputingContext::corex_stream;
    int* pos = (int*) pos_->cx_int()->data();
    if ( DT == DataType::Float ) {

        float* in = (float *)data();
        float* cos_sin = (float *)cached->cx_float()->data();
        float* out = (float *)y->cx_float()->data();

        corex::kr_rotary_embed<float>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t* in = (device_fp16_t *)data();
        device_fp16_t* out = (device_fp16_t *)y->cx_fp16()->data();
        float* cos_sin = (float *)cached->cx_float()->data();

        corex::kr_rotary_embed<device_fp16_t>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CXTensor<DT>::op_transpose_0213(tensor_t self, tensor_t y) {
    auto x = this;
    auto stream = ComputingContext::corex_stream;

    int sz0 = self->shape()[0];
    int sz1 = self->shape()[1];
    int sz2 = self->shape()[2];
    int sz3 = self->shape()[3];

    if ( DT == DataType::Float ) {
        auto out = y->cx_float();
        corex::kr_transpose_0213<float>((float *)x->data(), (float *)out->data(), sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto out = y->cx_fp16();
        corex::kr_transpose_0213<device_fp16_t>((device_fp16_t *)x->data(), (device_fp16_t *)out->data(), sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CXTensor<DT>::op_qk(tensor_t self, tensor_t k_, tensor_t qk_) {
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
        void *A_ = k_->cx_float()->data();
        void *B_ = data();
        void *C_ = qk_->cx_float()->data();

        CXBLAS_CHECK( cublasGemmStridedBatchedEx(
                        ComputingContext::cxblas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, CUDA_R_32F, k, HnT,
                        B_, CUDA_R_32F, k, HfT,
                        &beta, C_, CUDA_R_32F, m, TT,
                        batch*heads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT) );

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        void *A_ = k_->cx_fp16()->data();
        void *B_ = data();
        void *C_ = qk_->cx_float()->data();

        CXBLAS_CHECK( cublasGemmStridedBatchedEx(
                        ComputingContext::cxblas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        m, n, k,
                        &alpha, A_, CUDA_R_16F, k, HnT,
                        B_, CUDA_R_16F, k, HfT,
                        &beta, C_, CUDA_R_32F, m, TT,
                        batch*heads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
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
