#include <cmath>
#include <utility>
#include <algorithm>

#include <kernels.hpp>
#include "vt.hpp"
#include "context.hpp"
#include "cuda_tensor.hpp"
#include "host_tensor.hpp"

namespace vt {

using device_fp16_t = __half;


template<DataType _DTYPE_>
CUDATensor<_DTYPE_>::CUDATensor(const ShapeType& shape) : owner_(true) {
    if ( _DTYPE_ == DataType::Float ) {
        CUDA_CHECK(cudaMalloc(&mem_, shape.numel() * sizeof(float)));
    } else if ( _DTYPE_ == DataType::Int ) {
        CUDA_CHECK(cudaMalloc(&mem_, shape.numel() * sizeof(int)));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        CUDA_CHECK(cudaMalloc(&mem_, shape.numel() * sizeof(device_fp16_t)));
    } else if ( _DTYPE_ == DataType::Q8 ) {
        size_t last_dim = shape.vec().back();
        size_t feature_num = shape.numel() / last_dim;
        last_dim += sizeof(float) * 2;
        CUDA_CHECK(cudaMalloc(&mem_, last_dim * feature_num));
    } else if ( _DTYPE_ == DataType::Q4 ) {
        size_t last_dim = shape.vec().back();
        vt_assert( (last_dim % Q4_BLOCK_SIZE) == 0, "Q4 tensor last dim must be 32 aligened.");

        size_t numel = shape.numel();
        size_t blk_num = numel / Q4_BLOCK_SIZE;
        CUDA_CHECK(cudaMalloc(&mem_, blk_num * sizeof( q4_block_t )));
    } else {
        vt_panic("Don't support DataType for CUDA");
    }
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_dump(tensor_t self) {
    size_t first8 = std::min(self->items(), (size_t)8);
    auto stream = ComputingContext::cuda_stream;

    if ( DT == DataType::Float ) {
        std::vector<float> local_first;
        std::vector<float> local_last;

        local_first.resize(first8, 0);
        local_last.resize(first8, 0);

        float *x = (float *)self->cuda_float()->data();
        CUDA_CHECK(cudaMemcpyAsync(local_first.data(), x, local_first.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        std::vector<size_t> pos = self->shape().vec();
        auto shape_ = self->shape().vec();
        for(int i = 0; i < (int)pos.size() - 1; i++) {
            pos[i] = shape_[i] - 1;
        }
        pos.back() = shape_.back() - first8;
        float* src = x + (self->items() - first8);
        CUDA_CHECK(cudaMemcpyAsync(local_last.data(), src, local_last.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_first[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_last[i] << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<local_fp16_t> local_first;
        std::vector<local_fp16_t> local_last;

        local_first.resize(first8, 0);
        local_last.resize(first8, 0);

        auto x = self->cuda_fp16();
        CUDA_CHECK(cudaMemcpyAsync(local_first.data(), x->data(), local_first.size() * sizeof(local_fp16_t), cudaMemcpyDeviceToHost, stream));

        std::vector<size_t> pos = self->shape().vec();
        auto shape_ = self->shape().vec();
        for(int i = 0; i < (int)pos.size() - 1; i++) {
            pos[i] = shape_[i] - 1;
        }
        pos.back() = shape_.back() - first8;
        void* src = (device_fp16_t *)x->data() + self->items() - first8;
        CUDA_CHECK(cudaMemcpyAsync(local_last.data(), src, local_last.size() * sizeof(local_fp16_t), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(local_first[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(local_last[i]) << " ";
        }
        std::cout << std::endl;
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        std::vector<int> local_first;
        std::vector<int> local_last;

        local_first.resize(first8, 0);
        local_last.resize(first8, 0);

        auto x = self->cuda_int();
        CUDA_CHECK(cudaMemcpyAsync(local_first.data(), x->data(), local_first.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));

        std::vector<size_t> pos = self->shape().vec();
        auto shape_ = self->shape().vec();
        for(int i = 0; i < (int)pos.size() - 1; i++) {
            pos[i] = shape_[i] - 1;
        }
        pos.back() = shape_.back() - first8;
        void* src = (int *)x->data() + self->items() - first8;
        CUDA_CHECK(cudaMemcpyAsync(local_last.data(), src, local_last.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_first[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << local_last[i] << " ";
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
        CUDA_CHECK(cudaMemcpyAsync(local_data.data(), data(), last_dim, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const q8_head_t* target = (const q8_head_t *)local_data.data();
        std::cout << "First " << first8 << " : ";
        for (size_t i = 0; i < 8; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;

        CUDA_CHECK(cudaMemcpyAsync(local_data.data(), (char *) data() + (feature_num - 1) * last_dim, last_dim, cudaMemcpyDeviceToHost, stream));
        last_dim = last_dim - sizeof(float) * 2;
        std::cout << "Last " << first8 << " : ";
        for (size_t i = last_dim - 8; i < last_dim; i++) {
            std::cout << dequantize_q8(target, i) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( DT == DataType::Q4 ) {
        size_t block_num = self->items() / Q4_BLOCK_SIZE;

        q4_block_t local_block;
        CUDA_CHECK(cudaMemcpyAsync(&local_block, data(), sizeof(q4_block_t), cudaMemcpyDeviceToHost, stream));

        std::cout << "First 8 : ";
        for (size_t i = 0; i < 8; i++) {
            std::cout << dequantize_q4(&local_block, i) << " ";
        }
        std::cout << std::endl;

        void *src = (char *)data() + sizeof(q4_block_t) * ( block_num - 1);
        CUDA_CHECK(cudaMemcpyAsync(&local_block, src, sizeof(q4_block_t), cudaMemcpyDeviceToHost, stream));

        std::cout << "Last 8 : ";
        for (size_t i = Q4_BLOCK_SIZE - 8; i < Q4_BLOCK_SIZE; i++) {
            std::cout << dequantize_q4(&local_block, i) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_load(tensor_t self, const char* fileName) {
    if ( DT == DataType::Float ) {
        std::vector<float> src;
        read_data(fileName, src);

        vt_assert(src.size() == self->items() , "loaded data must has same size");
        void* x = src.data();
        void* y = data();

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(y, x, src.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        std::vector<int> src;
        read_data(fileName, src);

        vt_assert(src.size() == self->items() , "loaded data must has same size");
        void* x = src.data();
        void* y = data();

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(y, x, src.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<unsigned short> src;
        read_data(fileName, src);

        vt_assert(src.size() == self->items() , "loaded data must has same size");
        void* x = src.data();
        void* y = data();

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(y, x, src.size() * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q8 ) {
        std::vector<char> src;
        read_data(fileName, src);

        size_t len = std::get<1>(self->op_sizeof(self));
        vt_assert(src.size() == len , "loaded data must has same size");
        void* x = src.data();
        void* y = data();

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(y, x, src.size(), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }

    if ( DT == DataType::Q4 ) {
        std::vector<q4_block_t> src;
        read_data(fileName, src);

        vt_assert(src.size() == self->items() / Q4_BLOCK_SIZE , "loaded data must has same size");
        void* x = src.data();
        void* y = data();

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(y, x, src.size() * sizeof(q4_block_t), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_nccl_send(tensor_t self, int dst) {
    auto stream = ComputingContext::cuda_stream;
    if ( DT == DataType::Float ) {
        NCCL_CHECK( ncclSend(data(), self->items(), ncclFloat32, dst,
                             CollectiveContext::nccl_comm,
                             stream) );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        NCCL_CHECK( ncclSend(data(), self->items(), ncclFloat16, dst,
                             CollectiveContext::nccl_comm,
                             stream) );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        NCCL_CHECK( ncclSend(data(), self->items(), ncclInt32, dst,
                             CollectiveContext::nccl_comm,
                             stream) );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::io_nccl_recv(tensor_t self, int dst) {
    auto stream = ComputingContext::cuda_stream;
    if ( DT == DataType::Float ) {
        NCCL_CHECK( ncclRecv(data(), self->items(), ncclFloat32, dst,
                             CollectiveContext::nccl_comm,
                             stream) );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return OP_OK;
    }

    if ( DT == DataType::FP16 ) {
        NCCL_CHECK( ncclRecv(data(), self->items(), ncclFloat16, dst,
                             CollectiveContext::nccl_comm,
                             stream) );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return OP_OK;
    }

    if ( DT == DataType::Int ) {
        NCCL_CHECK( ncclRecv(data(), self->items(), ncclInt32, dst,
                             CollectiveContext::nccl_comm,
                             stream) );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> CUDATensor<_DTYPE_>::op_sizeof(tensor_t self) {
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
ComputingReturn CUDATensor<DT>::op_zero(tensor_t self) {
    void *dst = data();
    auto s = std::get<1>(self->op_sizeof(self));
    CUDA_CHECK( cudaMemset(dst, 0, s) );
    return OP_OK;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_scale(tensor_t self, float scale) {
    if (   DT == DataType::Float
        || DT == DataType::Int
        || DT == DataType::FP16 ) {
        void *dst = data();
        auto desc = create_cudnn_td_with( self->shape().vec() );
        CUDNN_CHECK( cudnnScaleTensor( ComputingContext::cudnn_handle,
                                        desc,
                                        dst,
                                        &scale) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_fill(tensor_t self, float value) {
    if (   DT == DataType::Float
        || DT == DataType::Int
        || DT == DataType::FP16 ) {
        float* dst = (float *)data();
        auto desc = create_cudnn_td_with( {self->items()} );
        CUDNN_CHECK( cudnnSetTensor( ComputingContext::cudnn_handle,
                                        desc,
                                        dst,
                                        &value) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_alibi(tensor_t self) {
    int heads = self->shape()[1];
    int tokens = self->shape()[3];

    auto stream = ComputingContext::cuda_stream;
    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        std::vector<float> buffer;
        vt::fill_alibi<float>(buffer, heads, tokens);

        CUDA_CHECK(cudaMemcpyAsync(data(), buffer.data(), s,  cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<local_fp16_t> buffer;
        vt::fill_alibi<local_fp16_t>(buffer, heads, tokens);
        CUDA_CHECK(cudaMemcpyAsync(data(), buffer.data(), s,  cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_rotary_cache(tensor_t self, float base) {
    if ( DT == DataType::Float ) {
        // building inv_freq
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync( data(), cos_sin.data(), self->items() * sizeof(float), cudaMemcpyHostToDevice, stream));
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];
    auto stream = ComputingContext::cuda_stream;

    int* mask  = (int *)data();
    if ( out->dtype() == DataType::Float ) {
        float* dst = (float *)out->cuda_float()->data();
        cuda::causal_mask<float>(mask, dst, batch, new_tokens, full_tokens, stream);
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_copy(tensor_t self, tensor_t src) {
    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        if ( src->is_host() ) {
            void* x = src->host_float()->data();
            void* y = data();

            auto stream = ComputingContext::cuda_stream;
            CUDA_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_float()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        if ( src->is_host() ) {
            void* x = src->host_int()->data();
            void* y = data();

            auto stream = ComputingContext::cuda_stream;
            CUDA_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_int()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        if ( src->is_host() ) {
            void* x = src->host_fp16()->data();
            void* y = data();

            auto stream = ComputingContext::cuda_stream;
            CUDA_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_fp16()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q8 ) {
        if ( src->is_host() ) {
            void* x = src->host_q8()->data();
            void* y = data();

            auto stream = ComputingContext::cuda_stream;
            CUDA_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_q8()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }
    if ( DT == DataType::Q4 ) {
        if ( src->is_host() ) {
            void* x = src->host_q4()->data();
            void* y = data();

            auto stream = ComputingContext::cuda_stream;
            CUDA_CHECK(cudaMemcpyAsync(y, x, s, cudaMemcpyHostToDevice, stream));
            return OP_OK;
        }
        auto stream = ComputingContext::cuda_stream;
        CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_q4()->data(), s, cudaMemcpyDeviceToDevice, stream));
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_convert(tensor_t self, tensor_t src) {
    if ( DT == DataType::FP16 && src->is_float() ) {
        auto stream = ComputingContext::cuda_stream;
        cuda::float_to_half((float *)src->cuda_float()->data(), (device_fp16_t *)data(), self->items(), stream);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_linear(tensor_t self, tensor_t w_, tensor_t b_, tensor_t y_) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w_->shape()[0];

    float alpha = 1.0;
    float beta = 0.0;

    if ( DT == DataType::Float && w_->is_float() ) {
        void* A = w_->cuda_float()->data();
        void* B = data();
        void* C = y_->cuda_float()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;
        cuda::LtSgemm(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_32F, k,
                B, CUDA_R_32F, k, &beta,
                C, CUDA_R_32F, m,
                ComputingContext::cuda_workspace,
                ComputingContext::workspace_size);

        if ( b_ != nullptr ) {
            auto ydesc = y_->cuda_float()->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b_->cuda_float()->create_cudnn_td_with({1, 1, 1, outSize});
            void* bias = b_->cuda_float()->data();

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
        return OP_OK;
    }

    if ( DT == DataType::Float && w_->is_q8() ) {
        float* src = (float *)data();
        float* dst = (float *)y_->cuda_fp16()->data();
        void* w = w_->cuda_q8()->data();

        //ComputingContext::cuda_event(0);
        cuda::linear2d_q8<float>((float *)src, (void*)w, (float*)dst, batch * tokens, outSize, inSize, ComputingContext::cuda_stream);
        //std::cout << "Kernel using " << ComputingContext::cuda_event(1);

        if ( b_ != nullptr ) {
            auto ydesc = y_->cuda_float()->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b_->cuda_float()->create_cudnn_td_with({1, 1, 1, outSize});
            void* bias = b_->cuda_float()->data();
            void* C = y_->cuda_float()->data();

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }

        return OP_OK;
    }

    if ( DT == DataType::Float && w_->is_q4() ) {
        float* src = (float *)data();
        float* dst = (float *)y_->cuda_fp16()->data();
        void* w = w_->cuda_q4()->data();

        //ComputingContext::cuda_event(0);
        cuda::linear2d_q4<float>((float *)src, (void*)w, (float*)dst, batch * tokens, outSize, inSize, ComputingContext::cuda_stream);
        //std::cout << "Kernel using " << ComputingContext::cuda_event(1);

        if ( b_ != nullptr ) {
            auto ydesc = y_->cuda_float()->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b_->cuda_float()->create_cudnn_td_with({1, 1, 1, outSize});
            void* bias = b_->cuda_float()->data();
            void* C = y_->cuda_float()->data();

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }

        return OP_OK;
    }

    if ( DT == DataType::FP16 && w_->is_fp16() ) {
        void* A = w_->cuda_fp16()->data();
        void* B = data();
        void* C = y_->cuda_fp16()->data();

        int m = outSize;
        int n = batch * tokens;
        int k = inSize;

        cuda::LtSgemm(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_16F, k,
                B, CUDA_R_16F, k, &beta,
                C, CUDA_R_16F, m,
                ComputingContext::cuda_workspace,
                ComputingContext::workspace_size);

        if ( b_ != nullptr ) {
            auto ydesc = y_->cuda_fp16()->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b_->cuda_fp16()->create_cudnn_td_with({1, 1, 1, outSize});
            void* bias = b_->cuda_fp16()->data();

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
        return OP_OK;
    }

    if ( DT == DataType::FP16 && w_->is_q8() ) {
        device_fp16_t* src = (device_fp16_t *)data();
        device_fp16_t* dst = (device_fp16_t *)y_->cuda_fp16()->data();
        void* w = w_->cuda_q8()->data();

        //ComputingContext::cuda_event(0);
        cuda::linear2d_q8<device_fp16_t>((device_fp16_t *)src, (void*)w, (device_fp16_t*)dst, batch * tokens, outSize, inSize, ComputingContext::cuda_stream);
        //std::cout << "Kernel using " << ComputingContext::cuda_event(1);


        if ( b_ != nullptr ) {
            auto ydesc = y_->cuda_fp16()->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b_->cuda_fp16()->create_cudnn_td_with({1, 1, 1, outSize});
            void* bias = b_->cuda_fp16()->data();
            void* C = y_->cuda_fp16()->data();

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
        return OP_OK;
    }


    if ( DT == DataType::FP16 && w_->is_q4() ) {
        device_fp16_t* src = (device_fp16_t *)data();
        device_fp16_t* dst = (device_fp16_t *)y_->cuda_fp16()->data();
        void* w = w_->cuda_q4()->data();

        //ComputingContext::cuda_event(0);
        cuda::linear2d_q4<device_fp16_t>((device_fp16_t *)src, (void*)w, (device_fp16_t*)dst, batch * tokens, outSize, inSize, ComputingContext::cuda_stream);
        //std::cout << "Kernel using " << ComputingContext::cuda_event(1);

        if ( b_ != nullptr ) {
            auto ydesc = y_->cuda_fp16()->create_cudnn_td_with({batch, 1, tokens, outSize});
            auto bdesc = b_->cuda_fp16()->create_cudnn_td_with({1, 1, 1, outSize});
            void* bias = b_->cuda_fp16()->data();
            void* C = y_->cuda_fp16()->data();

            beta = 1.0;
            CUDNN_CHECK( cudnnAddTensor(ComputingContext::cudnn_handle,
                                        &alpha, bdesc, bias,
                                        &beta, ydesc, C));
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t> CUDATensor<DT>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);

    if ( DT == DataType::Float ) {
        void *newData = (char *)data() + offset * sizeof(float);
        auto* newCudaTensor = new CUDATensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        void *newData = (char *)data() + offset * sizeof(int);
        auto* newCudaTensor = new CUDATensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        void *newData = (char *)data() + offset * sizeof(device_fp16_t);
        auto* newCudaTensor = new CUDATensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    if ( DT == DataType::Q8 ) {
        auto last_dim = self->shape().vec().back();
        vt_assert(offset % last_dim == 0, "Q8's view must aligen with last dim");
        vt_assert(newShape_.back() == last_dim, "Q8's view must aligen with last dim");

        void *newData = (char *)data() + (offset / last_dim) * ( last_dim + sizeof(float) * 2 );
        auto* newCudaTensor = new CUDATensor<DataType::Q8>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    if ( DT == DataType::Q4 ) {
        vt_assert(offset % Q4_BLOCK_SIZE == 0, "Q4's view must aligen with Q4_BLOCK_T");
        void *newData = (char *)data() + (offset / Q4_BLOCK_SIZE) * sizeof(q4_block_t);
        auto* newCudaTensor = new CUDATensor<DataType::Q4>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> CUDATensor<_DT_>::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
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
        auto* newCudaTensor = new CUDATensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        auto* newCudaTensor = new CUDATensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        auto* newCudaTensor = new CUDATensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newCudaTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);

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
ComputingReturn CUDATensor<_DTYPE_>::op_quantize(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::cuda_stream;
    if ( _DTYPE_ == DataType::Float && out->is_q4() ) {
        const float* src = (float *)data();
        void* dst = out->cuda_q4()->data();
        cuda::quantize_q4<float>(src, dst, self->items(), stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::FP16 && out->is_q4() ) {
        const device_fp16_t* src = (device_fp16_t *)data();
        void* dst = out->cuda_q4()->data();
        cuda::quantize_q4<device_fp16_t>(src, dst, self->items(), stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::Float && out->is_q8() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        const float* src = (float *)data();
        void* dst = out->cuda_q8()->data();
        cuda::quantize_q8<float>(src, dst, feature_num, feature_size, stream);
        return OP_OK;
    }

    if ( _DTYPE_ == DataType::FP16 && out->is_q8() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        const device_fp16_t* src = (device_fp16_t *)data();
        void* dst = out->cuda_q8()->data();
        cuda::quantize_q8<device_fp16_t>(src, dst, feature_num, feature_size, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CUDATensor<_DTYPE_>::op_dequantize(tensor_t self, tensor_t out) {
    auto stream = ComputingContext::cuda_stream;
    if ( _DTYPE_ == DataType::Q4 && out->is_fp16() ) {
        void* src = data();
        device_fp16_t* dst =(device_fp16_t *) out->cuda_fp16()->data();

        //ComputingContext::cuda_event(0);
        cuda::dequantize_q4<device_fp16_t>(src, dst, self->items(), stream);
        //std::cout << "Kernel using " << ComputingContext::cuda_event(1);

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Q8 && out->is_fp16() ) {
        size_t feature_size = self->shape().vec().back();
        size_t feature_num = self->items() / feature_size;

        //ComputingContext::cuda_event(0);
        void* src = data();
        device_fp16_t* dst =(device_fp16_t *) out->cuda_fp16()->data();
        cuda::dequantize_q8<device_fp16_t>(src, dst, feature_num, feature_size, stream);
        //std::cout << "Kernel using " << ComputingContext::cuda_event(1);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}


template <DataType _DTYPE_>
ComputingReturn CUDATensor<_DTYPE_>::op_embed(tensor_t self, tensor_t table, tensor_t outspace) {
    size_t batch = self->shape()[0];
    size_t len = self->shape()[1];
    size_t hidden = table->shape()[1];

    auto stream = ComputingContext::cuda_stream;
    int* text = (int *)data();

    if ( table->dtype() == DataType::Float ) {
        float* from = (float *)table->cuda_float()->data();
        float* out = (float *)outspace->cuda_float()->data();
        cuda::embed_forward<float>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    if ( table->dtype() == DataType::FP16 ) {
        device_fp16_t* from = (device_fp16_t *)table->cuda_fp16()->data();
        device_fp16_t* out = (device_fp16_t *)outspace->cuda_fp16()->data();
        cuda::embed_forward<device_fp16_t>(text, from, out, batch*len, hidden, stream);

        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    float alpha = 1.0;
    float beta = 0.0;
    cudnnOpTensorDescriptor_t opTensorDesc;

    CUDNN_CHECK( cudnnCreateOpTensorDescriptor(&opTensorDesc) );
    if ( DT == DataType::Float ) {
        auto adesc = create_cudnn_td_with( self->shape().vec() );
        auto bdesc = b->cuda_float()->create_cudnn_td_with( b->shape().vec() );
        auto cdesc = c->cuda_float()->create_cudnn_td_with( c->shape().vec() );

        CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );
        CUDNN_CHECK( cudnnOpTensor(ComputingContext::cudnn_handle,
                                    opTensorDesc,
                                    &alpha, adesc, data(),
                                    &alpha, bdesc, b->cuda_float()->data(),
                                    &beta,  cdesc, c->cuda_float()->data()) );

        CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto adesc = create_cudnn_td_with( self->shape().vec() );
        auto bdesc = b->cuda_fp16()->create_cudnn_td_with( b->shape().vec() );
        auto cdesc = c->cuda_fp16()->create_cudnn_td_with( c->shape().vec() );

        CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );
        CUDNN_CHECK( cudnnOpTensor(ComputingContext::cudnn_handle,
                                    opTensorDesc,
                                    &alpha, adesc, data(),
                                    &alpha, bdesc, b->cuda_fp16()->data(),
                                    &beta,  cdesc, c->cuda_fp16()->data()) );

        CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    float alpha = 1.0;
    float beta = 0.0;
    cudnnOpTensorDescriptor_t opTensorDesc;

    CUDNN_CHECK( cudnnCreateOpTensorDescriptor(&opTensorDesc) );
    if ( DT == DataType::Float ) {
        auto adesc = create_cudnn_td_with( self->shape().vec() );
        auto bdesc = b->cuda_float()->create_cudnn_td_with( b->shape().vec() );
        auto cdesc = c->cuda_float()->create_cudnn_td_with( c->shape().vec() );

        CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );
        CUDNN_CHECK( cudnnOpTensor(ComputingContext::cudnn_handle,
                                    opTensorDesc,
                                    &alpha, adesc, data(),
                                    &alpha, bdesc, b->cuda_float()->data(),
                                    &beta,  cdesc, c->cuda_float()->data()) );

        CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto adesc = create_cudnn_td_with( self->shape().vec() );
        auto bdesc = b->cuda_fp16()->create_cudnn_td_with( b->shape().vec() );
        auto cdesc = c->cuda_fp16()->create_cudnn_td_with( c->shape().vec() );

        CUDNN_CHECK( cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN) );
        CUDNN_CHECK( cudnnOpTensor(ComputingContext::cudnn_handle,
                                    opTensorDesc,
                                    &alpha, adesc, data(),
                                    &alpha, bdesc, b->cuda_fp16()->data(),
                                    &beta,  cdesc, c->cuda_fp16()->data()) );

        CUDNN_CHECK( cudnnDestroyOpTensorDescriptor(opTensorDesc) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    if ( DT == DataType::Float ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->cuda_float();
        auto v = var->cuda_float();
        auto s = scale->cuda_float();
        auto b = bias->cuda_float();
        auto out = y->cuda_float();

        // TODO using eps inside kernel
        auto stream = ComputingContext::cuda_stream;
        lightseq::cuda::launch_layer_norm<float>((float *)out->data(), (float *)v->data(), (float *)m->data(),
                                 (float *)x->data(), (float *)s->data(), (float *)b->data(), batch, hidden, stream);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto x = this;
        size_t batch = self->shape()[0] * self->shape()[1];
        size_t hidden = self->shape()[2];

        auto m = mean->cuda_fp16();
        auto v = var->cuda_fp16();
        auto s = scale->cuda_fp16();
        auto b = bias->cuda_fp16();
        auto out = y->cuda_fp16();

        // TODO using eps inside kernel
        auto stream = ComputingContext::cuda_stream;
        lightseq::cuda::launch_layer_norm<__half>((__half *)out->data(), (__half *)v->data(), (__half *)m->data(),
                                 (__half *)x->data(), (__half *)s->data(), (__half *)b->data(), batch, hidden, stream);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t hidden = self->shape()[2];

    if ( DT != DataType::Float && DT != DataType::FP16) {
        return OP_TODO_ERROR;
    }


   if ( DT == DataType::FP16 ) {
        device_fp16_t* norm2_ = (device_fp16_t *)norm2->cuda_fp16()->data();
        device_fp16_t* feature = (device_fp16_t *)self->cuda_fp16()->data();
        device_fp16_t* w = (device_fp16_t *)scale->cuda_fp16()->data();
        device_fp16_t* out = (device_fp16_t *)y->cuda_fp16()->data();

        auto stream = ComputingContext::cuda_stream;
        cuda::rms_norm<device_fp16_t>(feature, w, out, norm2_, batch * tokens, hidden, eps, stream);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos_, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

    vt_assert(hidden == cached->shape()[1], "heads number must be same with cache");

    int* pos = (int*) pos_->cuda_int()->data();
    if ( DT == DataType::Float ) {

        float* in = (float *)data();
        float* cos_sin = (float *)cached->cuda_float()->data();
        float* out = (float *)y->cuda_float()->data();

        auto stream = ComputingContext::cuda_stream;
        cuda::rotary_embed<float>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t* in = (device_fp16_t *)data();
        device_fp16_t* out = (device_fp16_t *)y->cuda_fp16()->data();
        float* cos_sin = (float *)cached->cuda_float()->data();

        auto stream = ComputingContext::cuda_stream;
        cuda::rotary_embed<device_fp16_t>(in, cos_sin, pos, out, batch, heads, tokens, hidden, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_transpos_0213(tensor_t self, tensor_t y) {
    auto x = this;
    auto stream = ComputingContext::cuda_stream;

    int sz0 = self->shape()[0];
    int sz1 = self->shape()[1];
    int sz2 = self->shape()[2];
    int sz3 = self->shape()[3];

    if ( DT == DataType::Float ) {
        auto out = y->cuda_float();
        lightseq::cuda::launch_transform_0213<float>((float *)x->data(), (float *)out->data(), sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto out = y->cuda_fp16();
        lightseq::cuda::launch_transform_0213<device_fp16_t>((device_fp16_t *)x->data(), (device_fp16_t *)out->data(), sz0, sz1, sz2, sz3, stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_qk(tensor_t self, tensor_t k_, tensor_t qk_) {
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

    if ( DT == DataType::Float ) {
#if 0
        int HnT = hhidden * ntokens ;
        int HfT = hhidden * ftokens ;
        int TT = ftokens * ntokens;
        for (int i = 0; i < batch * heads; i++) {
            float* B = (float *)data() + i * HnT;
            float* A = (float *)(k_->cuda_float()->data()) + i * HfT;
            float* C = (float *)(qk_->cuda_float()->data()) + i * TT;
            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_32F, k,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }
#else
        float* B = (float *)data();
        float* A = (float *)(k_->cuda_float()->data());
        float* C = (float *)(qk_->cuda_float()->data());
        cuda::LtSgemmBatched(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_32F, k,
                B, CUDA_R_32F, k, &beta,
                C, CUDA_R_32F, m, batch*heads,
                ComputingContext::cuda_workspace,
                ComputingContext::workspace_size);
#endif
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
#if 0
        for (int i = 0; i < batch * heads; i++) {
            device_fp16_t* B = (device_fp16_t *)data() + i * HnT;
            device_fp16_t* A = (device_fp16_t *)(k_->cuda_fp16()->data()) + i * HfT;
            float* C = (float *)(qk_->cuda_float()->data()) + i * TT;
            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_16F, k,
                    B, CUDA_R_16F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }
#else
        device_fp16_t* B = (device_fp16_t *)data();
        device_fp16_t* A = (device_fp16_t *)(k_->cuda_fp16()->data());
        float* C = (float *)(qk_->cuda_float()->data());
        cuda::LtSgemmBatched(ComputingContext::cublasLt_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_16F, k,
                B, CUDA_R_16F, k, &beta,
                C, CUDA_R_32F, m, batch * heads,
                ComputingContext::cuda_workspace,
                ComputingContext::workspace_size);
#endif
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_softmax(tensor_t self, tensor_t y) {
    if ( DT != DataType::Float && DT != DataType::FP16) {
        return OP_TODO_ERROR;
    }
    float alpha = 1.0;
    float beta = 0.0;

    auto shape_ = self->shape().vec();
    if ( shape_.size() == 4 )  {

        size_t batch = shape_[0];
        size_t heads = shape_[1];
        size_t tokens = shape_[2];
        size_t hidden = shape_[3];

        void* xdata = data();
        void* ydata = y->device_data( self->impl_index() );

        auto xdesc = create_cudnn_td_with({ batch * heads * tokens, hidden, 1, 1});
        auto ydesc = create_cudnn_td_with({ batch * heads * tokens, hidden, 1, 1});
        CUDNN_CHECK( cudnnSoftmaxForward( ComputingContext::cudnn_handle,
                                        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &alpha, xdesc, xdata, &beta, ydesc, ydata) );

        return OP_OK;
    }

    if ( shape_.size() == 2 )  {
        size_t number = shape_[0];
        size_t tokens = shape_[1];

        void* xdata = data();
        void* ydata = y->device_data( self->impl_index() );

        auto xdesc = create_cudnn_td_with({ number, tokens, 1, 1});
        auto ydesc = create_cudnn_td_with({ number, tokens, 1, 1});
        CUDNN_CHECK( cudnnSoftmaxForward( ComputingContext::cudnn_handle,
                                        CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &alpha, xdesc, xdata, &beta, ydesc, ydata) );
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_attn(tensor_t self, tensor_t value_, tensor_t out_) {
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
#if 0
        for (int i = 0; i < batch * heads; i++) {
            float* A = (float *)(value_->cuda_float()->data()) + i * HfT;
            float* B = (float *)data() + i * TT;
            float* C = (float *)(out_->cuda_float()->data()) + i * HnT;

            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_32F, m,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }
#else
        float* A = (float *)(value_->cuda_float()->data()) ;
        float* B = (float *)data() ;
        float* C = (float *)(out_->cuda_float()->data());

        cuda::LtSgemmBatched(ComputingContext::cublasLt_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_32F, m,
                B, CUDA_R_32F, k, &beta,
                C, CUDA_R_32F, m, batch * heads,
                ComputingContext::cuda_workspace,
                ComputingContext::workspace_size);

#endif
        return OP_OK;
    } else if ( value_->is_fp16() && self->is_float() ) {
        vt_assert( ComputingContext::workspace_size > (size_t)TT * 2, "Working memory is not enough!");
        device_fp16_t* half_B = (device_fp16_t*)ComputingContext::cuda_workspace;
        void* workspace = (char *)ComputingContext::cuda_workspace + TT * 2;
        int workspace_size = ComputingContext::workspace_size - TT * 2;

        auto stream = ComputingContext::cuda_stream;
        for (int i = 0; i < batch * heads; i++) {

            float* B_ = (float *)data() + i * TT;
            auto* A = (device_fp16_t *)(value_->cuda_fp16()->data()) + i * HfT;
            auto* C = (device_fp16_t *)(out_->cuda_fp16()->data()) + i * HnT;

            cuda::float_to_half(B_, half_B, TT, stream);
            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_16F, m,
                    half_B, CUDA_R_16F, k, &beta,
                    C, CUDA_R_16F, m,
                    workspace,
                    workspace_size);

        }
        return OP_OK;
    } else if ( value_->is_fp16() && self->is_fp16()  ) {
        auto* B = (device_fp16_t *)data();
        auto* A = (device_fp16_t *)(value_->cuda_fp16()->data());
        auto* C = (device_fp16_t *)(out_->cuda_fp16()->data());

        cuda::LtSgemmBatched(ComputingContext::cublasLt_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha, A, CUDA_R_16F, m,
                B, CUDA_R_16F, k, &beta,
                C, CUDA_R_16F, m, batch * heads,
                ComputingContext::cuda_workspace,
                ComputingContext::workspace_size);

        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_gelu(tensor_t self, tensor_t out) {
    if ( DT == DataType::Float ) {
        float* src = (float *)data();
        float* dst = (float *)out->cuda_float()->data();

        auto stream = ComputingContext::cuda_stream;
        cuda::gelu_forward(src, dst, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  CUDATensor<DT>::op_silu_product(tensor_t self, tensor_t in, tensor_t out) {
    if ( DT == DataType::Float ) {
        float* src = (float *)data();
        float* in_ = (float *)in->cuda_float()->data();
        float* dst = (float *)out->cuda_float()->data();

        auto stream = ComputingContext::cuda_stream;
        cuda::silu_product(src, in_, dst, self->items(), stream);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto* src = (device_fp16_t *)data();
        auto* in_ = (device_fp16_t *)in->cuda_fp16()->data();
        auto* dst = (device_fp16_t *)out->cuda_fp16()->data();

        auto stream = ComputingContext::cuda_stream;
        cuda::silu_product(src, in_, dst, self->items(), stream);
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
std::variant<ComputingReturn,int> CUDATensor<DT>::op_all_logits(tensor_t self, tensor_t mask_,  tensor_t lm_head, tensor_t output) {
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
                float* dst = (float *)output->cuda_float()->data() + pred * vocab_size;
                float* x = (float *)data() + b * new_tokens * hidden_size + target * hidden_size;

                float* A = (float *)lm_head->cuda_float()->data();
                float* B = x;
                float* C = dst;

                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_32F, k,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
            } else if ( DT == DataType::FP16 ) {
                auto* dst = (device_fp16_t *)output->cuda_fp16()->data() + pred * vocab_size;
                auto* x = (device_fp16_t *)data() + b * new_tokens * hidden_size + target * hidden_size;

                auto* A = (device_fp16_t *)lm_head->cuda_fp16()->data();
                auto* B = x;
                auto* C = dst;

                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_16F, k,
                    B, CUDA_R_16F, k, &beta,
                    C, CUDA_R_16F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
            } else {
                return OP_TODO_ERROR;
            }

            pred ++;
        }
     }

    return pred;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t>  CUDATensor<DT>::op_sampling_top3(tensor_t self, float temp) {
    if ( DT != DataType::Float && DT != DataType::FP16 ) {
        return OP_INPUT_ERROR;
    }

    int batch = self->shape()[0];
    int vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_int( ret_shape );

    auto stream = ComputingContext::cuda_stream;
    int* out = (int *)ComputingContext::cuda_workspace;
    device_fp16_t* logits = (device_fp16_t *) self->device_data();

    std::uniform_real_distribution<> dist(0.0, 1.0);
    float randx = dist( *ComputingContext::rng );

    cuda::easy_top3<device_fp16_t>(logits, out, batch, vocab_size, temp, randx, stream);

    CUDA_CHECK(cudaMemcpyAsync( ret->device_data(), out, batch * sizeof(int), cudaMemcpyDeviceToHost, stream));

    return ret;

#if 0
    float scale = 1.0 / temp;
    self->op_scale(self, scale);
    self->op_softmax(self, self);

    int preds = self->shape()[0];
    int vocab_size = self->shape()[1];

    std::vector<int> probs;
    std::vector<float> logits_f32;
    std::vector<local_fp16_t> logits_fp16;

    std::vector< std::pair<int, float> > scores;
    if ( DT == DataType::Float ) {
        logits_f32.resize(vocab_size);
    }
    if ( DT == DataType::FP16 ) {
        logits_fp16.resize(vocab_size);
    }
    scores.resize(vocab_size);

    auto stream = ComputingContext::cuda_stream;

    std::vector<size_t> ret_shape{ (size_t)preds};
    tensor_t ret = vt::create_host_int( ret_shape );

    for (int p = 0;  p < preds; p++) {
        int *dst = (int *)ret->device_data() + p;

        if ( DT == DataType::Float ) {
            float* logits_ = (float *)self->cuda_float()->data() + p * vocab_size;
            CUDA_CHECK(cudaMemcpyAsync(logits_f32.data(), logits_, vocab_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
            for(size_t i = 0; i < vocab_size; i++) {
                scores[i].first = i;
                scores[i].second = logits_f32[i];
            }
        } else if ( DT == DataType::FP16 ) {
            device_fp16_t* logits_ = (device_fp16_t *)self->cuda_fp16()->data() + p * vocab_size;
            CUDA_CHECK(cudaMemcpyAsync(logits_fp16.data(), logits_, vocab_size * sizeof(device_fp16_t), cudaMemcpyDeviceToHost, stream));
            for(size_t i = 0; i < vocab_size; i++) {
                scores[i].first = i;
                scores[i].second = fp16_to_fp32( logits_fp16[i] );
            }
        } else {
            return OP_TODO_ERROR;
        }

        std::sort( scores.begin(), scores.end(),
                [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
                    return a.second > b.second;
                });

        float sum = 0.0;
        probs.clear();
        for (size_t i = 0; i < vocab_size; i++) {
            probs.push_back( scores[i].first );
            sum = sum + scores[i].second;
            if ( sum >= top_p ) {
                break;
            }
        }

        // do random sampling ...
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int idx = dist( *ComputingContext::rng );
        *dst = scores[idx].first;
    }
    return ret;
#endif

}


template<DataType DT>
std::variant<ComputingReturn, float> CUDATensor<DT>::op_loss_backward(tensor_t self, tensor_t ids_, tensor_t mask_, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) {
#if 0
    struct _ {
        static void vocab_embedding(int vsize, int gsize, int hsize, float* lm_head, float* x, float* dst) {
            // computing vocab embedding
            int m = vsize;
            int n = gsize;
            int k = hsize;

            float alpha = 1.0;
            float beta = 0.0;

            float* A = lm_head;
            float* B = x;
            float* C = dst;

            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            m, n, k,
                            &alpha, A, CUDA_R_32F,  k,
                            B, CUDA_R_32F, k, &beta,
                            C, CUDA_R_32F, m,
                            ComputingContext::cuda_workspace,
                            ComputingContext::workspace_size);

        }

        static void vocab_grad_x(int vsize, int gsize, int hsize, float* lm_head, float* dx, float* dst) {
            // computing vocab embedding
            int m = hsize;
            int n = gsize;
            int k = vsize;

            float alpha = 1.0;
            float beta = 0;

            float* A = lm_head;
            float* B = dst;
            float* C = dx;

            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k,
                            &alpha, A, CUDA_R_32F,  m,
                            B, CUDA_R_32F, k, &beta,
                            C, CUDA_R_32F, m,
                            ComputingContext::cuda_workspace,
                            ComputingContext::workspace_size);

        }

        static void vocab_grad_w(int vsize, int gsize, int hsize, float* lm_head_w, float* x, float* dout) {
            // GEMM in CPU
            int m = hsize;
            int n = vsize;
            int k = gsize;

            float alpha = 1.0;
            float beta = 1.0;

            float* A = x;
            float* B = dout;
            float* C = lm_head_w;

            cblas_sgemm(::CblasColMajor, CblasNoTrans, CblasTrans,
                        m, n, k,
                        alpha, A, m,
                        B, n, beta,
                        C, m);
        }

        static float logits_loss(int vsize, int gsize, float loss_scale,  const int* id, float* logsoftmax, float* dout) {
            int split = gsize + 1024;
            split = split - split % 1024;
            vt_assert( split * 2 * sizeof(float) < vt::ComputingContext::workspace_size, " workspace size is too small!" );

            auto stream = ComputingContext::cuda_stream;
            int* id_ = (int *) vt::ComputingContext::cuda_workspace;
            float* out_ = (float *)id_ + split;

            CUDA_CHECK(cudaMemcpyAsync(id_, id, gsize*sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK( cudaMemset(dout, 0, gsize * vsize * sizeof(float)) );
            cuda::nllloss_forward(id_, logsoftmax, out_, dout, gsize, vsize, loss_scale, stream);

            float sum_loss;
            CUDA_CHECK(cudaMemcpyAsync(&sum_loss, out_, sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            return sum_loss;
        }

        static int count_loss_tokens(int batch, int tokens, const int* mask) {
            int loss_items = 0;
            for (int b = 0;  b < batch; b++) {
                const int* m = &mask[b * tokens];

                for(int t = 0; t < tokens - 1; t++) {
                    if ( m[t] != 0 && m[t+1] != 0) {
                        loss_items ++;
                    }
                }
            }

            return loss_items;
        }
    };

    if ( DT == DataType::Float ) {

        const int batch = self->shape()[0];
        const int tokens = self->shape()[1];
        const int hidden_size = self->shape()[2];

        const int vocab_size = lm_head->shape()[0];

        const int group_max_size = vt::ComputingContext::workspace_size / ( vocab_size * sizeof(float) );
        vt_assert(  (int)all_logits->items() / vocab_size > 2 * group_max_size, " logits workspace is not enough !");

        int* mask = (int *)mask_->host_int()->data();
        int* ids = (int *)ids_->host_int()->data();

        double total_loss = 0.0;
        int total_items = _::count_loss_tokens(batch, tokens, mask);
        float loss_scale = 1.0 / (float)total_items;

        float* local_x = new float[self->items()];
        CUDA_CHECK(cudaMemcpyAsync(local_x, data(), self->items() * sizeof(float), cudaMemcpyDeviceToHost, vt::ComputingContext::cuda_stream));

        x_g->op_zero(x_g);
        for (int b = 0;  b < batch; b++) {
            const int* m = &mask[b * tokens];
            const int* id = &ids[b * tokens];

            std::vector<int> id_group;
            for(int t = 0; t < tokens - 1; t++) {
                id_group.push_back(t);

                if ( t == (tokens - 2) || (int)id_group.size() == group_max_size ) {
                    // droped last continued masked tokens
                    while( id_group.size() > 0 ) {
                        if ( m[ id_group.back() ] == 0 ) {
                            id_group.pop_back();
                        } else {
                            break;
                        }
                    }
                    if ( id_group.size() == 0 ) {
                        continue;
                    }

                    int begin_t = id_group[0];
                    int loss_items = 0;
                    for(size_t i = 0; i < id_group.size(); i++) {
                        int tt = id_group[i];

                        if ( m[tt] == 0 || m[tt+1] == 0) {
                            id_group[i] = -100;
                        } else {
                            id_group[i] = id[tt+1];
                            loss_items ++;
                        }
                    }

                    if ( loss_items >  0) {
                        float* x = (float *)data() + b * tokens * hidden_size + begin_t * hidden_size;
                        float* dx = (float *)x_g->cuda_float()->data() + b * tokens * hidden_size + begin_t * hidden_size;
                        float* lm = (float *)lm_head->cuda_float()->data();

                        float* x_ = local_x + b * tokens * hidden_size + begin_t * hidden_size;
                        float* local_dst = (float *)vt::ComputingContext::host_workspace;
                        float* dst_a = (float *)all_logits->cuda_float()->data();
                        float* dst_b = dst_a + id_group.size() * vocab_size;

                        // do vocab embedding
                        _::vocab_embedding(vocab_size, id_group.size(), hidden_size,
                                           lm, x, dst_a);

                        // do log softmax
                        {
                            float alpha = 1.0;
                            float beta = 0.0;
                            auto xdesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            auto ydesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            CUDNN_CHECK( cudnnSoftmaxForward( ComputingContext::cudnn_handle,
                                            CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                                            &alpha, xdesc, dst_a, &beta, ydesc, dst_a) );
                        }

                        // computing final loss
                        float loss = _::logits_loss(vocab_size, id_group.size(), loss_scale, id_group.data(), dst_a, dst_b);
                        total_loss += loss;


                        // log softmax backward
                        {
                            float alpha = 1.0;
                            float beta = 0.0;
                            auto ydesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            auto dxdesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});
                            auto dydesc = create_cudnn_td_with({ id_group.size(), (size_t)vocab_size, 1, 1});

                            CUDNN_CHECK( cudnnSoftmaxBackward( ComputingContext::cudnn_handle,
                                            CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                                            &alpha, ydesc, dst_a, dydesc, dst_b, &beta, dxdesc, dst_b) );


                            CUDA_CHECK(cudaMemcpyAsync(local_dst, dst_b, id_group.size() * vocab_size * sizeof(float), cudaMemcpyDeviceToHost, vt::ComputingContext::cuda_stream));
                        }

                        // computing x_grad
                        _::vocab_grad_x(vocab_size, id_group.size(), hidden_size,
                                        lm, dx, dst_b);


                        // at local (CPU), computing grad for lm_head's weight
                        CUDA_CHECK(cudaStreamSynchronize( vt::ComputingContext::cuda_stream ) );
                        _::vocab_grad_w(vocab_size, id_group.size(), hidden_size,
                                        (float *)lm_head_g->host_float()->data(), x_, local_dst);

                    }
                    id_group.clear();
                }
            }
        }

        delete local_x;

        float ret = total_loss / total_items;
        std::cout << "#################  " << total_items << " "  <<  ret << std::endl;

        return ret;
    }
#endif
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_layernorm_backward(tensor_t self, tensor_t scale_, tensor_t bias_, tensor_t var_, tensor_t y_, tensor_t dscale_, tensor_t dbias_, tensor_t din_, float eps) {
    if ( DT == DataType::Float ) {
        cudaStream_t streams[] = {vt::ComputingContext::cuda_stream, vt::ComputingContext::cuda_stream};

        float* dout = (float *)self->cuda_float()->data();
        float* scale = (float *)scale_->cuda_float()->data();
        float* bias = (float *)bias_->cuda_float()->data();
        float* var = (float *)var_->cuda_float()->data();
        float* y = (float *)y_->cuda_float()->data();
        float* dscale = (float *)dscale_->cuda_float()->data();
        float* dbias = (float *)dbias_->cuda_float()->data();
        float* din = (float *)din_->cuda_float()->data();

        size_t batch = self->shape()[0];
        size_t tokens = self->shape()[1];
        int hidden = self->shape()[2];
        int num = batch * tokens;

        // TODO using eps value
        lightseq::cuda::launch_ln_bw<float>(dscale, dbias, din, dout,
                                    nullptr, y, scale, bias,
                                    var, nullptr,
                                    num, hidden, streams);


        return OP_OK;
    }
    return OP_TODO_ERROR;
}


template<DataType DT>
ComputingReturn CUDATensor<DT>::op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) {
    if ( DT == DataType::Float ) {
        size_t batch = self->shape()[0];
        size_t tokens = self->shape()[1];
        size_t outSize = self->shape()[2];
        size_t inSize = x->shape()[2];

        // do reduce follow batch
        {
            cudnnReduceTensorDescriptor_t reduceDesc;

            CUDNN_CHECK( cudnnCreateReduceTensorDescriptor(&reduceDesc) );
            CUDNN_CHECK( cudnnSetReduceTensorDescriptor(reduceDesc,
                                                        CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                                        CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES) );

            float alpha = 1.0;
            float beta = 0.0;
            auto  adesc = create_cudnn_td_with({batch * tokens, outSize, 1, 1});
            auto  cdesc = create_cudnn_td_with({1,              outSize, 1, 1});
            void* a = data();
            void* c = bias_g->cuda_float()->data();

            CUDNN_CHECK( cudnnReduceTensor(ComputingContext::cudnn_handle,
                                reduceDesc,
                                nullptr, 0,
                                vt::ComputingContext::cuda_workspace, vt::ComputingContext::workspace_size,
                                &alpha, adesc, a, &beta, cdesc, c) );

            CUDNN_CHECK( cudnnDestroyReduceTensorDescriptor(reduceDesc) );
        }

        // do twice gemm
        {
            int m = inSize;
            int n = outSize;
            int k = batch * tokens;

            float* A = (float *)x->cuda_float()->data();
            float* B = (float *)data();
            float* C = (float *)weight_g->cuda_float()->data();

            float alpha = 1.0;
            float beta = 0.0;

            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &alpha, A, CUDA_R_32F, m,
                    B, CUDA_R_32F, n, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }

        {
            int m = inSize;
            int n = batch * tokens;
            int k = outSize;

            float* A = (float *)weight->cuda_float()->data();
            float* B = (float *)data();
            float* C = (float *)x_g->cuda_float()->data();

            float alpha = 1.0;
            float beta = 0.0;

            cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_32F, m,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
        }


        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_gelu_backward(tensor_t self, tensor_t x_, tensor_t x_g_) {
    if ( DT == DataType::Float ) {
        auto stream = ComputingContext::cuda_stream;
        float* out_g = (float *)data();
        float* x = (float *)x_->cuda_float()->data();
        float* x_g = (float *)x_g_->cuda_float()->data();

        cuda::gelu_backward(out_g, x, x_g, self->items(), stream);


        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
    if ( DT == DataType::Float ) {
        float alpha = 1.0;
        float beta = 0.0;

        auto shape_ = self->shape().vec();
        int batch = shape_[0];
        int heads = shape_[1];
        int tokens = shape_[2];
        int hidden = shape_[3];

        int HT = hidden * tokens ;
        int TT = tokens * tokens;
        for (int i = 0; i < batch * heads; i++) {
            // computing value_g
            {
                float* A = (float *)data() + i * HT;
                float* B = (float *)attn->cuda_float()->data() + i * TT;
                float* C = (float *)v_g->cuda_float()->data() + i * HT;

                int m = hidden;
                int n = tokens;
                int k = tokens;

                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &alpha, A, CUDA_R_32F, m,
                    B, CUDA_R_32F, n, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
            }

            // computing attn_g
            {
                float* A = (float *)v->cuda_float()->data() + i * HT;
                float* B = (float *)data() + i * HT;
                float* C = (float *)attn_g->cuda_float()->data() + i * TT;

                int m = tokens;
                int n = tokens;
                int k = hidden;

                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_32F,  k,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
            }
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) {
    if ( DT == DataType::Float ) {
        float alpha = 1.0;
        float beta = 0.0;

        auto shape_ = self->shape().vec();

        size_t batch = shape_[0];
        size_t heads = shape_[1];
        size_t tokens = shape_[2];

        auto dydesc = create_cudnn_td_with({ batch * heads * tokens, tokens, 1, 1});
        auto dxdesc = create_cudnn_td_with({ batch * heads * tokens, tokens, 1, 1});
        auto ydesc = create_cudnn_td_with({ batch * heads * tokens, tokens, 1, 1});

        float*  dy = (float *)data();
        float*  dx = (float *)x_g->cuda_float()->data();
        float*  y = (float *)out->cuda_float()->data();

        CUDNN_CHECK( cudnnSoftmaxBackward( ComputingContext::cudnn_handle,
                                          CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                          &alpha, ydesc, y, dydesc, dy, &beta, dxdesc, dx) );


        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
    if ( DT == DataType::Float ) {
        float alpha = 1.0;
        float beta = 0.0;

        auto shape_ = self->shape().vec();
        int batch = shape_[0];
        int heads = shape_[1];
        int tokens = shape_[2];
        int hidden = shape_[3];

        int HT = hidden * tokens ;
        int TT = tokens * tokens;

        float* softmax_out = (float *)ComputingContext::cuda_workspace;
        float* wp = softmax_out + TT;
        size_t wp_size = ComputingContext::workspace_size - TT * sizeof(float);
        for (int i = 0; i < batch * heads; i++) {
            // computing value_g
            {
                float* A = (float *)data() + i * HT;
                float* B = (float *)attn->cuda_float()->data() + i * TT;
                float* C = (float *)v_g->cuda_float()->data() + i * HT;

                int m = hidden;
                int n = tokens;
                int k = tokens;

                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &alpha, A, CUDA_R_32F,  m,
                    B, CUDA_R_32F, n, &beta,
                    C, CUDA_R_32F, m,
                    wp, wp_size);
            }

            // copy softmax out to a temp place
            {
                auto stream = ComputingContext::cuda_stream;
                float* src = (float *)attn->cuda_float()->data() + i * TT;
                CUDA_CHECK(cudaMemcpyAsync(softmax_out, src, TT * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            }

            // computing attn_g
            {
                float* A = (float *)v->cuda_float()->data() + i * HT;
                float* B = (float *)data() + i * HT;
                float* C = (float *)attn_g->cuda_float()->data() + i * TT;

                int m = tokens;
                int n = tokens;
                int k = hidden;

                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A , CUDA_R_32F, k,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    wp, wp_size);
            }

            // apply softmax backward
            {
                auto dydesc = create_cudnn_td_with({ (size_t)tokens, (size_t)tokens, 1, 1});
                auto dxdesc = create_cudnn_td_with({ (size_t)tokens, (size_t)tokens, 1, 1});
                auto  ydesc = create_cudnn_td_with({ (size_t)tokens, (size_t)tokens, 1, 1});

                float*  dy = (float *)attn_g->cuda_float()->data() + i * TT;
                float*  dx = dy;
                float*  y = softmax_out;

                CUDNN_CHECK( cudnnSoftmaxBackward( ComputingContext::cudnn_handle,
                            CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                            &alpha, ydesc, y, dydesc, dy, &beta, dxdesc, dx) );
            }
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn CUDATensor<DT>::
op_qk_backward(tensor_t self, tensor_t query, tensor_t key, tensor_t query_g, tensor_t key_g) {
    if ( DT == DataType::Float ) {
        auto shape_ = query->shape().vec();

        int batch = shape_[0];
        int heads = shape_[1];
        int tokens = shape_[2];
        int hhidden = shape_[3];

        int HT = hhidden * tokens ;
        int TT = tokens * tokens;

        float alpha = 1.0 / sqrt(hhidden);
        float beta = 0.0;
        for (int i = 0; i < batch * heads; i++) {
            // computing query_g
            {
                int m = hhidden;
                int n = tokens;
                int k = tokens;

                float* A = (float *)(key->cuda_float()->data()) + i * HT;
                float* B = (float *)data() + i * TT;
                float* C = (float *)(query_g->cuda_float()->data()) + i * HT;
                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha, A, CUDA_R_32F,  m,
                    B, CUDA_R_32F, k, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);
            }

            // computing key_g
            {
                int m = hhidden;
                int n = tokens;
                int k = tokens;

                float* A = (float *)(query->cuda_float()->data()) + i * HT;
                float* B = (float *)data() + i * TT;
                float* C = (float *)(key_g->cuda_float()->data()) + i * HT;
                cuda::LtSgemm(ComputingContext::cublasLt_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &alpha, A, CUDA_R_32F,  m,
                    B, CUDA_R_32F, n, &beta,
                    C, CUDA_R_32F, m,
                    ComputingContext::cuda_workspace,
                    ComputingContext::workspace_size);

            }
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

tensor_t create_cuda_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Float>* tensor = new CUDATensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::FP16>* tensor = new CUDATensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cuda_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CUDATensor<DataType::Int>* tensor = new CUDATensor<DataType::Int>(shape);
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


} // end of namespace br
