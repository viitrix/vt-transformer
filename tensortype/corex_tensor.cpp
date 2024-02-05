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

}
