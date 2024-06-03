#ifndef _TENSORTYPE_HPP_
#define _TENSORTYPE_HPP_

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>

#include "vt.hpp"
#include "computing.hpp"

namespace vt {

enum DataType {
    UDF = -1,
    F32 = 0,
    BF16 = 1,
    F16 = 2,
    I32 = 3,
    Q8 = 4,     // 1024 grouped with zero and scale
    Q4 = 5,     // 128  grouped with zero and scale
    PQ = 6,     // Focal Product Quantizer
};

inline DataType DataType_from(const char* dtype) {
    if ( strcmp(dtype, "f32") == 0) {
        return DataType::F32;
    }
    if ( strcmp(dtype, "i32") == 0) {
        return DataType::I32;
    }
    if ( strcmp(dtype, "f16") == 0) {
        return DataType::F16;
    }
    if ( strcmp(dtype, "bf16") == 0) {
        return DataType::BF16;
    }
    if ( strcmp(dtype, "q8") == 0) {
        return DataType::Q8;
    }
    if ( strcmp(dtype, "q4") == 0) {
        return DataType::Q4;
    }
    if ( strcmp(dtype, "pq") == 0) {
        return DataType::PQ;
    }
    std::cout << ">>>>>>>>>>>>" << dtype << std::endl;
    vt_fatal_error();
    return DataType::UDF;
}

inline const char* DataType_name(DataType type_) {
    switch( type_ ) {
        case F32:
            return "f32";
        case I32:
            return "i32";
        case F16:
            return "f16";
        case BF16:
            return "bf16";
        case Q8:
            return "q8";
        case Q4:
            return "q4";
        case PQ:
            return "pq";
        default:
            break;
    }
    vt_panic("Can't be here");
    return NULL;
}

// Logical/Math shape of a tensor
struct ShapeType {
public:
    ~ShapeType() {}
    ShapeType() = delete;
    ShapeType(const std::vector<size_t>& dims) {
        size_t ND = dims.size();
        dims_.resize(ND);
        for(size_t i = 0; i < ND; i++) {
            dims_[i] = dims[i];
        }

        numel_ = 1;
        for(size_t i = 0; i < dims_.size(); i++) {
            vt_assert( dims_[i] > 0, "Don't support zero dim vector");
            numel_ *= dims_[i];
        }
    }
    // all kinds accessors
    size_t numel() const {
        return numel_;
    }
    const std::vector<size_t>& vec() const {
        return dims_;
    }
    const size_t* dims() const {
        return &dims_[0];
    }
    const size_t dim() const {
        return dims_.size();
    }
    const size_t operator[](int i) const {
        if ( i < 0) {
            i = (int)dims_.size() + i;
        }
        if ( i >= (int)dims_.size() || i < 0 ) {
            vt_panic("Access shape out of dims");
        }
        return dims_[i];
    }
    bool operator == (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return false;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other[i] != dims_[i] ) {
                return false;
            }
        }
        return true;
    }
    bool operator != (const ShapeType& other) const {
        if ( other.dim() != dim() ) {
            return true;
        }
        for (size_t i = 0; i < dim(); i++) {
            if ( other[i] != dims_[i] ) {
                return true;
            }
        }
        return false;
    }
    std::string to_string() const {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < dim(); i++) {
            ss << dims_[i] << " ";
        }
        ss << "]";
        return ss.str();
    }

private:
    std::vector<size_t>  dims_;
    mutable size_t numel_;
};

// forward declare
template <DataType _DTYPE_> struct HostTensor;
using host_f32_t = HostTensor<DataType::F32>;
using host_i32_t = HostTensor<DataType::I32>;
using host_f16_t = HostTensor<DataType::F16>;
using host_bf16_t = HostTensor<DataType::BF16>;
using host_q8_t = HostTensor<DataType::Q8>;
using host_q4_t = HostTensor<DataType::Q4>;
using host_pq_t = HostTensor<DataType::PQ>;

#ifdef _USING_DEVICE_CUDA_
template <DataType _DTYPE_> struct CUDATensor;
using cuda_f32_t = CUDATensor<DataType::F32>;
using cuda_i32_t = CUDATensor<DataType::I32>;
using cuda_f16_t = CUDATensor<DataType::F16>;
using cuda_bf16_t = CUDATensor<DataType::BF16>;
using cuda_q8_t = CUDATensor<DataType::Q8>;
using cuda_q4_t = CUDATensor<DataType::Q4>;
using cuda_pq_t = CUDATensor<DataType::PQ>;
#endif

// TensorType is all you need
struct TensorType: public TransformerComputing {
public:
    // init functions
    virtual ~TensorType();
    TensorType() = delete;

    TensorType(host_f32_t* tensor, const ShapeType& shape);
    TensorType(host_i32_t* tensor, const ShapeType& shape);
    TensorType(host_f16_t* tensor, const ShapeType& shape);
    TensorType(host_bf16_t* tensor, const ShapeType& shape);
    TensorType(host_q8_t* tensor, const ShapeType& shape);
    TensorType(host_q4_t* tensor, const ShapeType& shape);
    TensorType(host_pq_t* tensor, const ShapeType& shape);
#ifdef _USING_DEVICE_CUDA_
    TensorType(cuda_f32_t* tensor, const ShapeType& shape);
    TensorType(cuda_i32_t* tensor, const ShapeType& shape);
    TensorType(cuda_f16_t* tensor, const ShapeType& shape);
    TensorType(cuda_bf16_t* tensor, const ShapeType& shape);
    TensorType(cuda_q8_t* tensor, const ShapeType& shape);
    TensorType(cuda_q4_t* tensor, const ShapeType& shape);
    TensorType(cuda_pq_t* tensor, const ShapeType& shape);
#endif

public:
    ComputingReturn io_load(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_save(ComputingContext* ctx, tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn io_pipe_read(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn io_pipe_write(ComputingContext* ctx, tensor_t self, int dst) override;
    std::variant<ComputingReturn, size_t> op_sizeof(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, void *> op_data( ComputingContext* ctx, tensor_t self) override;
    ComputingReturn op_zero(ComputingContext* ctx, tensor_t self) override;
    ComputingReturn op_fill(ComputingContext* ctx, tensor_t self, float value) override;
    ComputingReturn op_rotary_cache(ComputingContext* ctx, tensor_t self, float base) override;
    ComputingReturn op_causal_mask(ComputingContext* ctx, tensor_t self, tensor_t output) override;
    ComputingReturn op_copy_from(ComputingContext* ctx, tensor_t self, tensor_t src) override;
    ComputingReturn op_copy_to(ComputingContext* ctx, tensor_t self, tensor_t dst) override;
    ComputingReturn op_convert(ComputingContext* ctx, tensor_t self, tensor_t src) override;
    std::variant<ComputingReturn, tensor_t> op_view(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(ComputingContext* ctx, tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    ComputingReturn op_quantize(ComputingContext* ctx, tensor_t self, tensor_t out) override;
    ComputingReturn op_dequantize(ComputingContext* ctx, tensor_t self, tensor_t out) override;
    ComputingReturn op_embed(ComputingContext* ctx, tensor_t self, tensor_t table, tensor_t output) override;
    ComputingReturn op_add(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(ComputingContext* ctx, tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_linear(ComputingContext* ctx, tensor_t self, tensor_t w, tensor_t bias, tensor_t y) override;
    ComputingReturn op_layernorm(ComputingContext* ctx, tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;
    ComputingReturn op_rmsnorm(ComputingContext* ctx, tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) override;
    ComputingReturn op_rotary_embed(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;
    ComputingReturn op_transpose_0213(ComputingContext* ctx, tensor_t self, tensor_t y) override;
    ComputingReturn op_transpose_0213_rotary(ComputingContext* ctx, tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;
    ComputingReturn op_transpose_0213_repeated(ComputingContext* ctx, tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(ComputingContext* ctx, tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(ComputingContext* ctx, tensor_t self, tensor_t out) override;
    ComputingReturn op_attn(ComputingContext* ctx, tensor_t self, tensor_t v, tensor_t attn) override;
    ComputingReturn op_gelu(ComputingContext* ctx, tensor_t self, tensor_t dst) override;
    ComputingReturn op_silu_product(ComputingContext* ctx, tensor_t self, tensor_t up, tensor_t dst) override;
    std::variant<ComputingReturn, int> op_all_logits(ComputingContext* ctx, tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output ) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top1(ComputingContext* ctx, tensor_t self) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top3(ComputingContext* ctx, tensor_t self, float temp) override;
    ComputingReturn op_conv2d(ComputingContext* ctx, tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) override;

   // fast access
    const ShapeType& shape() const {
        return shape_;
    }
    const DataType& dtype() const {
        return dtype_;
    }
    const size_t items() {
        return shape_.numel();
    }
    size_t impl_index() {
        return impl_.index();
    }

    bool is_host() {
        auto ii = impl_index();
        if ( (ii >= HOST_F32) && (ii <= HOST_PQ) ) {
            return true;
        }
        return false;
    }

#ifdef _USING_DEVICE_CUDA_
    bool is_cuda() {
        auto ii = impl_index();
        if ( (ii >= CUDA_F32) && (ii <= CUDA_PQ) ) {
            return true;
        }
        return false;
    }
#endif

    bool is_quantized() {
        if ( dtype_ == DataType::Q8 ) {
            return true;
        }
        if ( dtype_ == DataType::Q4 ) {
            return true;
        }
        if ( dtype_ == DataType::PQ ) {
            return true;
        }
        return false;
    }

#define _CONVERT_(TT) \
        if ( impl_.index() == ImplType::TT ) { \
            return (TransformerComputing *)std::get<ImplType::TT>(impl_); \
        }

    TransformerComputing* impl() {
        _CONVERT_(HOST_F32)
        _CONVERT_(HOST_I32)
        _CONVERT_(HOST_F16)
        _CONVERT_(HOST_BF16)
        _CONVERT_(HOST_Q8)
        _CONVERT_(HOST_Q4)
        _CONVERT_(HOST_PQ)

#ifdef _USING_DEVICE_CUDA_
        _CONVERT_(CUDA_F32)
        _CONVERT_(CUDA_I32)
        _CONVERT_(CUDA_F16)
        _CONVERT_(CUDA_BF16)
        _CONVERT_(CUDA_Q8)
        _CONVERT_(CUDA_Q4)
        _CONVERT_(CUDA_PQ)
#endif
        vt_fatal_error();
        return nullptr;
    }

#define _ACCESSOR_(T, TT) \
    T##_t * T() {             \
        if ( impl_.index() == ImplType::TT ) { \
            return std::get<ImplType::TT>(impl_);       \
        }                               \
        vt::_M_Panic(__FILE__, __LINE__, "Can't be here!"); \
        return nullptr; \
    }
    _ACCESSOR_(host_f32, HOST_F32)
    _ACCESSOR_(host_i32, HOST_I32)
    _ACCESSOR_(host_f16, HOST_F16)
    _ACCESSOR_(host_bf16, HOST_BF16)
    _ACCESSOR_(host_q8,  HOST_Q8)
    _ACCESSOR_(host_q4,  HOST_Q4)
    _ACCESSOR_(host_pq,  HOST_PQ)
#ifdef _USING_DEVICE_CUDA_
    _ACCESSOR_(cuda_f32,  CUDA_F32)
    _ACCESSOR_(cuda_i32,  CUDA_I32)
    _ACCESSOR_(cuda_f16,  CUDA_F16)
    _ACCESSOR_(cuda_bf16, CUDA_BF16)
    _ACCESSOR_(cuda_q8,   CUDA_Q8)
    _ACCESSOR_(cuda_q4,   CUDA_Q4)
    _ACCESSOR_(cuda_pq,   CUDA_PQ)
#endif

    // help functions
    const char* device_name() {
        if ( (impl_index() <= ImplType::HOST_PQ) && (impl_index() >= ImplType::HOST_F32) ) {
            return "host";
        }
    #ifdef _USING_DEVICE_CUDA_
        if ( (impl_index() <= ImplType::CUDA_PQ) && (impl_index() >= ImplType::CUDA_F32) ) {
            return "cuda";
        }
    #endif
        vt_panic("Can't be here!");
        return "";
    }

    std::string to_string() {
        std::stringstream ss;
        ss << device_name() << ":" <<  DataType_name( dtype() ) ;
        ss << ":[";
        for (size_t i = 0; i < shape_.dim(); i++) {
            ss << shape_[i];
            if (i != shape_.dim() - 1) {
                ss << " ";
            }
        }
        ss << "]";
        return ss.str();
    }

private:
    // basic info about tensor
    const ShapeType shape_;
    const DataType  dtype_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
#ifdef _USING_DEVICE_CUDA_
        CUDA_F32,
        CUDA_I32,
        CUDA_F16,
        CUDA_BF16,
        CUDA_Q8,
        CUDA_Q4,
        CUDA_PQ,
#endif
        HOST_F32,
        HOST_I32,
        HOST_F16,
        HOST_BF16,
        HOST_Q8,
        HOST_Q4,
        HOST_PQ
    };

    // internal pointer based on unique_ptr auto delete
    using TensorImpl = std::variant<
#ifdef _USING_DEVICE_CUDA_
                  cuda_f32_t*,
                  cuda_i32_t*,
                  cuda_f16_t*,
                  cuda_bf16_t*,
                  cuda_q8_t*,
                  cuda_q4_t*,
                  cuda_pq_t*,
#endif
                  host_f32_t*,
                  host_i32_t*,
                  host_f16_t*,
                  host_bf16_t*,
                  host_q8_t*,
                  host_q4_t*,
                  host_pq_t* >;

    TensorImpl impl_;
};

tensor_t create_host_f32(std::vector<size_t>& shape);
tensor_t create_host_i32(std::vector<size_t>& shape);
tensor_t create_host_f16(std::vector<size_t>& shape);
tensor_t create_host_bf16(std::vector<size_t>& shape);
tensor_t create_host_q8(std::vector<size_t>& shape);
tensor_t create_host_q4(std::vector<size_t>& shape);
tensor_t create_host_pq(std::vector<size_t>& shape);

#ifdef _USING_DEVICE_CUDA_
tensor_t create_cuda_f32(std::vector<size_t>& shape);
tensor_t create_cuda_i32(std::vector<size_t>& shape);
tensor_t create_cuda_f16(std::vector<size_t>& shape);
tensor_t create_cuda_bf16(std::vector<size_t>& shape);
tensor_t create_cuda_q8(std::vector<size_t>& shape);
tensor_t create_cuda_q4(std::vector<size_t>& shape);
tensor_t create_cuda_pq(std::vector<size_t>& shape);
#endif

}
#endif
