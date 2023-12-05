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
    Float = 0,
    BF16 = 1,
    FP16 = 2,
    Int = 3,
    Q8 = 4,
    Q4 = 5,
};

inline DataType DataType_from(const char* dtype) {
    if ( strcmp(dtype, "float") == 0) {
        return DataType::Float;
    }
    if ( strcmp(dtype, "fp16") == 0) {
        return DataType::FP16;
    }
    if ( strcmp(dtype, "bf16") == 0) {
        return DataType::BF16;
    }
    if ( strcmp(dtype, "int") == 0) {
        return DataType::Int;
    }
    if ( strcmp(dtype, "q8") == 0) {
        return DataType::Q8;
    }
    if ( strcmp(dtype, "q4") == 0) {
        return DataType::Q4;
    }
    vt_panic("Can't be here");
    return DataType::Float;
}

inline const char* DataType_name(DataType type_) {
    switch( type_ ) {
        case Float:
            return "float";
        case FP16:
            return "fp16";
        case BF16:
            return "bf16";
        case Int:
            return "int";
        case Q8:
            return "q8";
        case Q4:
            return "q4";
        default:
            break;
    }
    vt_panic("Can't be here");
    return NULL;
}

// Logical/Math shape of a tensor
struct ShapeType {
public:
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
using host_float_t = HostTensor<DataType::Float>;
using host_fp16_t = HostTensor<DataType::FP16>;
using host_int_t = HostTensor<DataType::Int>;
using host_q8_t = HostTensor<DataType::Q8>;
using host_q4_t = HostTensor<DataType::Q4>;

#ifdef _USING_DEVICE_CUDA_
template <DataType _DTYPE_> struct CUDATensor;
using cuda_float_t = CUDATensor<DataType::Float>;
using cuda_fp16_t = CUDATensor<DataType::FP16>;
using cuda_int_t = CUDATensor<DataType::Int>;
using cuda_q8_t = CUDATensor<DataType::Q8>;
using cuda_q4_t = CUDATensor<DataType::Q4>;
#endif

#ifdef _USING_DEVICE_DCU_
template <DataType _DTYPE_> struct DCUTensor;
using dcu_float_t = DCUTensor<DataType::Float>;
using dcu_fp16_t = DCUTensor<DataType::FP16>;
using dcu_int_t = DCUTensor<DataType::Int>;
using dcu_q8_t = DCUTensor<DataType::Q8>;
using dcu_q4_t = DCUTensor<DataType::Q4>;
#endif

#ifdef _USING_DEVICE_DNNL_
template <DataType _DTYPE_> struct DNNLTensor;
using dnnl_float_t = DNNLTensor<DataType::Float>;
using dnnl_fp16_t = DNNLTensor<DataType::FP16>;
using dnnl_int_t = DNNLTensor<DataType::Int>;
#endif

// TensorType is all you need
struct TensorType: public TransformerComputing {
public:
    // init functions
    TensorType() = delete;
    TensorType(host_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(host_fp16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::FP16), impl_(tensor) {};
    TensorType(host_int_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Int), impl_(tensor) {};
    TensorType(host_q8_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q8), impl_(tensor) {};
    TensorType(host_q4_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q4), impl_(tensor) {};

#ifdef _USING_DEVICE_CUDA_
    TensorType(cuda_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(cuda_fp16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::FP16), impl_(tensor) {};
    TensorType(cuda_int_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Int), impl_(tensor) {};
    TensorType(cuda_q8_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q8), impl_(tensor) {};
    TensorType(cuda_q4_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q4), impl_(tensor) {};
#endif

#ifdef _USING_DEVICE_DCU_
    TensorType(dcu_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(dcu_fp16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::FP16), impl_(tensor) {};
    TensorType(dcu_int_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Int), impl_(tensor) {};
    TensorType(dcu_q8_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q8), impl_(tensor) {};
    TensorType(dcu_q4_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Q4), impl_(tensor) {};
#endif

#ifdef _USING_DEVICE_DNNL_
    TensorType(dnnl_float_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Float), impl_(tensor) {};
    TensorType(dnnl_fp16_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::FP16), impl_(tensor) {};
    TensorType(dnnl_int_t* tensor, const ShapeType& shape) : shape_(shape), dtype_(DataType::Int), impl_(tensor) {};
#endif

    virtual ~TensorType();

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
    size_t impl_index() const {
        return impl_.index();
    }

    host_float_t* host_float() {
        if ( impl_.index() != HOST_FLOAT ) {
            vt_panic("Cant get host_float from a tensor");
        }
        return std::get<HOST_FLOAT>(impl_);
    }
    host_fp16_t* host_fp16() {
        if ( impl_.index() != HOST_FP16 ) {
            vt_panic("Cant get host_fp16 from a tensor");
        }
        return std::get<HOST_FP16>(impl_);
    }
    host_int_t* host_int() {
        if ( impl_.index() != HOST_INT ) {
            vt_panic("Cant get host_int from a tensor");
        }
        return std::get<HOST_INT>(impl_);
    }
    host_q8_t* host_q8() {
        if ( impl_.index() != HOST_Q8 ) {
            vt_panic("Cant get host_q8 from a tensor");
        }
        return std::get<HOST_Q8>(impl_);
    }
    host_q4_t* host_q4() {
        if ( impl_.index() != HOST_Q4 ) {
            vt_panic("Cant get host_q4 from a tensor");
        }
        return std::get<HOST_Q4>(impl_);
    }

#ifdef _USING_DEVICE_CUDA_
    cuda_float_t* cuda_float() {
        if ( impl_.index() != CUDA_FLOAT ) {
            vt_panic("Cant get cuda_float from a tensor");
        }
        return std::get<CUDA_FLOAT>(impl_);
    }
    cuda_fp16_t* cuda_fp16() {
        if ( impl_.index() != CUDA_FP16 ) {
            vt_panic("Cant get cuda_fp16 from a tensor");
        }
        return std::get<CUDA_FP16>(impl_);
    }
    cuda_int_t* cuda_int() {
        if ( impl_.index() != CUDA_INT ) {
            vt_panic("Cant get cuda_int from a tensor");
        }
        return std::get<CUDA_INT>(impl_);
    }
    cuda_q8_t* cuda_q8() {
        if ( impl_.index() != CUDA_Q8 ) {
            vt_panic("Cant get cuda_q8 from a tensor");
        }
        return std::get<CUDA_Q8>(impl_);
    }
    cuda_q4_t* cuda_q4() {
        if ( impl_.index() != CUDA_Q4 ) {
            vt_panic("Cant get cuda_q4 from a tensor");
        }
        return std::get<CUDA_Q4>(impl_);
    }
#endif

#if _USING_DEVICE_DCU_
    dcu_float_t* dcu_float() {
        if ( impl_.index() != DCU_FLOAT ) {
            vt_panic("Cant get dcu_float from a tensor");
        }
        return std::get<DCU_FLOAT>(impl_);
    }
    dcu_fp16_t* dcu_fp16() {
        if ( impl_.index() != DCU_FP16 ) {
            vt_panic("Cant get dcu_fp16 from a tensor");
        }
        return std::get<DCU_FP16>(impl_);
    }
    dcu_int_t* dcu_int() {
        if ( impl_.index() != DCU_INT ) {
            vt_panic("Cant get dcu_int from a tensor");
        }
        return std::get<DCU_INT>(impl_);
    }
    dcu_q8_t* dcu_q8() {
        if ( impl_.index() != DCU_Q8 ) {
            vt_panic("Cant get dcu_q8 from a tensor");
        }
        return std::get<DCU_Q8>(impl_);
    }
    dcu_q4_t* dcu_q4() {
        if ( impl_.index() != DCU_Q4 ) {
            vt_panic("Cant get dcu_q4 from a tensor");
        }
        return std::get<DCU_Q4>(impl_);
    }
#endif

#if _USING_DEVICE_DNNL_
    dnnl_float_t* dnnl_float() {
        if ( impl_.index() != DNNL_FLOAT ) {
            vt_panic("Cant get dnnl_float from a tensor");
        }
        return std::get<DNNL_FLOAT>(impl_);
    }
    dnnl_fp16_t* dnnl_fp16() {
        if ( impl_.index() != DNNL_FP16 ) {
            vt_panic("Cant get dnnl_fp16 from a tensor");
        }
        return std::get<DNNL_FP16>(impl_);
    }
    dnnl_int_t* dnnl_int() {
        if ( impl_.index() != DNNL_INT ) {
            vt_panic("Cant get dnnl_int from a tensor");
        }
        return std::get<DNNL_INT>(impl_);
    }
#endif

    // help functions
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

    const char* device_name() {
        if ( (impl_index() <= ImplType::HOST_Q4) && (impl_index() >= ImplType::HOST_FLOAT) ) {
            return "host";
        }
#ifdef _USING_DEVICE_CUDA_
        if ( (impl_index() <= ImplType::CUDA_Q4) && (impl_index() >= ImplType::CUDA_FLOAT) ) {
            return "cuda";
        }
#endif

#ifdef _USING_DEVICE_DCU_
        if ( (impl_index() <= ImplType::DCU_Q4) && (impl_index() >= ImplType::DCU_FLOAT) ) {
            return "dcu";
        }
#endif

#ifdef _USING_DEVICE_DNNL_
        if ( (impl_index() <= ImplType::DNNL_INT) && (impl_index() >= ImplType::DNNL_FLOAT) ) {
            return "nccl";
        }
#endif
        vt_panic("Can't be here!");
        return "";
    }

    bool is_host() const {
        auto ii = impl_index();
        if ( (ii >= ImplType::HOST_FLOAT) && (ii <= ImplType::HOST_Q4) ) {
            return true;
        }
        return false;
    }

#ifdef _USING_DEVICE_CUDA_
    bool is_cuda() const {
        auto ii = impl_index();
        if ( (ii >= ImplType::CUDA_FLOAT) && (ii <= ImplType::CUDA_Q4) ) {
            return true;
        }
        return false;
    }
#endif

#ifdef _USING_DEVICE_DCU_
    bool is_dcu() const {
        auto ii = impl_index();
        if ( (ii >= ImplType::DCU_FLOAT) && (ii <= ImplType::DCU_Q4) ) {
            return true;
        }
        return false;
    }
#endif

    bool is_float() const {
        if (impl_index() == ImplType::HOST_FLOAT) {
            return true;
        }
#ifdef _USING_DEVICE_CUDA_
        if (impl_index() == ImplType::CUDA_FLOAT) {
            return true;
        }
#endif

#ifdef _USING_DEVICE_DCU_
        if (impl_index() == ImplType::DCU_FLOAT) {
            return true;
        }
#endif

#ifdef _USING_DEVICE_DNNL_
        if (impl_index() == ImplType::DNNL_FLOAT) {
            return true;
        }
#endif
        return false;
    }

    bool is_fp16() const {
        if (impl_index() == ImplType::HOST_FP16) {
            return true;
        }
#ifdef _USING_DEVICE_CUDA_
        if (impl_index() == ImplType::CUDA_FP16) {
            return true;
        }
#endif

#ifdef _USING_DEVICE_DCU_
        if (impl_index() == ImplType::DCU_FP16) {
            return true;
        }
#endif

#ifdef _USING_DEVICE_DNNL_
        if (impl_index() == ImplType::DNNL_FP16) {
            return true;
        }
#endif
        return false;
    }

    bool is_int() const {
        if (impl_index() == ImplType::HOST_INT) {
            return true;
        }
#ifdef _USING_DEVICE_CUDA_
        if (impl_index() == ImplType::CUDA_INT) {
            return true;
        }
#endif

#ifdef _USING_DEVICE_DCU_
        if (impl_index() == ImplType::DCU_INT) {
            return true;
        }
#endif

#ifdef _USING_DEVICE_DNNL_
        if (impl_index() == ImplType::DNNL_INT) {
            return true;
        }
#endif
        return false;
    }
    bool is_quantized() {
        return is_q4() || is_q8();
    }
    bool is_q8() const {
        if (impl_index() == ImplType::HOST_Q8) {
            return true;
        }
#ifdef _USING_DEVICE_CUDA_
        if (impl_index() == ImplType::CUDA_Q8) {
            return true;
        }
#endif
#ifdef _USING_DEVICE_DCU_
        if (impl_index() == ImplType::DCU_Q8) {
            return true;
        }
#endif
        return false;
    }
    bool is_q4() const {
        if (impl_index() == ImplType::HOST_Q4) {
            return true;
        }
#ifdef _USING_DEVICE_CUDA_
        if (impl_index() == ImplType::CUDA_Q4) {
            return true;
        }
#endif
#ifdef _USING_DEVICE_DCU_
        if (impl_index() == ImplType::DCU_Q4) {
            return true;
        }
#endif
        return false;
    }

    bool same_impl(tensor_t& other) {
        if ( impl_index() != other->impl_index() ) {
            return false;
        }
        return true;
    }
    bool same_dtype(tensor_t& other) {
        if ( dtype_ != other->dtype() ) {
            return false;
        }
        return true;
    }
    bool same_shape(tensor_t& other) {
        if ( shape_ != other->shape() ) {
            return false;
        }
        return true;
    }

    TransformerComputing* impl();
    void* device_data();
    void* device_data(size_t index);

public:
    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_copy(tensor_t self, tensor_t src) override;
    ComputingReturn op_convert(tensor_t self, tensor_t src) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;
    ComputingReturn op_quantize(tensor_t self, tensor_t out) override;
    ComputingReturn op_dequantize(tensor_t self, tensor_t out) override;
    ComputingReturn op_embed(tensor_t self, tensor_t table, tensor_t out) override;
    ComputingReturn op_scale(tensor_t self, float scale) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) override;
    ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;
    ComputingReturn op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) override;
    ComputingReturn op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;
    ComputingReturn op_transpose_0213(tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(tensor_t self, tensor_t out) override ;
    ComputingReturn op_attn(tensor_t self, tensor_t v, tensor_t attn) override;
    ComputingReturn op_xattn(tensor_t self, tensor_t k, tensor_t v, tensor_t qk, tensor_t attn) override;
    ComputingReturn op_gelu(tensor_t self, tensor_t dst) override;
    ComputingReturn op_silu_product(tensor_t self, tensor_t up, tensor_t dst) override;
    std::variant<ComputingReturn, int> op_all_logits(tensor_t self, tensor_t mask,  tensor_t lm_head, tensor_t output) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top3(tensor_t self, float temp) override;
    std::variant<ComputingReturn, float> op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) override;
    ComputingReturn op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) override;
    ComputingReturn op_rmsnorm_backward(tensor_t self, tensor_t x, tensor_t scale, tensor_t norm2, tensor_t dscale, tensor_t dx, float eps) override;
    ComputingReturn op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) override;
    ComputingReturn op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) override;
    ComputingReturn op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) override;
    ComputingReturn op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) override;
    ComputingReturn op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) override;
    ComputingReturn op_qk_backward(tensor_t self, tensor_t q, tensor_t k, tensor_t q_g, tensor_t k_g) override;

    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_mpi_bcast(tensor_t self, int root) override;
    ComputingReturn io_mpi_recv(tensor_t self, int source) override;
    ComputingReturn io_mpi_send(tensor_t self, int dst) override;
    ComputingReturn io_nccl_recv(tensor_t self, int source) override;
    ComputingReturn io_nccl_send(tensor_t self, int dst) override;
    ComputingReturn io_pipe_read(tensor_t self) override;
    ComputingReturn io_pipe_write(tensor_t self, int n) override;

private:
    // basic info about tensor
    ShapeType shape_;
    const DataType  dtype_;

    // ImplType enum order is same as TensorImpl's variant
    enum ImplType {
#ifdef _USING_DEVICE_CUDA_
        CUDA_FLOAT,
        CUDA_FP16,
        CUDA_INT,
        CUDA_Q8,
        CUDA_Q4,
#endif

#ifdef _USING_DEVICE_DCU_
        DCU_FLOAT,
        DCU_FP16,
        DCU_INT,
        DCU_Q8,
        DCU_Q4,
#endif

#ifdef _USING_DEVICE_DNNL_
        DNNL_FLOAT,
        DNNL_FP16,
        DNNL_INT,
#endif
        HOST_FLOAT,
        HOST_FP16,
        HOST_INT,
        HOST_Q8,
        HOST_Q4
    };
    using TensorImpl =   std::variant<
#ifdef _USING_DEVICE_CUDA_
                                        cuda_float_t*,
                                        cuda_fp16_t*,
                                        cuda_int_t*,
                                        cuda_q8_t*,
                                        cuda_q4_t*,
#endif

#ifdef _USING_DEVICE_DCU_
                                        dcu_float_t*,
                                        dcu_fp16_t*,
                                        dcu_int_t*,
                                        dcu_q8_t*,
                                        dcu_q4_t*,
#endif

#ifdef _USING_DEVICE_DNNL_
                                        dnnl_float_t*,
                                        dnnl_fp16_t*,
                                        dnnl_int_t*,
#endif
                                        host_float_t*,
                                        host_fp16_t*,
                                        host_int_t*,
                                        host_q8_t*,
                                        host_q4_t* >;
    TensorImpl impl_;
};

tensor_t create_host_float(std::vector<size_t>& shape);
tensor_t create_host_fp16(std::vector<size_t>& shape);
tensor_t create_host_int(std::vector<size_t>& shape);
tensor_t create_host_q8(std::vector<size_t>& shape);
tensor_t create_host_q4(std::vector<size_t>& shape);

#if _USING_DEVICE_CUDA_
tensor_t create_cuda_float(std::vector<size_t>& shape);
tensor_t create_cuda_fp16(std::vector<size_t>& shape);
tensor_t create_cuda_int(std::vector<size_t>& shape);
tensor_t create_cuda_q8(std::vector<size_t>& shape);
tensor_t create_cuda_q4(std::vector<size_t>& shape);
#endif

#if _USING_DEVICE_DCU_
tensor_t create_dcu_float(std::vector<size_t>& shape);
tensor_t create_dcu_fp16(std::vector<size_t>& shape);
tensor_t create_dcu_int(std::vector<size_t>& shape);
tensor_t create_dcu_q8(std::vector<size_t>& shape);
tensor_t create_dcu_q4(std::vector<size_t>& shape);
#endif

#if _USING_DEVICE_DNNL_
tensor_t create_dnnl_float(std::vector<size_t>& shape);
tensor_t create_dnnl_fp16(std::vector<size_t>& shape);
tensor_t create_dnnl_int(std::vector<size_t>& shape);
#endif


} // end of namespace br

#endif
