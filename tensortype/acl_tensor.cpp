#include <arm_compute/core/Types.h>
#include <arm_compute/core/utils/quantization/AsymmHelpers.h>
#include <arm_compute/core/WindowIterator.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>

#include "acl_tensor.hpp"
#include "acl_kernels/impl.hpp"
#include "host_tensor.hpp"

namespace vt {

using device_fp16_t = __fp16;

inline arm_compute::TensorShape buildShape(const ShapeType& s) {
    auto vs = s.vec();
    if ( vs.size() == 1 ) {
        return arm_compute::TensorShape( {vs[0]} );
    }
    if ( vs.size() == 2 ) {
        return arm_compute::TensorShape( {vs[0], 1, 1, vs[1]} );
    }
    if ( vs.size() == 3 ) {
        return arm_compute::TensorShape( {vs[0], 1, vs[1], vs[2]} );
    }
    if ( vs.size() == 4 ) {
        return arm_compute::TensorShape( {vs[0], vs[1], vs[2], vs[3]} );
    }
    return arm_compute::TensorShape();
}

template <DataType _DTYPE_>
void ACLTensor<_DTYPE_>::buildTensorWithShape(arm_compute::Tensor& target, const std::vector<size_t> newShape) {
    auto ts = buildShape(newShape);
    if ( _DTYPE_ == DataType::Float ) {
        target.allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
    } else if ( _DTYPE_ == DataType::Int ) {
        target.allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        target.allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
    }

    target.allocator()->import_memory(mem_);
}

template <DataType _DTYPE_>
void ACLTensor<_DTYPE_>::buildTensorWithShape(arm_compute::Tensor& target, const std::vector<size_t> newShape, void* mem) {
    auto ts = buildShape(newShape);
    if ( _DTYPE_ == DataType::Float ) {
        target.allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
    } else if ( _DTYPE_ == DataType::Int ) {
        target.allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        target.allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
    }

    target.allocator()->import_memory(mem);
}

template <DataType _DTYPE_>
ACLTensor<_DTYPE_>::~ACLTensor()  {
    if ( owner_ ) {
        MemoryContext::free(mem_, size_);
    }
    delete t_;
}

template <DataType _DTYPE_>
ACLTensor<_DTYPE_>::ACLTensor(const ShapeType& shape) : owner_(true) {
    t_ = new arm_compute::Tensor();
    arm_compute::TensorShape ts = buildShape(shape);

    if ( _DTYPE_ == DataType::Float ) {
        size_ = shape.numel() * sizeof(float);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
    } else if ( _DTYPE_ == DataType::Int ) {
        size_ = shape.numel() * sizeof(int);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        size_ =  shape.numel() * sizeof(device_fp16_t);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
    }
    mem_ = MemoryContext::alloc(size_);
    t_->allocator()->import_memory(mem_);
}

template <DataType _DTYPE_>
ACLTensor<_DTYPE_>::ACLTensor(const ShapeType& shape,  void *mem) : owner_(false), mem_(mem) {
    t_ = new arm_compute::Tensor();
    arm_compute::TensorShape ts = buildShape(shape);
    size_ = 0;

    if ( _DTYPE_ == DataType::Float ) {
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));
    } else if ( _DTYPE_ == DataType::Int ) {
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( _DTYPE_ == DataType::FP16 ) {
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
    }

    t_->allocator()->import_memory(mem_);
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::io_dump(tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    if ( _DTYPE_ == DataType::Float ) {
        float* d = (float *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (float *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        int* d = (int *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (int *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        device_fp16_t* d = (device_fp16_t *)data();
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << float(d[i]) << " ";
        }
        std::cout << std::endl;
        d = (device_fp16_t *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << float(d[i]) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, size_t> ACLTensor<_DTYPE_>::op_sizeof(tensor_t self) {
    if ( _DTYPE_ == DataType::Float ) {
        return (size_t) self->items() * sizeof(float);
    }
    if ( _DTYPE_ == DataType::Int ) {
        return (size_t) self->items() * sizeof(int);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        return (size_t) self->items() * sizeof(device_fp16_t);
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::op_zero(tensor_t self) {
    memset(mem_, 0, size_);
    return OP_OK;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_alibi(tensor_t self) {
    int heads = self->shape()[1];
    int tokens = self->shape()[3];

    auto s = std::get<1>(self->op_sizeof(self));
    if ( DT == DataType::Float ) {
        std::vector<float> buffer;
        vt::fill_alibi<float>(buffer, heads, tokens);

        memcpy( data(), buffer.data(), s);
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        std::vector<local_fp16_t> buffer;
        vt::fill_alibi<local_fp16_t>(buffer, heads, tokens);
        memcpy( data(), buffer.data(), s);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_causal_mask(tensor_t self, tensor_t out) {
    int batch = self->shape()[0];
    int full_tokens = self->shape()[1];
    int new_tokens = out->shape()[2];

    int* mask  = (int *)data();

    float*          out32 = nullptr;
    device_fp16_t*   out16 = nullptr;

    if ( out->dtype() == DataType::Float ) {
        out32 = (float *)out->acl_float()->data();
    } else if ( out->dtype() == DataType::FP16 ) {
        out16 = (device_fp16_t *)out->acl_fp16()->data();
    } else {
        return OP_TODO_ERROR;
    }

    for (int e = 0; e < batch * new_tokens; e++) {
        int b = e / new_tokens;
        int nt = e % new_tokens;
        int nt_end = full_tokens - new_tokens + nt;

        int* m = &mask[ b * full_tokens ];
        if ( out32 != nullptr) {
            float* o = &out32[ b * new_tokens * full_tokens + nt * full_tokens ];
            float minv = std::numeric_limits<float>::lowest();
            acl_kernels::fill_causal_mask<float>(m, o, minv, full_tokens, nt_end);
        }
        if ( out16 != nullptr ) {
            device_fp16_t* o = &out16[ b * new_tokens * full_tokens + nt * full_tokens ];
            device_fp16_t minv = (device_fp16_t)0xFC00U;
            acl_kernels::fill_causal_mask<device_fp16_t>(m, o, minv, full_tokens, nt_end);
        }
    }

    return OP_OK;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_rotary_cache(tensor_t self, float base) {
    if ( DT == DataType::Float ) {
        // building inv_freq
        int len = self->shape()[0];
        int dims = self->shape()[1];

        std::vector<float> cos_sin;
        vt::fill_rotary_cache(cos_sin, len, dims, base);

        memcpy( data(), cos_sin.data(), self->items() * sizeof(float));
        return OP_OK;
    }
    return OP_OUTPUT_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    std::ofstream wf(fileName, std::ios::out | std::ios::binary);

    const char* d = (const char *)data();
    size_t len = std::get<1>(self->op_sizeof(self));
    wf.write(d, len);
    wf.close();
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {
    std::ifstream inf(fileName, std::ios::binary);
    if ( ! inf.is_open() ) {
        std::cout << "Can't open " << fileName << std::endl;
        vt_panic("Can't open file");
    }

    if (_DTYPE_ == DataType::Float) {
        size_t ret = inf.read( (char *)data(), sizeof(float) * self->items() ).gcount();
        vt_assert(ret == sizeof(float) * self->items(), "file size dont't match tensor");
    } else  if (_DTYPE_ == DataType::Int) {
        size_t ret = inf.read( (char *)data(), sizeof(int) * self->items() ).gcount();
        vt_assert(ret == sizeof(int) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::FP16) {
        size_t ret = inf.read( (char *)data(), sizeof(device_fp16_t) * self->items() ).gcount();
        vt_assert(ret == sizeof(device_fp16_t) * self->items(), "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::Q8 ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)data(), s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::Q4 ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)data(), s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else if (_DTYPE_ == DataType::PQ ) {
        size_t s = std::get<1>(self->op_sizeof(self));
        size_t ret = inf.read( (char *)mem_, s).gcount();
        vt_assert(ret == s, "file size dont't match tensor");
    } else {
        vt_panic("DataType don't support");
    }

    inf.close();
    return OP_OK;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_fill(tensor_t self, float value) {
    size_t items = self->items();
    if ( DT == DataType::Float ) {
        float *dst = (float *)mem_;
        for (size_t i = 0; i < items; i++) {
            dst[i] = value;
        }
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t *dst = (device_fp16_t *)mem_;
        device_fp16_t v = float(value);
        for (size_t i = 0; i < items; i++) {
            dst[i] = v;
        }
        return OP_OK;
    }
    if ( DT == DataType::Int ) {
        int *dst = (int *)mem_;
        int v = value;
        for (size_t i = 0; i < items; i++) {
            dst[i] = v;
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_copy(tensor_t self, tensor_t from) {
    arm_compute::NECast op;
    if ( DT == DataType::Float) {
        op.configure(  from->acl_float()->t_, t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if ( DT == DataType::FP16) {
        op.configure(  from->acl_fp16()->t_, t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if ( DT == DataType::Int) {
        op.configure(  from->acl_int()->t_, t_,  arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_convert(tensor_t self, tensor_t from) {
    vt_assert( self->shape().dim() == 4, "convert support 4D tensor only!");
    if ( DT == DataType::FP16 && from->is_float() ) {

        return OP_OK;
    }
    if ( DT == DataType::Float && from->is_fp16() ) {

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> ACLTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    if ( _DTYPE_ == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCpuTensor = new ACLTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Int ) {
        ShapeType newShape(newShape_);
        int *newData = (int *)data() + offset;
        auto* newCpuTensor = new ACLTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        ShapeType newShape(newShape_);
        device_fp16_t *newData = (device_fp16_t *)data() + offset;
        auto* newCpuTensor = new ACLTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType _DT_>
std::variant<ComputingReturn, tensor_t> ACLTensor<_DT_>::op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape_, const char* dtype) {
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
        auto* newTensor = new ACLTensor<DataType::Float>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    if ( DT == DataType::Int ) {
        auto* newTensor = new ACLTensor<DataType::Int>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    if ( DT == DataType::FP16 ) {
        auto* newTensor = new ACLTensor<DataType::FP16>(newShape, newData);
        return std::make_shared<TensorType>(newTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    ShapeType newShape(newShape_);
    if ( owner_ == true ) {
        return OP_INPUT_ERROR;
    }

    if ( newShape.numel() + offset > self->items()  ) {
        return OP_INPUT_ERROR;
    }

    delete t_;
    t_ = new arm_compute::Tensor();
    arm_compute::TensorShape ts = buildShape(newShape);

    if ( DT == DataType::Float ) {
        mem_  = (char *)data() + offset * sizeof(float);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F32));

    } else if ( DT == DataType::Int ) {
        mem_  = (char *)data() + offset * sizeof(int);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::S32));
    } else if ( DT == DataType::FP16 ) {
        mem_  = (char *)data() + offset * sizeof(device_fp16_t);
        t_->allocator()->init( arm_compute::TensorInfo(ts, arm_compute::Format::F16));
    } else {
        vt_panic("Can't be here!");
        return OP_INPUT_ERROR;
    }

    mem_ = MemoryContext::alloc(size_);
    t_->allocator()->import_memory(mem_);
    return OP_OK;
}


template<DataType DT>
ComputingReturn ACLTensor<DT>::op_add(tensor_t self, tensor_t b, tensor_t c) {
    arm_compute::NEArithmeticAddition op;
    if (   DT == DataType::Float) {
        op.configure(t_, b->acl_float()->t_, c->acl_float()->t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if (   DT == DataType::FP16) {
        op.configure(t_, b->acl_fp16()->t_, c->acl_fp16()->t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    if (   DT == DataType::Int) {
        op.configure(t_, b->acl_int()->t_, c->acl_int()->t_, arm_compute::ConvertPolicy::SATURATE);
        op.run();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn ACLTensor<DT>::op_linear(tensor_t self, tensor_t w, tensor_t bias, tensor_t dst) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t inSize = self->shape()[2];
    size_t outSize = w->shape()[0];

    size_t num = batch * tokens;

    arm_compute::NEGEMM op;
    arm_compute::GEMMInfo info;
    info.set_pretranspose_A(true);

    arm_compute::Tensor A;
    arm_compute::Tensor B;
    arm_compute::Tensor C;
    arm_compute::Tensor D;

    buildTensorWithShape(B, {num, inSize});
    if ( DT == DataType::Float ) {
        w->acl_float()->buildTensorWithShape(A, {inSize, outSize});
        if ( bias != nullptr ) {
            bias->acl_float()->buildTensorWithShape(C, {1, outSize});
        }
        dst->acl_float()->buildTensorWithShape(D, {num, outSize});
    } else if ( DT == DataType::FP16 ) {
        w->acl_fp16()->buildTensorWithShape(A, {inSize, outSize});
        if ( bias != nullptr ) {
            bias->acl_fp16()->buildTensorWithShape(C, {1, outSize});
        }
        dst->acl_fp16()->buildTensorWithShape(D, {num, outSize});
    } else {
        return OP_TODO_ERROR;
    }

    float alpha = 1.0f;
    float beta  = 0.0f;
    if ( bias != nullptr ) {
        op.configure(&A, &B, &C, &D, alpha, beta, info);
    } else {
        op.configure(&A, &B, nullptr, &D, alpha, beta, info);
    }
    op.run();
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t feature = self->shape()[2];

    size_t num = batch * tokens;
    if (   _DTYPE_ == DataType::Float) {
        acl_kernels::rmsnorm<DataType::Float>(self->acl_float(), scale->acl_float(),
            norm2->acl_float(), y->acl_float(),
            num, feature, eps);
        return OP_OK;
    }
    if (   _DTYPE_ == DataType::FP16) {
        acl_kernels::rmsnorm<DataType::FP16>(self->acl_fp16(), scale->acl_fp16(),
            norm2->acl_fp16(), y->acl_fp16(),
            num, feature, eps);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType DT>
ComputingReturn ACLTensor<DT>::op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos_, tensor_t y) {
    size_t batch = self->shape()[0];
    size_t tokens = self->shape()[1];
    size_t heads = self->shape()[2];
    size_t hidden = self->shape()[3];

    vt_assert(hidden == cached->shape()[1], "heads number must be same with cache");

    int* pos = (int*) pos_->acl_int()->data();
    if ( DT == DataType::Float ) {

        float* in = (float *)data();
        float* cos_sin = (float *)cached->acl_float()->data();
        float* out = (float *)y->acl_float()->data();

        acl_kernels::rotary_embed<float>(in, cos_sin, pos, out, batch, heads, tokens, hidden);

        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        device_fp16_t* in = (device_fp16_t *)data();
        device_fp16_t* out = (device_fp16_t *)y->acl_fp16()->data();
        float* cos_sin = (float *)cached->acl_float()->data();

        acl_kernels::rotary_embed<device_fp16_t>(in, cos_sin, pos, out, batch, heads, tokens, hidden);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType DT>
ComputingReturn ACLTensor<DT>::op_transpose_0213(tensor_t self, tensor_t y) {
    arm_compute::NEPermute op;
    const arm_compute::PermutationVector  perm(0U,2U,1U,3U);
    if ( DT == DataType::Float ) {
        op.configure(t_, y->acl_float()->t_,  perm);
        op.run();
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        op.configure(t_, y->acl_fp16()->t_,  perm);
        op.run();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::op_qk(tensor_t self, tensor_t key, tensor_t qk) {
    auto shape_ = self->shape().vec();

    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int hhidden = shape_[3];
    int ftokens = key->shape()[2];

    float alpha = 1.0 / sqrt(hhidden);
    float beta = 0.0;

    int HnT = hhidden * ntokens ;
    int HfT = hhidden * ftokens ;
    int TT = ftokens * ntokens;

    arm_compute::GEMMInfo info;
    info.set_pretranspose_A(true);

    arm_compute::NEGEMM op;
    if ( _DTYPE_ == DataType::Float ) {
        for (int i = 0; i < batch * heads; i++) {
            float* A = (float *)data() + i * HnT;
            float* B = (float *)(key->acl_float()->data()) + i * HfT;
            float* C = (float *)(qk->acl_float()->data()) + i * TT;

            arm_compute::Tensor src0;
            key->acl_float()->buildTensorWithShape(src0, {(size_t)ftokens, (size_t)hhidden}, B);
            arm_compute::Tensor src1;
            buildTensorWithShape(src1, {(size_t)ntokens, (size_t)hhidden}, A);
            arm_compute::Tensor dst;
            qk->acl_float()->buildTensorWithShape(dst, {(size_t)ntokens, (size_t)ftokens}, C);

            op.configure(&src0, &src1, nullptr, &dst, alpha, beta, info);
            op.run();
        }
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        for (int i = 0; i < batch * heads; i++) {
            device_fp16_t* A = (device_fp16_t *)data() + i * HnT;
            device_fp16_t* B = (device_fp16_t *)(key->acl_fp16()->data()) + i * HfT;
            device_fp16_t* C = (device_fp16_t *)(qk->acl_fp16()->data()) + i * TT;

            arm_compute::Tensor src0;
            key->acl_fp16()->buildTensorWithShape(src0, {(size_t)ftokens, (size_t)hhidden}, B);
            arm_compute::Tensor src1;
            buildTensorWithShape(src1, {(size_t)ntokens, (size_t)hhidden}, A);
            arm_compute::Tensor dst;
            qk->acl_fp16()->buildTensorWithShape(dst, {(size_t)ntokens, (size_t)ftokens}, C);

            op.configure(&src0, &src1, nullptr, &dst, alpha, beta, info);
            op.run();
        }
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn ACLTensor<_DTYPE_>::op_softmax(tensor_t self, tensor_t dst_) {
    auto shape_ = self->shape().vec();

    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int ftokens = shape_[3];

    size_t num = batch * heads * ntokens;

    arm_compute::NESoftmaxLayerGeneric<false> op;

    if ( _DTYPE_ == DataType::Float) {
        arm_compute::Tensor src;
        buildTensorWithShape(src, {num, (size_t)ftokens});
        arm_compute::Tensor dst;
        dst_->acl_float()->buildTensorWithShape(dst, {num, (size_t)ftokens});
        op.configure(&src, &dst);
        op.run();
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16) {
        arm_compute::Tensor src;
        buildTensorWithShape(src, {num, (size_t)ftokens});
        arm_compute::Tensor dst;
        dst_->acl_fp16()->buildTensorWithShape(dst, {num, (size_t)ftokens});
        op.configure(&src, &dst);
        op.run();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType _DTYPE_>
ComputingReturn  ACLTensor<_DTYPE_>::op_attn(tensor_t self, tensor_t value_, tensor_t out_) {
    auto shape_ = self->shape().vec();
    int batch = shape_[0];
    int heads = shape_[1];
    int ntokens = shape_[2];
    int ftokens = shape_[3];
    int hhidden = value_->shape()[3];

    int num = batch * heads;
    int HfT = hhidden * ftokens;
    int HnT = hhidden * ntokens;
    int TT = ftokens * ntokens;

    float alpha = 1.0;
    float beta = 0.0;

    arm_compute::NEGEMM op;
    if ( _DTYPE_ == DataType::Float) {
        for (int i = 0; i < num; i++) {
            float* A = (float *)(value_->acl_float()->data()) + i * HfT;
            float* B = (float *)data() + i * TT;
            float* C = (float *)(out_->acl_float()->data()) + i * HnT;

            arm_compute::Tensor src0;
            value_->acl_float()->buildTensorWithShape(src0, {(size_t)ftokens, (size_t)hhidden}, A);
            arm_compute::Tensor src1;
            buildTensorWithShape(src1, {(size_t)ntokens, (size_t)ftokens}, B);
            arm_compute::Tensor dst;
            out_->acl_float()->buildTensorWithShape(dst, {(size_t)ntokens, (size_t)hhidden}, C);

            op.configure(&src0, &src1, nullptr, &dst, alpha, beta);
            op.run();
        }
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16) {
        for (int i = 0; i < batch * heads; i++) {
            device_fp16_t* A = (device_fp16_t *)(value_->acl_fp16()->data()) + i * HfT;
            device_fp16_t* B = (device_fp16_t *)data() + i * TT;
            device_fp16_t* C = (device_fp16_t *)(out_->acl_fp16()->data()) + i * HnT;

            arm_compute::Tensor src0;
            value_->acl_float()->buildTensorWithShape(src0, {(size_t)ftokens, (size_t)hhidden}, A);
            arm_compute::Tensor src1;
            buildTensorWithShape(src1, {(size_t)ntokens, (size_t)ftokens}, B);
            arm_compute::Tensor dst;
            out_->acl_float()->buildTensorWithShape(dst, {(size_t)ntokens, (size_t)hhidden}, C);

            op.configure(&src0, &src1, nullptr, &dst, alpha, beta);
            op.run();
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template<DataType DT>
ComputingReturn  ACLTensor<DT>::op_silu_product(tensor_t self, tensor_t in, tensor_t out) {
    if ( DT == DataType::Float ) {
        float* src = (float *)data();
        float* in_ = (float *)in->acl_float()->data();
        float* dst = (float *)out->acl_float()->data();

        acl_kernels::silu_product(src, in_, dst, self->items());
        return OP_OK;
    }
    if ( DT == DataType::FP16 ) {
        auto* src = (device_fp16_t *)data();
        auto* in_ = (device_fp16_t *)in->acl_fp16()->data();
        auto* dst = (device_fp16_t *)out->acl_fp16()->data();

        acl_kernels::silu_product(src, in_, dst, self->items());
        return OP_OK;
    }

    return OP_TODO_ERROR;
}

template<DataType DT>
std::variant<ComputingReturn,int> ACLTensor<DT>::op_all_logits(tensor_t self, tensor_t mask_,  tensor_t lm_head, tensor_t output) {
    int batch = self->shape()[0];
    int new_tokens = self->shape()[1];
    size_t hidden_size = self->shape()[2];
    int full_tokens = mask_->shape()[1];

    size_t vocab_size = lm_head->shape()[0];

    int* mask = (int *)mask_->host_int()->data();
    int pred = 0;

    float alpha = 1.0f;
    float beta  = 0.0f;
    arm_compute::NEGEMM op;
    arm_compute::GEMMInfo info;
    info.set_pretranspose_A(true);

    for (int b = 0;  b < batch; b++) {
        int* mk = &mask[b * full_tokens];
        for ( int i = 0; i < new_tokens ; i++) {
            int ii = full_tokens - new_tokens + i;
            if ( mk[ii] != 2 ) {
                continue;
            }
            int target = i;

            if ( DT == DataType::Float ) {
                void* dst = (float *)output->acl_float()->data() + pred * vocab_size;
                void* src = (float *)data() + b * new_tokens * hidden_size + target * hidden_size;
                void* w = (float *)lm_head->acl_float()->data();

                arm_compute::Tensor B;
                buildTensorWithShape(B, {1,  hidden_size}, src);
                arm_compute::Tensor A;
                lm_head->acl_float()->buildTensorWithShape(A, {vocab_size,  hidden_size}, w);

                arm_compute::Tensor D;
                output->acl_float()->buildTensorWithShape(D, {1,  vocab_size}, dst);
                op.configure(&A, &B, nullptr, &D, alpha, beta, info);
                op.run();
            } else if ( DT == DataType::FP16 ) {
                auto* dst = (device_fp16_t *)output->acl_fp16()->data() + pred * vocab_size;
                auto* src = (device_fp16_t *)data() + b * new_tokens * hidden_size + target * hidden_size;
                auto* w = (device_fp16_t *)lm_head->acl_fp16()->data();

                arm_compute::Tensor B;
                arm_compute::Tensor A;
                buildTensorWithShape(B, {1,  hidden_size}, src);
                lm_head->acl_fp16()->buildTensorWithShape(A, {vocab_size,  hidden_size}, w);

                arm_compute::Tensor D;
                output->acl_fp16()->buildTensorWithShape(D, {1,  vocab_size}, dst);
                op.configure(&A, &B, nullptr, &D, alpha, beta, info);
                op.run();

            } else {
                return OP_TODO_ERROR;
            }
            pred ++;
        }
    }

    return pred;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t>  ACLTensor<DT>::op_sampling_top1(tensor_t self) {
    if ( DT != DataType::Float && DT != DataType::FP16 ) {
        return OP_INPUT_ERROR;
    }

    int batch = self->shape()[0];
    int vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_int( ret_shape );
    int* out = (int *)ret->device_data();

    if ( DT == DataType::FP16 ) {
        device_fp16_t* logits = (device_fp16_t *) self->device_data();
        acl_kernels::easy_top1<device_fp16_t>(logits, out, batch, vocab_size);
    } else {
        float* logits = (float *) self->device_data();
        acl_kernels::easy_top1<float>(logits, out, batch, vocab_size);
    }
    return ret;
}

template<DataType DT>
std::variant<ComputingReturn, tensor_t>  ACLTensor<DT>::op_sampling_top3(tensor_t self, float temp) {
    if ( DT != DataType::Float && DT != DataType::FP16 ) {
        return OP_INPUT_ERROR;
    }

    int batch = self->shape()[0];
    int vocab_size = self->shape()[1];

    std::vector<size_t> ret_shape{ (size_t)batch};
    tensor_t ret = vt::create_host_int( ret_shape );
    int* out = (int *)ret->device_data();

    std::uniform_real_distribution<> dist(0.0, 1.0);
    float randx = dist( *ComputingContext::rng );

    if ( DT == DataType::FP16 ) {
        device_fp16_t* logits = (device_fp16_t *) self->device_data();
        acl_kernels::easy_top3<device_fp16_t>(logits, out, batch, vocab_size, temp, randx);
    } else {
        float* logits = (float *) self->device_data();
        acl_kernels::easy_top3<float>(logits, out, batch, vocab_size, temp, randx);
    }
    return ret;
}

tensor_t create_acl_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    ACLTensor<DataType::Float>* tensor = new ACLTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_acl_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    ACLTensor<DataType::FP16>* tensor = new ACLTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_acl_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    ACLTensor<DataType::Int>* tensor = new ACLTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
