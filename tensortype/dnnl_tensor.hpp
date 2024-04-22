#ifndef _DNNL_IMPL_HPP_
#define _DNNL_IMPL_HPP_



#include "vt.hpp"
#include "context.hpp"
#include "computing.hpp"
#include "tensortype.hpp"


namespace vt {

template <DataType _DTYPE_>
struct DNNLTensor : public TransformerComputing {
    virtual ~DNNLTensor();
    DNNLTensor(const ShapeType& shape, bool isGPU = false);
    DNNLTensor(const ShapeType& shape,  void *mem, bool isGPU = false);
    void* data() {
        return mem_;
    }

    dnnl::memory::desc build_memory_desc(const std::vector<size_t>& shape, DataType dt, dnnl::memory::format_tag tag) {
        dnnl::memory::dims dims;
        for(int i = 0; i < (int)shape.size(); i++) {
            dims.push_back(shape[i]);
        }
        if ( dt == DataType::Float ) {
            return dnnl::memory::desc(dims,  dnnl::memory::data_type::f32, tag);
        }
        if ( dt == DataType::FP16 ) {
            return dnnl::memory::desc(dims,  dnnl::memory::data_type::f16, tag);
        }

        vt_panic("Can't be here!");
        return dnnl::memory::desc();
    }
    dnnl::memory::desc build_memory_desc(const std::vector<size_t>& shape, dnnl::memory::format_tag tag);
    dnnl::memory build_memory(const dnnl::memory::desc& desc);

public:
    // Interfaces from TransformerComputing
    ComputingReturn io_dump(tensor_t self) override;
    ComputingReturn io_load(tensor_t self, const char* fileName) override;
    ComputingReturn io_save(tensor_t self, const char* fileName) override;

    std::variant<ComputingReturn, size_t> op_sizeof(tensor_t self) override;
    ComputingReturn op_zero(tensor_t self) override;
    ComputingReturn op_fill(tensor_t self, float value) override;
    ComputingReturn op_alibi(tensor_t self) override;
    ComputingReturn op_causal_mask(tensor_t self, tensor_t out) override;
    ComputingReturn op_rotary_cache(tensor_t self, float base) override;

    ComputingReturn op_copy(tensor_t self, tensor_t from) override;
    ComputingReturn op_convert(tensor_t self, tensor_t from) override;
    std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) override;
    std::variant<ComputingReturn, tensor_t> op_view_as(tensor_t self, size_t offset, const std::vector<size_t>& newShape, const char* dtype) override;
    ComputingReturn op_reshape(tensor_t self, size_t offset, const std::vector<size_t>& newShape) override;

    ComputingReturn op_scale(tensor_t self, float scale) override;
    ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) override;
    ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) override;
    
    ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) override;
    ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) override;
    ComputingReturn op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) override;
    ComputingReturn op_rotary_embed(tensor_t self, tensor_t cached, tensor_t pos, tensor_t y) override;

    ComputingReturn op_transpose_0213(tensor_t self, tensor_t y) override;
    ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) override;
    ComputingReturn op_softmax(tensor_t self, tensor_t out) override;
    ComputingReturn op_attn(tensor_t self, tensor_t value, tensor_t out) override;
    ComputingReturn op_gelu(tensor_t self, tensor_t dst) override;
    ComputingReturn op_silu_product(tensor_t self, tensor_t in, tensor_t dst) override;

    std::variant<ComputingReturn, int> op_all_logits(tensor_t self, tensor_t mask,  tensor_t lm_head, tensor_t output) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top1(tensor_t self) override;
    std::variant<ComputingReturn, tensor_t> op_sampling_top3(tensor_t self, float temp) override;
    
    ComputingReturn op_conv2d(tensor_t self, tensor_t weight, tensor_t bias, tensor_t dst, int stride, int padding) override;

protected:
    const bool owner_;
    const bool gpu_;
    void* mem_;
    size_t size_;

    friend struct DNNLTensor<DataType::Float>;
    friend struct DNNLTensor<DataType::Int>;
    friend struct DNNLTensor<DataType::FP16>;
};


} // end of namespace
#endif
