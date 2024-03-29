#ifndef _DNNL_KERNELS_HPP_
#define _DNNL_KERNELS_HPP_

namespace vt { namespace dnnl_kernels {

void binary_operate_float(tensor_t a, tensor_t b, tensor_t c, dnnl::algorithm op ) {
    auto amem_desc = a->dnnl_float()->build_memory_desc( a->shape().vec(),  dnnl::memory::format_tag::abcd);
    auto bmem_desc = b->dnnl_float()->build_memory_desc( b->shape().vec(),  dnnl::memory::format_tag::abcd);
    auto cmem_desc = c->dnnl_float()->build_memory_desc( c->shape().vec(),  dnnl::memory::format_tag::abcd);

    auto amem = a->dnnl_float()->build_memory(amem_desc);
    auto bmem = b->dnnl_float()->build_memory(bmem_desc);
    auto cmem = c->dnnl_float()->build_memory(cmem_desc);

    auto binary_pd = dnnl::binary::primitive_desc(*ComputingContext::dnnl_engine, op, amem_desc, bmem_desc, cmem_desc);
    auto binary_prim = dnnl::binary(binary_pd);

    std::unordered_map<int, dnnl::memory> binary_args;
    binary_args[DNNL_ARG_SRC_0] = amem;
    binary_args[DNNL_ARG_SRC_1] = bmem;
    binary_args[DNNL_ARG_DST] = cmem;

    binary_prim.execute(*ComputingContext::dnnl_stream, binary_args);
}

void binary_operate_fp16(tensor_t a, tensor_t b, tensor_t c, dnnl::algorithm op ) {
    auto amem_desc = a->dnnl_fp16()->build_memory_desc( a->shape().vec(),  dnnl::memory::format_tag::abcd);
    auto bmem_desc = b->dnnl_fp16()->build_memory_desc( b->shape().vec(),  dnnl::memory::format_tag::abcd);
    auto cmem_desc = c->dnnl_fp16()->build_memory_desc( c->shape().vec(),  dnnl::memory::format_tag::abcd);

    auto amem = a->dnnl_fp16()->build_memory(amem_desc);
    auto bmem = b->dnnl_fp16()->build_memory(bmem_desc);
    auto cmem = c->dnnl_fp16()->build_memory(cmem_desc);

    auto binary_pd = dnnl::binary::primitive_desc(*ComputingContext::dnnl_engine, op, amem_desc, bmem_desc, cmem_desc);
    auto binary_prim = dnnl::binary(binary_pd);

    std::unordered_map<int, dnnl::memory> binary_args;
    binary_args[DNNL_ARG_SRC_0] = amem;
    binary_args[DNNL_ARG_SRC_1] = bmem;
    binary_args[DNNL_ARG_DST] = cmem;

    binary_prim.execute(*ComputingContext::dnnl_stream, binary_args);
}

template<typename T>
void eltwise_operate(T* in, T* out,  size_t items, ::dnnl::algorithm op, float alpha, float beta) {
    auto src_md = in->build_memory_desc( {items},  dnnl::memory::format_tag::a);
    auto dst_md = out->build_memory_desc( {items},  dnnl::memory::format_tag::a);
    auto src_mem = in->build_memory(src_md);
    auto dst_mem = out->build_memory(dst_md);

    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(*ComputingContext::dnnl_engine,
        dnnl::prop_kind::forward_inference, op, src_md, dst_md, alpha, beta);

    auto eltwise_prim = dnnl::eltwise_forward(eltwise_pd);
    std::unordered_map<int, dnnl::memory> eltwise_args;
    eltwise_args[DNNL_ARG_SRC] = src_mem;
    eltwise_args[DNNL_ARG_DST] = dst_mem;
    eltwise_prim.execute(*ComputingContext::dnnl_stream, eltwise_args);
}

template<typename T>
void linear_operate(T* src, T* weight, T* bias, T* dst, size_t batch, size_t outFeature, size_t inFeature ) {
    auto src_md = src->build_memory_desc( {batch, inFeature},  dnnl::memory::format_tag::ab);
    auto w_md = weight->build_memory_desc( {inFeature, outFeature}, dnnl::memory::format_tag::ba);
    dnnl::memory::desc b_md;
    if ( bias != nullptr) {
        b_md = bias->build_memory_desc( {outFeature}, dnnl::memory::format_tag::a);
    }
    auto dst_md = dst->build_memory_desc( {batch, outFeature}, dnnl::memory::format_tag::ab);
    
    dnnl::matmul::primitive_desc matmul_pd;
    if ( bias == nullptr) {
        matmul_pd = dnnl::matmul::primitive_desc(
            *ComputingContext::dnnl_engine, src_md, w_md, dst_md);
    } else {
        matmul_pd = dnnl::matmul::primitive_desc(
            *ComputingContext::dnnl_engine, src_md, w_md, b_md, dst_md);
    }
    auto matmul_prim = dnnl::matmul(matmul_pd);

    std::unordered_map<int, dnnl::memory> matmul_args;

    matmul_args[DNNL_ARG_SRC] = src->build_memory(src_md);
    matmul_args[DNNL_ARG_WEIGHTS] = weight->build_memory(w_md);
    matmul_args[DNNL_ARG_DST] = dst->build_memory(dst_md);
    if ( bias != nullptr ) {
         matmul_args[DNNL_ARG_BIAS] = bias->build_memory( b_md );
    }
    matmul_prim.execute(*ComputingContext::dnnl_stream, matmul_args);
}



}}
#endif
