#ifndef _DNNL_MISC_HPP_
#define _DNNL_MISC_HPP_

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
}

}}
#endif
