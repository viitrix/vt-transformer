#include <chrono>
#include "tensortype.hpp"
#include "context.hpp"
#include "dag.hpp"

namespace vt {

namespace op {
    std::vector<size_t> fetch_shape(Stack& stack) {
        auto nums = stack.pop_number_list();
        std::vector<size_t> shape;
        for ( size_t i = 0; i < nums.size(); i++) {
            shape.push_back( nums[i] );
        }
        return shape;
    }
    struct Sync : public NativeWord {
        void run(Stack& stack) override {
#ifdef _USING_DEVICE_CUDA_
            auto stream = vt::ComputingContext::cuda_stream;
            CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        }
        NWORD_CREATOR_DEFINE_LR(Sync)
    };

    struct CheckPoint : public NativeWord {
        static std::chrono::time_point<std::chrono::high_resolution_clock> ck;

        void run(Stack& stack) override {
            int flag = stack.pop_number();
            if ( flag == 0 ) {
                ck = std::chrono::high_resolution_clock::now();
            } else {
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - ck);
                std::cout << "+++++ " << duration.count() / 1000.0 << " +++++"<< std::endl;
            }
        }
        NWORD_CREATOR_DEFINE_LR(CheckPoint)
    };
    std::chrono::time_point<std::chrono::high_resolution_clock> CheckPoint::ck;

#ifdef _USING_DEVICE_CUDA_
    struct CudaEvent: public NativeWord {
        void run(Stack& stack) override {
            int flag = stack.pop_number();
            float ret = ComputingContext::cuda_event(flag);
            std::cout << "CudaEvent " << flag << " : " << ret << std::endl;
        }
        NWORD_CREATOR_DEFINE_LR(CudaEvent)
    };
#endif

    struct Shape : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            for (size_t i = 0; i < x->shape().dim(); i++) {
                stack.push_number( x->shape()[i] );
            }
            stack.push_number( x->shape().dim() );
        }
        NWORD_CREATOR_DEFINE_LR(Shape)
    };

    struct Device : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            std::string dname = x->device_name();
            stack.push_string(dname);
        }
        NWORD_CREATOR_DEFINE_LR(Device)
    };

    struct DataType : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            stack.push_string( DataType_name( x->dtype() ));
        }
        NWORD_CREATOR_DEFINE_LR(DataType)
    };

    struct Create : public NativeWord {
        void run(Stack& stack) override {
            vt::DataType dtype = DataType_from( stack.pop_string().c_str() );
            auto device = stack.pop_string();
            auto shape = fetch_shape(stack);
            tensor_t t;
            if ( device == "cuda" ) {
#ifdef _USING_DEVICE_CUDA_
                if ( dtype == vt::Float ) {
                    t = vt::create_cuda_float(shape);
                } else if ( dtype == vt::FP16 ) {
                    t = vt::create_cuda_fp16(shape);
                } else if ( dtype == vt::Int ) {
                    t = vt::create_cuda_int(shape);
                } else if ( dtype == vt::Q8 ) {
                    t = vt::create_cuda_q8(shape);
                } else if ( dtype == vt::Q4 ) {
                    t = vt::create_cuda_q4(shape);
                } else {
                    vt_panic("Can' be here!");
                }
#else
                vt_panic("Can' be here!");
#endif
            } else if ( device == "dcu" ) {
#ifdef _USING_DEVICE_DCU_
                if ( dtype == vt::Float ) {
                    t = vt::create_dcu_float(shape);
                } else if ( dtype == vt::FP16 ) {
                    t = vt::create_dcu_fp16(shape);
                } else if ( dtype == vt::Int ) {
                    t = vt::create_dcu_int(shape);
                } else {
                    vt_panic("Can' be here!");
                }
#else
                vt_panic("Can' be here!");
#endif
            } else if ( device == "host" ) {
                if ( dtype == vt::Float ) {
                    t = vt::create_host_float(shape);
                } else if ( dtype == vt::FP16 ) {
                    t = vt::create_host_fp16(shape);
                } else if ( dtype == vt::Int ) {
                    t = vt::create_host_int(shape);
                } else if ( dtype == vt::Q8 ) {
                    t = vt::create_host_q8(shape);
                } else if ( dtype == vt::Q4 ) {
                    t = vt::create_host_q4(shape);
                } else {
                    vt_panic("Can' be here!");
                }
            } else {
                vt_panic("Can' be here!");
            }
            stack.push_tensor(t);
        }
        NWORD_CREATOR_DEFINE_LR(Create)
    };

    struct Null : public NativeWord {
        void run(Stack& stack) override {
            tensor_t ret = nullptr;
            stack.push_tensor(ret);
        }

        NWORD_CREATOR_DEFINE_LR(Null);
    };

    struct Zero : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->op_zero(t);
        }
        NWORD_CREATOR_DEFINE_LR(Zero)
    };

    struct Fill : public NativeWord {
        void run(Stack& stack) override {
            double value = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_fill(t, value);
        }
        NWORD_CREATOR_DEFINE_LR(Fill)
    };

    struct Alibi : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->op_alibi(t);
        }
        NWORD_CREATOR_DEFINE_LR(Alibi)
    };

    struct RotaryCache : public NativeWord {
        void run(Stack& stack) override {
            float base = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_rotary_cache(t, base);
        }
        NWORD_CREATOR_DEFINE_LR(RotaryCache)
    };

    struct CausalMask : public NativeWord {
        void run(Stack& stack) override {
            auto out = stack.pop_tensor();
            auto self = stack.pop_tensor();
            self->op_causal_mask(self, out);
        }
        NWORD_CREATOR_DEFINE_LR(CausalMask)
    };

    struct View : public NativeWord {
        void run(Stack& stack) override {
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            auto ret = t->op_view(t, offset, shape);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_LR(View)
    };

    struct ViewAs : public NativeWord {
        void run(Stack& stack) override {
            auto dtype = stack.pop_string();
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            auto ret = t->op_view_as(t, offset, shape, dtype.c_str());
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_LR(ViewAs)
    };

    struct Reshape : public NativeWord {
        void run(Stack& stack) override {
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_reshape(t, offset, shape);
        }
        NWORD_CREATOR_DEFINE_LR(Reshape)
    };

    struct Quantize : public NativeWord {
        void run(Stack& stack) override {
            tensor_t dst = stack.pop_tensor();
            tensor_t src = stack.pop_tensor();
            src->op_quantize(src, dst);
        }
        NWORD_CREATOR_DEFINE_LR(Quantize)
    };

    struct DeQuantize : public NativeWord {
        void run(Stack& stack) override {
            tensor_t dst = stack.pop_tensor();
            tensor_t src = stack.pop_tensor();
            src->op_dequantize(src, dst);
        }
        NWORD_CREATOR_DEFINE_LR(DeQuantize)
    };

    struct Embed : public NativeWord {
        void run(Stack& stack) override {
            auto out = stack.pop_tensor();
            auto table = stack.pop_tensor();
            auto self = stack.pop_tensor();

            self->op_embed(self, table, out);
        }
        NWORD_CREATOR_DEFINE_LR(Embed)
    };

    struct Copy : public NativeWord {
        void run(Stack& stack) override {
            tensor_t src = stack.pop_tensor();
            tensor_t dst = stack.pop_tensor();
            dst->op_copy(dst, src);
        }
        NWORD_CREATOR_DEFINE_LR(Copy)
    };

    struct Convert : public NativeWord {
        void run(Stack& stack) override {
            tensor_t src = stack.pop_tensor();
            tensor_t dst = stack.pop_tensor();
            dst->op_convert(dst, src);
        }
        NWORD_CREATOR_DEFINE_LR(Convert)
    };

    struct Linear : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t w = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_linear(x, w, b, y);
        }
        NWORD_CREATOR_DEFINE_LR(Linear)
    };

    struct Layernorm : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t y = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t var = stack.pop_tensor();
            tensor_t mean = stack.pop_tensor();

            tensor_t x = stack.pop_tensor();

            x->op_layernorm(x, mean, var, scale, bias, y, eps);
        }
        NWORD_CREATOR_DEFINE_LR(Layernorm)
    };

    struct RMSnorm : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t y = stack.pop_tensor();
            tensor_t norm2 = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_rmsnorm(x, scale, norm2, y, eps);
        }
        NWORD_CREATOR_DEFINE_LR(RMSnorm)
    };

    struct RotaryEmbed : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t pos = stack.pop_tensor();
            tensor_t cached = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_rotary_embed(x, cached, pos, out);
        }
        NWORD_CREATOR_DEFINE_LR( RotaryEmbed );
    };

    struct Transpos0213 : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_transpos_0213(x,y);
        }
        NWORD_CREATOR_DEFINE_LR(Transpos0213)
    };

    struct QueryKey : public NativeWord {
        void run(Stack& stack) override {
            tensor_t qk = stack.pop_tensor();
            tensor_t k = stack.pop_tensor();
            tensor_t q = stack.pop_tensor();

            q->op_qk(q, k, qk);
        }
        NWORD_CREATOR_DEFINE_LR(QueryKey)
    };

    struct Scale : public NativeWord {
        void run(Stack& stack) override {
            float scale = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->op_scale(x, scale);
        }
        NWORD_CREATOR_DEFINE_LR(Scale)
    };

    struct Add : public NativeWord {
        void run(Stack& stack) override {
            tensor_t c = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_add(x, b, c);
        }
        NWORD_CREATOR_DEFINE_LR(Add)
    };

    struct Mul : public NativeWord {
        void run(Stack& stack) override {
            tensor_t c = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_mul(x, b, c);
        }
        NWORD_CREATOR_DEFINE_LR(Mul)
    };

    struct Softmax : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_softmax(x, out);
        }
        NWORD_CREATOR_DEFINE_LR(Softmax)
    };

    struct Attn : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t value = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_attn(x, value, out);
        }
        NWORD_CREATOR_DEFINE_LR(Attn)
    };

    struct XAttn : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t qk = stack.pop_tensor();
            tensor_t value = stack.pop_tensor();
            tensor_t key = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_xattn(x, key, value, qk, out);
        }
        NWORD_CREATOR_DEFINE_LR(XAttn)
    };

    struct Gelu : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_gelu(x, out);
        }
        NWORD_CREATOR_DEFINE_LR(Gelu)
    };

    struct SiluProduct : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t in = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_silu_product(x, in, out);
        }
        NWORD_CREATOR_DEFINE_LR(SiluProduct)
    };

    struct AllLogits : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t lm_head = stack.pop_tensor();
            tensor_t mask = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            auto ret = x->op_all_logits(x, mask, lm_head, out);

            stack.push_number( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_LR(AllLogits)
    };

    struct SamplingTop3 : public NativeWord {
        void run(Stack& stack) override {
            float temp = stack.pop_number();
            tensor_t logits = stack.pop_tensor();
            auto ret = logits->op_sampling_top3(logits, temp);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_LR(SamplingTop3);
    };

    struct LossBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t lm_head_g = stack.pop_tensor();
            tensor_t x_g = stack.pop_tensor();
            tensor_t workspace = stack.pop_tensor();
            tensor_t lm_head = stack.pop_tensor();
            tensor_t mask = stack.pop_tensor();
            tensor_t ids = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_loss_backward(x, ids, mask, lm_head, workspace, x_g, lm_head_g);
        }
        NWORD_CREATOR_DEFINE_LR(LossBackward)
    };

    struct LayernormBackward : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t din = stack.pop_tensor();
            tensor_t dbias = stack.pop_tensor();
            tensor_t dscale = stack.pop_tensor();
            tensor_t y = stack.pop_tensor();
            tensor_t var = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_layernorm_backward(self, scale, bias, var, y, dscale, dbias, din, eps);
        }
        NWORD_CREATOR_DEFINE_LR(LayernormBackward)
    };
    struct LinearBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t bias_g = stack.pop_tensor();
            tensor_t weight_g = stack.pop_tensor();
            tensor_t x_g = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t weight = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_linear_backward(self, x, weight, bias, x_g, weight_g, bias_g);
        }
        NWORD_CREATOR_DEFINE_LR(LinearBackward)
    };
    struct GeluBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x_g = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_gelu_backward(self, x, x_g);
        }
        NWORD_CREATOR_DEFINE_LR(GeluBackward)
    };
    struct AttnBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t v_g = stack.pop_tensor();
            tensor_t attn_g = stack.pop_tensor();
            tensor_t v = stack.pop_tensor();
            tensor_t attn = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_attn_backward(self, attn, v, attn_g, v_g);
        }
        NWORD_CREATOR_DEFINE_LR(AttnBackward)
    };
    struct SoftmaxBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x_g = stack.pop_tensor();
            tensor_t out = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_softmax_backward(self, out, x_g);
        }
        NWORD_CREATOR_DEFINE_LR(SoftmaxBackward)
    };
    struct SoftmaxAttnBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t v_g = stack.pop_tensor();
            tensor_t attn_g = stack.pop_tensor();
            tensor_t v = stack.pop_tensor();
            tensor_t attn = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_softmax_attn_backward(self, attn, v, attn_g, v_g);
        }
        NWORD_CREATOR_DEFINE_LR(SoftmaxAttnBackward)
    };

    struct QueryKeyBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t k_g = stack.pop_tensor();
            tensor_t q_g = stack.pop_tensor();
            tensor_t k = stack.pop_tensor();
            tensor_t q = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_qk_backward(self, q, k, q_g, k_g);
        }
        NWORD_CREATOR_DEFINE_LR(QueryKeyBackward)
    };
}

namespace io {
    struct Dump : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->io_dump(t);
        }
        NWORD_CREATOR_DEFINE_LR(Dump)
    };

    struct Load : public NativeWord {
        void run(Stack& stack) override {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_load(x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_LR(Load)
    };

    struct Save : public NativeWord {
        void run(Stack& stack) override {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_save(x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_LR(Save)
    };

    struct MPIRank : public NativeWord {
        void run(Stack& stack) override {
#ifdef _USING_HPC_OPENMPI_
            stack.push_number( CollectiveContext::mpi_rank );
#else

            stack.push_number( 0 );
#endif
        }
        NWORD_CREATOR_DEFINE_LR(MPIRank)
    };

    struct PipeRank : public NativeWord {
        void run(Stack& stack) override {
            stack.push_number( CollectiveContext::pipe_rank );
        }
        NWORD_CREATOR_DEFINE_LR(PipeRank)
    };

    struct NcclRank : public NativeWord {
        void run(Stack& stack) override {
#ifdef _USING_DEVICE_CUDA_
            stack.push_number( CollectiveContext::nccl_rank );
#else
            stack.push_number(0);
#endif
        }
        NWORD_CREATOR_DEFINE_LR(NcclRank)
    };

    struct MPIRecv : public NativeWord {
        void run(Stack& stack) override {
            int source = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_recv(x, source);
        }
        NWORD_CREATOR_DEFINE_LR(MPIRecv)
    };

    struct MPISend : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_send(x, dst);
        }
        NWORD_CREATOR_DEFINE_LR(MPISend)
    };

    struct MPIBcast : public NativeWord {
        void run(Stack& stack) override {
            int root = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_bcast(x, root);
        }
        NWORD_CREATOR_DEFINE_LR(MPIBcast)
    };

    struct NcclRecv : public NativeWord {
        void run(Stack& stack) override {
            int source = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_nccl_recv(x, source);
        }
        NWORD_CREATOR_DEFINE_LR(NcclRecv)
    };

    struct NcclSend : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_nccl_send(x, dst);
        }
        NWORD_CREATOR_DEFINE_LR(NcclSend)
    };

    struct PipeRead : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            x->io_pipe_read(x);
        }
        NWORD_CREATOR_DEFINE_LR(PipeRead)
    };

    struct PipeWrite : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_pipe_write(x, dst);
        }
        NWORD_CREATOR_DEFINE_LR(PipeWrite)
    };

}

void load_nn_operators(Enviroment& env) {
    env.insert_native_word("io.dump", io::Dump::creator );
    env.insert_native_word("io.load", io::Load::creator );
    env.insert_native_word("io.save", io::Save::creator );

    env.insert_native_word("io.mpi_rank", io::MPIRank::creator );
    env.insert_native_word("io.pipe_rank", io::PipeRank::creator );
    env.insert_native_word("io.nccl_rank", io::NcclRank::creator );
    env.insert_native_word("io.mpi.bcast", io::MPIBcast::creator );
    env.insert_native_word("io.mpi.recv", io::MPIRecv::creator );
    env.insert_native_word("io.mpi.send", io::MPISend::creator );
    env.insert_native_word("io.nccl.send", io::NcclSend::creator );
    env.insert_native_word("io.nccl.recv", io::NcclRecv::creator );
    env.insert_native_word("io.pipe.read", io::PipeRead::creator );
    env.insert_native_word("io.pipe.write", io::PipeWrite::creator );

    env.insert_native_word("op.sync", op::Sync::creator );
    env.insert_native_word("op.check", op::CheckPoint::creator );
#if _USING_DEVICE_CUDA_
    env.insert_native_word("op.cuda_event", op::CudaEvent::creator );
#endif
    env.insert_native_word("op.get_shape", op::Shape::creator);
    env.insert_native_word("op.get_device", op::Device::creator);
    env.insert_native_word("op.get_dtype", op::DataType::creator);
    env.insert_native_word("op.create", op::Create::creator );
    env.insert_native_word("op.null", op::Null::creator );
    env.insert_native_word("op.zero", op::Zero::creator );
    env.insert_native_word("op.fill", op::Fill::creator );
    env.insert_native_word("op.alibi", op::Alibi::creator );
    env.insert_native_word("op.rotary_cache", op::RotaryCache::creator );
    env.insert_native_word("op.causal_mask", op::CausalMask::creator );
    env.insert_native_word("op.scale", op::Scale::creator );
    env.insert_native_word("op.view", op::View::creator );
    env.insert_native_word("op.view_as", op::ViewAs::creator );
    env.insert_native_word("op.reshape", op::Reshape::creator );
    env.insert_native_word("op.quantize", op::Quantize::creator );
    env.insert_native_word("op.dequantize", op::DeQuantize::creator );
    env.insert_native_word("op.embed", op::Embed::creator );
    env.insert_native_word("op.copy", op::Copy::creator );
    env.insert_native_word("op.convert", op::Convert::creator );
    env.insert_native_word("op.linear", op::Linear::creator );
    env.insert_native_word("op.layernorm", op::Layernorm::creator );
    env.insert_native_word("op.rmsnorm", op::RMSnorm::creator );
    env.insert_native_word("op.rotary_embed", op::RotaryEmbed::creator );
    env.insert_native_word("op.transpos_0213", op::Transpos0213::creator );
    env.insert_native_word("op.add", op::Add::creator);
    env.insert_native_word("op.mul", op::Mul::creator);
    env.insert_native_word("op.querykey", op::QueryKey::creator);
    env.insert_native_word("op.softmax", op::Softmax::creator);
    env.insert_native_word("op.attn", op::Attn::creator);
    env.insert_native_word("op.xattn", op::XAttn::creator);
    env.insert_native_word("op.gelu", op::Gelu::creator);
    env.insert_native_word("op.silu_product", op::SiluProduct::creator);
    env.insert_native_word("op.all_logits", op::AllLogits::creator);
    env.insert_native_word("op.sampling_top3", op::SamplingTop3::creator);
    env.insert_native_word("op.loss_backward", op::LossBackward::creator);
    env.insert_native_word("op.layernorm_backward", op::LayernormBackward::creator);
    env.insert_native_word("op.linear_backward", op::LinearBackward::creator);
    env.insert_native_word("op.gelu_backward", op::GeluBackward::creator);
    env.insert_native_word("op.attn_backward", op::AttnBackward::creator);
    env.insert_native_word("op.softmax_backward", op::SoftmaxBackward::creator);
    env.insert_native_word("op.softmax_attn_backward", op::SoftmaxAttnBackward::creator);
    env.insert_native_word("op.qk_backward", op::QueryKeyBackward::creator);
}

}// end of namespace br
