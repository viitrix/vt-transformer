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
        NWORD_CREATOR_DEFINE_CTX(CheckPoint)
    };
    std::chrono::time_point<std::chrono::high_resolution_clock> CheckPoint::ck;

    struct Shape : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            for (size_t i = 0; i < x->shape().dim(); i++) {
                stack.push_number( x->shape()[i] );
            }
            stack.push_number( x->shape().dim() );
        }
        NWORD_CREATOR_DEFINE_CTX(Shape)
    };

    struct Device : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            std::string dname = x->device_name();
            stack.push_string(dname);
        }
        NWORD_CREATOR_DEFINE_CTX(Device)
    };

    struct DataType : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            stack.push_string( DataType_name( x->dtype() ));
        }
        NWORD_CREATOR_DEFINE_CTX(DataType)
    };

    struct Create : public NativeWord {
        void run(Stack& stack) override {
            vt::DataType dtype = DataType_from( stack.pop_string().c_str() );
            auto device = stack.pop_string();
            std::vector<size_t> shape;
            shape = fetch_shape(stack);

            tensor_t t;
            if ( device == "cuda" ) {
#ifdef _USING_DEVICE_CUDA_
                if ( dtype == vt::F32 ) {
                    t = vt::create_cuda_f32(shape);
                } else if ( dtype == vt::F16 ) {
                    t = vt::create_cuda_f16(shape);
                 } else if ( dtype == vt::BF16 ) {
                    t = vt::create_cuda_bf16(shape);
                } else if ( dtype == vt::I32 ) {
                    t = vt::create_cuda_i32(shape);
                } else if ( dtype == vt::Q8 ) {
                    t = vt::create_cuda_q8(shape);
                } else if ( dtype == vt::Q4 ) {
                    t = vt::create_cuda_q4(shape);
                } else if ( dtype == vt::PQ ) {
                    t = vt::create_cuda_pq(shape);
                } else {
                    vt_fatal_error();
                }
#else
                vt_fatal_error();
#endif
            } else if ( device == "hip" ) {
#ifdef _USING_DEVICE_HIP_
                if ( dtype == vt::F32 ) {
                    t = vt::create_hip_f32(shape);
                } else if ( dtype == vt::F16 ) {
                    t = vt::create_hip_f16(shape);
                 } else if ( dtype == vt::BF16 ) {
                    t = vt::create_hip_bf16(shape);
                } else if ( dtype == vt::I32 ) {
                    t = vt::create_hip_i32(shape);
                } else if ( dtype == vt::Q8 ) {
                    t = vt::create_hip_q8(shape);
                } else if ( dtype == vt::Q4 ) {
                    t = vt::create_hip_q4(shape);
                } else if ( dtype == vt::PQ ) {
                    t = vt::create_hip_pq(shape);
                } else {
                    vt_fatal_error();
                }
#else
                vt_fatal_error();
#endif
            } else if ( device == "host" ) {
                if ( dtype == vt::F32 ) {
                    t = vt::create_host_f32(shape);
                } else if ( dtype == vt::F16 ) {
                    t = vt::create_host_f16(shape);
                } else if ( dtype == vt::BF16 ) {
                    t = vt::create_host_bf16(shape);
                } else if ( dtype == vt::I32 ) {
                    t = vt::create_host_i32(shape);
                } else if ( dtype == vt::Q8 ) {
                    t = vt::create_host_q8(shape);
                } else if ( dtype == vt::Q4 ) {
                    t = vt::create_host_q4(shape);
                } else if ( dtype == vt::PQ ) {
                    t = vt::create_host_pq(shape);
                } else {
                    vt_fatal_error();
                }
            } else {
                vt_fatal_error();
            }
            stack.push_tensor(t);
        }
        NWORD_CREATOR_DEFINE_CTX(Create)
    };

    struct Null : public NativeWord {
        void run(Stack& stack) override {
            tensor_t ret = nullptr;
            stack.push_tensor(ret);
        }

        NWORD_CREATOR_DEFINE_CTX(Null);
    };

    struct Sizeof : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            auto ret = x->op_sizeof(ctx_, x);
            stack.push_number( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_CTX(Sizeof)
    };

    struct Zero : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->op_zero(ctx_, t);
        }
        NWORD_CREATOR_DEFINE_CTX(Zero)
    };

    struct Fill : public NativeWord {
        void run(Stack& stack) override {
            double value = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_fill(ctx_, t, value);
        }
        NWORD_CREATOR_DEFINE_CTX(Fill)
    };

    struct RotaryCache : public NativeWord {
        void run(Stack& stack) override {
            float base = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_rotary_cache(ctx_, t, base);
        }
        NWORD_CREATOR_DEFINE_CTX(RotaryCache)
    };

    struct CausalMask : public NativeWord {
        void run(Stack& stack) override {
            auto out = stack.pop_tensor();
            auto self = stack.pop_tensor();
            self->op_causal_mask(ctx_, self, out);
        }
        NWORD_CREATOR_DEFINE_CTX(CausalMask)
    };

    struct View : public NativeWord {
        void run(Stack& stack) override {
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            auto ret = t->op_view(ctx_, t, offset, shape);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_CTX(View)
    };

    struct ViewAs : public NativeWord {
        void run(Stack& stack) override {
            auto dtype = stack.pop_string();
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            auto ret = t->op_view_as(ctx_, t, offset, shape, dtype.c_str());
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_CTX(ViewAs)
    };

    struct Reshape : public NativeWord {
        void run(Stack& stack) override {
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_reshape(ctx_, t, offset, shape);
        }
        NWORD_CREATOR_DEFINE_CTX(Reshape)
    };

    struct Quantize : public NativeWord {
        void run(Stack& stack) override {
            tensor_t dst = stack.pop_tensor();
            tensor_t src = stack.pop_tensor();
            src->op_quantize(ctx_, src, dst);
        }
        NWORD_CREATOR_DEFINE_CTX(Quantize)
    };

    struct DeQuantize : public NativeWord {
        void run(Stack& stack) override {
            tensor_t dst = stack.pop_tensor();
            tensor_t src = stack.pop_tensor();
            src->op_dequantize(ctx_, src, dst);
        }
        NWORD_CREATOR_DEFINE_CTX(DeQuantize)
    };

    struct Embed : public NativeWord {
        void run(Stack& stack) override {
            auto out = stack.pop_tensor();
            auto table = stack.pop_tensor();
            auto self = stack.pop_tensor();

            self->op_embed(ctx_, self, table, out);
        }
        NWORD_CREATOR_DEFINE_CTX(Embed)
    };

    struct CopyFrom : public NativeWord {
        void run(Stack& stack) override {
            tensor_t src = stack.pop_tensor();
            tensor_t dst = stack.pop_tensor();
            dst->op_copy_from(ctx_, dst, src);
        }
        NWORD_CREATOR_DEFINE_CTX(CopyFrom)
    };

    struct CopyTo : public NativeWord {
        void run(Stack& stack) override {
            tensor_t dst = stack.pop_tensor();
            tensor_t src = stack.pop_tensor();
            src->op_copy_to(ctx_, src, dst);
        }
        NWORD_CREATOR_DEFINE_CTX(CopyTo)
    };

    struct Convert : public NativeWord {
        void run(Stack& stack) override {
            tensor_t src = stack.pop_tensor();
            tensor_t dst = stack.pop_tensor();
            dst->op_convert(ctx_, dst, src);
        }
        NWORD_CREATOR_DEFINE_CTX(Convert)
    };

    struct Linear : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t w = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_linear(ctx_, x, w, b, y);
        }
        NWORD_CREATOR_DEFINE_CTX(Linear)
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

            x->op_layernorm(ctx_, x, mean, var, scale, bias, y, eps);
        }
        NWORD_CREATOR_DEFINE_CTX(Layernorm)
    };

    struct RMSnorm : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t y = stack.pop_tensor();
            tensor_t norm2 = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_rmsnorm(ctx_, x, scale, norm2, y, eps);
        }
        NWORD_CREATOR_DEFINE_CTX(RMSnorm)
    };

    struct RotaryEmbed : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t pos = stack.pop_tensor();
            tensor_t cached = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_rotary_embed(ctx_, x, cached, pos, out);
        }
        NWORD_CREATOR_DEFINE_CTX( RotaryEmbed );
    };

    struct Transpose0213 : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_transpose_0213(ctx_, x,y);
        }
        NWORD_CREATOR_DEFINE_CTX(Transpose0213)
    };

    struct Transpose0213Rotary : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t pos = stack.pop_tensor();
            tensor_t cached = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_transpose_0213_rotary(ctx_, x, cached, pos, out);
        }
        NWORD_CREATOR_DEFINE_CTX( Transpose0213Rotary );
    };

    struct Transpose0213Repeated : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_transpose_0213_repeated(ctx_, x,y);
        }
        NWORD_CREATOR_DEFINE_CTX(Transpose0213Repeated)
    };


    struct QueryKey : public NativeWord {
        void run(Stack& stack) override {
            tensor_t qk = stack.pop_tensor();
            tensor_t k = stack.pop_tensor();
            tensor_t q = stack.pop_tensor();

            q->op_qk(ctx_, q, k, qk);
        }
        NWORD_CREATOR_DEFINE_CTX(QueryKey)
    };

    struct Add : public NativeWord {
        void run(Stack& stack) override {
            tensor_t c = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_add(ctx_, x, b, c);
        }
        NWORD_CREATOR_DEFINE_CTX(Add)
    };

    struct Mul : public NativeWord {
        void run(Stack& stack) override {
            tensor_t c = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_mul(ctx_, x, b, c);
        }
        NWORD_CREATOR_DEFINE_CTX(Mul)
    };

    struct Softmax : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_softmax(ctx_, x, out);
        }
        NWORD_CREATOR_DEFINE_CTX(Softmax)
    };

    struct Attn : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t value = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_attn(ctx_, x, value, out);
        }
        NWORD_CREATOR_DEFINE_CTX(Attn)
    };

    struct Gelu : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_gelu(ctx_, x, out);
        }
        NWORD_CREATOR_DEFINE_CTX(Gelu)
    };

    struct SiluProduct : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t in = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_silu_product(ctx_, x, in, out);
        }
        NWORD_CREATOR_DEFINE_CTX(SiluProduct)
    };

    struct AllLogits : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t lm_head = stack.pop_tensor();
            tensor_t mask = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            auto ret = x->op_all_logits(ctx_, x, mask, lm_head, out);

            stack.push_number( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_CTX(AllLogits)
    };

    struct SamplingTop1 : public NativeWord {
        void run(Stack& stack) override {
            tensor_t logits = stack.pop_tensor();
            auto ret = logits->op_sampling_top1(ctx_, logits);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_CTX(SamplingTop1);
    };

    struct SamplingTop3 : public NativeWord {
        void run(Stack& stack) override {
            float temp = stack.pop_number();
            tensor_t logits = stack.pop_tensor();
            auto ret = logits->op_sampling_top3(ctx_, logits, temp);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_CTX(SamplingTop3);
    };

    struct Conv2D : public NativeWord {
        void run(Stack& stack) override {
            int padding = stack.pop_number();
            int stride = stack.pop_number();
            tensor_t dst = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t weight = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_conv2d(ctx_, x, weight, bias, dst, stride, padding);
        }
        NWORD_CREATOR_DEFINE_CTX(Conv2D);
    };
}

namespace io {
    struct Dump : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->io_dump(ctx_, t);
        }
        NWORD_CREATOR_DEFINE_CTX(Dump)
    };

    struct Load : public NativeWord {
        void run(Stack& stack) override {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_load(ctx_, x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_CTX(Load)
    };

    struct Save : public NativeWord {
        void run(Stack& stack) override {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_save(ctx_, x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_CTX(Save)
    };

   struct PipeRead : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            x->io_pipe_read(ctx_, x);
        }
        NWORD_CREATOR_DEFINE_CTX(PipeRead)
    };

    struct PipeWrite : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_pipe_write(ctx_, x, dst);
        }
        NWORD_CREATOR_DEFINE_CTX(PipeWrite)
    };

}

void load_nn_operators(Enviroment& env) {
    env.insert_native_word("io.dump", io::Dump::creator );
    env.insert_native_word("io.load", io::Load::creator );
    env.insert_native_word("io.save", io::Save::creator );
    env.insert_native_word("io.pipe.read", io::PipeRead::creator );
    env.insert_native_word("io.pipe.write", io::PipeWrite::creator );

    env.insert_native_word("op.check", op::CheckPoint::creator );
    env.insert_native_word("op.get_shape", op::Shape::creator);
    env.insert_native_word("op.get_device", op::Device::creator);
    env.insert_native_word("op.get_dtype", op::DataType::creator);
    env.insert_native_word("op.create", op::Create::creator );
    env.insert_native_word("op.null", op::Null::creator );
    env.insert_native_word("op.zero", op::Zero::creator );
    env.insert_native_word("op.sizeof", op::Sizeof::creator);
    env.insert_native_word("op.fill", op::Fill::creator );
    env.insert_native_word("op.rotary_cache", op::RotaryCache::creator );
    env.insert_native_word("op.causal_mask", op::CausalMask::creator );
    env.insert_native_word("op.view", op::View::creator );
    env.insert_native_word("op.view_as", op::ViewAs::creator );
    env.insert_native_word("op.reshape", op::Reshape::creator );
    env.insert_native_word("op.quantize", op::Quantize::creator );
    env.insert_native_word("op.dequantize", op::DeQuantize::creator );
    env.insert_native_word("op.embed", op::Embed::creator );
    env.insert_native_word("op.copy", op::CopyFrom::creator );
    env.insert_native_word("op.copy_from", op::CopyFrom::creator );
    env.insert_native_word("op.copy_to", op::CopyTo::creator );
    env.insert_native_word("op.convert", op::Convert::creator );
    env.insert_native_word("op.linear", op::Linear::creator );
    env.insert_native_word("op.layernorm", op::Layernorm::creator );
    env.insert_native_word("op.rmsnorm", op::RMSnorm::creator );
    env.insert_native_word("op.rotary_embed", op::RotaryEmbed::creator );
    env.insert_native_word("op.transpose_0213", op::Transpose0213::creator );
    env.insert_native_word("op.transpose_0213_rotary", op::Transpose0213Rotary::creator );
    env.insert_native_word("op.transpose_0213_repeated", op::Transpose0213Repeated::creator );
    env.insert_native_word("op.add", op::Add::creator);
    env.insert_native_word("op.mul", op::Mul::creator);
    env.insert_native_word("op.querykey", op::QueryKey::creator);
    env.insert_native_word("op.softmax", op::Softmax::creator);
    env.insert_native_word("op.attn", op::Attn::creator);
    env.insert_native_word("op.gelu", op::Gelu::creator);
    env.insert_native_word("op.silu_product", op::SiluProduct::creator);
    env.insert_native_word("op.all_logits", op::AllLogits::creator);
    env.insert_native_word("op.sampling_top1", op::SamplingTop1::creator);
    env.insert_native_word("op.sampling_top3", op::SamplingTop3::creator);
    env.insert_native_word("op.conv2d", op::Conv2D::creator);
}

}// end of namespace br
