#ifndef _MEMORY_HPP_

#include <sys/ipc.h>
#include <sys/shm.h>

const int VOCAB_SIZE = 151936;
const int HIDDEN_SIZE = 4096;
const int INTERMEDIATE_SIZE = 11008;
const int HEADS_NUM = 32;
const int HEAD_HIDDEN = 128;

const int IMAGE_BEGIN = 151857;
const int IMAGE_END = 151858;
const int IMAGE_PAD_BEGIN = 151859;                 // <imgpad_0>
const int IMAGE_PAD_END = IMAGE_PAD_BEGIN + 15;     // <imgpad_15>

const size_t shsize = 448 * 448 * 3 * 2;

struct MemoryFill : public vt::NativeWord {
    static void *img_;
    static void fill(std::vector<uint16_t>& source) {
        if ( img_ == nullptr) {
            img_ = malloc(source.size() * sizeof(uint16_t));
        }
        memcpy(img_, source.data(), source.size() * sizeof(uint16_t));
    }
    void run(vt::Stack& stack) override {
        auto tensor = stack.pop_tensor();
        void* dst = std::get<1>(tensor->op_data(ctx_, tensor));

#ifdef _USING_DEVICE_CUDA_
        if ( tensor->is_cuda() ) {
            CUDA_CHECK(cudaMemcpyAsync(dst, img_, shsize, cudaMemcpyHostToDevice, ctx_->cuda_stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx_->cuda_stream));
        }
#endif
    }
    NWORD_CREATOR_DEFINE_CTX(MemoryFill)
};

struct PositionId : public vt::NativeWord {
    void run(vt::Stack& stack) override {
        int wid = stack.pop_number();
        int hei = stack.pop_number();
        int bound = stack.pop_number();
        auto tensor = stack.pop_tensor();

        int* dst = (int *)std::get<1>(tensor->op_data(ctx_, tensor));

        float bucket = 1.0 / bound;

        for(int h = 0; h < hei; h++) {
            for(int w = 0; w < wid; w++) {
                float x = 1.0 * w / wid ;
                float y = 1.0 * h / hei;

                int hh = y / bucket;
                int ww = x / bucket;
                int ii = hh * bound + ww;
                *dst = ii;
                dst++;
            }
        }
    }
    NWORD_CREATOR_DEFINE_CTX(PositionId)
};

struct InsertImage : public vt::NativeWord {
    void run(vt::Stack& stack) override {
        auto ids = stack.pop_tensor();
        auto image = stack.pop_tensor();
        auto embed = stack.pop_tensor();

        int* tokens = (int *)std::get<1>(ids->op_data(ctx_, ids));
        for (int i = 0; i < (int)ids->items(); i++) {
            if ( tokens[i] >= IMAGE_PAD_BEGIN ) {
                vt_assert(i + 256 < (int)ids->items() - 1, "Token's length error for image");
                vt_assert(tokens[i + 256] == IMAGE_END, "Token's image format error!");

                auto ret = embed->op_view(ctx_, embed, i * HIDDEN_SIZE, {1, 256, 4096});
                auto target = std::get<1>(ret);

                target->op_copy_from(ctx_, target, image);
                break;
            }
        }
    }
    NWORD_CREATOR_DEFINE_CTX(InsertImage)
};

struct MemoryAlign : public vt::NativeWord {
    void run(vt::Stack& stack) override {
        size_t align = stack.pop_number();
        size_t offset = stack.pop_number();
        offset += align;
        offset = offset - ( offset % align );
        stack.push_number(offset);
    }
    NWORD_CREATOR_DEFINE_CTX(MemoryAlign)
};

struct MemoryCounting : public vt::NativeWord {
    void run(vt::Stack& stack) override {
        size_t tokens = stack.pop_number();
        size_t batch = stack.pop_number();
        size_t full_tokens = tokens + 8;

        size_t xinput = batch * tokens * HIDDEN_SIZE;
        size_t causal_mask = batch * tokens * full_tokens;
        size_t norm2 = batch * tokens;
        size_t xa = batch * tokens * HIDDEN_SIZE;
        size_t xb = xa;

        size_t base = xinput + xa + xb + norm2 + causal_mask;
        size_t attn = 0;
        {
            size_t xc = xa;
            size_t xd = xa;
            size_t xfa = batch * full_tokens * HIDDEN_SIZE;
            size_t xfb = batch * full_tokens * HIDDEN_SIZE;
            size_t xll_half = batch * HEADS_NUM * tokens * full_tokens;
            size_t xll = batch * HEADS_NUM * tokens * full_tokens * 2;
            attn = xc + xd + xfa + xfb + xll_half + xll;
        }

        size_t mlp = 0;
        {
           size_t x4a = batch * tokens * INTERMEDIATE_SIZE;
           size_t x4b = x4a;
           mlp = x4a + x4b;
        }

        size_t logits = batch * tokens * INTERMEDIATE_SIZE;

        size_t all = base + std::max(std::max(attn, mlp), logits);
        all = (all + 1024*1024) - all % (1024 * 1024);

        size_t oneG = 1024 * 1024 * 1024;
        size_t kv = 32 * batch * tokens * HIDDEN_SIZE * 2;
        std::cout << "Allocating " << all * 2.0 / oneG << " GB for internal memory." << std::endl;
        std::cout << "Allocating " << kv * 2.0 / oneG << " GB for kv caches memory." << std::endl;
        stack.push_number(all);
    }
    NWORD_CREATOR_DEFINE_CTX(MemoryCounting)
};



#endif
