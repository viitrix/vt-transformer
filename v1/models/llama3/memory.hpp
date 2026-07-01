#ifndef _MEMORY_HPP_

const int VOCAB_SIZE = 151936;
const int HIDDEN_SIZE = 4096;
const int INTERMEDIATE_SIZE = 11008;
const int HEADS_NUM = 32;
const int HEAD_HIDDEN = 128;

struct MemoryAlign : public vt::NativeWord {
    void run(vt::Stack& stack) override {
        size_t align = stack.pop_number();
        size_t offset = stack.pop_number();
        offset += align;
        offset = offset - ( offset % align );
        stack.push_number(offset);
    }
    NWORD_CREATOR_DEFINE_LR(MemoryAlign)
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
    NWORD_CREATOR_DEFINE_LR(MemoryCounting)
};



#endif
