#ifndef _VT_TOKENIZER_HPP_
#define _VT_TOKENIZER_HPP_

#include <string>
#include <vector>

namespace vt {
struct Tokenizer {
    virtual ~Tokenizer() {}

    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const int id) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
    virtual int token_unk() = 0;
    virtual int token_eos() = 0;
    virtual int token_bos() = 0;
};

struct ImageLoader {
    virtual ~ImageLoader() {}
    virtual size_t width() = 0;
    virtual size_t height() = 0;
    virtual void preprocess(std::vector<float>& out) = 0;
    virtual void preprocess(std::vector<uint16_t>& out) = 0;
};

// Qwen serial models
Tokenizer* build_tokenizer_qwen(const char* file_name, bool is_visual=false);
ImageLoader* build_imageloader_qwen(const char* file_name);

// Llama3
Tokenizer* build_tokenizer_llama3(const char* file_name);

// MiniCPM
ImageLoader* build_imageloader_minicpm(const char* file_name);

}
#endif
