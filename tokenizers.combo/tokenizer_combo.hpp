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

Tokenizer* build_tokenizer_qwen(const char* file_name);

}
#endif
