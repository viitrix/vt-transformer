#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include <vt.hpp>
#include "tiktoken_c.h"
#include "tokenizer_combo.hpp"

namespace vt {

struct QwenTokenizer : public Tokenizer {
    TiktokenHandle rustObj;
    QwenTokenizer(const char* hash_file) {
        // building special tokens manual ..
        std::string specs;
        specs.append("<|endoftext|>\n");
        specs.append("<|im_start|>\n");
        specs.append("<|im_end|>\n");
        for (int i = 0; i < 205; i++) {
            std::stringstream ss;
            ss << "<|extra_" << i << "|>" << std::endl;
            specs.append(ss.str());
        }

        const std::string reg = R""""((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"""";
        std::string hash = vt::fileToString(hash_file);

        rustObj = tiktoken_new_from_str(hash.c_str(), hash.size(),
                                        specs.c_str(), specs.size(),
                                        reg.c_str(), reg.size());
    }

    ~QwenTokenizer() {
        if ( rustObj != nullptr) {
            tiktoken_free(rustObj);
        }
        rustObj = nullptr;
    }

    virtual int token_unk() override {
        return 151643;
    }
    virtual int token_eos() override {
        return 151645;
    }
    virtual int token_bos() override {
        return 151644;
    }

    virtual std::vector<int> encode(const std::string& text) override {
        const uint32_t* ids;
        size_t ids_num;

        tiktoken_encode(rustObj, text.c_str(), text.size(), token_unk());
        tiktoken_get_encode_ids(rustObj, &ids, &ids_num);

        std::vector<int> res;
        for(size_t i = 0; i < ids_num; i++) {
            res.push_back( ids[i] );
        }
        return res;
    }

    virtual std::string decode(const int id) override {
        std::vector<int> ids;
        ids.push_back(id);
        return std::move( decode(ids) );
    }

    virtual std::string decode(const std::vector<int>& ids) override {
        const char* out;
        size_t out_len;

        tiktoken_decode(rustObj, (const uint32_t *)ids.data(), ids.size(), token_unk());
        tiktoken_get_decode_str(rustObj, &out, &out_len);

        std::string res;
        for (size_t i = 0; i < out_len; i++) {
            res = res + out[i];
        }
        return res;
    }
};

Tokenizer* build_tokenizer_qwen(const char* file_name) {
    return new QwenTokenizer(file_name);
}

}
