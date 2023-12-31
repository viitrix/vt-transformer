#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include <vt.hpp>
#include "tokenizers_c.h"
#include "tokenizer_combo.hpp"

namespace vt {

struct BloomzTokenizer : public Tokenizer {
    TokenizerHandle rustObj;
    BloomzTokenizer( const char* file_name) {
        std::string json = vt::fileToString(file_name);
        rustObj = tokenizers_new_from_str(json.c_str(), json.size());
    }
    ~BloomzTokenizer() {
        if ( rustObj != nullptr) {
            tokenizers_free(rustObj);
        }
        rustObj = nullptr;
    }

    virtual int token_unk() override {
        return 0;
    }
    virtual int token_bos() override {
        return 1;
    }
    virtual int token_eos() override {
        return 2;
    }
    virtual std::vector<int> encode(const std::string& text) override {
        const uint32_t* ids;
        size_t ids_num;

        tokenizers_encode(rustObj, text.c_str(), text.size(), token_unk());
        tokenizers_get_encode_ids(rustObj, &ids, &ids_num);

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

        tokenizers_decode(rustObj, (const uint32_t *)ids.data(), ids.size(), token_unk());
        tokenizers_get_decode_str(rustObj, &out, &out_len);

        std::string res;
        for (size_t i = 0; i < out_len; i++) {
            res = res + out[i];
        }
        return res;
    }
};

Tokenizer* build_tokenizer_bloomz(const char* file_name) {
    return new BloomzTokenizer(file_name);
}

}
