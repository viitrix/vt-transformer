#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include <vt.hpp>
#include <sentencepiece_processor.h>
#include "tokenizer_combo.hpp"

namespace vt {

struct BaichuanTokenizer : public Tokenizer {
    sentencepiece::SentencePieceProcessor processor;

    BaichuanTokenizer( const char* file_name) {
        const auto status = processor.Load(file_name);
        if (!status.ok()) {
            vt_panic("Can't crate BaichuanTokenizer from model file!");
        }
    }
    ~BaichuanTokenizer() {
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
        std::vector<int> res;
        processor.Encode(text, &res);
        return res;
    }

    virtual std::string decode(const int id) override {
        std::vector<int> ids = { id };
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }

    virtual std::string decode(const std::vector<int>& ids) override {
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }
};

Tokenizer* build_tokenizer_baichuan(const char* file_name) {
    return new BaichuanTokenizer(file_name);
}

}
