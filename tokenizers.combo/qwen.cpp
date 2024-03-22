
#include <string>
#include <sstream>
#include <fstream>
#include <map>

#include <vt.hpp>
#include <context.hpp>

#include "tiktoken_c.h"
#include "image_c.h"
#include "tokenizer_combo.hpp"

namespace vt {

struct QwenTokenizer : public Tokenizer {
    const bool is_visual;
    TiktokenHandle rustObj;
    QwenTokenizer(const char* hash_file, bool _visual) : is_visual(_visual) {
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
        if ( is_visual ) {
            specs.append("<ref>\n");
            specs.append("</ref>\n");
            specs.append("<box>\n");
            specs.append("</box>\n");
            specs.append("<quad>\n");
            specs.append("</quad>\n");
            specs.append("<img>\n");
            specs.append("</img>\n");
            for ( int i = 0; i < 16; i++) {
                std::stringstream ss;
                ss << "<imgpad_" << i << ">" << std::endl;
                specs.append(ss.str());
            }
        }

        const std::string reg = R""""((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"""";
        std::string hash = vt::fileToString(hash_file);

        rustObj = tiktoken_new_from_str(hash.c_str(), hash.size(),
                                        specs.c_str(), specs.size(),
                                        reg.c_str(), reg.size());
    }

    virtual ~QwenTokenizer() {
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

Tokenizer* build_tokenizer_qwen(const char* file_name, bool is_visual) {
    return new QwenTokenizer(file_name, is_visual);
}

// ================================================================
struct QwenImageLoader : public ImageLoader {
    static const float mean[];
    static const float std[];

    ImageHandle rustObj;
    QwenImageLoader(const std::string& filename) {
        rustObj = imgobj_load(filename.c_str(), filename.size());
        imgobj_resize(rustObj, 448, 448);
    }
    virtual ~QwenImageLoader() override{
        imgobj_free(rustObj);
    }
    virtual size_t width() override {
        return imgobj_width(rustObj);
    }
    virtual size_t height() override {
        return imgobj_height(rustObj);
    }

    virtual void preprocess(std::vector<local_fp16_t>& out) {
        const size_t s = width() * height();
        std::vector<unsigned char> rgb;
        rgb.resize(s * 3);
        out.resize(s * 3);

        imgobj_rgb_plane(rustObj, rgb.data());
        for (size_t h = 0; h < height(); h++) {
            for (size_t w = 0; w < width(); w++) {
                int i = w + h * width();

                float r,g,b;
                r = ((int)rgb[i] / 255.0 - mean[0]) / std[0];
                g = ((int)rgb[i + s] / 255.0 - mean[1]) / std[1];
                b = ((int)rgb[i + s * 2] / 255.0 - mean[2]) / std[2];

                out[i] = fp32_to_fp16(r);
                out[i + s] = fp32_to_fp16(g);
                out[i + s * 2] = fp32_to_fp16(b);
            }
        }
    }

    virtual void preprocess(std::vector<float>& out) {
        const size_t s = width() * height();
        std::vector<unsigned char> rgb;
        rgb.resize(s * 3);
        out.resize(s * 3);

        imgobj_rgb_plane(rustObj, rgb.data());
        for (size_t h = 0; h < height(); h++) {
            for (size_t w = 0; w < width(); w++) {
                int i = w + h * width();

                out[i] = ((int)rgb[i] / 255.0 - mean[0]) / std[0];
                out[i + s] = ((int)rgb[i + s] / 255.0 - mean[1]) / std[1];
                out[i + s * 2] = ((int)rgb[i + s * 2] / 255.0 - mean[2]) / std[2];
            }
        }
    }
};
const float QwenImageLoader::mean[3] = {0.48145466, 0.4578275, 0.40821073};
const float QwenImageLoader::std[3] = {0.26862954, 0.26130258, 0.27577711};

ImageLoader* build_imageloader_qwen(const char* file_name) {
    return new QwenImageLoader(file_name);
}


}
