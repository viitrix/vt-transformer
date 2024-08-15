
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <math.h>
#include <map>

#include "tiktoken_c.h"
#include "image_c.h"
#include "tokenizer_combo.hpp"

namespace vt {

inline std::string fileToString(const char* filename) {
    std::ifstream t(filename);
    if ( ! t.is_open() ) {
        std::cout << "Can't open " << filename << std::endl;
        exit(0);
    }

    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

static inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
	return as_float(w);
#elif defined(__CUDA_ARCH__)
	return __uint_as_float((unsigned int) w);
#elif defined(__INTEL_COMPILER)
	return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return _CopyFloatFromInt32((__int32) w);
#else
	union {
		uint32_t as_bits;
		float as_value;
	} fp32 = { w };
	return fp32.as_value;
#endif
}

static inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
	return as_uint(f);
#elif defined(__CUDA_ARCH__)
	return (uint32_t) __float_as_uint(f);
#elif defined(__INTEL_COMPILER)
	return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
	return (uint32_t) _CopyInt32FromFloat(f);
#else
	union {
		float as_value;
		uint32_t as_bits;
	} fp32 = { f };
	return fp32.as_bits;
#endif
}

static unsigned short fp32_to_fp16(float value) {
    float f = value;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
	const float scale_to_inf = 0x1.0p+112f;
	const float scale_to_zero = 0x1.0p-110f;
#else
	const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
	const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
	float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

	const uint32_t w = fp32_to_bits(f);
	const uint32_t shl1_w = w + w;
	const uint32_t sign = w & UINT32_C(0x80000000);
	uint32_t bias = shl1_w & UINT32_C(0xFF000000);
	if (bias < UINT32_C(0x71000000)) {
		bias = UINT32_C(0x71000000);
	}

	base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
	const uint32_t bits = fp32_to_bits(base);
	const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
	const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
	const uint32_t nonsign = exp_bits + mantissa_bits;
	return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}


struct MinicpmImageLoader : public ImageLoader {
    static const float mean[];
    static const float std[];
    static const int PATCH_SIZE;

    ImageHandle rustObj;
    MinicpmImageLoader(const std::string& filename) {
        rustObj = imgobj_load(filename.c_str(), filename.size());
        imgobj_resize(rustObj, 448, 448);
    }
    virtual ~MinicpmImageLoader() override{
        imgobj_free(rustObj);
    }
    virtual size_t width() override {
        return imgobj_width(rustObj);
    }
    virtual size_t height() override {
        return imgobj_height(rustObj);
    }

    virtual void preprocess(std::vector<unsigned short>& out) {
        const size_t s = width() * height();
        const int PW = width() / PATCH_SIZE;
        const int PH = height() / PATCH_SIZE;

        std::vector<unsigned char> rgb;
        rgb.resize(s * 3);
        out.resize(s * 3);

        imgobj_rgb_plane(rustObj, rgb.data());
        for (size_t h = 0; h < height(); h++) {
            for (size_t w = 0; w < width(); w++) {
                int i = w + h * width();
                int ii = 0;
                {
                    // do unflod
                    int hh = h / PATCH_SIZE;
                    int ww = w / PATCH_SIZE;

                    int pi = hh * PW + ww;
                    int iiy = h % PATCH_SIZE;
                    int iix = pi * PATCH_SIZE + (w % PATCH_SIZE);
                    ii = iiy * PW * PH * PATCH_SIZE + iix;
                }

                float r,g,b;
                r = ((int)rgb[i] / 255.0 - mean[0]) / std[0];
                g = ((int)rgb[i + s] / 255.0 - mean[1]) / std[1];
                b = ((int)rgb[i + s * 2] / 255.0 - mean[2]) / std[2];

                out[ii] = fp32_to_fp16(r);
                out[ii + s] = fp32_to_fp16(g);
                out[ii + s * 2] = fp32_to_fp16(b);
            }
        }
    }

    virtual void preprocess(std::vector<float>& out) {
        const size_t s = width() * height();
        const int PW = width() / PATCH_SIZE;
        const int PH = height() / PATCH_SIZE;

        std::vector<unsigned char> rgb;
        rgb.resize(s * 3);
        out.resize(s * 3);

        imgobj_rgb_plane(rustObj, rgb.data());
        for (size_t h = 0; h < height(); h++) {
            for (size_t w = 0; w < width(); w++) {
                int i = w + h * width();
                int ii = 0;
                {
                    // do unflod
                    int hh = h / PATCH_SIZE;
                    int ww = w / PATCH_SIZE;

                    int pi = hh * PW + ww;
                    int iiy = h % PATCH_SIZE;
                    int iix = pi * PATCH_SIZE + (w % PATCH_SIZE);
                    ii = iiy * PH * PW * PATCH_SIZE + iix;
                }

                float r,g,b;
                r = ((int)rgb[i] / 255.0 - mean[0]) / std[0];
                g = ((int)rgb[i + s] / 255.0 - mean[1]) / std[1];
                b = ((int)rgb[i + s * 2] / 255.0 - mean[2]) / std[2];

                out[ii] = r;
                out[ii + s] = g;
                out[ii + s * 2] = b;
            }
        }
    }
};
const float MinicpmImageLoader::mean[3] = {0.5, 0.5, 0.5};
const float MinicpmImageLoader::std[3] = {0.5, 0.5, 0.5};
const int MinicpmImageLoader::PATCH_SIZE = 14;

ImageLoader* build_imageloader_minicpm(const char* file_name) {
    return new MinicpmImageLoader(file_name);
}

}
