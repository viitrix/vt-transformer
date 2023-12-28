#include <string>
#include <iostream>
#include <fstream>

#include <dkm_parallel.hpp>
#include "vt.hpp"

using local_fp16_t = uint16_t;

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

float fp16_to_fp32(local_fp16_t value) {
    uint16_t h = value;
    /*
	 * Extend the half-precision floating-point number to 32 bits and shift to the upper part of the 32-bit word:
	 *      +---+-----+------------+-------------------+
	 *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
	 *      +---+-----+------------+-------------------+
	 * Bits  31  26-30    16-25            0-15
	 *
	 * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0 - zero bits.
	 */
	const uint32_t w = (uint32_t) h << 16;
	/*
	 * Extract the sign of the input number into the high bit of the 32-bit word:
	 *
	 *      +---+----------------------------------+
	 *      | S |0000000 00000000 00000000 00000000|
	 *      +---+----------------------------------+
	 * Bits  31                 0-31
	 */
	const uint32_t sign = w & UINT32_C(0x80000000);
	/*
	 * Extract mantissa and biased exponent of the input number into the high bits of the 32-bit word:
	 *
	 *      +-----+------------+---------------------+
	 *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
	 *      +-----+------------+---------------------+
	 * Bits  27-31    17-26            0-16
	 */
	const uint32_t two_w = w + w;

	/*
	 * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become mantissa and exponent
	 * of a single-precision floating-point number:
	 *
	 *       S|Exponent |          Mantissa
	 *      +-+---+-----+------------+----------------+
	 *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
	 *      +-+---+-----+------------+----------------+
	 * Bits   | 23-31   |           0-22
	 *
	 * Next, there are some adjustments to the exponent:
	 * - The exponent needs to be corrected by the difference in exponent bias between single-precision and half-precision
	 *   formats (0x7F - 0xF = 0x70)
	 * - Inf and NaN values in the inputs should become Inf and NaN values after conversion to the single-precision number.
	 *   Therefore, if the biased exponent of the half-precision input was 0x1F (max possible value), the biased exponent
	 *   of the single-precision output must be 0xFF (max possible value). We do this correction in two steps:
	 *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset below) rather than by 0x70 suggested
	 *     by the difference in the exponent bias (see above).
	 *   - Then we multiply the single-precision result of exponent adjustment by 2**(-112) to reverse the effect of
	 *     exponent adjustment by 0xE0 less the necessary exponent adjustment by 0x70 due to difference in exponent bias.
	 *     The floating-point multiplication hardware would ensure than Inf and NaN would retain their value on at least
	 *     partially IEEE754-compliant implementations.
	 *
	 * Note that the above operations do not handle denormal inputs (where biased exponent == 0). However, they also do not
	 * operate on denormal inputs, and do not produce denormal results.
	 */
	const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
	const float exp_scale = 0x1.0p-112f;
#else
	const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
	const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	/*
	 * Convert denormalized half-precision inputs into single-precision results (always normalized).
	 * Zero inputs are also handled here.
	 *
	 * In a denormalized number the biased exponent is zero, and mantissa has on-zero bits.
	 * First, we shift mantissa into bits 0-9 of the 32-bit word.
	 *
	 *                  zeros           |  mantissa
	 *      +---------------------------+------------+
	 *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
	 *      +---------------------------+------------+
	 * Bits             10-31                0-9
	 *
	 * Now, remember that denormalized half-precision numbers are represented as:
	 *    FP16 = mantissa * 2**(-24).
	 * The trick is to construct a normalized single-precision number with the same mantissa and thehalf-precision input
	 * and with an exponent which would scale the corresponding mantissa bits to 2**(-24).
	 * A normalized single-precision floating-point number is represented as:
	 *    FP32 = (1 + mantissa * 2**(-23)) * 2**(exponent - 127)
	 * Therefore, when the biased exponent is 126, a unit change in the mantissa of the input denormalized half-precision
	 * number causes a change of the constructud single-precision number by 2**(-24), i.e. the same ammount.
	 *
	 * The last step is to adjust the bias of the constructed single-precision number. When the input half-precision number
	 * is zero, the constructed single-precision number has the value of
	 *    FP32 = 1 * 2**(126 - 127) = 2**(-1) = 0.5
	 * Therefore, we need to subtract 0.5 from the constructed single-precision number to get the numerical equivalent of
	 * the input half-precision number.
	 */
	const uint32_t magic_mask = UINT32_C(126) << 23;
	const float magic_bias = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	/*
	 * - Choose either results of conversion of input as a normalized number, or as a denormalized number, depending on the
	 *   input exponent. The variable two_w contains input exponent in bits 27-31, therefore if its smaller than 2**27, the
	 *   input is either a denormal number, or zero.
	 * - Combine the result of conversion of exponent and mantissa with the sign of the input number.
	 */
	const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
	const uint32_t result = sign |
		(two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

local_fp16_t fp32_to_fp16(float value) {
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

std::vector<float> load_data_from_file(const std::string& fname) {
    std::cout << ">>>>> Loading " << fname << std::endl;

    std::vector<float> y;
    std::vector<local_fp16_t> y_;
    vt::read_data(fname.c_str(), y_);
    y.resize( y_.size() );
    for(size_t i = 0; i < y.size(); i++) {
        y[i] = fp16_to_fp32(y_[i]);
    }
    return y;
}

const int D = 4;
const int K = 256;
const int T = 256;
std::vector<float> train_one(std::vector<float>& src_, std::vector<uint8_t>& codes) {
    float SCALE = 0;
    for(size_t i = 0; i < src_.size(); i += D) {
        if ( fabs(src_[i]) > SCALE ) {
            SCALE = fabs(src_[i]);
        }
    }
    std::vector<std::array<float, D>> src;
    for(size_t i = 0; i < src_.size(); i += D) {
        std::array<float, D> v;
        for(int j = 0; j < D; j++) {
            v[j] = src_[i+j] / SCALE;
        }
        src.push_back(std::move(v));
    }

    std::cout << "DKMing..." << std::endl;
	auto cluster_data = dkm::kmeans_lloyd_parallel(src, K, T);

    std::vector<float> target;
    auto& centers = std::get<0>(cluster_data);
    for (size_t i = 0; i < centers.size(); i++) {
        for(int j = 0; j < D; j++) {
            target.push_back( centers[i][j] * SCALE );
        }
    }

    auto all_codes = std::get<1>(cluster_data);
    codes.resize(all_codes.size());
    for(size_t i = 0; i < all_codes.size(); i++) {
        vt_assert( all_codes[i] >= 0 && all_codes[i] <= 255, " bit8 codes must be !");
        codes[i] = (uint8_t)all_codes[i];
    }

    {
        double error = 0.0;
        float max = 0.0;
        std::vector<float> vars;
        for(size_t i = 0; i < src.size(); i++) {
            float* coded_value = centers[ all_codes[i] ].data();
            for (int j = 0; j < D; j++) {
                float e = fabs( coded_value[j] * SCALE - src_[i * D + j] );
                error += e;
                if ( e > max ) {
                    max = e;
                }
                vars.push_back(e);
            }
        }
        std::cout << "ERROR = " << error / src_.size() << std::endl;
        std::cout << "MAX = " << max << std::endl;

        std::ofstream wf("vars.data", std::ios::out | std::ios::binary);
        wf.write((char *)vars.data(), vars.size() * sizeof(float));
        wf.close();
    }


    return target;
}

void do_file(const std::string &fname) {
    auto data = std::move( load_data_from_file(fname + ".fp16"));
    vt_assert( data.size() % D == 0, "Can't convert from un aligned tensor");

    std::vector<uint8_t> codes;
    auto centers = train_one(data, codes);

    {
        std::string out_fname = fname + ".pq";

        std::ofstream wf(out_fname, std::ios::out | std::ios::binary);
        wf.write((char *)centers.data(), centers.size() * sizeof(float));
        wf.write((char *)codes.data(), codes.size());
        wf.close();
    }
}

int main(int argc, const char* argv[]) {
    std::vector<std::string> all_files;
    {
        std::ifstream in("./pq_files.txt");
        while( !in.eof() ) {
            std::string pqf;
            in >> pqf;
            if ( pqf != "" )
                all_files.push_back(pqf);
        }
        std::cout << "Total " << all_files.size() << " fp16 files." << std::endl;
    }

    for(size_t i = 0; i < all_files.size(); i++) {
        do_file( all_files[i] );
    }
}

