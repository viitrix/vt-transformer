#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/safetensors.hpp"

using vt::SafeTensorFile;
using vt::SafeTensorInfo;
using vt::safetensors_dtype_bytes;

namespace fs = std::filesystem;

static const char* kModelPath = "/home/teaonly/workspace/qwen3-0.6b/model.safetensors";

// FP32 位段 → FP16 位段（IEEE 754, round-to-nearest-even）。
// 参考 IEEE 754-2008 规范手写实现：
//   FP32: 1 sign | 8 exp (bias 127) | 23 mant
//   FP16: 1 sign | 5 exp (bias 15)  | 10 mant
// BF16 是 FP32 的高 16 位（截掉低 16 位尾数），所以 BF16→FP32 只需左移 16 位。
static uint16_t fp32_bits_to_fp16_bits(uint32_t bits) {
    const uint32_t sign = (bits >> 31) & 0x1;
    const uint32_t exp  = (bits >> 23) & 0xFF;
    const uint32_t mant = bits & 0x7FFFFF;
    const uint16_t h_sign = uint16_t(sign << 15);

    // Inf / NaN
    if (exp == 0xFF) {
        uint16_t h_mant = uint16_t(mant >> 13);
        if (mant != 0 && h_mant == 0) h_mant = 1;  // 保留 NaN-ness
        return h_sign | 0x7C00 | h_mant;
    }

    // 重新加偏置：FP32 bias 127 → FP16 bias 15
    const int32_t h_exp = int32_t(exp) - 127 + 15;

    if (h_exp >= 0x1F) {
        return h_sign | 0x7C00;  // 溢出 → Inf
    }

    if (h_exp > 0) {
        // FP16 规格化数：尾数 23 → 10，丢弃 13 位，banker's rounding
        uint32_t m10  = mant >> 13;
        uint32_t drop = mant & 0x1FFF;
        const uint32_t halfway = 0x1000;  // 1 << 12
        if (drop > halfway || (drop == halfway && (m10 & 1))) {
            ++m10;
            if (m10 == 0x400) { m10 = 0; /* 进位到下一指数，下面再判溢出 */ }
        }
        int32_t e = h_exp;
        if (m10 == 0) {
            ++e;
            if (e >= 0x1F) return h_sign | 0x7C00;
        }
        return h_sign | (uint16_t(e) << 10) | uint16_t(m10);
    }

    if (h_exp >= -9) {
        // FP16 非规格化数。尾数展开成 24 位（含隐式 1），按 shift 右移并舍入。
        const int      shift    = 14 - h_exp;          // h_exp=0 → 14, h_exp=-9 → 23
        const uint32_t mant_full = mant | 0x800000;    // 24-bit
        const uint32_t mask     = (1u << shift) - 1;
        const uint32_t halfway  = 1u << (shift - 1);
        const uint32_t drop     = mant_full & mask;
        uint32_t       h_mant   = mant_full >> shift;
        if (drop > halfway || (drop == halfway && (h_mant & 1))) ++h_mant;
        // 舍入到 0x400 时正好是最小规格化数 0x0400
        return h_sign | uint16_t(h_mant & 0x3FF);
    }

    // 下溢 → 0
    return h_sign;
}

static uint16_t bf16_bits_to_fp16_bits(uint16_t bf16_bits) {
    return fp32_bits_to_fp16_bits(uint32_t(bf16_bits) << 16);
}

static uint16_t f32_bits_to_fp16_bits(uint32_t f32_bits) {
    return fp32_bits_to_fp16_bits(f32_bits);
}

// src: 源数据原始字节（小端 u16/u32 元素数组）。返回转成 FP16 的字节流。
// 失败返回 false（不支持的 dtype）。
static bool convert_to_fp16(const std::string& src,
                            const std::string& dtype,
                            size_t numel,
                            std::string& out) {
    out.assign(numel * 2, '\0');
    auto* dst = reinterpret_cast<uint8_t*>(out.data());
    const auto* src_u8 = reinterpret_cast<const uint8_t*>(src.data());

    if (dtype == "F16") {
        std::memcpy(dst, src_u8, numel * 2);
        return true;
    }
    if (dtype == "BF16") {
        for (size_t i = 0; i < numel; ++i) {
            uint16_t bf = uint16_t(src_u8[2 * i]) | (uint16_t(src_u8[2 * i + 1]) << 8);
            uint16_t h  = bf16_bits_to_fp16_bits(bf);
            dst[2 * i]     = uint8_t(h & 0xFF);
            dst[2 * i + 1] = uint8_t(h >> 8);
        }
        return true;
    }
    if (dtype == "F32") {
        for (size_t i = 0; i < numel; ++i) {
            uint32_t f = uint32_t(src_u8[4 * i])
                       | (uint32_t(src_u8[4 * i + 1]) << 8)
                       | (uint32_t(src_u8[4 * i + 2]) << 16)
                       | (uint32_t(src_u8[4 * i + 3]) << 24);
            uint16_t h = f32_bits_to_fp16_bits(f);
            dst[2 * i]     = uint8_t(h & 0xFF);
            dst[2 * i + 1] = uint8_t(h >> 8);
        }
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <output_dir>" << std::endl;
        return 1;
    }
    const fs::path out_dir = argv[1];
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "cannot create output dir " << out_dir << ": " << ec.message() << std::endl;
        return 1;
    }

    SafeTensorFile top;
    if (!top.open(kModelPath)) {
        std::cerr << "failed to open model: " << kModelPath << std::endl;
        return 1;
    }

    size_t ok = 0, skip = 0, fail = 0;
    for (const auto& [name, info] : top.tensors()) {
        if (!info.verify_size()) {
            std::cerr << "[warn] size mismatch: " << name
                      << " (dtype=" << info.dtype
                      << ", bytes=" << info.bytes << ")" << std::endl;
        }

        std::string raw;
        if (!top.read_tensor(name, raw)) {
            std::cerr << "[fail] read: " << name << std::endl;
            ++fail;
            continue;
        }

        const size_t db = safetensors_dtype_bytes(info.dtype);
        if (db == 0) {
            std::cerr << "[skip] unsupported dtype " << info.dtype << ": " << name << std::endl;
            ++skip;
            continue;
        }
        const size_t numel = info.numel();
        if (raw.size() != numel * db) {
            std::cerr << "[skip] byte size mismatch: " << name
                      << " (got " << raw.size() << ", expect " << numel * db << ")" << std::endl;
            ++skip;
            continue;
        }

        std::string fp16;
        if (!convert_to_fp16(raw, info.dtype, numel, fp16)) {
            std::cerr << "[skip] cannot convert dtype " << info.dtype << ": " << name << std::endl;
            ++skip;
            continue;
        }

        fs::path out_path = out_dir / (name + ".fp16");
        std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            std::cerr << "[fail] open output: " << out_path << std::endl;
            ++fail;
            continue;
        }
        f.write(fp16.data(), static_cast<std::streamsize>(fp16.size()));
        if (!f) {
            std::cerr << "[fail] write: " << out_path << std::endl;
            ++fail;
            continue;
        }

        std::cout << "wrote " << name << ".fp16"
                  << " [" << info.dtype << "→F16]"
                  << " (" << fp16.size() << " bytes, " << numel << " elts)" << std::endl;
        ++ok;
    }

    std::cout << "done: " << ok << " ok, " << skip << " skipped, " << fail << " failed" << std::endl;
    return fail ? 1 : 0;
}
