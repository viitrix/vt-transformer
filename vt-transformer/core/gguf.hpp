// gguf.hpp — Header-only reader for GGUF (GPT-Generated Unified Format) files.
//
// GGUF 是 llama.cpp 系生态的模型容器格式，对应文件布局：
//   [4  bytes u32 LE]  : magic 'GGUF' (0x46554747)
//   [4  bytes u32 LE]  : version (支持 v2 / v3)
//   [8  bytes u64 LE]  : tensor_count
//   [8  bytes u64 LE]  : metadata_kv_count
//   [variable]         : metadata_kv_count 个 KV 记录
//   [variable]         : tensor_count    个 TensorInfo 记录
//   [padding]          : 对齐到 general.alignment（默认 32）字节边界
//   [remaining]        : 原始 tensor bytes，按 TensorInfo.offset 切片
//
// Metadata KV 记录布局：
//   key        : gguf_string  (u64 len + len bytes UTF-8)
//   value_type : u32 LE       (见 GgufValueType 枚举)
//   value      : 由 value_type 决定；array 为 u32 elem_type + u64 len + len 个元素
//
// TensorInfo 记录布局：
//   name   : gguf_string
//   n_dims : u32 LE
//   dims   : n_dims × u64 LE        （注意：ggml 风格，dims[0] 为最内层维度）
//   type   : u32 LE                  (见 GgufGgmlType 枚举)
//   offset : u64 LE                  (相对 data_start 的偏移)
//
// 入口：open(path) 解析 header 与所有 TensorInfo；之后通过 find / read_tensor 取数据。
//
//   vt::GgufFile f;
//   if (!f.open("model.gguf")) { ... }
//   const auto* info = f.find("token_embd.weight");
//   std::string bytes;
//   f.read_tensor("token_embd.weight", bytes);
//
// Metadata 通过 JSON 暴露（GGUF 的 value 类型异构，json 提供统一接口）：
//   const json& md = f.metadata();
//   uint64_t n_layers = md.value("llama.block_count", 0ULL);
//
// Dependencies:
//   - opt/json.hpp        (nlohmann/json)
//
// Limitations:
//   - 仅 little-endian 主机（GGUF 本身是 LE；safetensors.hpp 同样假设）。
//   - 量化类型 bytes 按 llama.cpp 的 block_size/type_size 表推算；遇到未列入表的特殊
//     类型时 bytes=0，read_tensor 会失败但不会误读。
//
// Thread-safety: open() is not; after open() the object is read-only.

#ifndef _VT_GGUF_HPP_
#define _VT_GGUF_HPP_

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "opt/json.hpp"

using nlohmann::json;

namespace vt {

// GGUF metadata value type 枚举（与官方 llama.cpp 一致）。
enum GgufValueType : uint32_t {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

// GGML tensor type 枚举（仅列出常见类型；type_id → 字符串名见 ggml_type_name()）。
enum GgufGgmlType : uint32_t {
    GGML_TYPE_F32      = 0,
    GGML_TYPE_F16      = 1,
    GGML_TYPE_Q4_0     = 2,
    GGML_TYPE_Q4_1     = 3,
    GGML_TYPE_Q5_0     = 6,
    GGML_TYPE_Q5_1     = 7,
    GGML_TYPE_Q8_0     = 8,
    GGML_TYPE_Q8_1     = 9,
    GGML_TYPE_Q2_K     = 10,
    GGML_TYPE_Q3_K     = 11,
    GGML_TYPE_Q4_K     = 12,
    GGML_TYPE_Q5_K     = 13,
    GGML_TYPE_Q6_K     = 14,
    GGML_TYPE_Q8_K     = 15,
    GGML_TYPE_IQ2_XXS  = 16,
    GGML_TYPE_IQ2_XS   = 17,
    GGML_TYPE_IQ3_XXS  = 18,
    GGML_TYPE_IQ1_S    = 19,
    GGML_TYPE_IQ4_NL   = 20,
    GGML_TYPE_IQ3_S    = 21,
    GGML_TYPE_IQ2_S    = 22,
    GGML_TYPE_IQ4_XS   = 23,
    GGML_TYPE_I8       = 24,
    GGML_TYPE_I16      = 25,
    GGML_TYPE_I32      = 26,
    GGML_TYPE_I64      = 27,
    GGML_TYPE_F64      = 28,
    GGML_TYPE_IQ1_M    = 29,
    GGML_TYPE_BF16     = 30,
};

// 返回 ggml type 的可读名（未知 type 返回 "UNKNOWN(<id>)"）。
inline std::string ggml_type_name(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:     return "F32";
        case GGML_TYPE_F16:     return "F16";
        case GGML_TYPE_BF16:    return "BF16";
        case GGML_TYPE_Q4_0:    return "Q4_0";
        case GGML_TYPE_Q4_1:    return "Q4_1";
        case GGML_TYPE_Q5_0:    return "Q5_0";
        case GGML_TYPE_Q5_1:    return "Q5_1";
        case GGML_TYPE_Q8_0:    return "Q8_0";
        case GGML_TYPE_Q8_1:    return "Q8_1";
        case GGML_TYPE_Q2_K:    return "Q2_K";
        case GGML_TYPE_Q3_K:    return "Q3_K";
        case GGML_TYPE_Q4_K:    return "Q4_K";
        case GGML_TYPE_Q5_K:    return "Q5_K";
        case GGML_TYPE_Q6_K:    return "Q6_K";
        case GGML_TYPE_Q8_K:    return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ1_S:   return "IQ1_S";
        case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case GGML_TYPE_IQ3_S:   return "IQ3_S";
        case GGML_TYPE_IQ2_S:   return "IQ2_S";
        case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case GGML_TYPE_I8:      return "I8";
        case GGML_TYPE_I16:     return "I16";
        case GGML_TYPE_I32:     return "I32";
        case GGML_TYPE_I64:     return "I64";
        case GGML_TYPE_F64:     return "F64";
        case GGML_TYPE_IQ1_M:   return "IQ1_M";
        default:                return "UNKNOWN(" + std::to_string(type) + ")";
    }
}

// 返回 ggml type 的 (block_size, type_size_in_bytes)。
//   - 非量化类型：block_size = 1，type_size = 每元素字节数
//   - 量化类型：  block_size = 每块包含的元素数，type_size = 每块字节数
//   - 未知类型：  返回 (0, 0)
inline std::pair<uint64_t, uint64_t> ggml_type_sizes(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:     return {1, 4};
        case GGML_TYPE_F16:     return {1, 2};
        case GGML_TYPE_BF16:    return {1, 2};
        case GGML_TYPE_F64:     return {1, 8};
        case GGML_TYPE_I8:      return {1, 1};
        case GGML_TYPE_I16:     return {1, 2};
        case GGML_TYPE_I32:     return {1, 4};
        case GGML_TYPE_I64:     return {1, 8};
        case GGML_TYPE_Q4_0:    return {32, 18};
        case GGML_TYPE_Q4_1:    return {32, 20};
        case GGML_TYPE_Q5_0:    return {32, 22};
        case GGML_TYPE_Q5_1:    return {32, 24};
        case GGML_TYPE_Q8_0:    return {32, 34};
        case GGML_TYPE_Q8_1:    return {32, 36};
        case GGML_TYPE_Q2_K:    return {256, 84};
        case GGML_TYPE_Q3_K:    return {256, 110};
        case GGML_TYPE_Q4_K:    return {256, 144};
        case GGML_TYPE_Q5_K:    return {256, 176};
        case GGML_TYPE_Q6_K:    return {256, 210};
        case GGML_TYPE_Q8_K:    return {256, 292};
        case GGML_TYPE_IQ2_XXS: return {256, 66};
        case GGML_TYPE_IQ2_XS:  return {256, 74};
        case GGML_TYPE_IQ3_XXS: return {256, 98};
        case GGML_TYPE_IQ1_S:   return {256, 54};
        case GGML_TYPE_IQ4_NL:  return {18, 9};
        case GGML_TYPE_IQ3_S:   return {256, 110};
        case GGML_TYPE_IQ2_S:   return {256, 82};
        case GGML_TYPE_IQ4_XS:  return {256, 136};
        case GGML_TYPE_IQ1_M:   return {256, 56};
        default:                return {0, 0};
    }
}

// 给定 ggml type 与元素数，计算 tensor 总字节数。未知 type 返回 0。
inline uint64_t ggml_tensor_bytes(uint32_t type, uint64_t numel) {
    auto [block_size, type_size] = ggml_type_sizes(type);
    if (block_size == 0) return 0;
    if (block_size == 1) return numel * type_size;
    if (numel % block_size != 0) return 0;  // 量化要求 numel 是 block_size 的整数倍
    return (numel / block_size) * type_size;
}

struct GgufTensorInfo {
    std::string                name;
    std::vector<uint64_t>      dims;        // ggml 风格：dims[0] 为最内层（最快变化）维度
    uint32_t                   ggml_type = 0;
    uint64_t                   offset    = 0;  // 相对 data_start_ 的偏移
    uint64_t                   bytes     = 0;  // 推算出的字节数（按 type × numel）

    uint64_t numel() const {
        uint64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }

    std::string type_name() const { return ggml_type_name(ggml_type); }
};

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile() = default;
    GgufFile(const GgufFile&)            = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    // 解析 GGUF header 与所有 TensorInfo。失败返回 false（对象状态保持空）。
    bool open(const std::string& path) {
        path_ = path;
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        uint32_t magic = 0;
        if (!read_u32(f, magic)) return false;
        if (magic != 0x46554747) return false;   // "GGUF" LE

        uint32_t version = 0;
        if (!read_u32(f, version)) return false;
        if (version < 2 || version > 3) return false;  // 仅支持 v2/v3
        version_ = version;

        uint64_t tensor_count = 0, kv_count = 0;
        if (!read_u64(f, tensor_count)) return false;
        if (!read_u64(f, kv_count))     return false;

        // 1) metadata KV
        metadata_ = json::object();
        value_types_.clear();
        array_elem_types_.clear();
        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key;
            if (!read_string(f, key)) return false;

            uint32_t vtype = 0;
            if (!read_u32(f, vtype)) return false;

            json v;
            if (!read_value(f, vtype, v, key)) return false;

            // 同名 key 重复 —— 文件损坏
            if (metadata_.contains(key)) return false;
            metadata_[key] = std::move(v);
            value_types_[key] = static_cast<GgufValueType>(vtype);
        }

        // 2) tensor infos
        tensors_.clear();
        tensors_.reserve(static_cast<size_t>(tensor_count));
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GgufTensorInfo info;

            if (!read_string(f, info.name)) return false;

            uint32_t n_dims = 0;
            if (!read_u32(f, n_dims)) return false;
            if (n_dims == 0 || n_dims > 8) return false;  // 合理性上限

            info.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                if (!read_u64(f, info.dims[d])) return false;
            }

            if (!read_u32(f, info.ggml_type)) return false;
            if (!read_u64(f, info.offset))    return false;

            info.bytes = ggml_tensor_bytes(info.ggml_type, info.numel());

            if (tensors_by_name_.count(info.name)) return false;  // 重名
            tensors_by_name_[info.name] = tensors_.size();
            tensors_.push_back(std::move(info));
        }

        // 3) data_start：当前 stream 位置对齐到 alignment
        alignment_ = 32;
        if (metadata_.contains("general.alignment")) {
            const auto& a = metadata_["general.alignment"];
            if (a.is_number_unsigned()) {
                uint64_t parsed = a.get<uint64_t>();
                if (parsed > 0) alignment_ = parsed;
            }
        }
        uint64_t cur = static_cast<uint64_t>(f.tellg());
        data_start_ = align_up(cur, alignment_);

        opened_ = true;
        return true;
    }

    bool is_open() const { return opened_; }
    uint32_t version()     const { return version_; }
    uint64_t alignment()   const { return alignment_; }
    uint64_t data_start()  const { return data_start_; }
    size_t   num_tensors() const { return tensors_.size(); }

    // metadata 访问：JSON 视图 + 类型信息
    const json& metadata() const { return metadata_; }

    // 顶层 value_type（含 ARRAY）；不存在返回 GGUF_VALUE_UINT8 是无法区分的，所以调用方
    // 应先 contains()。数组元素类型另用 array_elem_type()。
    GgufValueType value_type(const std::string& key) const {
        auto it = value_types_.find(key);
        return it == value_types_.end() ? GGUF_VALUE_UINT8 : it->second;
    }
    bool has_key(const std::string& key) const { return value_types_.count(key) > 0; }

    // 数组元素的 value_type；非数组或不存在返回 GGUF_VALUE_UINT8（调用方应先校验）。
    GgufValueType array_elem_type(const std::string& key) const {
        auto it = array_elem_types_.find(key);
        return it == array_elem_types_.end() ? GGUF_VALUE_UINT8 : it->second;
    }

    // tensor 视图
    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }

    const GgufTensorInfo* find(const std::string& name) const {
        auto it = tensors_by_name_.find(name);
        if (it == tensors_by_name_.end()) return nullptr;
        return &tensors_[it->second];
    }

    // 按 info.offset / info.bytes 读出 tensor 原始 bytes。
    // 失败（unknown type / 越界 / IO 错误）返回 false，out 保持不变。
    bool read_raw(const GgufTensorInfo& info, std::string& out) const {
        if (!opened_) return false;
        if (info.bytes == 0) return false;  // 包含 unknown type 推不出 size 的情形
        std::ifstream f(path_, std::ios::binary);
        if (!f) return false;
        uint64_t abs_off = data_start_ + info.offset;
        f.seekg(static_cast<std::streamoff>(abs_off), std::ios::beg);
        if (!f) return false;
        std::string buf(static_cast<size_t>(info.bytes), '\0');
        f.read(&buf[0], static_cast<std::streamsize>(info.bytes));
        if (f.gcount() != static_cast<std::streamsize>(info.bytes)) return false;
        out.swap(buf);
        return true;
    }

    // 便捷接口：按名字定位 tensor 并读出 bytes。失败返回 false。
    bool read_tensor(const std::string& name, std::string& out) const {
        const auto* info = find(name);
        if (!info) return false;
        return read_raw(*info, out);
    }

private:
    // ---------- 低层读取 ----------
    static bool read_u32(std::ifstream& f, uint32_t& out) {
        f.read(reinterpret_cast<char*>(&out), 4);
        return f.good();
    }
    static bool read_u64(std::ifstream& f, uint64_t& out) {
        f.read(reinterpret_cast<char*>(&out), 8);
        return f.good();
    }
    static bool read_i32(std::ifstream& f, int32_t& out) {
        f.read(reinterpret_cast<char*>(&out), 4);
        return f.good();
    }
    static bool read_i64(std::ifstream& f, int64_t& out) {
        f.read(reinterpret_cast<char*>(&out), 8);
        return f.good();
    }
    static bool read_f32(std::ifstream& f, float& out) {
        f.read(reinterpret_cast<char*>(&out), 4);
        return f.good();
    }
    static bool read_f64(std::ifstream& f, double& out) {
        f.read(reinterpret_cast<char*>(&out), 8);
        return f.good();
    }
    static bool read_u8(std::ifstream& f, uint8_t& out) {
        f.read(reinterpret_cast<char*>(&out), 1);
        return f.good();
    }
    static bool read_i8(std::ifstream& f, int8_t& out) {
        f.read(reinterpret_cast<char*>(&out), 1);
        return f.good();
    }
    static bool read_u16(std::ifstream& f, uint16_t& out) {
        f.read(reinterpret_cast<char*>(&out), 2);
        return f.good();
    }
    static bool read_i16(std::ifstream& f, int16_t& out) {
        f.read(reinterpret_cast<char*>(&out), 2);
        return f.good();
    }

    static bool read_string(std::ifstream& f, std::string& out) {
        uint64_t len = 0;
        if (!read_u64(f, len)) return false;
        if (len > (1ULL << 32)) return false;  // 合理性上限 ~4GB
        out.assign(static_cast<size_t>(len), '\0');
        if (len > 0) {
            f.read(&out[0], static_cast<std::streamsize>(len));
            if (!f.good()) return false;
        }
        return true;
    }

    // 按 vtype 读取一个值，写入 json。array 的 elem_type 同时记录到 array_elem_types_。
    // 注：本函数会在 array 分支里写入 array_elem_types_，所以 key 需要传入。
    bool read_value(std::ifstream& f, uint32_t vtype, json& out, const std::string& key = "") {
        switch (vtype) {
            case GGUF_VALUE_UINT8: {
                uint8_t v;  if (!read_u8(f, v)) return false;
                out = static_cast<uint64_t>(v);
                return true;
            }
            case GGUF_VALUE_INT8: {
                int8_t v;   if (!read_i8(f, v)) return false;
                out = static_cast<int64_t>(v);
                return true;
            }
            case GGUF_VALUE_UINT16: {
                uint16_t v; if (!read_u16(f, v)) return false;
                out = static_cast<uint64_t>(v);
                return true;
            }
            case GGUF_VALUE_INT16: {
                int16_t v;  if (!read_i16(f, v)) return false;
                out = static_cast<int64_t>(v);
                return true;
            }
            case GGUF_VALUE_UINT32: {
                uint32_t v; if (!read_u32(f, v)) return false;
                out = static_cast<uint64_t>(v);
                return true;
            }
            case GGUF_VALUE_INT32: {
                int32_t v;  if (!read_i32(f, v)) return false;
                out = static_cast<int64_t>(v);
                return true;
            }
            case GGUF_VALUE_UINT64: {
                uint64_t v; if (!read_u64(f, v)) return false;
                out = v;
                return true;
            }
            case GGUF_VALUE_INT64: {
                int64_t v;  if (!read_i64(f, v)) return false;
                out = v;
                return true;
            }
            case GGUF_VALUE_FLOAT32: {
                float v;    if (!read_f32(f, v)) return false;
                out = static_cast<double>(v);
                return true;
            }
            case GGUF_VALUE_FLOAT64: {
                double v;   if (!read_f64(f, v)) return false;
                out = v;
                return true;
            }
            case GGUF_VALUE_BOOL: {
                uint8_t v;  if (!read_u8(f, v)) return false;
                out = (v != 0);
                return true;
            }
            case GGUF_VALUE_STRING: {
                std::string s;
                if (!read_string(f, s)) return false;
                out = std::move(s);
                return true;
            }
            case GGUF_VALUE_ARRAY: {
                uint32_t elem_type = 0;
                uint64_t len = 0;
                if (!read_u32(f, elem_type)) return false;
                if (!read_u64(f, len))        return false;
                if (len > (1ULL << 32))       return false;

                json arr = json::array();
                for (uint64_t i = 0; i < len; ++i) {
                    json elem;
                    if (!read_value(f, elem_type, elem)) return false;
                    arr.push_back(std::move(elem));
                }
                out = std::move(arr);
                if (!key.empty()) {
                    array_elem_types_[key] = static_cast<GgufValueType>(elem_type);
                }
                return true;
            }
            default:
                return false;
        }
    }

    static uint64_t align_up(uint64_t v, uint64_t align) {
        if (align == 0) return v;
        return (v + align - 1) / align * align;
    }

private:
    std::string        path_;
    bool               opened_   = false;
    uint32_t           version_  = 0;
    uint64_t           alignment_ = 32;
    uint64_t           data_start_ = 0;

    json                                                  metadata_;
    std::unordered_map<std::string, GgufValueType>        value_types_;
    std::unordered_map<std::string, GgufValueType>        array_elem_types_;

    std::vector<GgufTensorInfo>                           tensors_;
    std::unordered_map<std::string, size_t>               tensors_by_name_;
};

} // namespace vt

#endif // _VT_GGUF_HPP_
