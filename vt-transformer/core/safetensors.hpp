// safetensors.hpp — Header-only reader for Hugging Face safetensors files.
//
// 支持两种格式：
//   1. 单文件：model.safetensors
//   2. 分片：  model.safetensors.index.json + model-00001-of-00005.safetensors + ...
//
// 入口：open(path) 自动按扩展名分发
//   - "*.safetensors"                  → 单文件模式
//   - "*.index.json" / "*_index.json"  → 分片模式（解析 index，自动加载所有 shard）
//
// 读取 tensor 原始 bytes（两种模式通用）：
//   std::string bytes;
//   top.read_tensor("layer.0.weight", bytes);   // 自动定位 owner shard
//
// 或显式分两步（需要先拿到 SafeTensorInfo）：
//   auto* info   = top.find("layer.0.weight");
//   auto* owner  = top.find_owner("layer.0.weight");
//   owner->read_raw(*info, bytes);              // 必须在 owner 上调用
//
// 文件布局（单文件）：
//   [8 bytes u64 LE]  : JSON header length N
//   [N bytes UTF-8]   : JSON metadata describing tensors
//   [remaining]       : raw tensor bytes, sliced by data_offsets (relative to here)
//
// Index JSON 布局（分片）：
//   { "metadata": { "total_size": N },
//     "weight_map": { "tensor_name": "shard_filename.safetensors", ... } }
//
// Dependencies:
//   - opt/json.hpp        (nlohmann/json)
//
// Thread-safety: open() is not; after open() the object is read-only.

#ifndef _VT_SAFETENSORS_HPP_
#define _VT_SAFETENSORS_HPP_

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "opt/json.hpp"

using nlohmann::json;

namespace vt {

// dtype 字节数。返回 0 表示未知 dtype。
inline size_t safetensors_dtype_bytes(const std::string& dtype) {
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "I8"  || dtype == "U8"  || dtype == "BOOL") return 1;
    return 0;
}

struct SafeTensorInfo {
    std::string           dtype;       // "F32" / "BF16" / "F16" / "I32" / ...
    std::vector<int64_t>  shape;
    uint64_t              offset = 0;  // 绝对文件偏移（已含 data_start，单文件模式有效）
    uint64_t              bytes  = 0;

    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    bool verify_size() const {
        if (bytes == 0) return true;
        size_t db = safetensors_dtype_bytes(dtype);
        if (db == 0) return false;
        return bytes == numel() * db;
    }
};

class SafeTensorFile {
public:
    SafeTensorFile() = default;
    ~SafeTensorFile() = default;
    SafeTensorFile(const SafeTensorFile&)            = delete;
    SafeTensorFile& operator=(const SafeTensorFile&) = delete;

    // 智能入口：按扩展名自动分发。
    //   "*.safetensors"                  → open_single
    //   "*.index.json" / "*_index.json"  → open_index
    // 失败返回 false。
    bool open(const std::string& path) {
        auto ends_with = [](const std::string& s, const std::string& suffix) {
            return s.size() >= suffix.size() &&
                   s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        if (ends_with(path, ".index.json") || ends_with(path, "_index.json")) {
            return open_index(path);
        }
        return open_single(path);
    }

    bool is_sharded() const { return !shards_.empty(); }

    const std::unordered_map<std::string, SafeTensorInfo>& tensors() const { return tensors_; }

    const SafeTensorInfo* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    // 返回持有 name 的物理文件。
    //   单文件模式：返回 this
    //   分片模式：  返回对应 shard
    //   未 open / 未找到：返回 nullptr
    // 调用方应在此返回值上调用 read_raw —— info.offset 是 owner path_ 的绝对偏移。
    const SafeTensorFile* find_owner(const std::string& name) const {
        if (tensors_.find(name) == tensors_.end()) return nullptr;
        if (shards_.empty()) return this;
        auto it = tensor_owner_.find(name);
        return it == tensor_owner_.end() ? nullptr : it->second;
    }

    // 在本文件 path_ 上按 info.offset / info.bytes 读出原始 bytes。
    // 必须在 owner 上调用（即 find_owner 返回值）；失败返回 false 且 out 保持不变。
    bool read_raw(const SafeTensorInfo& info, std::string& out) const {
        if (path_.empty()) return false;
        if (info.bytes == 0) {
            out.clear();
            return true;
        }
        std::ifstream f(path_, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
        if (!f) return false;
        std::string buf(static_cast<size_t>(info.bytes), '\0');
        f.read(&buf[0], static_cast<std::streamsize>(info.bytes));
        if (f.gcount() != static_cast<std::streamsize>(info.bytes)) return false;
        out.swap(buf);
        return true;
    }

    // 便捷接口：自动定位 owner 并读出 tensor 数据。失败返回 false。
    bool read_tensor(const std::string& name, std::string& out) const {
        const auto* owner = find_owner(name);
        if (!owner) return false;
        const auto* info = owner->find(name);
        if (!info) return false;
        return owner->read_raw(*info, out);
    }

    size_t num_tensors() const { return tensors_.size(); }
    size_t num_shards()  const { return shards_.size(); }

    // 分片模式下从 index.json 读出的总字节数；单文件模式返回 0。
    uint64_t total_size() const { return total_size_; }

private:
    // 单文件解析（或被分片模式当作子 shard 调用）
    bool open_single(const std::string& path) {
        path_ = path;
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        uint64_t header_len = 0;
        f.read(reinterpret_cast<char*>(&header_len), 8);
        if (!f || header_len == 0 || header_len > (1ULL << 30)) return false;

        std::string header_str(header_len, '\0');
        f.read(header_str.data(), header_len);
        if (!f) return false;

        data_start_ = 8 + header_len;

        json j;
        try {
            j = json::parse(header_str);
        } catch (...) {
            return false;
        }

        tensors_.clear();
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            const auto& v = it.value();
            if (!v.is_object()) continue;

            SafeTensorInfo info;
            info.dtype = v.value("dtype", std::string());
            if (v.contains("shape") && v["shape"].is_array()) {
                info.shape = v["shape"].get<std::vector<int64_t>>();
            }
            if (v.contains("data_offsets") && v["data_offsets"].size() == 2) {
                uint64_t lo = v["data_offsets"][0].get<uint64_t>();
                uint64_t hi = v["data_offsets"][1].get<uint64_t>();
                info.offset = data_start_ + lo;
                info.bytes  = hi - lo;
            }
            tensors_[it.key()] = std::move(info);
        }
        return true;
    }

    // 分片解析：读 index.json，加载所有 shard，合并 tensor 视图
    bool open_index(const std::string& index_path) {
        std::ifstream f(index_path);
        if (!f) return false;

        json j;
        try {
            j = json::parse(f);
        } catch (...) {
            return false;
        }

        if (!j.contains("weight_map") || !j["weight_map"].is_object()) return false;
        const auto& weight_map = j["weight_map"];

        // index.json 所在目录（用于解析 shard 的相对路径）
        auto slash_pos = index_path.find_last_of('/');
        std::string base_dir = (slash_pos == std::string::npos)
                                 ? std::string()
                                 : index_path.substr(0, slash_pos + 1);

        // metadata.total_size（可选）
        if (j.contains("metadata") && j["metadata"].contains("total_size")) {
            total_size_ = j["metadata"]["total_size"].get<uint64_t>();
        }

        // 收集 unique shard 文件名，按发现顺序加载
        std::vector<std::string> shard_order;
        std::unordered_map<std::string, size_t> shard_index;
        for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
            std::string shard_name = it.value().get<std::string>();
            if (shard_index.find(shard_name) == shard_index.end()) {
                shard_index[shard_name] = shard_order.size();
                shard_order.push_back(shard_name);
            }
        }

        // 加载每个 shard
        shards_.clear();
        shards_.reserve(shard_order.size());
        for (const auto& shard_name : shard_order) {
            auto shard = std::make_unique<SafeTensorFile>();
            std::string shard_path = base_dir + shard_name;
            if (!shard->open_single(shard_path)) {
                shards_.clear();
                return false;
            }
            shards_.push_back(std::move(shard));
        }

        // 合并视图
        tensors_.clear();
        tensor_owner_.clear();
        for (auto& shard_ptr : shards_) {
            for (auto& [tensor_name, info] : shard_ptr->tensors_) {
                if (tensors_.count(tensor_name)) {
                    // 同名 tensor 出现在多个 shard —— index.json 损坏
                    return false;
                }
                tensors_[tensor_name] = info;
                tensor_owner_[tensor_name] = shard_ptr.get();
            }
        }
        return true;
    }

private:
    // 单文件模式有效
    std::string                                         path_;
    uint64_t                                            data_start_ = 0;

    // 合并视图（两种模式都填）
    std::unordered_map<std::string, SafeTensorInfo>     tensors_;

    // 分片模式：tensor_name → owning shard
    std::unordered_map<std::string, SafeTensorFile*>    tensor_owner_;
    std::vector<std::unique_ptr<SafeTensorFile>>        shards_;
    uint64_t                                            total_size_ = 0;
};

} // namespace vt

#endif // _VT_SAFETENSORS_HPP_
