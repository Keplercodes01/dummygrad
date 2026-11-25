#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <iostream>

#include "json.h"
#include "mapped_file.h"
#include "tensor.h"
#include "types.h"

// Pure C++ Zero-Copy SafeTensors & Binary Serialization Engine
// Interoperable with Hugging Face SafeTensors checkpoints with zero Python dependencies
namespace io {

// Exact 64-bit scalars (step counters, metrics) stored as the raw bytes of a 2-element Float32
// tensor, so they survive a SafeTensors round trip without float rounding.
inline std::shared_ptr<Tensor> encode_int64(int64_t value) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{2}, Device::CPU, DType::Float32, false);
    std::memcpy(t->data_ptr<void>(), &value, sizeof(value));
    return t;
}

inline std::shared_ptr<Tensor> encode_double(double value) {
    auto t = std::make_shared<Tensor>(std::vector<int64_t>{2}, Device::CPU, DType::Float32, false);
    std::memcpy(t->data_ptr<void>(), &value, sizeof(value));
    return t;
}

inline int64_t decode_int64(const std::shared_ptr<Tensor>& t) {
    if (!t || t->size() * dtype_size(t->dtype) != sizeof(int64_t)) throw std::runtime_error("decode_int64: not an encoded scalar");
    int64_t value;
    auto host = make_contiguous(t)->cpu();
    std::memcpy(&value, host->data_ptr<void>(), sizeof(value));
    return value;
}

inline double decode_double(const std::shared_ptr<Tensor>& t) {
    if (!t || t->size() * dtype_size(t->dtype) != sizeof(double)) throw std::runtime_error("decode_double: not an encoded scalar");
    double value;
    auto host = make_contiguous(t)->cpu();
    std::memcpy(&value, host->data_ptr<void>(), sizeof(value));
    return value;
}


inline std::string dtype_to_safetensors_str(DType dt) {
    switch (dt) {
        case DType::Float32:  return "F32";
        case DType::Float16:  return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Int8:     return "I8";
        default:              return "F32";
    }
}

// Bytes per element of a safetensors dtype, or 0 if unknown
inline size_t safetensors_dtype_size(const std::string& s) {
    if (s == "F64" || s == "I64" || s == "U64") return 8;
    if (s == "F32" || s == "I32" || s == "U32") return 4;
    if (s == "F16" || s == "BF16" || s == "I16" || s == "U16") return 2;
    if (s == "I8" || s == "U8" || s == "BOOL") return 1;
    return 0;
}

// A tensor from a checkpoint's bytes. F32, F16, BF16 and I8 keep their dtype. The others become Float32, the
// type ids and counts have here: integers stay exact up to 2^24, F64 is rounded.
inline std::shared_ptr<Tensor> tensor_from_bytes(const uint8_t* src, size_t bytes, const std::string& dtype,
                                                 const std::vector<int64_t>& shape, const std::string& name) {
    DType native = dtype == "F16" ? DType::Float16 : dtype == "BF16" ? DType::BFloat16 : dtype == "I8" ? DType::Int8 : DType::Float32;
    auto t = std::make_shared<Tensor>(shape, Device::CPU, native, false);
    if (dtype == "F32" || dtype == "F16" || dtype == "BF16" || dtype == "I8") {
        if (bytes > 0) std::memcpy(t->data_ptr<char>(), src, bytes);
        return t;
    }
    float* out = t->data_ptr<float>();
    int64_t n = t->size();
    auto convert = [&](auto zero) {
        using T = decltype(zero);
        for (int64_t i = 0; i < n; ++i) {
            T v;
            std::memcpy(&v, src + (size_t)i * sizeof(T), sizeof(T));
            out[i] = (float)v;
        }
    };
    if (dtype == "F64") convert(double{});
    else if (dtype == "I64") convert(int64_t{});
    else if (dtype == "U64") convert(uint64_t{});
    else if (dtype == "I32") convert(int32_t{});
    else if (dtype == "U32") convert(uint32_t{});
    else if (dtype == "I16") convert(int16_t{});
    else if (dtype == "U16") convert(uint16_t{});
    else if (dtype == "U8" || dtype == "BOOL") convert(uint8_t{});
    else throw std::runtime_error("load_safetensors: unsupported dtype \"" + dtype + "\" for tensor " + name);
    return t;
}

// Element count of a shape, or -1 if it is negative or overflows
inline int64_t checked_numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0 || (d > 0 && n > INT64_MAX / d)) return -1;
        n *= d;
    }
    return n;
}

// Save named tensors to a Hugging Face compatible .safetensors file
inline void save_safetensors(
    const std::string& filepath,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& named_tensors
) {
    // 1. Build JSON metadata header and calculate byte offsets
    std::ostringstream json_ss;
    json_ss << "{";

    uint64_t current_offset = 0;
    bool first = true;

    struct TensorMeta {
        std::string name;
        std::shared_ptr<Tensor> tensor;
        uint64_t start_byte;
        uint64_t end_byte;
    };
    std::vector<TensorMeta> ordered_tensors;

    for (const auto& pair : named_tensors) {
        const std::string& name = pair.first;
        const auto& tensor = pair.second;
        if (!tensor) continue;

        uint64_t bytes = static_cast<uint64_t>(tensor->size()) * dtype_size(tensor->dtype);
        uint64_t start = current_offset;
        uint64_t end = current_offset + bytes;
        current_offset = end;

        ordered_tensors.push_back({name, tensor, start, end});

        if (!first) json_ss << ",";
        first = false;

        json_ss << json::quote(name) << ":{"
                << "\"dtype\":\"" << dtype_to_safetensors_str(tensor->dtype) << "\","
                << "\"shape\":[";
        for (size_t s = 0; s < tensor->shape.size(); ++s) {
            json_ss << tensor->shape[s];
            if (s + 1 < tensor->shape.size()) json_ss << ",";
        }
        json_ss << "],"
                << "\"data_offsets\":[" << start << "," << end << "]}";
    }
    json_ss << "}";

    std::string json_header = json_ss.str();
    uint64_t header_len = static_cast<uint64_t>(json_header.size());

    // 2. Write 8-byte header size + JSON string + raw contiguous tensor payloads
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("save_safetensors: Unable to open file for writing: " + filepath);
    }

    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(json_header.data(), header_len);

    for (const auto& item : ordered_tensors) {
        auto host_tensor = make_contiguous(item.tensor)->cpu();
        uint64_t bytes = item.end_byte - item.start_byte;
        out.write(host_tensor->data_ptr<char>(), bytes);
    }

    out.close();
}

// One tensor of a .safetensors file
struct SafeTensorEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset_start = 0;  // bytes from the end of the header
    uint64_t offset_end = 0;
};

inline std::vector<SafeTensorEntry> parse_safetensors_header(const std::string& text) {
    std::vector<SafeTensorEntry> entries;
    auto header = json::parse(text);
    if (!header.is_object()) throw std::runtime_error("load_safetensors: the header is not an object");
    for (const auto& [name, v] : header.object) {
        if (name == "__metadata__") continue;
        SafeTensorEntry e;
        e.name = name;
        e.dtype = v.at("dtype").as_string();
        for (const auto& d : v.at("shape").array) e.shape.push_back(d.as_int());
        const auto& offsets = v.at("data_offsets").array;
        if (offsets.size() != 2 || offsets[0].as_int() < 0 || offsets[1].as_int() < 0) {
            throw std::runtime_error("load_safetensors: bad offsets for tensor: " + name);
        }
        e.offset_start = (uint64_t)offsets[0].as_int();
        e.offset_end = (uint64_t)offsets[1].as_int();
        entries.push_back(std::move(e));
    }
    return entries;
}

// A .safetensors file whose tensors are read one at a time (the file is memory-mapped), so that a large
// checkpoint never sits in memory twice
class SafeTensorsFile {
public:
    explicit SafeTensorsFile(const std::string& path) : path_(path), file_(path) {
        if (file_.size() < 8) throw std::runtime_error("load_safetensors: corrupt or empty file: " + path);
        uint64_t header_len;
        std::memcpy(&header_len, file_.data(), sizeof(header_len));
        if (header_len > file_.size() - 8) throw std::runtime_error("load_safetensors: corrupt header size in " + path);
        for (auto& e : parse_safetensors_header(std::string(reinterpret_cast<const char*>(file_.data() + 8), (size_t)header_len))) {
            size_t elem = safetensors_dtype_size(e.dtype);
            if (elem == 0) throw std::runtime_error("load_safetensors: unsupported dtype \"" + e.dtype + "\" for tensor " + e.name);
            if (e.offset_start > e.offset_end || e.offset_end > file_.size() - 8 - header_len) {
                throw std::runtime_error("load_safetensors: offsets out of bounds for tensor: " + e.name);
            }
            uint64_t bytes = e.offset_end - e.offset_start;
            int64_t n = checked_numel(e.shape);
            if (n < 0 || bytes % elem != 0 || (uint64_t)n != bytes / elem) {
                throw std::runtime_error("load_safetensors: byte size mismatch for tensor: " + e.name);
            }
            std::string name = e.name;
            entries_[name] = std::move(e);
        }
        payload_ = file_.data() + 8 + header_len;
    }

    const std::unordered_map<std::string, SafeTensorEntry>& entries() const { return entries_; }
    bool has(const std::string& name) const { return entries_.count(name) > 0; }

    // A copy of the tensor on the CPU
    std::shared_ptr<Tensor> read(const std::string& name) const {
        auto it = entries_.find(name);
        if (it == entries_.end()) throw std::runtime_error("load_safetensors: no tensor " + name + " in " + path_);
        const auto& e = it->second;
        return tensor_from_bytes(payload_ + e.offset_start, (size_t)(e.offset_end - e.offset_start), e.dtype, e.shape, name);
    }

private:
    std::string path_;
    MappedFile file_;
    const uint8_t* payload_ = nullptr;
    std::unordered_map<std::string, SafeTensorEntry> entries_;
};

// Every tensor of a .safetensors file
inline std::unordered_map<std::string, std::shared_ptr<Tensor>> load_safetensors(
    const std::string& filepath,
    Device device = Device::CPU
) {
    SafeTensorsFile file(filepath);
    std::unordered_map<std::string, std::shared_ptr<Tensor>> result;
    for (const auto& entry : file.entries()) {
        auto t = file.read(entry.first);
        result[entry.first] = device == Device::CPU ? t : t->to(device);
    }
    return result;
}

// -------------------------------------------------------------
// Ultra-fast Native Binary Checkpointing (.bin)
// -------------------------------------------------------------
inline void save_checkpoint(
    const std::string& filepath,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state_dict
) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("save_checkpoint: Unable to open file: " + filepath);
    }

    const uint32_t MAGIC = 0x44554D47; // "DUMG"
    out.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));

    uint32_t num_tensors = static_cast<uint32_t>(state_dict.size());
    out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

    for (const auto& pair : state_dict) {
        const std::string& name = pair.first;
        const auto& tensor = pair.second;

        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(name.data(), name_len);

        int32_t dt = static_cast<int32_t>(tensor->dtype);
        out.write(reinterpret_cast<const char*>(&dt), sizeof(dt));

        uint32_t ndim = static_cast<uint32_t>(tensor->ndim());
        out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (int64_t d : tensor->shape) {
            out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }

        auto contig_cpu = make_contiguous(tensor)->cpu();
        uint64_t bytes = static_cast<uint64_t>(tensor->size()) * dtype_size(tensor->dtype);
        out.write(contig_cpu->data_ptr<char>(), bytes);
    }
    out.close();
}

inline std::unordered_map<std::string, std::shared_ptr<Tensor>> load_checkpoint(
    const std::string& filepath,
    Device device = Device::CPU
) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("load_checkpoint: Unable to open file: " + filepath);
    }

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x44554D47) {
        throw std::runtime_error("load_checkpoint: Invalid magic header in " + filepath);
    }

    uint32_t num_tensors = 0;
    in.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        int32_t dt_raw = 0;
        in.read(reinterpret_cast<char*>(&dt_raw), sizeof(dt_raw));
        if (!in || dt_raw < 0 || dt_raw > static_cast<int32_t>(DType::Int8)) {
            throw std::runtime_error("load_checkpoint: corrupt tensor header in " + filepath);
        }
        DType dt = static_cast<DType>(dt_raw);

        uint32_t ndim = 0;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        if (!in || ndim > 64) throw std::runtime_error("load_checkpoint: corrupt tensor header in " + filepath);
        std::vector<int64_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            in.read(reinterpret_cast<char*>(&shape[d]), sizeof(shape[d]));
        }
        if (!in || checked_numel(shape) < 0) throw std::runtime_error("load_checkpoint: corrupt tensor shape in " + filepath);

        auto tensor = std::make_shared<Tensor>(shape, Device::CPU, dt, false);
        uint64_t bytes = static_cast<uint64_t>(tensor->size()) * dtype_size(dt);
        in.read(tensor->data_ptr<char>(), bytes);
        if (!in) throw std::runtime_error("load_checkpoint: " + filepath + " is truncated");

        if (device != Device::CPU) {
            state_dict[name] = tensor->to(device);
        } else {
            state_dict[name] = tensor;
        }
    }
    in.close();
    return state_dict;
}

} // namespace io
