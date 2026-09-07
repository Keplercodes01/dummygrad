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

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define HAS_MMAP 1
#else
#define HAS_MMAP 0
#endif

#include "tensor.h"
#include "types.h"

// Pure C++ Zero-Copy SafeTensors & Binary Serialization Engine
// Interoperable with Hugging Face SafeTensors checkpoints with zero Python dependencies
namespace io {

inline std::string dtype_to_safetensors_str(DType dt) {
    switch (dt) {
        case DType::Float32:  return "F32";
        case DType::Float16:  return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Int8:     return "I8";
        default:              return "F32";
    }
}

inline DType safetensors_str_to_dtype(const std::string& str) {
    if (str == "F32" || str == "float32") return DType::Float32;
    if (str == "F16" || str == "float16") return DType::Float16;
    if (str == "BF16" || str == "bfloat16") return DType::BFloat16;
    if (str == "I8" || str == "int8") return DType::Int8;
    return DType::Float32;
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

        uint64_t bytes = tensor->storage->total_bytes;
        uint64_t start = current_offset;
        uint64_t end = current_offset + bytes;
        current_offset = end;

        ordered_tensors.push_back({name, tensor, start, end});

        if (!first) json_ss << ",";
        first = false;

        json_ss << "\"" << name << "\":{"
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
        auto host_tensor = item.tensor->cpu();
        out.write(reinterpret_cast<const char*>(host_tensor->data_ptr<void>()), item.tensor->storage->total_bytes);
    }

    out.close();
    std::cout << "[dummygrad SafeTensors] Saved " << ordered_tensors.size() 
              << " tensor(s) (" << current_offset / (1024.0 * 1024.0) << " MB) to " << filepath << "\n";
}

// Minimal, zero-dependency parser for SafeTensors JSON header
struct SafeTensorEntry {
    std::string name;
    DType dtype = DType::Float32;
    std::vector<int64_t> shape;
    uint64_t offset_start = 0;
    uint64_t offset_end = 0;
};

inline std::vector<SafeTensorEntry> parse_safetensors_header(const std::string& json) {
    std::vector<SafeTensorEntry> entries;
    size_t pos = 0;

    while (pos < json.size()) {
        // Find next key name
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;

        std::string key = json.substr(key_start + 1, key_end - key_start - 1);
        pos = key_end + 1;

        if (key == "__metadata__") {
            continue; // Skip optional HuggingFace metadata dict
        }

        // Find entry dict
        size_t dict_start = json.find('{', pos);
        size_t dict_end = json.find('}', dict_start);
        if (dict_start == std::string::npos || dict_end == std::string::npos) break;

        std::string entry_str = json.substr(dict_start, dict_end - dict_start + 1);
        pos = dict_end + 1;

        SafeTensorEntry entry;
        entry.name = key;

        // Parse dtype
        size_t dt_pos = entry_str.find("\"dtype\":");
        if (dt_pos != std::string::npos) {
            size_t v_start = entry_str.find('"', dt_pos + 8);
            size_t v_end = entry_str.find('"', v_start + 1);
            if (v_start != std::string::npos && v_end != std::string::npos) {
                entry.dtype = safetensors_str_to_dtype(entry_str.substr(v_start + 1, v_end - v_start - 1));
            }
        }

        // Parse shape array
        size_t sh_pos = entry_str.find("\"shape\":[");
        if (sh_pos != std::string::npos) {
            size_t s_end = entry_str.find(']', sh_pos);
            std::string s_str = entry_str.substr(sh_pos + 9, s_end - sh_pos - 9);
            std::stringstream s_stream(s_str);
            std::string num_str;
            while (std::getline(s_stream, num_str, ',')) {
                if (!num_str.empty()) {
                    entry.shape.push_back(std::stoll(num_str));
                }
            }
        }

        // Parse data_offsets
        size_t off_pos = entry_str.find("\"data_offsets\":[");
        if (off_pos != std::string::npos) {
            size_t off_end = entry_str.find(']', off_pos);
            std::string off_str = entry_str.substr(off_pos + 16, off_end - off_pos - 16);
            size_t comma = off_str.find(',');
            if (comma != std::string::npos) {
                entry.offset_start = std::stoull(off_str.substr(0, comma));
                entry.offset_end = std::stoull(off_str.substr(comma + 1));
            }
        }

        if (!entry.name.empty() && !entry.shape.empty()) {
            entries.push_back(entry);
        }
    }

    return entries;
}

// Load tensors from a SafeTensors file using fast zero-copy memory mapping
inline std::unordered_map<std::string, std::shared_ptr<Tensor>> load_safetensors(
    const std::string& filepath,
    Device device = Device::CPU
) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> result;

#if HAS_MMAP
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("load_safetensors: Unable to open file: " + filepath);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("load_safetensors: Failed to stat file: " + filepath);
    }
    size_t file_size = sb.st_size;

    void* mmapped_data = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmapped_data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("load_safetensors: mmap failed for: " + filepath);
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(mmapped_data);
    uint64_t header_len = *reinterpret_cast<const uint64_t*>(ptr);

    if (8 + header_len > file_size) {
        munmap(mmapped_data, file_size);
        close(fd);
        throw std::runtime_error("load_safetensors: Corrupt file header size");
    }

    std::string json_header(reinterpret_cast<const char*>(ptr + 8), header_len);
    auto entries = parse_safetensors_header(json_header);
    const uint8_t* payload_base = ptr + 8 + header_len;

    for (const auto& entry : entries) {
        auto tensor = std::make_shared<Tensor>(entry.shape, Device::CPU, entry.dtype, false);
        const uint8_t* src = payload_base + entry.offset_start;
        size_t bytes = entry.offset_end - entry.offset_start;
        std::memcpy(tensor->data_ptr<void>(), src, bytes);

        if (device != Device::CPU) {
            result[entry.name] = tensor->to(device);
        } else {
            result[entry.name] = tensor;
        }
    }

    munmap(mmapped_data, file_size);
    close(fd);
#else
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("load_safetensors: Unable to open file: " + filepath);
    }

    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));

    std::string json_header(header_len, '\0');
    in.read(&json_header[0], header_len);

    auto entries = parse_safetensors_header(json_header);
    uint64_t payload_base = 8 + header_len;

    for (const auto& entry : entries) {
        auto tensor = std::make_shared<Tensor>(entry.shape, Device::CPU, entry.dtype, false);
        in.seekg(payload_base + entry.offset_start);
        size_t bytes = entry.offset_end - entry.offset_start;
        in.read(reinterpret_cast<char*>(tensor->data_ptr<void>()), bytes);

        if (device != Device::CPU) {
            result[entry.name] = tensor->to(device);
        } else {
            result[entry.name] = tensor;
        }
    }
    in.close();
#endif

    std::cout << "[dummygrad SafeTensors] Successfully loaded " << result.size() 
              << " tensor(s) from " << filepath << "\n";
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

        auto cpu_tensor = tensor->cpu();
        out.write(reinterpret_cast<const char*>(cpu_tensor->data_ptr<void>()), tensor->storage->total_bytes);
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
        DType dt = static_cast<DType>(dt_raw);

        uint32_t ndim = 0;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int64_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            in.read(reinterpret_cast<char*>(&shape[d]), sizeof(shape[d]));
        }

        auto tensor = std::make_shared<Tensor>(shape, Device::CPU, dt, false);
        in.read(reinterpret_cast<char*>(tensor->data_ptr<void>()), tensor->storage->total_bytes);

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
