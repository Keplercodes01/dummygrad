#pragma once
// A file's bytes, read-only. Memory-mapped where the system can, so opening a 100 GB file is instant and only
// the parts used are read; read into memory elsewhere.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    std::vector<char> bytes_;  // without memory mapping
};
