#pragma once
// A file's bytes, read-only. Memory-mapped where the system can, so opening a 100 GB file is instant and only
// the parts used are read; read into memory elsewhere.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define DUMMYGRAD_HAS_MMAP 1
#else
#define DUMMYGRAD_HAS_MMAP 0
#endif

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
#if DUMMYGRAD_HAS_MMAP
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open file: " + path);
        struct stat st;
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
            ::close(fd);
            throw std::runtime_error("not a regular file: " + path);
        }
        size_ = (size_t)st.st_size;
        void* p = size_ > 0 ? ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0) : nullptr;
        ::close(fd);  // the mapping keeps the file open
        if (p == MAP_FAILED) throw std::runtime_error("cannot map file: " + path);
        data_ = static_cast<const uint8_t*>(p);
#else
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open file: " + path);
        bytes_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        size_ = bytes_.size();
        data_ = reinterpret_cast<const uint8_t*>(bytes_.data());
#endif
    }
    ~MappedFile() {
#if DUMMYGRAD_HAS_MMAP
        if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
#endif
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#if !DUMMYGRAD_HAS_MMAP
    std::vector<char> bytes_;
#endif
};
