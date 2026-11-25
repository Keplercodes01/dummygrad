#include "mapped_file.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

MappedFile::MappedFile(const std::string& path) {
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
}

MappedFile::~MappedFile() {
    if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
}

#else

MappedFile::MappedFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    bytes_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    size_ = bytes_.size();
    data_ = reinterpret_cast<const uint8_t*>(bytes_.data());
}

MappedFile::~MappedFile() = default;

#endif
