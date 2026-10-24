#include "storage.h"
#include "allocator.h"
#include <stdexcept>

Storage::Storage(size_t total_elements, Device d, DType type) 
    : device(d), dtype(type) {
    total_bytes = total_elements * dtype_size(type);
    data = get_memory(device, total_bytes);
}

Storage::~Storage() {
    if(data) free_memory(device, data, total_bytes);
}

Storage::Storage(Storage&& other) noexcept
    : data(other.data), device(other.device), dtype(other.dtype),
      total_bytes(other.total_bytes), tpu_handle(std::move(other.tpu_handle)) {
    other.data = nullptr;
    other.total_bytes = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (data) free_memory(device, data, total_bytes);
        data = other.data;
        device = other.device;
        dtype = other.dtype;
        total_bytes = other.total_bytes;
        tpu_handle = std::move(other.tpu_handle);
        other.data = nullptr;
        other.total_bytes = 0;
    }
    return *this;
}
