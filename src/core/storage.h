#pragma once
#include "types.h"
#include <cstddef>
#include <memory>

struct TPUBufferHandle;

struct Storage {
    void* data = nullptr;
    Device device;
    DType dtype;
    size_t total_bytes;
    std::shared_ptr<TPUBufferHandle> tpu_handle = nullptr;

    explicit Storage(size_t total_elements, Device d, DType type = DType::Float32);
    ~Storage();
};
