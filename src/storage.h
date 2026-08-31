#pragma once
#include "types.h"
#include <cstddef>

struct Storage {
    void* cpu_data = nullptr;
    void* cuda_data = nullptr;
    Device device;
    DType dtype;
    size_t total_bytes;

    explicit Storage(size_t total_elements, Device d, DType type = DType::Float32);
    ~Storage();
};
