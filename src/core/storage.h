#pragma once
#include "types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

struct TPUBufferHandle;

struct Storage {
    void* data = nullptr;
    Device device;
    DType dtype;
    size_t total_bytes;
    std::shared_ptr<TPUBufferHandle> tpu_handle = nullptr;

    // Counts in-place writes made through ops (fill, add_inplace, copies) and optimizer steps, so
    // copies derived from this memory know when they are stale. Writes through a raw data_ptr do
    // not count: call bump_version() after one if a cached copy may exist.
    uint64_t version = 0;
    void bump_version() { ++version; }

    // A copy of this storage's contents in another dtype, valid while cast_version == version.
    // Autocast keeps a parameter's half-precision copy here instead of converting it every forward.
    std::shared_ptr<Storage> cast_copy;
    DType cast_dtype = DType::Float32;
    uint64_t cast_version = 0;
    std::mutex cast_mutex;

    explicit Storage(size_t total_elements, Device d, DType type = DType::Float32);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;
};
