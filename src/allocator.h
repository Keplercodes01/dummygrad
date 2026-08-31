#pragma once
#include <cstddef>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "types.h"

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void free(void* ptr, size_t bytes) = 0;
    virtual void empty_cache() = 0;
};

class CPUCachingAllocator : public Allocator {
private:
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    std::mutex mtx;
    size_t round_up(size_t bytes) { return (bytes + 63) & ~63; }
public:
    static CPUCachingAllocator& get() { static CPUCachingAllocator instance; return instance; }
    void* allocate(size_t bytes) override;
    void free(void* ptr, size_t bytes) override;
    void empty_cache() override;
};

void* get_memory(Device device, size_t bytes);
void free_memory(Device device, void* ptr, size_t bytes);
