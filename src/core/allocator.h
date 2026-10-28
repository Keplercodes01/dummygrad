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
    ~CPUCachingAllocator() override { empty_cache(); }
    void* allocate(size_t bytes) override;
    void free(void* ptr, size_t bytes) override;
    void empty_cache() override;
};

// GPU memory held by tensors and by the caching allocator, summed over devices, in bytes.
// in_use counts live blocks (rounded to 512 bytes), like torch.cuda.memory_allocated; reserved
// adds the cached free blocks, like torch.cuda.memory_reserved; peak_in_use is the highest
// in_use since the last reset_peak_memory_stats().
#define DUMMYGRAD_HAS_MEMORY_STATS 1  // lets tools built against several versions (bench.cpp) check
struct MemoryStats {
    size_t in_use = 0;
    size_t peak_in_use = 0;
    size_t reserved = 0;
};
MemoryStats cuda_memory_stats();
void reset_peak_memory_stats();

class CUDACachingAllocator : public Allocator {
private:
    // Pools are per device: a block cudaMalloc'd on one GPU must never be reused on another
    std::unordered_map<int, std::unordered_map<size_t, std::vector<void*>>> free_blocks;
    std::unordered_map<void*, int> ptr_device;
    std::mutex mtx;
    MemoryStats stats;  // guarded by mtx
    size_t round_up(size_t bytes) { return (bytes + 511) & ~511; }
public:
    MemoryStats memory_stats() {
        std::lock_guard<std::mutex> lock(mtx);
        return stats;
    }
    void reset_peak() {
        std::lock_guard<std::mutex> lock(mtx);
        stats.peak_in_use = stats.in_use;
    }
    static CUDACachingAllocator& get() { static CUDACachingAllocator instance; return instance; }
    ~CUDACachingAllocator() override { empty_cache(); }
    void* allocate(size_t bytes) override;
    void free(void* ptr, size_t bytes) override;
    void empty_cache() override;
};

void* get_memory(Device device, size_t bytes);
void free_memory(Device device, void* ptr, size_t bytes);
