#pragma once
#include <cstddef>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include "block_pool.h"
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
    // Never destroyed: tensors in static storage may be freed after any static allocator would be
    static CPUCachingAllocator& get() { static auto* instance = new CPUCachingAllocator; return *instance; }
    ~CPUCachingAllocator() override { empty_cache(); }
    void* allocate(size_t bytes) override;
    void free(void* ptr, size_t bytes) override;
    void empty_cache() override;
};

// GPU memory held by tensors and by the caching allocator, summed over devices, in bytes.
// in_use counts live blocks, like torch.cuda.memory_allocated; reserved adds the cached free ones, like
// torch.cuda.memory_reserved; peak_in_use is the highest in_use since the last reset_peak_memory_stats().
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
    // One pool per device: memory of one GPU must never be handed out for another
    std::unordered_map<int, std::unique_ptr<BlockPool>> pools;
    std::mutex mtx;
    size_t peak_in_use = 0;  // guarded by mtx
    BlockPool& pool(int device);
    MemoryStats stats_locked() const;
public:
    MemoryStats memory_stats();
    void reset_peak();
    static CUDACachingAllocator& get() { static auto* instance = new CUDACachingAllocator; return *instance; }
    ~CUDACachingAllocator() override { empty_cache(); }
    void* allocate(size_t bytes) override;
    void free(void* ptr, size_t bytes) override;
    // Returns memory no tensor uses to the driver
    void empty_cache() override;
};

void* get_memory(Device device, size_t bytes);
void free_memory(Device device, void* ptr, size_t bytes);
