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

class CUDACachingAllocator : public Allocator {
private:
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    std::mutex mtx;
    size_t round_up(size_t bytes) { return (bytes + 511) & ~511; }
public:
    static CUDACachingAllocator& get() { static CUDACachingAllocator instance; return instance; }
    void* allocate(size_t bytes) override;
    void free(void* ptr, size_t bytes) override;
    void empty_cache() override;
};

// Zero-allocation static scratchpad arena for GPU activation buffers
class CUDAScratchpadArena {
private:
    void* base_ptr = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::mutex mtx;

public:
    static CUDAScratchpadArena& get() { static CUDAScratchpadArena instance; return instance; }
    void init(size_t total_bytes);
    void* allocate(size_t bytes);
    void reset();
    bool contains(void* ptr) const {
        if (!base_ptr || !ptr) return false;
        return ptr >= base_ptr && ptr < static_cast<const char*>(base_ptr) + capacity;
    }
    ~CUDAScratchpadArena();
};

void set_arena_active(bool active);
bool is_arena_active();

// RAII Scope: All intermediate tensors inside this scope use the zero-allocation arena
struct ArenaScope {
    explicit ArenaScope(size_t init_capacity_bytes = 0) {
        if (init_capacity_bytes > 0) {
            CUDAScratchpadArena::get().init(init_capacity_bytes);
        }
        set_arena_active(true);
    }
    ~ArenaScope() {
        set_arena_active(false);
        CUDAScratchpadArena::get().reset();
    }
};

void* get_memory(Device device, size_t bytes);
void free_memory(Device device, void* ptr, size_t bytes);
