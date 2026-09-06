#include "allocator.h"
#include <cstdlib>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void* CPUCachingAllocator::allocate(size_t bytes) {
    size_t alloc_size = round_up(bytes);
    std::lock_guard<std::mutex> lock(mtx);
    
    auto& blocks = free_blocks[alloc_size];
    if (!blocks.empty()) {
        void* ptr = blocks.back();
        blocks.pop_back(); // Remove from pool and return to engine
        return ptr;
    }
    
    // Cache miss: we actually have to hit the OS for memory
    void* ptr = std::malloc(alloc_size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void CPUCachingAllocator::free(void* ptr, size_t bytes) {
    if (!ptr) return;
    size_t alloc_size = round_up(bytes);
    std::lock_guard<std::mutex> lock(mtx);
    
    // Instead of OS free(), we put it in the cache for instant reuse
    free_blocks[alloc_size].push_back(ptr);
}

void CPUCachingAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto& pair : free_blocks) {
        for (void* ptr : pair.second) {
            std::free(ptr); // Actual OS free
        }
    }
    free_blocks.clear();
}

void* CUDACachingAllocator::allocate(size_t bytes) {
#ifdef USE_CUDA
    size_t alloc_size = round_up(bytes);
    std::lock_guard<std::mutex> lock(mtx);

    auto& blocks = free_blocks[alloc_size];
    if (!blocks.empty()) {
        void* ptr = blocks.back();
        blocks.pop_back();
        return ptr;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, alloc_size);
    if (err != cudaSuccess || !ptr) {
        throw std::bad_alloc();
    }
    return ptr;
#else
    (void)bytes;
    throw std::runtime_error("CUDACachingAllocator: dummygrad was built without CUDA support");
#endif
}

void CUDACachingAllocator::free(void* ptr, size_t bytes) {
    if (!ptr) return;
#ifdef USE_CUDA
    size_t alloc_size = round_up(bytes);
    std::lock_guard<std::mutex> lock(mtx);
    free_blocks[alloc_size].push_back(ptr);
#else
    (void)bytes;
#endif
}

void CUDACachingAllocator::empty_cache() {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(mtx);
    for (auto& pair : free_blocks) {
        for (void* ptr : pair.second) {
            cudaFree(ptr);
        }
    }
    free_blocks.clear();
#endif
}

static thread_local bool g_arena_active = false;

void set_arena_active(bool active) { g_arena_active = active; }
bool is_arena_active() { return g_arena_active; }

// Device dispatcher for memory requests
void* get_memory(Device device, size_t bytes) {
    if (device == Device::CPU || device == Device::TPU) {
        return CPUCachingAllocator::get().allocate(bytes);
    }
    if (device == Device::CUDA) {
        if (is_arena_active()) {
            return CUDAScratchpadArena::get().allocate(bytes);
        }
        return CUDACachingAllocator::get().allocate(bytes);
    }
    throw std::runtime_error("Allocator: Unsupported device");
}

void free_memory(Device device, void* ptr, size_t bytes) {
    if (device == Device::CPU || device == Device::TPU) {
        CPUCachingAllocator::get().free(ptr, bytes);
    } else if (device == Device::CUDA) {
        if (CUDAScratchpadArena::get().contains(ptr)) {
            return; // Managed by static scratchpad arena: instant zero-cost reset at step boundary!
        }
        CUDACachingAllocator::get().free(ptr, bytes);
    }
}

// -------------------------------------------------------------
// CUDAScratchpadArena Implementation
// -------------------------------------------------------------
void CUDAScratchpadArena::init(size_t total_bytes) {
    std::lock_guard<std::mutex> lock(mtx);
#ifdef USE_CUDA
    if (base_ptr) {
        cudaFree(base_ptr);
        base_ptr = nullptr;
    }
    capacity = (total_bytes + 511) & ~511;
    offset = 0;
    cudaError_t err = cudaMalloc(&base_ptr, capacity);
    if (err != cudaSuccess || !base_ptr) {
        throw std::runtime_error("CUDAScratchpadArena: failed to allocate " + std::to_string(capacity) + " bytes on GPU");
    }
#else
    (void)total_bytes;
#endif
}

void* CUDAScratchpadArena::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mtx);
    size_t aligned = (bytes + 511) & ~511;
    if (offset + aligned > capacity) {
        throw std::runtime_error("CUDAScratchpadArena out of memory: requested " + std::to_string(aligned) +
                                 ", remaining " + std::to_string(capacity - offset));
    }
    void* res = static_cast<char*>(base_ptr) + offset;
    offset += aligned;
    return res;
}

void CUDAScratchpadArena::reset() {
    std::lock_guard<std::mutex> lock(mtx);
    offset = 0;
}

CUDAScratchpadArena::~CUDAScratchpadArena() {
#ifdef USE_CUDA
    if (base_ptr) {
        cudaFree(base_ptr);
        base_ptr = nullptr;
    }
#endif
}
