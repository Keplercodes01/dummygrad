#include "allocator.h"
#include <cstdlib>
#include <stdexcept>

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

// Device dispatcher for memory requests
void* get_memory(Device device, size_t bytes) {
    if (device == Device::CPU) {
        return CPUCachingAllocator::get().allocate(bytes);
    }
    throw std::runtime_error("Allocator: Device not implemented yet");
}

void free_memory(Device device, void* ptr, size_t bytes) {
    if (device == Device::CPU) {
        CPUCachingAllocator::get().free(ptr, bytes);
    }
}
