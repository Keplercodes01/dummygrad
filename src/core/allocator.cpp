#include "allocator.h"
#include "metal/metal_backend.h"
#include <algorithm>
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
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mtx);

    auto& blocks = free_blocks[dev][alloc_size];
    void* ptr = nullptr;
    if (!blocks.empty()) {
        ptr = blocks.back();
        blocks.pop_back();
    } else {
        cudaError_t err = cudaMalloc(&ptr, alloc_size);
        if (err != cudaSuccess || !ptr) {
            throw std::bad_alloc();
        }
        ptr_device[ptr] = dev;
        stats.reserved += alloc_size;
    }
    stats.in_use += alloc_size;
    stats.peak_in_use = std::max(stats.peak_in_use, stats.in_use);
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
    // Return the block to the pool of the GPU it was allocated on, not the caller's current device
    auto it = ptr_device.find(ptr);
    int dev = 0;
    if (it != ptr_device.end()) {
        dev = it->second;
    } else {
        cudaGetDevice(&dev);
        ptr_device[ptr] = dev;
    }
    free_blocks[dev][alloc_size].push_back(ptr);
    stats.in_use -= std::min(stats.in_use, alloc_size);
#else
    (void)bytes;
#endif
}

void CUDACachingAllocator::empty_cache() {
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(mtx);
    for (auto& dev_pool : free_blocks) {
        for (auto& pair : dev_pool.second) {
            for (void* ptr : pair.second) {
                cudaFree(ptr);
                ptr_device.erase(ptr);
                stats.reserved -= std::min(stats.reserved, pair.first);
            }
        }
    }
    free_blocks.clear();
#endif
}

MemoryStats cuda_memory_stats() { return CUDACachingAllocator::get().memory_stats(); }
void reset_peak_memory_stats() { CUDACachingAllocator::get().reset_peak(); }

// Device dispatcher for memory requests
void* get_memory(Device device, size_t bytes) {
    if (device == Device::CPU || device == Device::TPU) {
        return CPUCachingAllocator::get().allocate(bytes);
    }
    if (device == Device::CUDA) {
        return CUDACachingAllocator::get().allocate(bytes);
    }
    if (device == Device::MPS) {
        void* ptr = MetalBackend::get().allocate_shared_buffer(bytes);
        if (ptr) return ptr;
        return CPUCachingAllocator::get().allocate(bytes);
    }
    throw std::runtime_error("Allocator: Unsupported device");
}

void free_memory(Device device, void* ptr, size_t bytes) {
    if (device == Device::CPU || device == Device::TPU) {
        CPUCachingAllocator::get().free(ptr, bytes);
    } else if (device == Device::CUDA) {
        CUDACachingAllocator::get().free(ptr, bytes);
    } else if (device == Device::MPS) {
        if (!MetalBackend::get().is_available()) {
            CPUCachingAllocator::get().free(ptr, bytes);
        } else {
            MetalBackend::get().free_buffer(ptr, bytes);
        }
    }
}
