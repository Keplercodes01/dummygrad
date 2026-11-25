#include "allocator.h"
#include "metal/metal_backend.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

void* CPUCachingAllocator::allocate(size_t bytes) {
    size_t alloc_size = round_up(std::max<size_t>(bytes, 1));  // an empty tensor still needs a pointer
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
    size_t alloc_size = round_up(std::max<size_t>(bytes, 1));
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

BlockPool& CUDACachingAllocator::pool(int device) {
    auto& p = pools[device];
    if (!p) {
#ifdef USE_CUDA
        p = std::make_unique<BlockPool>(
            [](size_t bytes) -> void* {
                void* ptr = nullptr;
                return cudaMalloc(&ptr, bytes) == cudaSuccess ? ptr : nullptr;
            },
            [](void* ptr) { cudaFree(ptr); });
#else
        p = std::make_unique<BlockPool>([](size_t) -> void* { return nullptr; }, [](void*) {});
#endif
    }
    return *p;
}

MemoryStats CUDACachingAllocator::stats_locked() const {
    MemoryStats s;
    for (const auto& p : pools) {
        s.in_use += p.second->in_use();
        s.reserved += p.second->reserved();
    }
    s.peak_in_use = std::max(peak_in_use, s.in_use);
    return s;
}

MemoryStats CUDACachingAllocator::memory_stats() {
    std::lock_guard<std::mutex> lock(mtx);
    return stats_locked();
}

void CUDACachingAllocator::reset_peak() {
    std::lock_guard<std::mutex> lock(mtx);
    peak_in_use = 0;
    peak_in_use = stats_locked().in_use;
}

void* CUDACachingAllocator::allocate(size_t bytes) {
#ifdef USE_CUDA
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) throw std::runtime_error("CUDACachingAllocator: no usable GPU");
    std::lock_guard<std::mutex> lock(mtx);
    auto& p = pool(dev);
    void* ptr = p.allocate(bytes);
    if (!ptr) {
        // Free blocks may be in segments too small for this request: give those back and try again
        cudaDeviceSynchronize();
        p.release_unused();
        ptr = p.allocate(bytes);
    }
    if (!ptr) {
        throw std::runtime_error("out of GPU memory: " + std::to_string(bytes >> 20) + " MB requested, " +
                                 std::to_string(p.in_use() >> 20) + " MB in use by tensors, " +
                                 std::to_string((p.reserved() - p.in_use()) >> 20) + " MB free in pieces");
    }
    size_t in_use = 0;
    for (const auto& each : pools) in_use += each.second->in_use();
    peak_in_use = std::max(peak_in_use, in_use);
    return ptr;
#else
    (void)bytes;
    throw std::runtime_error("CUDACachingAllocator: dummygrad was built without CUDA support");
#endif
}

void CUDACachingAllocator::free(void* ptr, size_t bytes) {
    (void)bytes;
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mtx);
    // Back to the GPU it was allocated on, whichever is current now
    for (auto& p : pools) if (p.second->free(ptr)) return;
}

void CUDACachingAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mtx);
#ifdef USE_CUDA
    // Kernels still queued may be reading blocks whose tensors are gone
    int current = 0;
    cudaGetDevice(&current);
    for (auto& p : pools) {
        cudaSetDevice(p.first);
        cudaDeviceSynchronize();
        p.second->release_unused();
    }
    cudaSetDevice(current);
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
