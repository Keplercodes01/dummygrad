#pragma once
// Page-locked host memory for copies to the GPU. Such a copy is queued behind the kernels before it and the program
// goes on at once; a copy from ordinary memory makes it wait until the GPU has caught up.

#include <cstddef>
#include <memory>

namespace staging {

// bytes of page-locked memory, reused once released; null without CUDA or when there is none to be had
std::shared_ptr<void> allocate(size_t bytes);

// Queues a copy of src to the current GPU and keeps src alive until it has run
void copy_to_gpu(void* dst, const std::shared_ptr<void>& src, size_t bytes);

}  // namespace staging
