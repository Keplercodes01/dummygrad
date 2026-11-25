// GPU backend of the kernels in kernels.h: the same functors, launched as CUDA kernels
#include "kernels.h"
#include "kernel_math.h"
#include "allocator.h"
#include "cuda/cuda_common.h"

#ifdef USE_CUDA

#include <algorithm>

namespace kern {
namespace gpu {

namespace {

constexpr int kThreads = 256;

template <typename F>
__global__ void k_launch(F f, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) f(i);
}

template <typename F>
void launch(int64_t n, const F& f, int64_t /*cost*/) {
    if (n <= 0) return;
    int64_t blocks = std::min<int64_t>((n + kThreads - 1) / kThreads, 1 << 20);
    k_launch<F><<<(int)blocks, kThreads>>>(f, n);
    CUDA_CHECK(cudaGetLastError());
}

// Temporary device memory, returned to the caching allocator at scope exit (kernels run in order on
// one stream, so a later kernel cannot overwrite it before the ones launched here are done)
struct Scratch {
    void* ptr;
    size_t bytes;
    explicit Scratch(size_t n) : ptr(nullptr), bytes(n < 4 ? 4 : n) { ptr = get_memory(Device::CUDA, bytes); }
    ~Scratch() { free_memory(Device::CUDA, ptr, bytes); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    template <typename T> T* as() const { return static_cast<T*>(ptr); }
    void upload(const void* src, size_t n) {
        if (n > 0) CUDA_CHECK(cudaMemcpy(ptr, src, n, cudaMemcpyHostToDevice));
    }
};

}  // namespace

#include "kernels_impl.inc"

}  // namespace gpu
}  // namespace kern

#endif
