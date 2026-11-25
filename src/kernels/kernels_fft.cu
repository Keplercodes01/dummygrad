// Fourier transforms on the GPU through cuFFT
#include "kernels.h"
#include "allocator.h"
#include "cuda/cuda_common.h"

#if defined(USE_CUDA) && defined(DUMMYGRAD_CUFFT)

#include <cufft.h>

#include <list>
#include <mutex>
#include <string>

namespace kern {
namespace gpu {

namespace {

void check(cufftResult r) {
    if (r != CUFFT_SUCCESS) throw std::runtime_error("cuFFT error " + std::to_string((int)r));
}

struct Plan {
    int device;
    int n, rows;
    cufftHandle handle;
    size_t work_bytes;
};

// The plans used last. Planning takes far longer than a transform.
std::list<Plan>& plans() {
    static auto* p = new std::list<Plan>;
    return *p;
}
std::mutex& plans_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

Plan& find_plan(int n, int rows) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    auto& all = plans();
    for (auto it = all.begin(); it != all.end(); ++it) {
        if (it->device == device && it->n == n && it->rows == rows) {
            all.splice(all.begin(), all, it);
            return all.front();
        }
    }
    Plan p{device, n, rows, 0, 0};
    check(cufftCreate(&p.handle));
    // The work area comes from the caching allocator, at each transform
    check(cufftSetAutoAllocation(p.handle, 0));
    check(cufftMakePlanMany(p.handle, 1, &p.n, nullptr, 1, n, nullptr, 1, n, CUFFT_C2C, rows, &p.work_bytes));
    all.push_front(p);
    if (all.size() > 16) {
        cufftDestroy(all.back().handle);
        all.pop_back();
    }
    return all.front();
}

}  // namespace

void fft(const float* x, float* out, int64_t rows, int64_t n, bool inverse) {
    if (rows == 0 || n == 0) return;
    if (rows > INT32_MAX || n > INT32_MAX) throw std::invalid_argument("fft: too large for one transform on the GPU");
    std::lock_guard<std::mutex> lock(plans_mutex());
    Plan& p = find_plan((int)n, (int)rows);
    size_t bytes = p.work_bytes < 4 ? 4 : p.work_bytes;
    void* work = get_memory(Device::CUDA, bytes);
    cufftResult r = cufftSetWorkArea(p.handle, work);
    if (r == CUFFT_SUCCESS) {
        r = cufftExecC2C(p.handle, reinterpret_cast<cufftComplex*>(const_cast<float*>(x)), reinterpret_cast<cufftComplex*>(out),
                         inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    }
    // Queued work still uses the area, but anything reusing it runs after, on the same stream
    free_memory(Device::CUDA, work, bytes);
    check(r);
}

}  // namespace gpu
}  // namespace kern

#endif
