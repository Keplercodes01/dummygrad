#include "staging.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#endif

namespace staging {

#ifdef USE_CUDA

namespace {

struct Copy {
    cudaEvent_t done;
    int device;
    std::shared_ptr<void> memory;
    size_t bytes;
};

struct State {
    std::mutex mutex;
    std::unordered_map<size_t, std::vector<void*>> unused;             // blocks by size, a power of two
    std::unordered_map<int, std::vector<cudaEvent_t>> unused_events;   // by device
    std::deque<Copy> queued;
    size_t queued_bytes = 0;
};

// Never destroyed: blocks may be released while the process exits
State& state() {
    static State* s = new State;
    return *s;
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Takes finished copies off the queue; with too many bytes queued, waits for the oldest ones. Their memory goes back to
// the pool when the returned list dies, which must be outside the lock.
std::vector<Copy> finished(State& s) {
    constexpr size_t kMostQueued = (size_t)256 << 20;
    std::vector<Copy> done;
    while (!s.queued.empty()) {
        Copy& oldest = s.queued.front();
        if (s.queued_bytes > kMostQueued) cudaEventSynchronize(oldest.done);
        else if (cudaEventQuery(oldest.done) == cudaErrorNotReady) break;
        s.unused_events[oldest.device].push_back(oldest.done);
        s.queued_bytes -= oldest.bytes;
        done.push_back(std::move(oldest));
        s.queued.pop_front();
    }
    return done;
}

}  // namespace

std::shared_ptr<void> allocate(size_t bytes) {
    State& s = state();
    {
        std::vector<Copy> done;
        std::lock_guard<std::mutex> lock(s.mutex);
        done = finished(s);
        // done is declared before the lock, so it is destroyed after the lock is released
    }
    size_t size = 4096;
    while (size < bytes) size *= 2;
    void* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto& blocks = s.unused[size];
        if (!blocks.empty()) {
            ptr = blocks.back();
            blocks.pop_back();
        }
    }
    if (!ptr && cudaMallocHost(&ptr, size) != cudaSuccess) return nullptr;
    return std::shared_ptr<void>(ptr, [size](void* p) {
        State& st = state();
        std::lock_guard<std::mutex> lock(st.mutex);
        st.unused[size].push_back(p);
    });
}

void copy_to_gpu(void* dst, const std::shared_ptr<void>& src, size_t bytes) {
    State& s = state();
    int device = 0;
    check(cudaGetDevice(&device), "copy_to_gpu");
    cudaEvent_t event = nullptr;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto& events = s.unused_events[device];
        if (!events.empty()) {
            event = events.back();
            events.pop_back();
        }
    }
    if (!event) check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "copy_to_gpu");
    check(cudaMemcpyAsync(dst, src.get(), bytes, cudaMemcpyHostToDevice), "copy_to_gpu");
    check(cudaEventRecord(event), "copy_to_gpu");
    std::lock_guard<std::mutex> lock(s.mutex);
    s.queued.push_back({event, device, src, bytes});
    s.queued_bytes += bytes;
}

#else

std::shared_ptr<void> allocate(size_t) { return nullptr; }
void copy_to_gpu(void*, const std::shared_ptr<void>&, size_t) {}

#endif

}  // namespace staging
