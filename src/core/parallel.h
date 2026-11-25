#pragma once
// CPU data parallelism. parallel_for splits a range over a persistent thread pool:
//
//   parallel_for(0, n, 4096, [&](int64_t begin, int64_t end) {
//       for (int64_t i = begin; i < end; ++i) out[i] = f(in[i]);
//   });
//
// grain is the smallest piece worth handing to another thread; ranges no longer than it run on the
// calling thread. Nested calls, and calls made while another thread's parallel region is running, run
// serially on the caller, so kernels may be called from anywhere (backward worker threads, replicas).
//
// Where the range is cut depends on the thread count, so a body must not let a floating-point result
// depend on where its piece begins. parallel_sum below reduces deterministically.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

// Threads kernels may use, including the caller. Defaults to the DUMMYGRAD_NUM_THREADS environment
// variable, else the hardware's core count.
int get_num_threads();
void set_num_threads(int n);

namespace parallel_detail {
// Runs body(begin, end) over [begin, end) cut into pieces of at least grain
void run(int64_t begin, int64_t end, int64_t grain, const std::function<void(int64_t, int64_t)>& body);
bool in_parallel_region();
}  // namespace parallel_detail

template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain, const F& body) {
    if (end <= begin) return;
    if (end - begin <= grain || get_num_threads() == 1 || parallel_detail::in_parallel_region()) {
        body(begin, end);
        return;
    }
    parallel_detail::run(begin, end, grain, std::function<void(int64_t, int64_t)>(std::cref(body)));
}

// Sum of chunk_sum(begin, end) over fixed 16K-element chunks, added in chunk order: the result does
// not depend on the thread count
template <typename F>
inline double parallel_sum(int64_t n, const F& chunk_sum) {
    const int64_t chunk = 16384;
    int64_t chunks = (n + chunk - 1) / chunk;
    if (chunks <= 1) return n > 0 ? (double)chunk_sum((int64_t)0, n) : 0.0;
    std::vector<double> partial((size_t)chunks);
    parallel_for(0, chunks, 4, [&](int64_t b, int64_t e) {
        for (int64_t c = b; c < e; ++c) partial[(size_t)c] = (double)chunk_sum(c * chunk, std::min(n, (c + 1) * chunk));
    });
    double total = 0.0;
    for (double p : partial) total += p;
    return total;
}
