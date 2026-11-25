// GPU kernels built on sorting: order along a dimension, and deterministic scatters, which bring the writes to each
// element together by sorting instead of racing with atomics
#include "kernels.h"
#include "allocator.h"
#include "cuda/cuda_common.h"

#ifdef USE_CUDA

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <string>

namespace kern {
namespace gpu {

namespace {

constexpr int kThreads = 256;

template <typename F>
__global__ void k_each(F f, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) f(i);
}
template <typename F>
void each(int64_t n, const F& f) {
    if (n <= 0) return;
    int64_t blocks = std::min<int64_t>((n + kThreads - 1) / kThreads, 1 << 20);
    k_each<F><<<(int)blocks, kThreads>>>(f, n);
    CUDA_CHECK(cudaGetLastError());
}

struct Memory {
    void* ptr;
    size_t bytes;
    explicit Memory(size_t n) : ptr(nullptr), bytes(n < 4 ? 4 : n) { ptr = get_memory(Device::CUDA, bytes); }
    ~Memory() { free_memory(Device::CUDA, ptr, bytes); }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <typename T> T* as() const { return static_cast<T*>(ptr); }
};

// Bits needed to count to n - 1 (at least 1)
int bits_for(int64_t n) {
    int b = 1;
    while (b < 63 && ((int64_t)1 << b) < n) ++b;
    return b;
}

// n (key, value) pairs and room to sort them by the low bits of the keys; equal keys keep their order
template <typename K>
struct Pairs {
    int64_t n;
    Memory k0, k1, v0, v1;
    K* keys;
    int* values;
    explicit Pairs(int64_t n)
        : n(n), k0((size_t)n * sizeof(K)), k1((size_t)n * sizeof(K)), v0((size_t)n * sizeof(int)), v1((size_t)n * sizeof(int)),
          keys(k0.template as<K>()), values(v0.template as<int>()) {
        if (n > INT32_MAX) throw std::invalid_argument("sorting more than 2^31 elements at once is not supported");
    }
    void sort(int bits) {
        cub::DoubleBuffer<K> dk(k0.template as<K>(), k1.template as<K>());
        cub::DoubleBuffer<int> dv(v0.template as<int>(), v1.template as<int>());
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, dk, dv, (int)n, 0, bits));
        Memory temp(bytes);
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp.ptr, bytes, dk, dv, (int)n, 0, bits));
        keys = dk.Current();
        values = dv.Current();
    }
};

// Unsigned integers in the order of the floats: -0 equals 0, NaN is the largest
__device__ __forceinline__ uint32_t order_key(float f, bool descending) {
    if (f == 0.0f) f = 0.0f;
    uint32_t u = __float_as_uint(f);
    uint32_t key = f != f ? 0xffffffffu : (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return descending ? ~key : key;
}

template <typename K>
struct FillOrder {
    const float* x;
    K* keys;
    int* values;
    int64_t size, inner;
    bool descending;
    __device__ void operator()(int64_t t) const {
        int64_t line = t / size, j = t - line * size;
        int64_t o = line / inner, i = line - o * inner;
        keys[t] = (K)(((uint64_t)line << 32) | order_key(x[(o * size + j) * inner + i], descending));
        values[t] = (int)j;
    }
};
struct TakeOrder {
    const int* values;
    float* index;
    int64_t size, inner, k;
    __device__ void operator()(int64_t t) const {  // t counts (line, rank)
        int64_t line = t / k, r = t - line * k;
        int64_t o = line / inner, i = line - o * inner;
        index[(o * k + r) * inner + i] = (float)values[line * size + r];
    }
};

template <typename K>
void sort_lines(const float* x, float* index, int64_t lines, int64_t size, int64_t inner, int64_t k, bool descending, int bits) {
    Pairs<K> p(lines * size);
    each(p.n, FillOrder<K>{x, p.keys, p.values, size, inner, descending});
    p.sort(bits);
    each(lines * k, TakeOrder{p.values, index, size, inner, k});
}

struct FillIndex {
    const float* index;
    uint32_t* keys;
    int* values;
    __device__ void operator()(int64_t m) const {
        keys[m] = (uint32_t)index[m];
        values[m] = (int)m;
    }
};

// First position of keys[0 .. n) whose key is not below k
__device__ __forceinline__ int64_t lower_bound(const uint32_t* keys, int64_t n, uint32_t k) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (keys[mid] < k) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

struct FillStarts {
    const uint32_t* keys;
    int64_t n;
    int* starts;
    __device__ void operator()(int64_t k) const { starts[k] = (int)lower_bound(keys, n, (uint32_t)k); }
};
struct AddRows {
    float* out;
    const float* src;
    const int *positions, *starts;
    int64_t K, inner, M;
    __device__ void operator()(int64_t t) const {  // an element of out
        int64_t row = t / inner, i = t - row * inner;
        int64_t o = row / K, k = row - o * K;
        int b = starts[k], e = starts[k + 1];
        if (b == e) return;
        float s = 0.0f;
        for (int p = b; p < e; ++p) s += src[(o * M + positions[p]) * inner + i];
        out[t] += s;
    }
};

// Where gather's output element e reads x
struct FillTargets {
    GatherPlan g;
    const float* index;
    uint32_t* keys;
    int* values;
    __device__ void operator()(int64_t e) const {
        int64_t rest = e, off = (int64_t)index[e] * g.x_stride[g.dim];
        for (int d = g.ndim - 1; d >= 0; --d) {
            int64_t c = rest % g.shape[d];
            rest /= g.shape[d];
            if (d != g.dim) off += c * g.x_stride[d];
        }
        keys[e] = (uint32_t)off;
        values[e] = (int)e;
    }
};
struct LastWins {
    const uint32_t* keys;
    const int* values;
    const float* src;
    float* out;
    int64_t n;
    __device__ void operator()(int64_t p) const {
        if (p + 1 < n && keys[p + 1] == keys[p]) return;
        out[keys[p]] = src[values[p]];
    }
};
// Sums of a run of equal keys: 64 at a time, then those sums, so that long runs are parallel too and every sum
// is always taken in the same order
constexpr int64_t kRunBlock = 64;
struct RunBlocks {
    const uint32_t* keys;
    const int* values;
    const float* src;
    float* sums;
    int64_t n;
    __device__ void operator()(int64_t p) const {
        uint32_t k = keys[p];
        if ((p - lower_bound(keys, n, k)) % kRunBlock) return;
        float s = 0.0f;
        for (int64_t q = p; q < n && q < p + kRunBlock && keys[q] == k; ++q) s += src[values[q]];
        sums[p] = s;
    }
};
struct RunTotals {
    const uint32_t* keys;
    const float* sums;
    float* out;
    int64_t n;
    __device__ void operator()(int64_t p) const {
        uint32_t k = keys[p];
        if (p > 0 && keys[p - 1] == k) return;
        float s = 0.0f;
        for (int64_t q = p; q < n && keys[q] == k; q += kRunBlock) s += sums[q];
        out[k] += s;
    }
};

int64_t plan_size(const GatherPlan& g) {
    int64_t n = 1;
    for (int d = 0; d < g.ndim; ++d) n *= g.x_shape[d];
    if (n > (int64_t)UINT32_MAX) throw std::invalid_argument("scatter: tensors past 2^32 elements are not supported on the GPU");
    return n;
}

}  // namespace

void sort_index(const float* x, float* index, int64_t outer, int64_t K, int64_t inner, int64_t k, bool descending) {
    int64_t lines = outer * inner;
    if (lines == 0 || K == 0 || k == 0) return;
    if (lines == 1) sort_lines<uint32_t>(x, index, 1, K, inner, k, descending, 32);
    else sort_lines<uint64_t>(x, index, lines, K, inner, k, descending, 32 + bits_for(lines));
}

void index_add_sorted(float* out, const float* src, const float* index, int64_t outer, int64_t K, int64_t inner, int64_t M) {
    if (M == 0 || outer * K * inner == 0) return;
    Pairs<uint32_t> p(M);
    each(M, FillIndex{index, p.keys, p.values});
    p.sort(bits_for(K));
    Memory starts((size_t)(K + 1) * sizeof(int));
    each(K + 1, FillStarts{p.keys, M, starts.as<int>()});
    each(outer * K * inner, AddRows{out, src, p.values, starts.as<int>(), K, inner, M});
}

void scatter_sorted(float* out, const float* index, const float* src, const GatherPlan& g) {
    if (g.n == 0) return;
    int bits = bits_for(plan_size(g));
    Pairs<uint32_t> p(g.n);
    each(g.n, FillTargets{g, index, p.keys, p.values});
    p.sort(bits);
    each(g.n, LastWins{p.keys, p.values, src, out, g.n});
}

void scatter_add_sorted(float* out, const float* index, const float* src, const GatherPlan& g) {
    if (g.n == 0) return;
    int bits = bits_for(plan_size(g));
    Pairs<uint32_t> p(g.n);
    each(g.n, FillTargets{g, index, p.keys, p.values});
    p.sort(bits);
    Memory sums((size_t)g.n * sizeof(float));
    each(g.n, RunBlocks{p.keys, p.values, src, sums.as<float>(), g.n});
    each(g.n, RunTotals{p.keys, sums.as<float>(), out, g.n});
}

}  // namespace gpu
}  // namespace kern

#endif
