// Tiled scaled-dot-product attention (FlashAttention-style) for q, k, v, out laid out as
// [batch * heads, seq, head_dim], with an optional causal mask.
//
// Forward: each query row is owned by a group of head_dim / 16 threads holding 16-element slices
// in registers; K/V stream through shared memory in tiles while an online softmax accumulates the
// output. The row's log-sum-exp is saved for backward.
//
// Backward (FlashAttention-2 split): delta_i = dO_i . O_i, then one kernel where thread groups own
// key rows and accumulate dK/dV over all queries, and one where they own query rows and accumulate
// dQ over all keys. Every output element is written by exactly one thread, so there are no atomics
// and results are deterministic.
//
// Keeping each row's state in registers split across a thread group (instead of one thread per row
// with rows in shared memory) keeps shared memory to the tiles all threads read together, which
// lets many more warps run per multiprocessor.

#include "cuda_ops.cuh"
#include "cuda_common.h"
#include "allocator.h"

#include <algorithm>
#include <cmath>

#ifdef USE_CUDA

namespace cuda {

namespace {

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(std::max<size_t>(n, 4)) { ptr = get_memory(Device::CUDA, bytes); }
    ~DeviceBuffer() { free_memory(Device::CUDA, ptr, bytes); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    float* f() const { return static_cast<float*>(ptr); }
};

// Each row (query or key) is owned by a group of SPLIT = D / 16 adjacent threads; thread `part`
// of the group holds elements [part * 16, part * 16 + 16) of the row in registers. Dot products
// are partial sums over the slice combined across the group with shuffles. Loops run over
// warp-uniform bounds so every lane reaches every shuffle; masks are applied after the shuffle.
constexpr int kSlice = 16;
constexpr int kRows = 64;   // rows per block (blockDim = kRows * SPLIT)
constexpr int kTile = 32;   // rows per shared-memory tile

template <int SPLIT>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
    for (int offset = SPLIT / 2; offset > 0; offset /= 2) v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// Loads rows [row0, row0 + kTile) of a [S, D] head into shared memory (row-major), zero-padded.
template <int D>
__device__ void load_tile(float* dst, const float* head, int row0, int S) {
    for (int idx = threadIdx.x; idx < kTile * D; idx += blockDim.x) {
        int row = row0 + idx / D;
        dst[idx] = (row < S) ? head[(int64_t)row0 * D + idx] : 0.0f;
    }
}

// Copies this thread's 16-element slice of a row from global memory (zeros past the end).
__device__ __forceinline__ void load_slice(float (&dst)[kSlice], const float* row, bool valid) {
#pragma unroll
    for (int e = 0; e < kSlice; ++e) dst[e] = valid ? row[e] : 0.0f;
}

// Dot product of a register slice with a 16-float slice of shared memory (16-byte aligned).
__device__ __forceinline__ float slice_dot(const float (&a)[kSlice], const float* smem_slice) {
    const float4* b = reinterpret_cast<const float4*>(smem_slice);
    float s = 0.0f;
#pragma unroll
    for (int e4 = 0; e4 < kSlice / 4; ++e4) {
        float4 v = b[e4];
        s += a[e4 * 4] * v.x + a[e4 * 4 + 1] * v.y + a[e4 * 4 + 2] * v.z + a[e4 * 4 + 3] * v.w;
    }
    return s;
}

// acc = acc * scale_acc + w * smem_slice
__device__ __forceinline__ void slice_axpby(float (&acc)[kSlice], float scale_acc, float w, const float* smem_slice) {
    const float4* b = reinterpret_cast<const float4*>(smem_slice);
#pragma unroll
    for (int e4 = 0; e4 < kSlice / 4; ++e4) {
        float4 v = b[e4];
        acc[e4 * 4] = acc[e4 * 4] * scale_acc + w * v.x;
        acc[e4 * 4 + 1] = acc[e4 * 4 + 1] * scale_acc + w * v.y;
        acc[e4 * 4 + 2] = acc[e4 * 4 + 2] * scale_acc + w * v.z;
        acc[e4 * 4 + 3] = acc[e4 * 4 + 3] * scale_acc + w * v.w;
    }
}

// grid (ceil(S / kRows), batch * heads); the thread group `row` owns query blockIdx.x * kRows + row
template <int D>
__global__ void k_attention_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                                    int S, bool causal, float scale) {
    constexpr int SPLIT = D / kSlice;
    extern __shared__ __align__(16) float smem[];
    float* Ks = smem;            // [kTile][D]
    float* Vs = Ks + kTile * D;  // [kTile][D]

    int64_t head = (int64_t)blockIdx.y * S * D;
    int row = threadIdx.x / SPLIT, part = threadIdx.x % SPLIT;
    int block_row0 = blockIdx.x * kRows;
    int i = block_row0 + row;
    bool active = i < S;
    int off = part * kSlice;

    float qv[kSlice], acc[kSlice];
    load_slice(qv, q + head + (int64_t)i * D + off, active);
#pragma unroll
    for (int e = 0; e < kSlice; ++e) acc[e] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    int key_end = causal ? min(S, block_row0 + kRows) : S;
    for (int j0 = 0; j0 < key_end; j0 += kTile) {
        __syncthreads();
        load_tile<D>(Ks, k + head, j0, S);
        load_tile<D>(Vs, v + head, j0, S);
        __syncthreads();
        int jn = min(kTile, S - j0);
        for (int jj = 0; jj < jn; ++jj) {
            float s = group_sum<SPLIT>(slice_dot(qv, Ks + jj * D + off)) * scale;
            if (causal && j0 + jj > i) continue;  // after the shuffle: masked keys contribute nothing
            float m_new = fmaxf(m, s);
            float correction = expf(m - m_new);
            float p = expf(s - m_new);
            l = l * correction + p;
            slice_axpby(acc, correction, p, Vs + jj * D + off);
            m = m_new;
        }
    }

    if (active) {
        float inv = 1.0f / l;
        float* o = out + head + (int64_t)i * D + off;
#pragma unroll
        for (int e = 0; e < kSlice; ++e) o[e] = acc[e] * inv;
        if (part == 0) lse[(int64_t)blockIdx.y * S + i] = m + logf(l);
    }
}

// delta_i = dO_i . O_i, one warp per row
__global__ void k_attention_delta(const float* out, const float* dout, float* delta, int64_t rows, int D) {
    int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;  // whole warps exit together
    const float* o = out + row * D;
    const float* g = dout + row * D;
    float s = 0.0f;
    for (int d = lane; d < D; d += 32) s += o[d] * g[d];
    for (int offset = 16; offset > 0; offset /= 2) s += __shfl_down_sync(0xffffffff, s, offset);
    if (lane == 0) delta[row] = s;
}

// grid (ceil(S / kRows), batch * heads); the thread group `row` owns key blockIdx.x * kRows + row
template <int D>
__global__ void k_attention_backward_kv(const float* q, const float* k, const float* v, const float* dout,
                                        const float* lse, const float* delta, float* dk, float* dv,
                                        int S, bool causal, float scale) {
    constexpr int SPLIT = D / kSlice;
    extern __shared__ __align__(16) float smem[];
    float* Qs = smem;            // [kTile][D]
    float* Gs = Qs + kTile * D;  // [kTile][D]  (dO)
    float* Ls = Gs + kTile * D;  // [kTile]     (lse)
    float* Ds = Ls + kTile;      // [kTile]     (delta)

    int64_t head = (int64_t)blockIdx.y * S * D;
    int64_t row_base = (int64_t)blockIdx.y * S;
    int row = threadIdx.x / SPLIT, part = threadIdx.x % SPLIT;
    int block_row0 = blockIdx.x * kRows;
    int j = block_row0 + row;
    bool active = j < S;
    int off = part * kSlice;

    float kv[kSlice], vv[kSlice], gk[kSlice], gv[kSlice];
    load_slice(kv, k + head + (int64_t)j * D + off, active);
    load_slice(vv, v + head + (int64_t)j * D + off, active);
#pragma unroll
    for (int e = 0; e < kSlice; ++e) gk[e] = gv[e] = 0.0f;

    int query_start = causal ? (block_row0 / kTile) * kTile : 0;  // queries i >= j only
    for (int i0 = query_start; i0 < S; i0 += kTile) {
        __syncthreads();
        load_tile<D>(Qs, q + head, i0, S);
        load_tile<D>(Gs, dout + head, i0, S);
        for (int r = threadIdx.x; r < kTile; r += blockDim.x) {
            int i = i0 + r;
            Ls[r] = (i < S) ? lse[row_base + i] : 0.0f;
            Ds[r] = (i < S) ? delta[row_base + i] : 0.0f;
        }
        __syncthreads();
        int in = min(kTile, S - i0);
        for (int ii = 0; ii < in; ++ii) {
            float s = group_sum<SPLIT>(slice_dot(kv, Qs + ii * D + off));
            float dp = group_sum<SPLIT>(slice_dot(vv, Gs + ii * D + off));
            if (causal && i0 + ii < j) continue;
            float p = expf(s * scale - Ls[ii]);
            float ds = p * (dp - Ds[ii]) * scale;
            slice_axpby(gv, 1.0f, p, Gs + ii * D + off);
            slice_axpby(gk, 1.0f, ds, Qs + ii * D + off);
        }
    }

    if (active) {
        float* dkr = dk + head + (int64_t)j * D + off;
        float* dvr = dv + head + (int64_t)j * D + off;
#pragma unroll
        for (int e = 0; e < kSlice; ++e) {
            dkr[e] = gk[e];
            dvr[e] = gv[e];
        }
    }
}

// grid (ceil(S / kRows), batch * heads); the thread group `row` owns query blockIdx.x * kRows + row
template <int D>
__global__ void k_attention_backward_q(const float* q, const float* k, const float* v, const float* dout,
                                       const float* lse, const float* delta, float* dq,
                                       int S, bool causal, float scale) {
    constexpr int SPLIT = D / kSlice;
    extern __shared__ __align__(16) float smem[];
    float* Ks = smem;            // [kTile][D]
    float* Vs = Ks + kTile * D;  // [kTile][D]

    int64_t head = (int64_t)blockIdx.y * S * D;
    int row = threadIdx.x / SPLIT, part = threadIdx.x % SPLIT;
    int block_row0 = blockIdx.x * kRows;
    int i = block_row0 + row;
    bool active = i < S;
    int off = part * kSlice;

    float qv[kSlice], gov[kSlice], gq[kSlice];
    load_slice(qv, q + head + (int64_t)i * D + off, active);
    load_slice(gov, dout + head + (int64_t)i * D + off, active);
#pragma unroll
    for (int e = 0; e < kSlice; ++e) gq[e] = 0.0f;
    float li = active ? lse[(int64_t)blockIdx.y * S + i] : 0.0f;
    float di = active ? delta[(int64_t)blockIdx.y * S + i] : 0.0f;

    int key_end = causal ? min(S, block_row0 + kRows) : S;
    for (int j0 = 0; j0 < key_end; j0 += kTile) {
        __syncthreads();
        load_tile<D>(Ks, k + head, j0, S);
        load_tile<D>(Vs, v + head, j0, S);
        __syncthreads();
        int jn = min(kTile, S - j0);
        for (int jj = 0; jj < jn; ++jj) {
            float s = group_sum<SPLIT>(slice_dot(qv, Ks + jj * D + off));
            float dp = group_sum<SPLIT>(slice_dot(gov, Vs + jj * D + off));
            if (causal && j0 + jj > i) continue;
            float p = expf(s * scale - li);
            float ds = p * (dp - di) * scale;
            slice_axpby(gq, 1.0f, ds, Ks + jj * D + off);
        }
    }

    if (active) {
        float* dqr = dq + head + (int64_t)i * D + off;
#pragma unroll
        for (int e = 0; e < kSlice; ++e) dqr[e] = gq[e];
    }
}

template <int D>
void launch_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                    int batch_heads, int S, bool causal, float scale) {
    dim3 grid((S + kRows - 1) / kRows, batch_heads);
    size_t smem = (size_t)(2 * kTile * D) * sizeof(float);
    k_attention_forward<D><<<grid, kRows * (D / kSlice), smem>>>(q, k, v, out, lse, S, causal, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <int D>
void launch_backward(const float* q, const float* k, const float* v, const float* dout, const float* lse,
                     const float* delta, float* dq, float* dk, float* dv, int batch_heads, int S, bool causal, float scale) {
    dim3 grid((S + kRows - 1) / kRows, batch_heads);
    int threads = kRows * (D / kSlice);
    size_t smem_kv = (size_t)(2 * kTile * D + 2 * kTile) * sizeof(float);
    k_attention_backward_kv<D><<<grid, threads, smem_kv>>>(q, k, v, dout, lse, delta, dk, dv, S, causal, scale);
    CUDA_CHECK(cudaGetLastError());
    size_t smem_q = (size_t)(2 * kTile * D) * sizeof(float);
    k_attention_backward_q<D><<<grid, threads, smem_q>>>(q, k, v, dout, lse, delta, dq, S, causal, scale);
    CUDA_CHECK(cudaGetLastError());
}

bool tiled_supported(int D) { return D == 16 || D == 32 || D == 64 || D == 128; }

}  // namespace

void attention_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                       int batch, int heads, int seq_len, int head_dim, bool causal, float scale) {
    int bh = batch * heads;
    if (bh == 0 || seq_len == 0) return;
    switch (head_dim) {
        case 16: launch_forward<16>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        case 32: launch_forward<32>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        case 64: launch_forward<64>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        case 128: launch_forward<128>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        default:
            // Untiled kernel for other head sizes; its backward recomputes the statistics itself
            flash_attention_forward(q, k, v, out, batch, heads, seq_len, head_dim, causal, scale);
            break;
    }
}

void attention_backward(const float* q, const float* k, const float* v, const float* out, const float* dout,
                        const float* lse, float* dq, float* dk, float* dv,
                        int batch, int heads, int seq_len, int head_dim, bool causal, float scale) {
    int bh = batch * heads;
    if (bh == 0 || seq_len == 0) return;
    if (!tiled_supported(head_dim)) {
        flash_attention_backward(q, k, v, out, dout, dq, dk, dv, batch, heads, seq_len, head_dim, causal, scale);
        return;
    }
    int64_t rows = (int64_t)bh * seq_len;
    DeviceBuffer delta((size_t)rows * sizeof(float));
    k_attention_delta<<<(unsigned)((rows * 32 + 255) / 256), 256>>>(out, dout, delta.f(), rows, head_dim);
    CUDA_CHECK(cudaGetLastError());
    switch (head_dim) {
        case 16: launch_backward<16>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        case 32: launch_backward<32>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        case 64: launch_backward<64>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        default: launch_backward<128>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
    }
}

}  // namespace cuda

#endif  // USE_CUDA
