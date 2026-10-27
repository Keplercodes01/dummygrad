// Scaled-dot-product attention for q, k, v, out laid out as [batch * heads, seq, head_dim], with an
// optional causal mask. Two implementations; forward reports which one it used and backward follows:
//
// GEMM path (seq <= 2048 by default, any head dim): the [seq, seq] score matrices are materialized a
// chunk of heads at a time and every product runs as a cuBLAS strided-batched GEMM. At these
// lengths the score matrices are small and cuBLAS runs near peak, several times faster than the
// tiled kernels. Forward: S = scale * Q K^T, P = softmax(S) in place (saving the log-sum-exp),
// O = P V. Backward: S and dP = dO V^T are recomputed, turned into P and dS = P * (dP - delta) * scale
// in place, then dV = P^T dO, dK = dS^T Q, dQ = dS K. cuBLAS and the row kernels reduce in a
// fixed order, so this path is deterministic too. Under fp16/bf16 autocast the GEMM operands
// (Q, K, V, dO, P, dS) are rounded to half precision for tensor cores; scores, the softmax, lse
// and every GEMM output stay fp32.
//
// Tiled path (longer sequences, head dims 16/32/64/128), whose memory stays linear in seq:
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
#include "cuda_reduce.cuh"
#include "allocator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

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

// ---------------------------------------------------------------------------------------------
// GEMM path
// ---------------------------------------------------------------------------------------------

constexpr size_t kGemmScratchBytes = size_t(256) << 20;  // score matrices per chunk of heads
std::atomic<int> g_gemm_max_seq{2048};

// Heads per chunk so that [S, S] matrices taking `bytes_per_score` bytes per element in total fit
// the scratch budget
int heads_per_chunk(int bh, int S, size_t bytes_per_score) {
    size_t per_head = bytes_per_score * S * S;
    return (int)std::max<size_t>(1, std::min<size_t>(bh, kGemmScratchBytes / per_head));
}

// Row-major strided-batched GEMM: C[b] = alpha * op(A[b]) op(B[b]) with op(A) [m, k], op(B) [k, n]
// and C [m, n] fp32. A is stored [m, k] ([k, m] if trans_a) and B [k, n] ([n, k] if trans_b), both
// of the compute dtype. cuBLAS is column-major, so this computes C^T = op(B)^T op(A)^T.
void gemm_batched(bool trans_a, bool trans_b, int m, int n, int k, float alpha, const void* A, int64_t stride_a,
                  const void* B, int64_t stride_b, float* C, int64_t stride_c, int batch, DType compute) {
    gemm_strided_batched(trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, alpha,
                         B, trans_b ? k : n, stride_b, A, trans_a ? m : k, stride_a, 0.0f, C, n, stride_c, batch, compute);
}

// GEMM element type for each compute dtype
template <typename T> constexpr DType gemm_dtype();
template <> constexpr DType gemm_dtype<float>() { return DType::Float32; }
template <> constexpr DType gemm_dtype<__half>() { return DType::Float16; }
template <> constexpr DType gemm_dtype<__nv_bfloat16>() { return DType::BFloat16; }

__device__ __forceinline__ void store_as(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_as(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ void store_as(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// An fp32 device array as a GEMM operand of type T: the array itself for float, else a rounded
// copy that lives as long as the operand
template <typename T>
struct Operand {
    std::unique_ptr<DeviceBuffer> copy;
    const T* ptr;
    Operand(const float* x, int64_t n) {
        if constexpr (std::is_same_v<T, float>) {
            ptr = x;
        } else {
            copy = std::make_unique<DeviceBuffer>((size_t)n * sizeof(T));
            cast_from_float(x, copy->ptr, n, gemm_dtype<T>());
            ptr = static_cast<const T*>(copy->ptr);
        }
    }
};

// Second [S, S] buffer of type T for a chunk: none for float (the fp32 scores are overwritten in
// place), else a separate half-precision array the next GEMMs read
template <typename T>
struct ScoresAs {
    std::unique_ptr<DeviceBuffer> own;
    T* ptr;
    ScoresAs(float* fp32, size_t n) {
        if constexpr (std::is_same_v<T, float>) {
            ptr = fp32;
        } else {
            own = std::make_unique<DeviceBuffer>(n * sizeof(T));
            ptr = static_cast<T*>(own->ptr);
        }
    }
};

template <typename T>
constexpr size_t half_bytes() { return std::is_same_v<T, float> ? 0 : sizeof(T); }

int row_threads(int S) { return std::min(256, std::max(32, (S + 31) / 32 * 32)); }

int grid_for(int64_t n, int threads) {
    return (int)std::max<int64_t>(1, std::min<int64_t>((n + threads - 1) / threads, 1 << 20));  // grid-stride loops
}

// Over rows of fp32 scaled scores: probs = softmax(S) as T (in place when T is float), with keys
// after the query masked to zero when causal; saves each row's log-sum-exp. One block per row
// (head * S + query).
template <typename T>
__global__ void k_attention_softmax(float* scores, T* probs, float* lse, int S, bool causal) {
    int64_t row = blockIdx.x;
    int i = (int)(row % S);
    float* r = scores + row * S;
    T* pr = probs + row * S;
    int n = causal ? i + 1 : S;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, r[j]);
    float row_max = block_reduce_max(m);
    __shared__ float s_max, s_inv;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    m = s_max;
    float l = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) l += expf(r[j] - m);
    float total = block_reduce_sum(l);
    if (threadIdx.x == 0) {
        s_inv = 1.0f / total;
        lse[row] = m + logf(total);
    }
    __syncthreads();
    float inv = s_inv;
    for (int j = threadIdx.x; j < S; j += blockDim.x) store_as(pr + j, (j < n) ? expf(r[j] - m) * inv : 0.0f);
}

// Over a chunk of heads, from fp32 scores S and dP: probs = P = exp(S - lse_i) and
// ds = dS = P * (dP - delta_i) * scale as T (in place over scores / dp when T is float), both zero
// above the diagonal when causal. lse and delta point at the chunk's first row.
template <typename T>
__global__ void k_attention_grad_scores(const float* scores, const float* dp, T* probs, T* ds, const float* lse,
                                        const float* delta, int64_t n, int S, bool causal, float scale) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t row = idx / S;
        int j = (int)(idx - row * S);
        int i = (int)(row % S);
        float p = 0.0f, g = 0.0f;
        if (!causal || j <= i) {
            p = expf(scores[idx] - lse[row]);
            g = p * (dp[idx] - delta[row]) * scale;
        }
        store_as(probs + idx, p);
        store_as(ds + idx, g);
    }
}

// T is the GEMM element type: float, or __half / __nv_bfloat16 under autocast. Scores, softmax and
// lse are fp32 either way; only the GEMM operands are rounded to T.
template <typename T>
void forward_gemm(const float* q, const float* k, const float* v, float* out, float* lse,
                  int bh, int S, int D, bool causal, float scale) {
    constexpr DType dt = gemm_dtype<T>();
    int64_t hs = (int64_t)S * D, ss = (int64_t)S * S, total = (int64_t)bh * hs;
    Operand<T> qh(q, total), kh(k, total), vh(v, total);
    int chunk = heads_per_chunk(bh, S, sizeof(float) + half_bytes<T>());
    DeviceBuffer scores((size_t)(chunk * ss) * sizeof(float));
    ScoresAs<T> probs(scores.f(), (size_t)(chunk * ss));
    for (int h0 = 0; h0 < bh; h0 += chunk) {
        int hc = std::min(chunk, bh - h0);
        int64_t o = h0 * hs;
        gemm_batched(false, true, S, S, D, scale, qh.ptr + o, hs, kh.ptr + o, hs, scores.f(), ss, hc, dt);
        k_attention_softmax<T><<<(unsigned)(hc * S), row_threads(S)>>>(scores.f(), probs.ptr, lse + (int64_t)h0 * S,
                                                                      S, causal);
        CUDA_CHECK(cudaGetLastError());
        gemm_batched(false, false, S, D, S, 1.0f, probs.ptr, ss, vh.ptr + o, hs, out + o, hs, hc, dt);
    }
}

template <typename T>
void backward_gemm(const float* q, const float* k, const float* v, const float* dout, const float* lse,
                   const float* delta, float* dq, float* dk, float* dv, int bh, int S, int D, bool causal, float scale) {
    constexpr DType dt = gemm_dtype<T>();
    int64_t hs = (int64_t)S * D, ss = (int64_t)S * S, total = (int64_t)bh * hs;
    Operand<T> qh(q, total), kh(k, total), vh(v, total), doh(dout, total);
    int chunk = heads_per_chunk(bh, S, 2 * (sizeof(float) + half_bytes<T>()));
    DeviceBuffer scores((size_t)(chunk * ss) * sizeof(float)), dp((size_t)(chunk * ss) * sizeof(float));
    ScoresAs<T> probs(scores.f(), (size_t)(chunk * ss)), ds(dp.f(), (size_t)(chunk * ss));
    for (int h0 = 0; h0 < bh; h0 += chunk) {
        int hc = std::min(chunk, bh - h0);
        int64_t o = h0 * hs;
        gemm_batched(false, true, S, S, D, scale, qh.ptr + o, hs, kh.ptr + o, hs, scores.f(), ss, hc, dt);
        gemm_batched(false, true, S, S, D, 1.0f, doh.ptr + o, hs, vh.ptr + o, hs, dp.f(), ss, hc, dt);
        int64_t n = hc * ss;
        k_attention_grad_scores<T><<<grid_for(n, 256), 256>>>(scores.f(), dp.f(), probs.ptr, ds.ptr,
                                                             lse + (int64_t)h0 * S, delta + (int64_t)h0 * S,
                                                             n, S, causal, scale);
        CUDA_CHECK(cudaGetLastError());
        gemm_batched(true, false, S, D, S, 1.0f, probs.ptr, ss, doh.ptr + o, hs, dv + o, hs, hc, dt);  // P^T dO
        gemm_batched(true, false, S, D, S, 1.0f, ds.ptr, ss, qh.ptr + o, hs, dk + o, hs, hc, dt);      // dS^T Q
        gemm_batched(false, false, S, D, S, 1.0f, ds.ptr, ss, kh.ptr + o, hs, dq + o, hs, hc, dt);     // dS K
    }
}

}  // namespace

void set_attention_gemm_max_seq(int seq_len) { g_gemm_max_seq = seq_len; }
int attention_gemm_max_seq() { return g_gemm_max_seq; }

AttentionImpl attention_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                                int batch, int heads, int seq_len, int head_dim, bool causal, float scale,
                                DType compute) {
    int bh = batch * heads;
    AttentionImpl impl = seq_len <= g_gemm_max_seq ? AttentionImpl::Gemm
                       : tiled_supported(head_dim) ? AttentionImpl::Tiled
                                                   : AttentionImpl::Untiled;
    if (bh == 0 || seq_len == 0) return impl;
    switch (impl) {
        case AttentionImpl::Gemm:
            switch (compute) {
                case DType::Float32: forward_gemm<float>(q, k, v, out, lse, bh, seq_len, head_dim, causal, scale); break;
                case DType::Float16: forward_gemm<__half>(q, k, v, out, lse, bh, seq_len, head_dim, causal, scale); break;
                case DType::BFloat16:
                    forward_gemm<__nv_bfloat16>(q, k, v, out, lse, bh, seq_len, head_dim, causal, scale);
                    break;
                default: throw std::runtime_error("cuda::attention_forward: compute dtype must be Float32, Float16 or BFloat16");
            }
            break;
        case AttentionImpl::Untiled:
            // Other head sizes at long lengths; writes no lse, its backward recomputes the statistics
            flash_attention_forward(q, k, v, out, batch, heads, seq_len, head_dim, causal, scale);
            break;
        case AttentionImpl::Tiled:
            switch (head_dim) {
                case 16: launch_forward<16>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
                case 32: launch_forward<32>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
                case 64: launch_forward<64>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
                default: launch_forward<128>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
            }
            break;
    }
    return impl;
}

void attention_backward(const float* q, const float* k, const float* v, const float* out, const float* dout,
                        const float* lse, float* dq, float* dk, float* dv,
                        int batch, int heads, int seq_len, int head_dim, bool causal, float scale, AttentionImpl impl,
                        DType compute) {
    int bh = batch * heads;
    if (bh == 0 || seq_len == 0) return;
    if (impl == AttentionImpl::Untiled) {
        flash_attention_backward(q, k, v, out, dout, dq, dk, dv, batch, heads, seq_len, head_dim, causal, scale);
        return;
    }
    int64_t rows = (int64_t)bh * seq_len;
    DeviceBuffer delta((size_t)rows * sizeof(float));
    k_attention_delta<<<(unsigned)((rows * 32 + 255) / 256), 256>>>(out, dout, delta.f(), rows, head_dim);
    CUDA_CHECK(cudaGetLastError());
    if (impl == AttentionImpl::Gemm) {
        switch (compute) {
            case DType::Float32:
                backward_gemm<float>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, head_dim, causal, scale);
                break;
            case DType::Float16:
                backward_gemm<__half>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, head_dim, causal, scale);
                break;
            case DType::BFloat16:
                backward_gemm<__nv_bfloat16>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, head_dim, causal,
                                             scale);
                break;
            default: throw std::runtime_error("cuda::attention_backward: compute dtype must be Float32, Float16 or BFloat16");
        }
        return;
    }
    switch (head_dim) {
        case 16: launch_backward<16>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        case 32: launch_backward<32>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        case 64: launch_backward<64>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        default: launch_backward<128>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
    }
}

}  // namespace cuda

#endif  // USE_CUDA
