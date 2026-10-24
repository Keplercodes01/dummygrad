// Tiled scaled-dot-product attention (FlashAttention-style) for q, k, v, out laid out as
// [batch * heads, seq, head_dim], with an optional causal mask.
//
// Forward: each thread owns one query row; K/V stream through shared memory in tiles while an
// online softmax accumulates the output in registers. The row's log-sum-exp is saved for backward.
//
// Backward (FlashAttention-2 split): delta_i = dO_i . O_i, then one kernel where each thread owns a
// key row and accumulates dK/dV over all queries, and one where each thread owns a query row and
// accumulates dQ over all keys. Every output element is written by exactly one thread, so there
// are no atomics and results are deterministic.
//
// Rows a thread owns are stored transposed in shared memory ([head_dim][rows]) so that threads
// reading their own rows hit consecutive banks; tiles that all threads read together are stored
// row-major and served as broadcasts.

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

// Loads rows [row0, row0 + ROWS) of a [S, D] head into shared memory, transposed to [D][ROWS].
template <int D, int ROWS>
__device__ void load_transposed(float* dst, const float* head, int row0, int S) {
    for (int idx = threadIdx.x; idx < ROWS * D; idx += blockDim.x) {
        int r = idx / D, d = idx - r * D;
        int row = row0 + r;
        dst[d * ROWS + r] = (row < S) ? head[(int64_t)row * D + d] : 0.0f;
    }
}

// Loads rows [row0, row0 + ROWS) of a [S, D] head into shared memory, row-major.
template <int D, int ROWS>
__device__ void load_rows(float* dst, const float* head, int row0, int S) {
    for (int idx = threadIdx.x; idx < ROWS * D; idx += blockDim.x) {
        int row = row0 + idx / D;
        dst[idx] = (row < S) ? head[(int64_t)row0 * D + idx] : 0.0f;
    }
}

// grid (ceil(S / BLOCK), batch * heads), BLOCK threads; thread t owns query row blockIdx.x * BLOCK + t
template <int D, int BLOCK, int TILE>
__global__ void k_attention_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                                    int S, bool causal, float scale) {
    extern __shared__ float smem[];
    float* Qt = smem;               // [D][BLOCK]
    float* Ks = Qt + D * BLOCK;     // [TILE][D]
    float* Vs = Ks + TILE * D;      // [TILE][D]

    int64_t head = (int64_t)blockIdx.y * S * D;
    int i0 = blockIdx.x * BLOCK;
    int t = threadIdx.x;
    int i = i0 + t;
    bool active = i < S;

    load_transposed<D, BLOCK>(Qt, q + head, i0, S);

    float acc[D];
#pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    int key_end = causal ? min(S, i0 + BLOCK) : S;
    for (int j0 = 0; j0 < key_end; j0 += TILE) {
        __syncthreads();  // previous tile fully consumed (and Q loaded, on the first pass)
        load_rows<D, TILE>(Ks, k + head, j0, S);
        load_rows<D, TILE>(Vs, v + head, j0, S);
        __syncthreads();
        if (!active) continue;
        int jmax = min(TILE, S - j0);
        if (causal) jmax = min(jmax, i - j0 + 1);  // keys j <= i
        for (int jj = 0; jj < jmax; ++jj) {
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) s += Qt[d * BLOCK + t] * Ks[jj * D + d];
            s *= scale;
            float m_new = fmaxf(m, s);
            float correction = expf(m - m_new);
            float p = expf(s - m_new);
            l = l * correction + p;
#pragma unroll
            for (int d = 0; d < D; ++d) acc[d] = acc[d] * correction + p * Vs[jj * D + d];
            m = m_new;
        }
    }

    if (active) {
        float inv = 1.0f / l;
        float* o = out + head + (int64_t)i * D;
#pragma unroll
        for (int d = 0; d < D; ++d) o[d] = acc[d] * inv;
        lse[(int64_t)blockIdx.y * S + i] = m + logf(l);
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

// grid (ceil(S / BLOCK), batch * heads), BLOCK threads; thread t owns key row blockIdx.x * BLOCK + t
template <int D, int BLOCK, int TILE>
__global__ void k_attention_backward_kv(const float* q, const float* k, const float* v, const float* dout,
                                        const float* lse, const float* delta, float* dk, float* dv,
                                        int S, bool causal, float scale) {
    extern __shared__ float smem[];
    float* Kt = smem;               // [D][BLOCK]
    float* Vt = Kt + D * BLOCK;     // [D][BLOCK]
    float* Qs = Vt + D * BLOCK;     // [TILE][D]
    float* Gs = Qs + TILE * D;      // [TILE][D]  (dO)
    float* Ls = Gs + TILE * D;      // [TILE]     (lse)
    float* Ds = Ls + TILE;          // [TILE]     (delta)

    int64_t head = (int64_t)blockIdx.y * S * D;
    int64_t row_base = (int64_t)blockIdx.y * S;
    int j0 = blockIdx.x * BLOCK;
    int t = threadIdx.x;
    int j = j0 + t;
    bool active = j < S;

    load_transposed<D, BLOCK>(Kt, k + head, j0, S);
    load_transposed<D, BLOCK>(Vt, v + head, j0, S);

    float gk[D], gv[D];
#pragma unroll
    for (int d = 0; d < D; ++d) gk[d] = gv[d] = 0.0f;

    int query_start = causal ? (j0 / TILE) * TILE : 0;  // queries i >= j only
    for (int i0 = query_start; i0 < S; i0 += TILE) {
        __syncthreads();
        load_rows<D, TILE>(Qs, q + head, i0, S);
        load_rows<D, TILE>(Gs, dout + head, i0, S);
        for (int r = t; r < TILE; r += blockDim.x) {
            int i = i0 + r;
            Ls[r] = (i < S) ? lse[row_base + i] : 0.0f;
            Ds[r] = (i < S) ? delta[row_base + i] : 0.0f;
        }
        __syncthreads();
        if (!active) continue;
        int imax = min(TILE, S - i0);
        for (int ii = 0; ii < imax; ++ii) {
            if (causal && i0 + ii < j) continue;
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                s += Qs[ii * D + d] * Kt[d * BLOCK + t];
                dp += Gs[ii * D + d] * Vt[d * BLOCK + t];
            }
            float p = expf(s * scale - Ls[ii]);
            float ds = p * (dp - Ds[ii]) * scale;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                gv[d] += p * Gs[ii * D + d];
                gk[d] += ds * Qs[ii * D + d];
            }
        }
    }

    if (active) {
        float* dkr = dk + head + (int64_t)j * D;
        float* dvr = dv + head + (int64_t)j * D;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            dkr[d] = gk[d];
            dvr[d] = gv[d];
        }
    }
}

// grid (ceil(S / BLOCK), batch * heads), BLOCK threads; thread t owns query row blockIdx.x * BLOCK + t
template <int D, int BLOCK, int TILE>
__global__ void k_attention_backward_q(const float* q, const float* k, const float* v, const float* dout,
                                       const float* lse, const float* delta, float* dq,
                                       int S, bool causal, float scale) {
    extern __shared__ float smem[];
    float* Qt = smem;               // [D][BLOCK]
    float* Gt = Qt + D * BLOCK;     // [D][BLOCK]  (dO)
    float* Ks = Gt + D * BLOCK;     // [TILE][D]
    float* Vs = Ks + TILE * D;      // [TILE][D]

    int64_t head = (int64_t)blockIdx.y * S * D;
    int i0 = blockIdx.x * BLOCK;
    int t = threadIdx.x;
    int i = i0 + t;
    bool active = i < S;

    load_transposed<D, BLOCK>(Qt, q + head, i0, S);
    load_transposed<D, BLOCK>(Gt, dout + head, i0, S);
    float li = active ? lse[(int64_t)blockIdx.y * S + i] : 0.0f;
    float di = active ? delta[(int64_t)blockIdx.y * S + i] : 0.0f;

    float gq[D];
#pragma unroll
    for (int d = 0; d < D; ++d) gq[d] = 0.0f;

    int key_end = causal ? min(S, i0 + BLOCK) : S;
    for (int j0 = 0; j0 < key_end; j0 += TILE) {
        __syncthreads();
        load_rows<D, TILE>(Ks, k + head, j0, S);
        load_rows<D, TILE>(Vs, v + head, j0, S);
        __syncthreads();
        if (!active) continue;
        int jmax = min(TILE, S - j0);
        if (causal) jmax = min(jmax, i - j0 + 1);
        for (int jj = 0; jj < jmax; ++jj) {
            float s = 0.0f, dp = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                s += Qt[d * BLOCK + t] * Ks[jj * D + d];
                dp += Gt[d * BLOCK + t] * Vs[jj * D + d];
            }
            float p = expf(s * scale - li);
            float ds = p * (dp - di) * scale;
#pragma unroll
            for (int d = 0; d < D; ++d) gq[d] += ds * Ks[jj * D + d];
        }
    }

    if (active) {
        float* dqr = dq + head + (int64_t)i * D;
#pragma unroll
        for (int d = 0; d < D; ++d) dqr[d] = gq[d];
    }
}

template <int D, int BLOCK, int TILE>
void launch_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                    int batch_heads, int S, bool causal, float scale) {
    dim3 grid((S + BLOCK - 1) / BLOCK, batch_heads);
    size_t smem = (size_t)(D * BLOCK + 2 * TILE * D) * sizeof(float);
    k_attention_forward<D, BLOCK, TILE><<<grid, BLOCK, smem>>>(q, k, v, out, lse, S, causal, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <int D, int BLOCK, int TILE>
void launch_backward(const float* q, const float* k, const float* v, const float* dout, const float* lse,
                     const float* delta, float* dq, float* dk, float* dv, int batch_heads, int S, bool causal, float scale) {
    dim3 grid((S + BLOCK - 1) / BLOCK, batch_heads);
    size_t smem_kv = (size_t)(2 * D * BLOCK + 2 * TILE * D + 2 * TILE) * sizeof(float);
    k_attention_backward_kv<D, BLOCK, TILE><<<grid, BLOCK, smem_kv>>>(q, k, v, dout, lse, delta, dk, dv, S, causal, scale);
    CUDA_CHECK(cudaGetLastError());
    size_t smem_q = (size_t)(2 * D * BLOCK + 2 * TILE * D) * sizeof(float);
    k_attention_backward_q<D, BLOCK, TILE><<<grid, BLOCK, smem_q>>>(q, k, v, dout, lse, delta, dq, S, causal, scale);
    CUDA_CHECK(cudaGetLastError());
}

// Tile shapes per head dimension, sized so every kernel stays under 48 KB of shared memory
bool tiled_supported(int D) { return D == 16 || D == 32 || D == 64 || D == 128; }

}  // namespace

void attention_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                       int batch, int heads, int seq_len, int head_dim, bool causal, float scale) {
    int bh = batch * heads;
    if (bh == 0 || seq_len == 0) return;
    switch (head_dim) {
        case 16: launch_forward<16, 64, 64>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        case 32: launch_forward<32, 64, 32>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        case 64: launch_forward<64, 64, 16>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
        case 128: launch_forward<128, 32, 8>(q, k, v, out, lse, bh, seq_len, causal, scale); break;
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
        case 16: launch_backward<16, 64, 64>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        case 32: launch_backward<32, 64, 32>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        case 64: launch_backward<64, 64, 16>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
        default: launch_backward<128, 32, 8>(q, k, v, dout, lse, delta.f(), dq, dk, dv, bh, seq_len, causal, scale); break;
    }
}

}  // namespace cuda

#endif  // USE_CUDA
