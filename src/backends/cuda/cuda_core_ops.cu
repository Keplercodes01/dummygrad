// GPU kernels for the ops a training step needs besides matmul and attention: broadcast binary
// ops, scalar ops, RMSNorm, fused cross entropy, embedding lookup, GQA head repetition, and
// sums of squares for gradient clipping.
//
// Every reduction is deterministic: each output is a fixed-order sum over a fixed partition of
// its terms (no atomics), so identical runs give bitwise-identical results.

#include "cuda_ops.cuh"
#include "cuda_common.h"
#include "cuda_reduce.cuh"
#include "allocator.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef USE_CUDA

namespace cuda {

namespace {

constexpr int kThreads = 256;

int grid_for(int64_t n) {
    int64_t blocks = (n + kThreads - 1) / kThreads;
    return (int)std::max<int64_t>(1, std::min<int64_t>(blocks, 1 << 20));  // kernels use grid-stride loops
}

// Scratch device memory from the caching allocator, released at scope exit.
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(std::max<size_t>(n, 4)) { ptr = get_memory(Device::CUDA, bytes); }
    ~DeviceBuffer() { free_memory(Device::CUDA, ptr, bytes); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T> T* as() const { return static_cast<T*>(ptr); }
};

// ---------------------------------------------------------------------------------------------
// Deterministic reduction: out[t] = sum_{k < K} term(ctx(t), k) for t < n_out. The K terms of
// every output are split into `splits` contiguous chunks; chunk sums are added in chunk order.
// The split count depends only on the shapes, so results are reproducible.
// ---------------------------------------------------------------------------------------------

template <typename Term>
__global__ void k_reduce_partials(Term term, int64_t n_out, int64_t K, int splits, float* partial) {
    int64_t items = n_out * splits;
    for (int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; w < items; w += (int64_t)gridDim.x * blockDim.x) {
        int64_t t = w % n_out;
        int64_t s = w / n_out;
        int64_t k0 = K * s / splits, k1 = K * (s + 1) / splits;
        auto ctx = term.context(t);
        float acc = 0.0f;
        for (int64_t k = k0; k < k1; ++k) acc += term(ctx, k);
        partial[s * n_out + t] = acc;
    }
}

__global__ void k_sum_partials(const float* partial, float* out, int64_t n_out, int splits) {
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < n_out; t += (int64_t)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s) acc += partial[(int64_t)s * n_out + t];
        out[t] = acc;
    }
}

template <typename Term>
void reduce_terms(const Term& term, int64_t n_out, int64_t K, float* out) {
    if (n_out <= 0) return;
    // Aim for ~128K threads of work, at least 32 terms per chunk, at most 4096 chunks.
    int64_t splits = std::min<int64_t>({std::max<int64_t>(1, K / 32), std::max<int64_t>(1, (131072 + n_out - 1) / n_out), 4096});
    if (splits <= 1) {
        k_reduce_partials<<<grid_for(n_out), kThreads>>>(term, n_out, K, 1, out);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    DeviceBuffer partial((size_t)(n_out * splits) * sizeof(float));
    k_reduce_partials<<<grid_for(n_out * splits), kThreads>>>(term, n_out, K, (int)splits, partial.as<float>());
    CUDA_CHECK(cudaGetLastError());
    k_sum_partials<<<grid_for(n_out), kThreads>>>(partial.as<float>(), out, n_out, (int)splits);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Broadcast binary ops. x is [batch, r, c]; y is [y_batch, yr, yc] with yr in {1, r}, yc in
// {1, c}, and y_batch equal to batch, 1, or a divisor of batch whose batches are shared by
// contiguous groups (GQA). Mirrors broadcast_binary in functional.h.
// ---------------------------------------------------------------------------------------------

struct BroadcastGeom {
    int64_t batch, r, c, y_batch, yr, yc;

    __device__ int64_t y_batch_of(int64_t b) const {
        if (y_batch == batch) return b;
        if (y_batch > 1) return b / (batch / y_batch);
        return 0;
    }
    __device__ int64_t y_index(int64_t x_idx) const {
        int64_t rc = r * c;
        int64_t b = x_idx / rc;
        int64_t rem = x_idx - b * rc;
        int64_t i = rem / c, j = rem - i * c;
        return (y_batch_of(b) * yr + (yr == 1 ? 0 : i)) * yc + (yc == 1 ? 0 : j);
    }
};

__device__ __forceinline__ float apply_binary(int op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        default: return a / b;
    }
}

__global__ void k_broadcast_forward(int op, const float* x, const float* y, float* out, BroadcastGeom g, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = apply_binary(op, x[i], y[g.y_index(i)]);
    }
}

__global__ void k_broadcast_backward_x(int op, const float* grad, const float* y, float* gx, BroadcastGeom g, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float d = grad[i];
        if (op == 2) d *= y[g.y_index(i)];
        else if (op == 3) d /= y[g.y_index(i)];
        gx[i] = d;
    }
}

// dL/dy[t] sums over every x element mapped to y element t.
struct BroadcastGradY {
    int op;
    const float* grad;
    const float* x;
    const float* y;
    BroadcastGeom g;
    int64_t ni, nj;  // rows and columns reduced per y element (r or 1, c or 1)

    struct Ctx {
        int64_t b0, i0, j0;
        float yv;
    };
    __device__ Ctx context(int64_t t) const {
        int64_t yb = t / (g.yr * g.yc);
        int64_t rem = t - yb * g.yr * g.yc;
        int64_t yi = rem / g.yc, yj = rem - yi * g.yc;
        Ctx ctx;
        if (g.y_batch == g.batch) ctx.b0 = yb;
        else if (g.y_batch > 1) ctx.b0 = yb * (g.batch / g.y_batch);
        else ctx.b0 = 0;
        ctx.i0 = (g.yr == 1) ? 0 : yi;
        ctx.j0 = (g.yc == 1) ? 0 : yj;
        ctx.yv = (op >= 2) ? y[t] : 0.0f;
        return ctx;
    }
    __device__ float operator()(const Ctx& ctx, int64_t k) const {
        int64_t per_batch = ni * nj;
        int64_t b = ctx.b0 + k / per_batch;
        int64_t rem = k - (k / per_batch) * per_batch;
        int64_t i = ctx.i0 + rem / nj, j = ctx.j0 + rem % nj;
        int64_t xi = (b * g.r + i) * g.c + j;
        float d = grad[xi];
        switch (op) {
            case 0: return d;
            case 1: return -d;
            case 2: return d * x[xi];
            default: return -(d * x[xi]) / (ctx.yv * ctx.yv);
        }
    }
};

BroadcastGeom make_geom(int batch, int r, int c, int y_batch, int yr, int yc) {
    return BroadcastGeom{batch, r, c, y_batch, yr, yc};
}

// ---------------------------------------------------------------------------------------------
// Scalar ops: 0 add, 1 sub, 2 mul, 3 div, 4 rsub (s - a), 5 rdiv (s / a)
// ---------------------------------------------------------------------------------------------

__global__ void k_scalar_forward(int op, const float* a, float s, float* out, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float v = a[i];
        switch (op) {
            case 0: out[i] = v + s; break;
            case 1: out[i] = v - s; break;
            case 2: out[i] = v * s; break;
            case 3: out[i] = v / s; break;
            case 4: out[i] = s - v; break;
            default: out[i] = s / v; break;
        }
    }
}

__global__ void k_scalar_backward(int op, const float* grad, const float* a, float s, float* ga, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float d = grad[i];
        switch (op) {
            case 0: case 1: ga[i] = d; break;
            case 2: ga[i] = d * s; break;
            case 3: ga[i] = d / s; break;
            case 4: ga[i] = -d; break;
            default: { float v = a[i]; ga[i] = -s * d / (v * v); } break;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// RMSNorm over rows of length cols: out = x * inv_rms * gamma, inv_rms = 1 / sqrt(mean(x^2) + eps)
// ---------------------------------------------------------------------------------------------

__global__ void k_rmsnorm_forward(const float* x, const float* gamma, float* out, float* inv_rms, int64_t cols, float eps) {
    int64_t row = blockIdx.x;
    const float* xr = x + row * cols;
    float sq = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) sq += xr[j] * xr[j];
    float total = block_reduce_sum(sq);
    __shared__ float s_irms;
    if (threadIdx.x == 0) {
        s_irms = rsqrtf(total / (float)cols + eps);
        inv_rms[row] = s_irms;
    }
    __syncthreads();
    float irms = s_irms;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) out[row * cols + j] = xr[j] * irms * gamma[j];
}

// dx = irms * (u - x * irms^2 * mean(u * x)),  u = grad * gamma
__global__ void k_rmsnorm_backward_x(const float* grad, const float* x, const float* gamma, const float* inv_rms,
                                     float* gx, int64_t cols) {
    int64_t row = blockIdx.x;
    const float* xr = x + row * cols;
    const float* gr = grad + row * cols;
    float dot = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) dot += gr[j] * gamma[j] * xr[j];
    float total = block_reduce_sum(dot);
    __shared__ float s_factor;
    float irms = inv_rms[row];
    if (threadIdx.x == 0) s_factor = irms * irms * total / (float)cols;
    __syncthreads();
    float factor = s_factor;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        gx[row * cols + j] = irms * (gr[j] * gamma[j] - xr[j] * factor);
    }
}

// dgamma[j] = sum over rows of grad * x * inv_rms
struct RMSNormGradGamma {
    const float* grad;
    const float* x;
    const float* inv_rms;
    int64_t cols;
    struct Ctx { int64_t j; };
    __device__ Ctx context(int64_t t) const { return Ctx{t}; }
    __device__ float operator()(const Ctx& ctx, int64_t row) const {
        int64_t i = row * cols + ctx.j;
        return grad[i] * x[i] * inv_rms[row];
    }
};

// ---------------------------------------------------------------------------------------------
// Fused cross entropy over logits [rows, vocab] with class ids stored as floats.
// ---------------------------------------------------------------------------------------------

__global__ void k_cross_entropy_rows(const float* logits, const float* targets, float* lse, float* row_loss,
                                     float* row_valid, int64_t vocab, int ignore_index) {
    int64_t row = blockIdx.x;
    const float* lr = logits + row * vocab;
    float m = -INFINITY;
    for (int64_t j = threadIdx.x; j < vocab; j += blockDim.x) m = fmaxf(m, lr[j]);
    float row_max = block_reduce_max(m);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    float mx = s_max;
    float se = 0.0f;
    for (int64_t j = threadIdx.x; j < vocab; j += blockDim.x) se += expf(lr[j] - mx);
    float total = block_reduce_sum(se);
    if (threadIdx.x == 0) {
        float l = logf(total) + mx;
        lse[row] = l;
        int t = (int)targets[row];
        if (t == ignore_index) {
            row_loss[row] = 0.0f;
            row_valid[row] = 0.0f;
        } else {
            // An out-of-range id cannot throw on the device; make the loss NaN so it is visible
            row_loss[row] = (t >= 0 && t < vocab) ? l - lr[t] : NAN;
            row_valid[row] = 1.0f;
        }
    }
}

// One block: mean loss over valid rows, in a fixed order.
__global__ void k_cross_entropy_mean(const float* row_loss, const float* row_valid, float* loss, float* count, int64_t rows) {
    float s = 0.0f, n = 0.0f;
    for (int64_t i = threadIdx.x; i < rows; i += blockDim.x) {
        s += row_loss[i];
        n += row_valid[i];
    }
    float total = block_reduce_sum(s);
    __syncthreads();
    float valid = block_reduce_sum(n);
    if (threadIdx.x == 0) {
        count[0] = valid;
        loss[0] = valid > 0.0f ? total / valid : 0.0f;
    }
}

__global__ void k_cross_entropy_backward(const float* logits, const float* targets, const float* lse,
                                         const float* count, const float* dloss, float* grad,
                                         int64_t rows, int64_t vocab, int ignore_index) {
    float scale = count[0] > 0.0f ? dloss[0] / count[0] : 0.0f;
    int64_t n = rows * vocab;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t row = i / vocab, j = i - row * vocab;
        int t = (int)targets[row];
        if (t == ignore_index) {
            grad[i] = 0.0f;
        } else {
            float p = expf(logits[i] - lse[row]);
            grad[i] = scale * (p - (j == t ? 1.0f : 0.0f));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------------------------

__global__ void k_embedding_forward(const float* indices, const float* weight, float* out, int64_t n, int64_t vocab, int64_t dim) {
    int64_t total = n * dim;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t pos = i / dim, d = i - pos * dim;
        int64_t id = (int64_t)indices[pos];
        out[i] = (id >= 0 && id < vocab) ? weight[id * dim + d] : NAN;
    }
}

// grad_weight[unique_ids[u]] = sum of grad rows at positions[starts[u] .. starts[u + 1]), in order
__global__ void k_embedding_backward(const float* grad, const int* positions, const int* unique_ids, const int* starts,
                                     float* grad_weight, int64_t n_unique, int64_t dim) {
    int64_t total = n_unique * dim;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t u = i / dim, d = i - u * dim;
        float acc = 0.0f;
        for (int p = starts[u]; p < starts[u + 1]; ++p) acc += grad[(int64_t)positions[p] * dim + d];
        grad_weight[(int64_t)unique_ids[u] * dim + d] = acc;
    }
}

// ---------------------------------------------------------------------------------------------
// GQA: [B, kv, S, D] -> [B, kv * group, S, D] (query head h reads kv head h / group)
// ---------------------------------------------------------------------------------------------

__global__ void k_repeat_kv_forward(const float* in, float* out, int64_t batch, int64_t kv, int64_t group, int64_t head) {
    int64_t heads = kv * group;
    int64_t total = batch * heads * head;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t e = i % head;
        int64_t h = (i / head) % heads;
        int64_t b = i / (head * heads);
        out[i] = in[(b * kv + h / group) * head + e];
    }
}

__global__ void k_repeat_kv_backward(const float* grad, float* gin, int64_t batch, int64_t kv, int64_t group, int64_t head) {
    int64_t total = batch * kv * head;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t e = i % head;
        int64_t k = (i / head) % kv;
        int64_t b = i / (head * kv);
        float acc = 0.0f;
        for (int64_t g = 0; g < group; ++g) acc += grad[((b * kv + k) * group + g) * head + e];
        gin[i] = acc;
    }
}

// ---------------------------------------------------------------------------------------------
// Sum of squares in double precision: a fixed grid writes one partial per thread (strided, so
// reads are coalesced), then one thread per buffer adds its partials in order.
// ---------------------------------------------------------------------------------------------

constexpr int kSqBlocks = 64;
constexpr int64_t kSqPartials = (int64_t)kSqBlocks * kThreads;

__global__ void k_sum_squares_partials(const float* x, int64_t n, double* partial) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double acc = 0.0;
    for (int64_t i = tid; i < n; i += kSqPartials) acc += (double)x[i] * (double)x[i];
    partial[tid] = acc;
}

__global__ void k_sum_squares_finish(const double* partial, double* out, int buffers) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= buffers) return;
    double acc = 0.0;
    for (int64_t i = 0; i < kSqPartials; ++i) acc += partial[(int64_t)b * kSqPartials + i];
    out[b] = acc;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------------------------

void broadcast_binary_forward(int op, const float* x, const float* y, float* out,
                              int batch, int r, int c, int y_batch, int yr, int yc) {
    int64_t n = (int64_t)batch * r * c;
    if (n == 0) return;
    k_broadcast_forward<<<grid_for(n), kThreads>>>(op, x, y, out, make_geom(batch, r, c, y_batch, yr, yc), n);
    CUDA_CHECK(cudaGetLastError());
}

void broadcast_binary_backward_x(int op, const float* grad, const float* y, float* gx,
                                 int batch, int r, int c, int y_batch, int yr, int yc) {
    int64_t n = (int64_t)batch * r * c;
    if (n == 0) return;
    k_broadcast_backward_x<<<grid_for(n), kThreads>>>(op, grad, y, gx, make_geom(batch, r, c, y_batch, yr, yc), n);
    CUDA_CHECK(cudaGetLastError());
}

void broadcast_binary_backward_y(int op, const float* grad, const float* x, const float* y, float* gy,
                                 int batch, int r, int c, int y_batch, int yr, int yc) {
    BroadcastGeom g = make_geom(batch, r, c, y_batch, yr, yc);
    int64_t ni = (yr == 1) ? r : 1, nj = (yc == 1) ? c : 1;
    int64_t nb = (y_batch == batch) ? 1 : (y_batch > 1 ? batch / y_batch : batch);
    BroadcastGradY term{op, grad, x, y, g, ni, nj};
    reduce_terms(term, (int64_t)y_batch * yr * yc, nb * ni * nj, gy);
}

void scalar_forward(int op, const float* a, float s, float* out, int64_t n) {
    if (n == 0) return;
    k_scalar_forward<<<grid_for(n), kThreads>>>(op, a, s, out, n);
    CUDA_CHECK(cudaGetLastError());
}

void scalar_backward(int op, const float* grad, const float* a, float s, float* ga, int64_t n) {
    if (n == 0) return;
    k_scalar_backward<<<grid_for(n), kThreads>>>(op, grad, a, s, ga, n);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_forward(const float* x, const float* gamma, float* out, float* inv_rms, int64_t rows, int64_t cols, float eps) {
    if (rows == 0) return;
    k_rmsnorm_forward<<<(unsigned)rows, kThreads>>>(x, gamma, out, inv_rms, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_backward(const float* grad, const float* x, const float* gamma, const float* inv_rms,
                      float* gx, float* ggamma, int64_t rows, int64_t cols) {
    if (rows == 0) return;
    if (gx) {
        k_rmsnorm_backward_x<<<(unsigned)rows, kThreads>>>(grad, x, gamma, inv_rms, gx, cols);
        CUDA_CHECK(cudaGetLastError());
    }
    if (ggamma) reduce_terms(RMSNormGradGamma{grad, x, inv_rms, cols}, cols, rows, ggamma);
}

void cross_entropy_forward(const float* logits, const float* targets, float* loss, float* lse, float* count,
                           int64_t rows, int64_t vocab, int ignore_index) {
    DeviceBuffer row_loss((size_t)rows * sizeof(float)), row_valid((size_t)rows * sizeof(float));
    if (rows > 0) {
        k_cross_entropy_rows<<<(unsigned)rows, kThreads>>>(logits, targets, lse, row_loss.as<float>(), row_valid.as<float>(),
                                                           vocab, ignore_index);
        CUDA_CHECK(cudaGetLastError());
    }
    k_cross_entropy_mean<<<1, 1024>>>(row_loss.as<float>(), row_valid.as<float>(), loss, count, rows);
    CUDA_CHECK(cudaGetLastError());
}

void cross_entropy_backward(const float* logits, const float* targets, const float* lse, const float* count,
                            const float* dloss, float* grad, int64_t rows, int64_t vocab, int ignore_index) {
    int64_t n = rows * vocab;
    if (n == 0) return;
    k_cross_entropy_backward<<<grid_for(n), kThreads>>>(logits, targets, lse, count, dloss, grad, rows, vocab, ignore_index);
    CUDA_CHECK(cudaGetLastError());
}

void embedding_forward(const float* indices, const float* weight, float* out, int64_t n, int64_t vocab, int64_t dim) {
    if (n * dim == 0) return;
    k_embedding_forward<<<grid_for(n * dim), kThreads>>>(indices, weight, out, n, vocab, dim);
    CUDA_CHECK(cudaGetLastError());
}

void embedding_backward(const float* grad, const std::vector<int>& positions, const std::vector<int>& unique_ids,
                        const std::vector<int>& starts, float* grad_weight, int64_t vocab, int64_t dim) {
    CUDA_CHECK(cudaMemset(grad_weight, 0, (size_t)(vocab * dim) * sizeof(float)));
    if (unique_ids.empty()) return;
    DeviceBuffer d_pos(positions.size() * sizeof(int)), d_ids(unique_ids.size() * sizeof(int)), d_starts(starts.size() * sizeof(int));
    CUDA_CHECK(cudaMemcpy(d_pos.ptr, positions.data(), positions.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids.ptr, unique_ids.data(), unique_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_starts.ptr, starts.data(), starts.size() * sizeof(int), cudaMemcpyHostToDevice));
    int64_t n_unique = (int64_t)unique_ids.size();
    k_embedding_backward<<<grid_for(n_unique * dim), kThreads>>>(grad, d_pos.as<int>(), d_ids.as<int>(), d_starts.as<int>(),
                                                                 grad_weight, n_unique, dim);
    CUDA_CHECK(cudaGetLastError());
}

void repeat_kv_forward(const float* in, float* out, int64_t batch, int64_t kv_heads, int64_t group, int64_t head_elems) {
    int64_t n = batch * kv_heads * group * head_elems;
    if (n == 0) return;
    k_repeat_kv_forward<<<grid_for(n), kThreads>>>(in, out, batch, kv_heads, group, head_elems);
    CUDA_CHECK(cudaGetLastError());
}

void repeat_kv_backward(const float* grad, float* gin, int64_t batch, int64_t kv_heads, int64_t group, int64_t head_elems) {
    int64_t n = batch * kv_heads * head_elems;
    if (n == 0) return;
    k_repeat_kv_backward<<<grid_for(n), kThreads>>>(grad, gin, batch, kv_heads, group, head_elems);
    CUDA_CHECK(cudaGetLastError());
}

double sum_of_squares(const std::vector<const float*>& buffers, const std::vector<int64_t>& sizes) {
    int count = (int)buffers.size();
    if (count == 0) return 0.0;
    DeviceBuffer partial((size_t)(count * kSqPartials) * sizeof(double)), sums((size_t)count * sizeof(double));
    for (int b = 0; b < count; ++b) {
        k_sum_squares_partials<<<kSqBlocks, kThreads>>>(buffers[b], sizes[b], partial.as<double>() + (int64_t)b * kSqPartials);
        CUDA_CHECK(cudaGetLastError());
    }
    k_sum_squares_finish<<<(count + kThreads - 1) / kThreads, kThreads>>>(partial.as<double>(), sums.as<double>(), count);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), sums.ptr, count * sizeof(double), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (double v : host) total += v;
    return total;
}

}  // namespace cuda

#endif  // USE_CUDA
