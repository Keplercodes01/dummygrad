// GPU kernels for the ops a training step needs besides matmul and attention: broadcast binary
// ops, scalar ops, RMSNorm, LayerNorm, fused cross entropy, embedding lookup, GQA head repetition, and
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

// Aim for ~128K threads of work, at least 32 terms per chunk, at most 4096 chunks.
int64_t reduction_splits(int64_t n_out, int64_t K) {
    return std::min<int64_t>({std::max<int64_t>(1, K / 32), std::max<int64_t>(1, (131072 + n_out - 1) / n_out), 4096});
}

template <typename Term>
void reduce_terms(const Term& term, int64_t n_out, int64_t K, float* out) {
    if (n_out <= 0) return;
    int64_t splits = reduction_splits(n_out, K);
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
// Broadcast binary ops (NumPy-style): y's shape is aligned to x's rank, and every y dimension is
// 1 or equal to x's. The output has x's shape. Mirrors broadcast_binary in functional.h.
// ---------------------------------------------------------------------------------------------

constexpr int kMaxDims = 8;

struct BroadcastDims {
    int ndim;
    int64_t x_shape[kMaxDims];
    int64_t x_strides[kMaxDims];
    int64_t y_strides[kMaxDims];  // 0 where y broadcasts
    // Dimensions y is broadcast along (y size 1, x size > 1), for reducing dL/dy
    int nred;
    int64_t red_size[kMaxDims];
    int64_t red_stride[kMaxDims];  // x strides of those dimensions
    // y's own shape (aligned) and the matching x strides, for locating y element t in x
    int64_t y_shape[kMaxDims];

    __device__ int64_t y_offset(int64_t x_idx) const {
        int64_t off = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t i = x_idx % x_shape[d];
            x_idx /= x_shape[d];
            off += i * y_strides[d];
        }
        return off;
    }
};

BroadcastDims make_dims(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
    if (x_shape.size() != y_shape.size() || x_shape.size() > (size_t)kMaxDims || x_shape.empty()) {
        throw std::runtime_error("cuda broadcast: shapes must be aligned to the same rank (1 to 8 dims)");
    }
    BroadcastDims b{};
    b.ndim = (int)x_shape.size();
    int64_t xs = 1, ys = 1;
    for (int d = b.ndim - 1; d >= 0; --d) {
        b.x_shape[d] = x_shape[d];
        b.y_shape[d] = y_shape[d];
        b.x_strides[d] = xs;
        b.y_strides[d] = (y_shape[d] == 1) ? 0 : ys;
        xs *= x_shape[d];
        ys *= y_shape[d];
    }
    b.nred = 0;
    for (int d = 0; d < b.ndim; ++d) {
        if (y_shape[d] == 1 && x_shape[d] > 1) {
            b.red_size[b.nred] = x_shape[d];
            b.red_stride[b.nred] = b.x_strides[d];
            ++b.nred;
        }
    }
    return b;
}

__device__ __forceinline__ float apply_binary(int op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        default: return a / b;
    }
}

__global__ void k_broadcast_forward(int op, const float* x, const float* y, float* out, BroadcastDims b, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = apply_binary(op, x[i], y[b.y_offset(i)]);
    }
}

__global__ void k_broadcast_backward_x(int op, const float* grad, const float* y, float* gx, BroadcastDims b, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float d = grad[i];
        if (op == 2) d *= y[b.y_offset(i)];
        else if (op == 3) d /= y[b.y_offset(i)];
        gx[i] = d;
    }
}

// dL/dy[t] sums over the x elements that read y element t: those varying only along the
// broadcast dimensions, visited in row-major order.
struct BroadcastGradY {
    int op;
    const float* grad;
    const float* x;
    const float* y;
    BroadcastDims b;

    struct Ctx {
        int64_t base;
        float yv;
    };
    __device__ Ctx context(int64_t t) const {
        int64_t base = 0, rest = t;
        for (int d = b.ndim - 1; d >= 0; --d) {
            int64_t i = rest % b.y_shape[d];
            rest /= b.y_shape[d];
            base += i * b.x_strides[d];  // i is 0 along broadcast dimensions
        }
        return Ctx{base, (op >= 2) ? y[t] : 0.0f};
    }
    __device__ float operator()(const Ctx& ctx, int64_t k) const {
        int64_t off = ctx.base;
        for (int d = b.nred - 1; d >= 0; --d) {
            off += (k % b.red_size[d]) * b.red_stride[d];
            k /= b.red_size[d];
        }
        float g = grad[off];
        switch (op) {
            case 0: return g;
            case 1: return -g;
            case 2: return g * x[off];
            default: return -(g * x[off]) / (ctx.yv * ctx.yv);
        }
    }
};

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
// LayerNorm over rows of length cols: out = (x - mean) * rstd * gamma + beta
// ---------------------------------------------------------------------------------------------

int norm_threads(int64_t cols) { return (int)std::min<int64_t>(kThreads, std::max<int64_t>(32, (cols + 31) / 32 * 32)); }

__global__ void k_layernorm_forward(const float* x, const float* gamma, const float* beta, float* out, float* mean,
                                    float* rstd, int64_t cols, float eps) {
    int64_t row = blockIdx.x;
    const float* xr = x + row * cols;
    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) sum += xr[j];
    float total = block_reduce_sum(sum);
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x == 0) s_mean = total / (float)cols;
    __syncthreads();
    float mu = s_mean;
    // Two-pass variance: sum((x - mean)^2) avoids the cancellation of E[x^2] - mean^2
    float sq = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = xr[j] - mu;
        sq += d * d;
    }
    float var_total = block_reduce_sum(sq);
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(var_total / (float)cols + eps);
        mean[row] = mu;
        rstd[row] = s_rstd;
    }
    __syncthreads();
    float rs = s_rstd;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) out[row * cols + j] = (xr[j] - mu) * rs * gamma[j] + beta[j];
}

// dx = rstd * (u - mean(u) - xhat * mean(u * xhat)),  u = grad * gamma, xhat = (x - mean) * rstd
__global__ void k_layernorm_backward_x(const float* grad, const float* x, const float* gamma, const float* mean,
                                       const float* rstd, float* gx, int64_t cols) {
    int64_t row = blockIdx.x;
    const float* xr = x + row * cols;
    const float* gr = grad + row * cols;
    float mu = mean[row], rs = rstd[row];
    float sum_u = 0.0f, sum_u_xhat = 0.0f;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float u = gr[j] * gamma[j];
        sum_u += u;
        sum_u_xhat += u * (xr[j] - mu) * rs;
    }
    float total_u = block_reduce_sum(sum_u);
    float total_u_xhat = block_reduce_sum(sum_u_xhat);
    __shared__ float s_mean_u, s_mean_u_xhat;
    if (threadIdx.x == 0) {
        s_mean_u = total_u / (float)cols;
        s_mean_u_xhat = total_u_xhat / (float)cols;
    }
    __syncthreads();
    float mean_u = s_mean_u, mean_u_xhat = s_mean_u_xhat;
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float xhat = (xr[j] - mu) * rs;
        gx[row * cols + j] = rs * (gr[j] * gamma[j] - mean_u - xhat * mean_u_xhat);
    }
}

// dgamma[j] = sum over rows of grad * xhat, dbeta[j] = sum over rows of grad. Both come from one
// pass over grad and x, split over rows like reduce_terms; partials for split s are at s * cols.
__global__ void k_layernorm_param_partials(const float* grad, const float* x, const float* mean, const float* rstd,
                                           int64_t rows, int64_t cols, int splits, float* pg, float* pb) {
    int64_t items = cols * splits;
    for (int64_t w = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; w < items; w += (int64_t)gridDim.x * blockDim.x) {
        int64_t j = w % cols;
        int64_t s = w / cols;
        int64_t r0 = rows * s / splits, r1 = rows * (s + 1) / splits;
        float acc_g = 0.0f, acc_b = 0.0f;
        for (int64_t r = r0; r < r1; ++r) {
            float g = grad[r * cols + j];
            acc_g += g * (x[r * cols + j] - mean[r]) * rstd[r];
            acc_b += g;
        }
        if (pg) pg[s * cols + j] = acc_g;
        if (pb) pb[s * cols + j] = acc_b;
    }
}

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

static int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

void broadcast_binary_forward(int op, const float* x, const float* y, float* out,
                              const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
    int64_t n = numel(x_shape);
    if (n == 0) return;
    k_broadcast_forward<<<grid_for(n), kThreads>>>(op, x, y, out, make_dims(x_shape, y_shape), n);
    CUDA_CHECK(cudaGetLastError());
}

void broadcast_binary_backward_x(int op, const float* grad, const float* y, float* gx,
                                 const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
    int64_t n = numel(x_shape);
    if (n == 0) return;
    k_broadcast_backward_x<<<grid_for(n), kThreads>>>(op, grad, y, gx, make_dims(x_shape, y_shape), n);
    CUDA_CHECK(cudaGetLastError());
}

void broadcast_binary_backward_y(int op, const float* grad, const float* x, const float* y, float* gy,
                                 const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
    BroadcastDims b = make_dims(x_shape, y_shape);
    int64_t K = 1;
    for (int d = 0; d < b.nred; ++d) K *= b.red_size[d];
    reduce_terms(BroadcastGradY{op, grad, x, y, b}, numel(y_shape), K, gy);
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

void layernorm_forward(const float* x, const float* gamma, const float* beta, float* out, float* mean,
                       float* rstd, int64_t rows, int64_t cols, float eps) {
    if (rows == 0) return;
    k_layernorm_forward<<<(unsigned)rows, norm_threads(cols)>>>(x, gamma, beta, out, mean, rstd, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward(const float* grad, const float* x, const float* gamma, const float* mean, const float* rstd,
                        float* gx, float* ggamma, float* gbeta, int64_t rows, int64_t cols) {
    if (rows == 0) return;
    if (gx) {
        k_layernorm_backward_x<<<(unsigned)rows, norm_threads(cols)>>>(grad, x, gamma, mean, rstd, gx, cols);
        CUDA_CHECK(cudaGetLastError());
    }
    if (!ggamma && !gbeta) return;
    int64_t splits = reduction_splits(cols, rows);
    if (splits <= 1) {
        k_layernorm_param_partials<<<grid_for(cols), kThreads>>>(grad, x, mean, rstd, rows, cols, 1, ggamma, gbeta);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    DeviceBuffer pg((size_t)(cols * splits) * sizeof(float)), pb((size_t)(cols * splits) * sizeof(float));
    k_layernorm_param_partials<<<grid_for(cols * splits), kThreads>>>(grad, x, mean, rstd, rows, cols, (int)splits,
                                                                      pg.as<float>(), pb.as<float>());
    CUDA_CHECK(cudaGetLastError());
    if (ggamma) {
        k_sum_partials<<<grid_for(cols), kThreads>>>(pg.as<float>(), ggamma, cols, (int)splits);
        CUDA_CHECK(cudaGetLastError());
    }
    if (gbeta) {
        k_sum_partials<<<grid_for(cols), kThreads>>>(pb.as<float>(), gbeta, cols, (int)splits);
        CUDA_CHECK(cudaGetLastError());
    }
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
