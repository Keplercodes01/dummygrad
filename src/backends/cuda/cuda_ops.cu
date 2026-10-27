#include "cuda_ops.cuh"
#include "cuda_common.h"
#include "cuda_reduce.cuh"
#include "cuda_dtype.cuh"
#include "profiler.h"
#include "allocator.h"
#include <cublasLt.h>
#include <atomic>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef USE_CUDA

namespace cuda {

// -------------------------------------------------------------
// CUDA Kernels
// -------------------------------------------------------------

template <typename T>
__global__ void k_fill(T* data, float val, int64_t size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = from_float<T>(val);
}

// Half-precision sums are formed in fp32 and rounded once
template <typename T>
__global__ void k_add_inplace(T* dst, const T* src, int64_t size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] = from_float<T>(to_float(dst[idx]) + to_float(src[idx]));
}

template <typename In, typename Out>
__global__ void k_convert(const In* in, Out* out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = from_float<Out>(to_float(in[idx]));
}

__global__ void k_relu_forward(const float* in, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
}

__global__ void k_relu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad_in[idx] = in[idx] > 0.0f ? grad_out[idx] : 0.0f;
}

__global__ void k_gelu_forward(const float* in, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void k_gelu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
        float g = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
        grad_in[idx] = g * grad_out[idx];
    }
}

__global__ void k_sigmoid_forward(const float* in, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

__global__ void k_sigmoid_backward(const float* out, const float* grad_out, float* grad_in, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float s = out[idx];
        grad_in[idx] = s * (1.0f - s) * grad_out[idx];
    }
}

__global__ void k_silu_forward(const float* in, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void k_silu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        float s = 1.0f / (1.0f + expf(-x));
        grad_in[idx] = (s * (1.0f + x * (1.0f - s))) * grad_out[idx];
    }
}

__global__ void k_leaky_relu_forward(const float* in, float* out, int64_t size, float negative_slope) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        out[idx] = x > 0.0f ? x : negative_slope * x;
    }
}

__global__ void k_leaky_relu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size, float negative_slope) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = in[idx] > 0.0f ? grad_out[idx] : negative_slope * grad_out[idx];
    }
}

__global__ void k_tanh_forward(const float* in, float* out, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = tanhf(in[idx]);
}

__global__ void k_tanh_backward(const float* out, const float* grad_out, float* grad_in, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = out[idx];
        grad_in[idx] = (1.0f - t * t) * grad_out[idx];
    }
}

__global__ void k_causal_mask(const float* in, float* out, int total_rows, int seq_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)total_rows * seq_len;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        out[idx] = (col > row) ? -1e9f : in[idx];
    }
}

__global__ void k_causal_mask_backward(const float* grad_out, float* grad_in, int total_rows, int seq_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)total_rows * seq_len;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        grad_in[idx] = (col <= row) ? grad_out[idx] : 0.0f;
    }
}

__global__ void k_sliding_window_mask(const float* in, float* out, int total_rows, int seq_len, int window_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)total_rows * seq_len;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        bool valid = (col <= row) && (row - col <= window_size);
        out[idx] = valid ? in[idx] : -1e9f;
    }
}

__global__ void k_sliding_window_mask_backward(const float* grad_out, float* grad_in, int total_rows, int seq_len, int window_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)total_rows * seq_len;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        bool valid = (col <= row) && (row - col <= window_size);
        grad_in[idx] = valid ? grad_out[idx] : 0.0f;
    }
}

__global__ void k_prefix_causal_mask(const float* in, float* out, int total_rows, int seq_len, int prefix_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)total_rows * seq_len;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        bool valid = (row < prefix_len) ? (col < prefix_len) : (col <= row);
        out[idx] = valid ? in[idx] : -1e9f;
    }
}

__global__ void k_prefix_causal_mask_backward(const float* grad_out, float* grad_in, int total_rows, int seq_len, int prefix_len) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)total_rows * seq_len;
    if (idx < total_elements) {
        int col = idx % seq_len;
        int row = (idx / seq_len) % seq_len;
        bool valid = (row < prefix_len) ? (col < prefix_len) : (col <= row);
        grad_in[idx] = valid ? grad_out[idx] : 0.0f;
    }
}

// Frontier Tiled FlashAttention forward kernel (online softmax, O(1) intermediate GPU memory)
// Each thread owns head-dim indices d = threadIdx.x + s * blockDim.x. Launches use
// >= FA_MIN_THREADS threads, so a thread needs at most FA_MAX_DK / FA_MIN_THREADS slots.
constexpr int FA_MAX_DK = 512;
constexpr int FA_MIN_THREADS = 32;
constexpr int FA_SLOTS = FA_MAX_DK / FA_MIN_THREADS;

__global__ void k_flash_attention_forward(const float* __restrict__ q,
                                          const float* __restrict__ k,
                                          const float* __restrict__ v,
                                          float* __restrict__ out,
                                          int batch_size, int num_heads, int seq_len, int d_k,
                                          bool causal, float scale) {
    int bh = blockIdx.x; // index in [0, batch_size * num_heads)
    int i = blockIdx.y;  // query token index in [0, seq_len)

    if (bh >= batch_size * num_heads || i >= seq_len) return;

    int64_t head_stride = (int64_t)seq_len * d_k;
    const float* q_row = q + bh * head_stride + (int64_t)i * d_k;
    const float* k_head = k + bh * head_stride;
    const float* v_head = v + bh * head_stride;
    float* out_row = out + bh * head_stride + (int64_t)i * d_k;

    extern __shared__ float s_q[];
    for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
        s_q[d] = q_row[d];
    }
    __syncthreads();

    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_o[FA_SLOTS];
    #pragma unroll
    for (int s = 0; s < FA_SLOTS; ++s) acc_o[s] = 0.0f;

    int max_j = causal ? (i + 1) : seq_len;

    for (int j = 0; j < max_j; ++j) {
        const float* k_j = k_head + (int64_t)j * d_k;
        const float* v_j = v_head + (int64_t)j * d_k;

        float score = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            score += s_q[d] * k_j[d];
        }
        score *= scale;

        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float beta = expf(score - m_new);

        l_i = l_i * alpha + beta;

        for (int d = threadIdx.x, s = 0; d < d_k; d += blockDim.x, ++s) {
            acc_o[s] = acc_o[s] * alpha + beta * v_j[d];
        }

        m_i = m_new;
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = threadIdx.x, s = 0; d < d_k; d += blockDim.x, ++s) {
        out_row[d] = acc_o[s] * inv_l;
    }
}

// Frontier Tiled FlashAttention backward kernel (online stats, O(1) intermediate memory)
__global__ void k_flash_attention_backward(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const float* __restrict__ out,
                                           const float* __restrict__ grad_out,
                                           float* __restrict__ grad_q,
                                           float* __restrict__ grad_k,
                                           float* __restrict__ grad_v,
                                           int batch_size, int num_heads, int seq_len, int d_k,
                                           bool causal, float scale) {
    int bh = blockIdx.x;
    int i = blockIdx.y;

    if (bh >= batch_size * num_heads || i >= seq_len) return;

    int64_t head_stride = (int64_t)seq_len * d_k;
    const float* q_row = q + bh * head_stride + (int64_t)i * d_k;
    const float* o_row = out + bh * head_stride + (int64_t)i * d_k;
    const float* do_row = grad_out + bh * head_stride + (int64_t)i * d_k;
    const float* k_head = k + bh * head_stride;
    const float* v_head = v + bh * head_stride;

    float* dq_row = grad_q + bh * head_stride + (int64_t)i * d_k;
    float* dk_head = grad_k + bh * head_stride;
    float* dv_head = grad_v + bh * head_stride;

    extern __shared__ float s_mem_bwd[];
    float* s_q = s_mem_bwd;
    float* s_do = s_mem_bwd + d_k;

    for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
        s_q[d] = q_row[d];
        s_do[d] = do_row[d];
    }
    __syncthreads();

    // 1. Compute Di = sum_d (dO_i,d * O_i,d)
    float d_i = 0.0f;
    for (int d = 0; d < d_k; ++d) {
        d_i += s_do[d] * o_row[d];
    }

    // 2. Recompute log-sum-exp (L_i) for query token i
    float m_i = -1e20f;
    float l_i = 0.0f;
    int max_j = causal ? (i + 1) : seq_len;

    for (int j = 0; j < max_j; ++j) {
        const float* k_j = k_head + (int64_t)j * d_k;
        float score = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            score += s_q[d] * k_j[d];
        }
        score *= scale;

        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float beta = expf(score - m_new);
        l_i = l_i * alpha + beta;
        m_i = m_new;
    }
    float l_inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

    // 3. Compute gradients dQ_i and accumulate dK_j, dV_j
    float acc_dq[FA_SLOTS];
    #pragma unroll
    for (int s = 0; s < FA_SLOTS; ++s) acc_dq[s] = 0.0f;

    for (int j = 0; j < max_j; ++j) {
        const float* k_j = k_head + (int64_t)j * d_k;
        const float* v_j = v_head + (int64_t)j * d_k;
        float* dk_j = dk_head + (int64_t)j * d_k;
        float* dv_j = dv_head + (int64_t)j * d_k;

        float score = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            score += s_q[d] * k_j[d];
        }
        score *= scale;

        float p_ij = expf(score - m_i) * l_inv;

        float dp_ij = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            dp_ij += s_do[d] * v_j[d];
        }

        float ds_ij = p_ij * (dp_ij - d_i) * scale;

        for (int d = threadIdx.x, s = 0; d < d_k; d += blockDim.x, ++s) {
            acc_dq[s] += ds_ij * k_j[d];
            atomicAdd(&dk_j[d], ds_ij * s_q[d]);
            atomicAdd(&dv_j[d], p_ij * s_do[d]);
        }
    }

    for (int d = threadIdx.x, s = 0; d < d_k; d += blockDim.x, ++s) {
        dq_row[d] = acc_dq[s];
    }
}

// Fused Bias and Activations
__global__ void k_add_bias(float* data, const float* bias, int rows, int cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * cols;
    if (idx < total) {
        int c = idx % cols;
        data[idx] += bias[c];
    }
}

__global__ void k_add_bias_and_gelu(const float* in, const float* bias, float* out, int rows, int cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * cols;
    if (idx < total) {
        int c = idx % cols;
        float x = in[idx] + bias[c];
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void k_add_bias_and_relu(const float* in, const float* bias, float* out, int rows, int cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * cols;
    if (idx < total) {
        int c = idx % cols;
        float x = in[idx] + bias[c];
        out[idx] = x > 0.0f ? x : 0.0f;
    }
}

__global__ void k_softmax_forward(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = in + row * cols;
    float* row_out = out + row * cols;

    float max_val = -1e20f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (row_in[i] > max_val) max_val = row_in[i];
    }

    float total_max = block_reduce_max(max_val);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = total_max;
    __syncthreads();
    max_val = s_max;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(row_in[i] - max_val);
        row_out[i] = exp_val;
        sum_exp += exp_val;
    }

    float total_sum = block_reduce_sum(sum_exp);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = total_sum;
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-12f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

__global__ void k_softmax_backward(const float* out, const float* grad_out, float* grad_in, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_out = out + row * cols;
    const float* row_grad = grad_out + row * cols;
    float* row_in = grad_in + row * cols;

    float sum_dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum_dot += row_grad[i] * row_out[i];
    }

    float total_dot = block_reduce_sum(sum_dot);
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = total_dot;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_in[i] = row_out[i] * (row_grad[i] - s_dot);
    }
}

__global__ void k_adamw_step(float* param, const float* grad, float* m, float* v,
                             float lr, float beta1, float beta2, float eps, float weight_decay,
                             float b1_corr, float b2_corr, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float p = param[idx];
        float g = grad[idx];

        // Weight decay
        p -= lr * weight_decay * p;

        // Moments update
        float m_t = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_t = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = m_t;
        v[idx] = v_t;

        // Bias correction
        float m_hat = m_t / b1_corr;
        float v_hat = v_t / b2_corr;

        // Parameter update
        param[idx] = p - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// -------------------------------------------------------------
// Host Dispatch Wrappers
// -------------------------------------------------------------

void fill(Tensor& tensor, float value) {
    int64_t size = tensor.size();
    if (size == 0) return;
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    DG_CUDA_DISPATCH_FLOAT(tensor.dtype, "cuda::fill", {
        k_fill<scalar_t><<<blocks, threads>>>(static_cast<scalar_t*>(tensor.data_ptr<void>()), value, size);
    });
    CUDA_CHECK(cudaGetLastError());
}

void add_inplace(Tensor& dst, const Tensor& src) {
    if (dst.dtype != src.dtype) throw std::runtime_error("cuda::add_inplace requires identical dtypes");
    if (dst.size() != src.size()) throw std::runtime_error("cuda::add_inplace requires identical sizes");
    int64_t size = dst.size();
    if (size == 0) return;
    int threads = 256;
    int blocks = (int)((size + threads - 1) / threads);
    DG_CUDA_DISPATCH_FLOAT(dst.dtype, "cuda::add_inplace", {
        k_add_inplace<scalar_t><<<blocks, threads>>>(static_cast<scalar_t*>(dst.data_ptr<void>()),
                                                     static_cast<const scalar_t*>(src.data_ptr<void>()), size);
    });
    CUDA_CHECK(cudaGetLastError());
}

void convert(const void* in, DType in_dtype, void* out, DType out_dtype, int64_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    DG_CUDA_DISPATCH_FLOAT(in_dtype, "cuda::convert", {
        using In = scalar_t;
        DG_CUDA_DISPATCH_FLOAT(out_dtype, "cuda::convert", {
            k_convert<In, scalar_t><<<blocks, threads>>>(static_cast<const In*>(in), static_cast<scalar_t*>(out), n);
        });
    });
    CUDA_CHECK(cudaGetLastError());
}

void copy_from_vector(Tensor& tensor, const std::vector<float>& values) {
    CUDA_CHECK(cudaMemcpy(tensor.data_ptr<float>(), values.data(),
                          values.size() * sizeof(float), cudaMemcpyHostToDevice));
}

struct TensorLayout {
    int ndim;
    int64_t shape[8];
    int64_t strides[8];
};

// Pure data movement, so T is just an element of the right width (4/2/1 bytes)
template <typename T>
__global__ void k_make_contiguous(T* dst, const T* src, TensorLayout layout, int64_t total_elements) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t temp = idx;
    int64_t src_offset = 0;
    for (int d = layout.ndim - 1; d >= 0; --d) {
        int64_t coord = temp % layout.shape[d];
        temp /= layout.shape[d];
        src_offset += coord * layout.strides[d];
    }
    dst[idx] = src[src_offset];
}

__global__ void k_rope_forward(const float* in, const float* cos_freqs, const float* sin_freqs,
                               float* out, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos) {
    int64_t total = (int64_t)batch_size * seq_len * n_heads * (head_dim / 2);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int half_dim = head_dim / 2;
    int j = idx % half_dim;
    int64_t rem = idx / half_dim;
    int h = rem % n_heads;
    rem /= n_heads;
    int s = rem % seq_len;
    int b = rem / seq_len;

    int pos = start_pos + s;
    float c = cos_freqs[pos * half_dim + j];
    float s_val = sin_freqs[pos * half_dim + j];

    int64_t base = ((int64_t)(b * seq_len + s) * n_heads + h) * head_dim;
    float x0 = in[base + 2 * j];
    float x1 = in[base + 2 * j + 1];

    out[base + 2 * j]     = x0 * c - x1 * s_val;
    out[base + 2 * j + 1] = x0 * s_val + x1 * c;
}

__global__ void k_rope_backward(const float* grad_out, const float* cos_freqs, const float* sin_freqs,
                                float* grad_in, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos) {
    int64_t total = (int64_t)batch_size * seq_len * n_heads * (head_dim / 2);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int half_dim = head_dim / 2;
    int j = idx % half_dim;
    int64_t rem = idx / half_dim;
    int h = rem % n_heads;
    rem /= n_heads;
    int s = rem % seq_len;
    int b = rem / seq_len;

    int pos = start_pos + s;
    float c = cos_freqs[pos * half_dim + j];
    float s_val = sin_freqs[pos * half_dim + j];

    int64_t base = ((int64_t)(b * seq_len + s) * n_heads + h) * head_dim;
    float d0 = grad_out[base + 2 * j];
    float d1 = grad_out[base + 2 * j + 1];

    grad_in[base + 2 * j]     = d0 * c + d1 * s_val;
    grad_in[base + 2 * j + 1] = -d0 * s_val + d1 * c;
}

template <typename T>
static void make_contiguous_typed(Tensor& dst, const Tensor& src) {
    int ndim = src.ndim();
    int64_t total = src.size();
    if (ndim <= 8) {
        TensorLayout layout;
        layout.ndim = ndim;
        for (int i = 0; i < ndim; ++i) {
            layout.shape[i] = src.shape[i];
            layout.strides[i] = src.strides[i];
        }
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        k_make_contiguous<T><<<blocks, threads>>>(dst.data_ptr<T>(), src.data_ptr<T>(), layout, total);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // CPU fallback for > 8 dimensions. Strided views can reach anywhere in the storage,
        // so stage from the view's base offset up to its furthest element.
        int64_t span = 1;
        for (int d = 0; d < ndim; d++) span += (src.shape[d] - 1) * src.strides[d];
        std::vector<T> host_src(span);
        std::vector<T> host_dst(total);
        CUDA_CHECK(cudaMemcpy(host_src.data(), src.data_ptr<T>(), span * sizeof(T), cudaMemcpyDeviceToHost));
        std::vector<int64_t> coords(ndim, 0);
        for (int64_t i = 0; i < total; i++) {
            int64_t src_idx = 0;
            for (int d = 0; d < ndim; d++) src_idx += coords[d] * src.strides[d];
            host_dst[i] = host_src[src_idx];
            for (int d = ndim - 1; d >= 0; d--) {
                coords[d]++;
                if (coords[d] < src.shape[d]) break;
                coords[d] = 0;
            }
        }
        CUDA_CHECK(cudaMemcpy(dst.data_ptr<T>(), host_dst.data(), total * sizeof(T), cudaMemcpyHostToDevice));
    }
}

void make_contiguous(Tensor& dst, const Tensor& src) {
    if (dst.dtype != src.dtype) throw std::runtime_error("cuda::make_contiguous requires identical dtypes");
    if (dst.size() != src.size()) throw std::runtime_error("cuda::make_contiguous requires identical sizes");
    int64_t total = src.size();
    if (total == 0) return;
    size_t elem = dtype_size(src.dtype);

    if (src.is_contiguous()) {
        CUDA_CHECK(cudaMemcpy(dst.data_ptr<void>(), src.data_ptr<void>(), total * elem, cudaMemcpyDeviceToDevice));
        return;
    }
    switch (elem) {
        case 4: make_contiguous_typed<uint32_t>(dst, src); break;
        case 2: make_contiguous_typed<uint16_t>(dst, src); break;
        case 1: make_contiguous_typed<uint8_t>(dst, src); break;
        default: throw std::runtime_error("cuda::make_contiguous: unsupported element size");
    }
}

void rope_forward(const float* in, const float* cos_freqs, const float* sin_freqs,
                  float* out, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos) {
    int64_t total = (int64_t)batch_size * seq_len * n_heads * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_rope_forward<<<blocks, threads>>>(in, cos_freqs, sin_freqs, out, batch_size, seq_len, n_heads, head_dim, start_pos);
    CUDA_CHECK(cudaGetLastError());
}

void rope_backward(const float* grad_out, const float* cos_freqs, const float* sin_freqs,
                   float* grad_in, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos) {
    int64_t total = (int64_t)batch_size * seq_len * n_heads * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_rope_backward<<<blocks, threads>>>(grad_out, cos_freqs, sin_freqs, grad_in, batch_size, seq_len, n_heads, head_dim, start_pos);
    CUDA_CHECK(cudaGetLastError());
}

void relu_forward(const float* in, float* out, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_relu_forward<<<blocks, threads>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_relu_backward<<<blocks, threads>>>(in, grad_out, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_forward(const float* in, float* out, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_gelu_forward<<<blocks, threads>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_gelu_backward<<<blocks, threads>>>(in, grad_out, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_forward(const float* in, float* out, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_sigmoid_forward<<<blocks, threads>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward(const float* out, const float* grad_out, float* grad_in, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_sigmoid_backward<<<blocks, threads>>>(out, grad_out, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}

void silu_forward(const float* in, float* out, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_silu_forward<<<blocks, threads>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
}

void silu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_silu_backward<<<blocks, threads>>>(in, grad_out, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu_forward(const float* in, float* out, int64_t size, float negative_slope) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_leaky_relu_forward<<<blocks, threads>>>(in, out, size, negative_slope);
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size, float negative_slope) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_leaky_relu_backward<<<blocks, threads>>>(in, grad_out, grad_in, size, negative_slope);
    CUDA_CHECK(cudaGetLastError());
}

void tanh_forward(const float* in, float* out, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_tanh_forward<<<blocks, threads>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
}

void tanh_backward(const float* out, const float* grad_out, float* grad_in, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_tanh_backward<<<blocks, threads>>>(out, grad_out, grad_in, size);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_forward(const float* in, float* out, int rows, int cols) {
    int threads = std::min(256, ((cols + 31) / 32) * 32);
    k_softmax_forward<<<rows, threads>>>(in, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_backward(const float* out, const float* grad_out, float* grad_in, int rows, int cols) {
    int threads = std::min(256, ((cols + 31) / 32) * 32);
    k_softmax_backward<<<rows, threads>>>(out, grad_out, grad_in, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void causal_mask(const float* in, float* out, int batch, int seq_len) {
    int64_t total = (int64_t)batch * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_causal_mask<<<blocks, threads>>>(in, out, batch * seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());
}

void causal_mask_backward(const float* grad_out, float* grad_in, int batch, int seq_len) {
    int64_t total = (int64_t)batch * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_causal_mask_backward<<<blocks, threads>>>(grad_out, grad_in, batch * seq_len, seq_len);
    CUDA_CHECK(cudaGetLastError());
}

void sliding_window_mask(const float* in, float* out, int batch, int seq_len, int window_size) {
    int64_t total = (int64_t)batch * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_sliding_window_mask<<<blocks, threads>>>(in, out, batch * seq_len, seq_len, window_size);
    CUDA_CHECK(cudaGetLastError());
}

void sliding_window_mask_backward(const float* grad_out, float* grad_in, int batch, int seq_len, int window_size) {
    int64_t total = (int64_t)batch * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_sliding_window_mask_backward<<<blocks, threads>>>(grad_out, grad_in, batch * seq_len, seq_len, window_size);
    CUDA_CHECK(cudaGetLastError());
}

void prefix_causal_mask(const float* in, float* out, int batch, int seq_len, int prefix_len) {
    int64_t total = (int64_t)batch * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_prefix_causal_mask<<<blocks, threads>>>(in, out, batch * seq_len, seq_len, prefix_len);
    CUDA_CHECK(cudaGetLastError());
}

void prefix_causal_mask_backward(const float* grad_out, float* grad_in, int batch, int seq_len, int prefix_len) {
    int64_t total = (int64_t)batch * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_prefix_causal_mask_backward<<<blocks, threads>>>(grad_out, grad_in, batch * seq_len, seq_len, prefix_len);
    CUDA_CHECK(cudaGetLastError());
}

void flash_attention_forward(const float* q, const float* k, const float* v,
                             float* out,
                             int batch_size, int num_heads, int seq_len, int d_k,
                             bool causal, float scale) {
    if (d_k > FA_MAX_DK) throw std::runtime_error("flash_attention_forward: head dim " + std::to_string(d_k) + " exceeds " + std::to_string(FA_MAX_DK));
    dim3 grid(batch_size * num_heads, seq_len);
    int threads = std::min(128, d_k);
    if (threads < FA_MIN_THREADS) threads = FA_MIN_THREADS;
    size_t smem = d_k * sizeof(float);
    k_flash_attention_forward<<<grid, threads, smem>>>(q, k, v, out,
                                                       batch_size, num_heads, seq_len, d_k,
                                                       causal, scale);
    CUDA_CHECK(cudaGetLastError());
}

void flash_attention_backward(const float* q, const float* k, const float* v,
                              const float* out, const float* grad_out,
                              float* grad_q, float* grad_k, float* grad_v,
                              int batch_size, int num_heads, int seq_len, int d_k,
                              bool causal, float scale) {
    if (d_k > FA_MAX_DK) throw std::runtime_error("flash_attention_backward: head dim " + std::to_string(d_k) + " exceeds " + std::to_string(FA_MAX_DK));
    int64_t head_stride = (int64_t)seq_len * d_k;
    int64_t total_elements = (int64_t)batch_size * num_heads * head_stride;
    CUDA_CHECK(cudaMemset(grad_k, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_v, 0, total_elements * sizeof(float)));

    dim3 grid(batch_size * num_heads, seq_len);
    int threads = std::min(128, d_k);
    if (threads < FA_MIN_THREADS) threads = FA_MIN_THREADS;
    size_t smem = 2 * d_k * sizeof(float);
    k_flash_attention_backward<<<grid, threads, smem>>>(q, k, v, out, grad_out,
                                                        grad_q, grad_k, grad_v,
                                                        batch_size, num_heads, seq_len, d_k,
                                                        causal, scale);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias(float* data, const float* bias, int rows, int cols) {
    int64_t total = (int64_t)rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_add_bias<<<blocks, threads>>>(data, bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias_and_gelu(const float* in, const float* bias, float* out, int rows, int cols) {
    int64_t total = (int64_t)rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_add_bias_and_gelu<<<blocks, threads>>>(in, bias, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias_and_relu(const float* in, const float* bias, float* out, int rows, int cols) {
    int64_t total = (int64_t)rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    k_add_bias_and_relu<<<blocks, threads>>>(in, bias, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void adamw_step(float* param, const float* grad, float* m, float* v,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                int step, int64_t size) {
    float b1_corr = 1.0f - std::pow(beta1, (float)step);
    float b2_corr = 1.0f - std::pow(beta2, (float)step);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_adamw_step<<<blocks, threads>>>(param, grad, m, v, lr, beta1, beta2, eps, weight_decay,
                                      b1_corr, b2_corr, size);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------------------------------------------------
// cuBLAS Matrix Multiplication (Row-Major via Column-Major Trick)
// fp32 tensors in and out. The compute dtype (from AutocastGuard) picks the GEMM: Float32 runs
// SGEMM (TF32 tensor cores where the GPU has them); Float16/BFloat16 round the operands to that
// type and run on tensor cores, accumulating and writing fp32.
// -------------------------------------------------------------

void cast_from_float(const float* in, void* out, int64_t n, DType dtype) {
    DG_PROFILE_OP("autocast_cast");
    if (dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("cuda::cast_from_float: target dtype must be Float16 or BFloat16");
    }
    convert(in, DType::Float32, out, dtype, n);
}

static void check_compute_dtype(DType compute, const char* who) {
    if (compute != DType::Float32 && compute != DType::Float16 && compute != DType::BFloat16) {
        throw std::runtime_error(std::string(who) + ": compute dtype must be Float32, Float16 or BFloat16");
    }
}

// A contiguous tensor as a GEMM operand of the compute dtype: itself when it already has that
// dtype, else a converted copy (rounded to nearest even when narrowing)
static std::shared_ptr<Tensor> gemm_operand(const std::shared_ptr<Tensor>& t, DType compute) {
    if (t->dtype == compute) return t;
    DG_PROFILE_OP("autocast_cast");
    auto out = std::make_shared<Tensor>(t->shape, Device::CUDA, compute, false);
    convert(t->data_ptr<void>(), t->dtype, out->data_ptr<void>(), compute, t->size());
    return out;
}

// Column-major C = op_a(A) op_b(B) + beta * C with alpha = 1 (see gemm_strided_batched)
static void gemm(cublasOperation_t op_a, cublasOperation_t op_b, int m, int n, int k,
                 const void* A, int lda, long long stride_a, const void* B, int ldb, long long stride_b,
                 float beta, float* C, int ldc, long long stride_c, int batch, DType compute) {
    gemm_strided_batched(op_a, op_b, m, n, k, 1.0f, A, lda, stride_a, B, ldb, stride_b, beta, C, ldc, stride_c,
                         batch, compute);
}

// ---- cuBLASLt GEMM with the bias fused into its epilogue (Linear) ----
// Column-major D (m x n, ld m) = A (m x k, ld m) B (k x n, ld k) + bias[m] added to every column,
// with no transposes. The row-major Linear out[M, N] = x[M, K] W[K, N] + b maps onto it with
// m = N, n = M, A = W, B = x, so the bias runs along D's rows. Plans (descriptors and the
// heuristic's algorithm) are cached per shape and dtype, so a call only sets the bias pointer.

struct LtKey {
    int m, n, k, dtype;
    bool operator==(const LtKey& o) const { return m == o.m && n == o.n && k == o.k && dtype == o.dtype; }
};
struct LtKeyHash {
    size_t operator()(const LtKey& key) const {
        size_t h = (size_t)key.m;
        for (int v : {key.n, key.k, key.dtype}) h = h * 1000003u ^ (size_t)v;
        return h;
    }
};
struct LtPlan {
    bool ok = false;  // false: cuBLASLt has no algorithm for this case; the caller falls back
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo{};
};
struct LtContext {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::unordered_map<LtKey, LtPlan, LtKeyHash> plans;
};
constexpr size_t kLtWorkspaceBytes = 4u << 20;

// Per thread and device, like get_cublas_handle: a plan's descriptor is modified on every call
static LtContext& lt_context() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    static thread_local std::unordered_map<int, LtContext> contexts;
    LtContext& ctx = contexts[dev];
    if (!ctx.handle) {
        CUBLAS_CHECK(cublasLtCreate(&ctx.handle));
        ctx.workspace = get_memory(Device::CUDA, kLtWorkspaceBytes);
    }
    return ctx;
}

static LtPlan& lt_bias_plan(LtContext& ctx, int m, int n, int k, DType compute) {
    LtKey key{m, n, k, (int)compute};
    auto it = ctx.plans.find(key);
    if (it != ctx.plans.end()) return it->second;
    LtPlan& p = ctx.plans[key];

    cudaDataType_t in_type = compute == DType::Float16 ? CUDA_R_16F : compute == DType::BFloat16 ? CUDA_R_16BF : CUDA_R_32F;
    // fp32 inputs use TF32 tensor cores where the GPU has them, like the SGEMM path's math mode
    cublasComputeType_t compute_type = compute == DType::Float32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.op, compute_type, CUDA_R_32F));
    cublasOperation_t no_trans = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans)));
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, in_type, m, k, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, in_type, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.d, CUDA_R_32F, m, n, m));

    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace = kLtWorkspaceBytes;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                                      sizeof(workspace)));
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(ctx.handle, p.op, p.a, p.b, p.d, p.d, pref, 1, &result, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status == CUBLAS_STATUS_SUCCESS && found > 0) {
        p.algo = result.algo;
        p.ok = true;
    }
    return p;
}

static std::atomic<int64_t> g_linear_unfused{0};
int64_t linear_unfused_count() { return g_linear_unfused; }

static bool aligned16(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0; }

// D = A B + bias as described above. Returns false (and does nothing) when cuBLASLt cannot run the
// case: no algorithm for the shape, or a pointer that is not 16-byte aligned (an offset view).
static bool gemm_bias(int m, int n, int k, const void* A, const void* B, const float* bias, float* D, DType compute) {
    if (!aligned16(A) || !aligned16(B) || !aligned16(bias) || !aligned16(D)) return false;
    LtContext& ctx = lt_context();
    LtPlan& p = lt_bias_plan(ctx, m, n, k, compute);
    if (!p.ok) return false;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(ctx.handle, p.op, &alpha, A, p.a, B, p.b, &beta, D, p.d, D, p.d, &p.algo,
                                ctx.workspace, kLtWorkspaceBytes, 0));
    return true;
}

static void check_matmul_inputs(const Tensor& a, const Tensor& b, const char* who) {
    if (!is_float_dtype(a.dtype) || !is_float_dtype(b.dtype)) {
        throw std::runtime_error(std::string(who) + ": tensors must be Float32, Float16 or BFloat16");
    }
}

// out = a @ b (+ bias over the last dimension); bias requires b to be a single 2-D matrix
static std::shared_ptr<Tensor> matmul_impl(const std::shared_ptr<Tensor>& _a, const std::shared_ptr<Tensor>& _b,
                                           const std::shared_ptr<Tensor>& bias, DType compute,
                                           std::shared_ptr<Tensor>* saved_a, std::shared_ptr<Tensor>* saved_b) {
    check_matmul_inputs(*_a, *_b, "cuda::matmul");
    check_compute_dtype(compute, "cuda::matmul");
    auto a = make_contiguous(_a);
    auto b = make_contiguous(_b);
    int n1 = a->ndim();
    int n2 = b->ndim();
    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    if (c1 != r2) throw std::runtime_error("cuda::matmul: inner dimensions mismatch");

    int batch_size_a = a->size() / (r1 * c1);
    int batch_size_b = b->size() / (r2 * c2);
    int batch_size = std::max(batch_size_a, batch_size_b);
    // Strided-batched GEMM can only broadcast a whole operand (batch 1), not individual dims
    if (batch_size_a != batch_size_b && batch_size_a != 1 && batch_size_b != 1)
        throw std::runtime_error("cuda::matmul: partial batch broadcasting is not supported");

    std::vector<int64_t> out_shape = (batch_size_a >= batch_size_b) ? a->shape : b->shape;
    out_shape[out_shape.size() - 2] = r1;
    out_shape[out_shape.size() - 1] = c2;

    auto out = std::make_shared<Tensor>(out_shape, Device::CUDA, DType::Float32, a->requires_grad || b->requires_grad);

    // A shared weight (b has batch 1) multiplies every row of a: run one GEMM over a viewed as
    // [batch * r1, c1] instead of a batched GEMM. The memory layout is identical.
    if (batch_size_b == 1 && batch_size_a > 1) {
        r1 *= batch_size_a;
        batch_size_a = 1;
        batch_size = 1;
    }

    long long int stride_a = (batch_size_a > 1) ? (long long)r1 * c1 : 0;
    long long int stride_b = (batch_size_b > 1) ? (long long)r2 * c2 : 0;
    long long int stride_c = (long long)r1 * c2;

    auto ah = gemm_operand(a, compute);
    auto bh = gemm_operand(b, compute);
    // Row-major C = A * B -> Col-major C^T = B^T * A^T. With a bias, b is one shared matrix, so
    // the product is a single folded GEMM (batch_size 1) and the bias rides in its epilogue.
    bool fused = bias && batch_size == 1 &&
                 gemm_bias(c2, r1, c1, bh->data_ptr<void>(), ah->data_ptr<void>(), bias->data_ptr<float>(),
                           out->data_ptr<float>(), compute);
    if (!fused) {
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, c2, r1, c1,
             bh->data_ptr<void>(), c2, stride_b, ah->data_ptr<void>(), c1, stride_a,
             0.0f, out->data_ptr<float>(), c2, stride_c, batch_size, compute);
        if (bias) {
            ++g_linear_unfused;
            int64_t rows = out->size() / c2;
            broadcast_binary_forward(0, out->data_ptr<float>(), bias->data_ptr<float>(), out->data_ptr<float>(),
                                     {rows, (int64_t)c2}, {1, (int64_t)c2});
        }
    }
    if (saved_a) *saved_a = std::move(ah);
    if (saved_b) *saved_b = std::move(bh);
    return out;
}

std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b, DType compute,
                               std::shared_ptr<Tensor>* saved_a, std::shared_ptr<Tensor>* saved_b) {
    return matmul_impl(a, b, nullptr, compute, saved_a, saved_b);
}

std::shared_ptr<Tensor> linear(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& w,
                               const std::shared_ptr<Tensor>& bias, DType compute,
                               std::shared_ptr<Tensor>* saved_x, std::shared_ptr<Tensor>* saved_w) {
    if (w->ndim() != 2) throw std::runtime_error("cuda::linear: weight must be 2-D [in, out]");
    if (bias->dtype != DType::Float32 || bias->device != Device::CUDA || bias->size() != w->shape[1]) {
        throw std::runtime_error("cuda::linear: bias must be a Float32 CUDA tensor with one value per output feature");
    }
    return matmul_impl(x, w, make_contiguous(bias), compute, saved_x, saved_w);
}

// A backward GEMM operand: contiguous and of the compute dtype. matmul's saved operands already
// are, so they pass through untouched; other tensors are densified and converted here.
static std::shared_ptr<Tensor> backward_operand(const std::shared_ptr<Tensor>& t, DType compute) {
    if (!t) return nullptr;
    if (!is_float_dtype(t->dtype)) {
        throw std::runtime_error("cuda::matmul_backward: operands must be Float32, Float16 or BFloat16");
    }
    return gemm_operand(make_contiguous(t), compute);
}

static int64_t shape_numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

void matmul_backward(const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
                     const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                     const std::shared_ptr<Tensor>& grad_out,
                     std::shared_ptr<Tensor>* grad_a, std::shared_ptr<Tensor>* grad_b, DType compute) {
    if (!grad_out || (!grad_a && !grad_b)) return;
    check_compute_dtype(compute, "cuda::matmul_backward");
    if (!is_float_dtype(grad_out->dtype)) {
        throw std::runtime_error("cuda::matmul_backward: gradient must be Float32, Float16 or BFloat16");
    }
    if ((grad_a && !b) || (grad_b && !a)) {
        throw std::runtime_error("cuda::matmul_backward: grad_a needs operand b and grad_b needs operand a");
    }
    if (a_shape.size() < 2 || b_shape.size() < 2) throw std::runtime_error("cuda::matmul_backward: operands must be at least 2-D");

    int r1 = a_shape[a_shape.size() - 2];
    int c1 = a_shape[a_shape.size() - 1];
    int r2 = b_shape[b_shape.size() - 2];
    int c2 = b_shape[b_shape.size() - 1];

    int batch_size_a = shape_numel(a_shape) / ((int64_t)r1 * c1);
    int batch_size_b = shape_numel(b_shape) / ((int64_t)r2 * c2);
    int batch_size = std::max(batch_size_a, batch_size_b);
    if (batch_size_a != batch_size_b && batch_size_a != 1 && batch_size_b != 1)
        throw std::runtime_error("cuda::matmul_backward: partial batch broadcasting is not supported");
    if ((a && a->size() != shape_numel(a_shape)) || (b && b->size() != shape_numel(b_shape)) ||
        grad_out->size() != (int64_t)batch_size * r1 * c2) {
        throw std::runtime_error("cuda::matmul_backward: operand sizes do not match the given shapes");
    }

    // No zero fill: every GEMM below writes its output with beta = 0 before anything accumulates
    if (grad_a) *grad_a = std::make_shared<Tensor>(a_shape, Device::CUDA, DType::Float32, false);
    if (grad_b) *grad_b = std::make_shared<Tensor>(b_shape, Device::CUDA, DType::Float32, false);

    // Shared weight: dA = dC @ B^T is one GEMM over [batch * r1] rows, and dB = A^T @ dC sums over
    // the batch inside a single GEMM whose inner dimension spans every row
    if (batch_size_b == 1 && batch_size_a > 1) {
        r1 *= batch_size_a;
        batch_size_a = 1;
        batch_size = 1;
    }

    long long int stride_a = (batch_size_a > 1) ? (long long)r1 * c1 : 0;
    long long int stride_b = (batch_size_b > 1) ? (long long)r2 * c2 : 0;
    long long int stride_g = (long long)r1 * c2;

    // Each operand is rounded at most once: a saved one not at all, dC once for both products
    auto ah = grad_b ? backward_operand(a, compute) : nullptr;
    auto bh = grad_a ? backward_operand(b, compute) : nullptr;
    auto gh = backward_operand(grad_out, compute);
    const char* A = ah ? ah->data_ptr<char>() : nullptr;
    const char* B = bh ? bh->data_ptr<char>() : nullptr;
    const char* G = gh->data_ptr<char>();
    float* dA = grad_a ? (*grad_a)->data_ptr<float>() : nullptr;
    float* dB = grad_b ? (*grad_b)->data_ptr<float>() : nullptr;

    if (batch_size_a == batch_size_b) {
        // dA = dC B^T, dB = A^T dC
        if (dA) gemm(CUBLAS_OP_T, CUBLAS_OP_N, c1, r1, c2, B, c2, stride_b, G, c2, stride_g,
                     0.0f, dA, c1, stride_a, batch_size, compute);
        if (dB) gemm(CUBLAS_OP_N, CUBLAS_OP_T, c2, c1, r1, G, c2, stride_g, A, c1, stride_a,
                     0.0f, dB, c2, stride_b, batch_size, compute);
        return;
    }

    // a has batch 1 and b does not (the reverse was folded above): dA sums every batch's
    // contribution, so the batches run one at a time and accumulate in fp32
    size_t elem = dtype_size(compute);
    for (int batch = 0; batch < batch_size; ++batch) {
        float beta_a = (batch_size_a == 1 && batch > 0) ? 1.0f : 0.0f;
        float beta_b = (batch_size_b == 1 && batch > 0) ? 1.0f : 0.0f;
        int64_t off_a = batch_size_a > 1 ? (int64_t)batch * r1 * c1 : 0;
        int64_t off_b = batch_size_b > 1 ? (int64_t)batch * r2 * c2 : 0;
        int64_t off_g = (int64_t)batch * r1 * c2;
        if (dA) gemm(CUBLAS_OP_T, CUBLAS_OP_N, c1, r1, c2, B + off_b * elem, c2, 0, G + off_g * elem, c2, 0,
                     beta_a, dA + off_a, c1, 0, 1, compute);
        if (dB) gemm(CUBLAS_OP_N, CUBLAS_OP_T, c2, c1, r1, G + off_g * elem, c2, 0, A + off_a * elem, c1, 0,
                     beta_b, dB + off_b, c2, 0, 1, compute);
    }
}

} // namespace cuda

#endif
