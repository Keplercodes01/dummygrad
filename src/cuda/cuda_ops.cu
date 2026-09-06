#include "cuda_ops.cuh"
#include "cuda_common.h"
#include <cmath>
#include <algorithm>

#ifdef USE_CUDA

namespace cuda {

// -------------------------------------------------------------
// CUDA Kernels
// -------------------------------------------------------------

__global__ void k_fill(float* data, float val, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = val;
}

__global__ void k_add_inplace(float* dst, const float* src, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] += src[idx];
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

__global__ void k_layernorm_forward(const float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    float* __restrict__ out,
                                    float* __restrict__ mean_out,
                                    float* __restrict__ rstd_out,
                                    int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + (int64_t)row * cols;
    float* row_out = out + (int64_t)row * cols;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_x[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / cols;
    __syncthreads();
    float mu = s_mean;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_x[i] - mu;
        var_sum += diff * diff;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(var_sum / cols + eps);
        if (mean_out) mean_out[row] = mu;
        if (rstd_out) rstd_out[row] = s_rstd;
    }
    __syncthreads();
    float rstd = s_rstd;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float norm = (row_x[i] - mu) * rstd;
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta ? beta[i] : 0.0f;
        row_out[i] = norm * g + b;
    }
}

__global__ void k_layernorm_backward(const float* __restrict__ grad_out,
                                     const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ mean_in,
                                     const float* __restrict__ rstd_in,
                                     float* __restrict__ grad_x,
                                     float* __restrict__ grad_gamma,
                                     float* __restrict__ grad_beta,
                                     int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_dout = grad_out + (int64_t)row * cols;
    const float* row_x = x + (int64_t)row * cols;
    float* row_dx = grad_x + (int64_t)row * cols;

    float mu = mean_in[row];
    float rstd = rstd_in[row];

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float dy = row_dout[i];
        float xhat = (row_x[i] - mu) * rstd;
        float g = gamma ? gamma[i] : 1.0f;
        float dy_g = dy * g;
        sum_dy += dy_g;
        sum_dy_xhat += dy_g * xhat;

        if (grad_gamma) atomicAdd(&grad_gamma[i], dy * xhat);
        if (grad_beta) atomicAdd(&grad_beta[i], dy);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dy += __shfl_down_sync(0xffffffff, sum_dy, offset);
        sum_dy_xhat += __shfl_down_sync(0xffffffff, sum_dy_xhat, offset);
    }
    __shared__ float s_sum_dy, s_sum_dy_xhat;
    if (threadIdx.x == 0) {
        s_sum_dy = sum_dy;
        s_sum_dy_xhat = sum_dy_xhat;
    }
    __syncthreads();

    float f_cols = (float)cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float dy = row_dout[i];
        float xhat = (row_x[i] - mu) * rstd;
        float g = gamma ? gamma[i] : 1.0f;
        float dy_g = dy * g;
        row_dx[i] = rstd * (dy_g - (s_sum_dy + xhat * s_sum_dy_xhat) / f_cols);
    }
}

// Frontier Tiled FlashAttention forward kernel (online softmax, O(1) intermediate GPU memory)
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
    float acc_o[128];
    #pragma unroll
    for (int d = 0; d < 128; ++d) acc_o[d] = 0.0f;

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

        for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
            acc_o[d] = acc_o[d] * alpha + beta * v_j[d];
        }

        m_i = m_new;
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
        out_row[d] = acc_o[d] * inv_l;
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
    float acc_dq[128];
    #pragma unroll
    for (int d = 0; d < 128; ++d) acc_dq[d] = 0.0f;

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

        for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
            acc_dq[d] += ds_ij * k_j[d];
            atomicAdd(&dk_j[d], ds_ij * s_q[d]);
            atomicAdd(&dv_j[d], p_ij * s_do[d]);
        }
    }

    for (int d = threadIdx.x; d < d_k; d += blockDim.x) {
        dq_row[d] = acc_dq[d];
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

    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(row_in[i] - max_val);
        row_out[i] = exp_val;
        sum_exp += exp_val;
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum_exp;
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

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
    }
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = sum_dot;
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
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_fill<<<blocks, threads>>>(tensor.data_ptr<float>(), value, size);
    CUDA_CHECK(cudaGetLastError());
}

void add_inplace(Tensor& dst, const Tensor& src) {
    int64_t size = dst.size();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    k_add_inplace<<<blocks, threads>>>(dst.data_ptr<float>(), src.data_ptr<float>(), size);
    CUDA_CHECK(cudaGetLastError());
}

void copy_from_vector(Tensor& tensor, const std::vector<float>& values) {
    CUDA_CHECK(cudaMemcpy(tensor.data_ptr<float>(), values.data(),
                          values.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void make_contiguous(Tensor& dst, const Tensor& src) {
    if (src.is_contiguous()) {
        CUDA_CHECK(cudaMemcpy(dst.data_ptr<float>(), src.data_ptr<float>(),
                              src.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    } else {
        // Fallback for non-contiguous: copy to host, make contiguous, copy back
        std::vector<float> host_src(src.size());
        CUDA_CHECK(cudaMemcpy(host_src.data(), src.data_ptr<float>(), src.size() * sizeof(float), cudaMemcpyDeviceToHost));
        // Direct linear copy to destination
        CUDA_CHECK(cudaMemcpy(dst.data_ptr<float>(), host_src.data(), dst.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
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

void layernorm_forward(const float* x, const float* gamma, const float* beta,
                       float* out, float* mean, float* rstd,
                       int rows, int cols, float eps) {
    int threads = std::min(256, ((cols + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    k_layernorm_forward<<<rows, threads>>>(x, gamma, beta, out, mean, rstd, rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward(const float* grad_out, const float* x, const float* gamma,
                        const float* mean, const float* rstd,
                        float* grad_x, float* grad_gamma, float* grad_beta,
                        int rows, int cols) {
    if (grad_gamma) CUDA_CHECK(cudaMemset(grad_gamma, 0, cols * sizeof(float)));
    if (grad_beta) CUDA_CHECK(cudaMemset(grad_beta, 0, cols * sizeof(float)));
    int threads = std::min(256, ((cols + 31) / 32) * 32);
    if (threads < 32) threads = 32;
    k_layernorm_backward<<<rows, threads>>>(grad_out, x, gamma, mean, rstd,
                                            grad_x, grad_gamma, grad_beta,
                                            rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void flash_attention_forward(const float* q, const float* k, const float* v,
                             float* out,
                             int batch_size, int num_heads, int seq_len, int d_k,
                             bool causal, float scale) {
    dim3 grid(batch_size * num_heads, seq_len);
    int threads = std::min(128, d_k);
    if (threads < 32) threads = 32;
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
    int64_t head_stride = (int64_t)seq_len * d_k;
    int64_t total_elements = (int64_t)batch_size * num_heads * head_stride;
    CUDA_CHECK(cudaMemset(grad_k, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_v, 0, total_elements * sizeof(float)));

    dim3 grid(batch_size * num_heads, seq_len);
    int threads = std::min(128, d_k);
    if (threads < 32) threads = 32;
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
// Supports TF32 (FP32), FP16, and BF16 with Tensor Core Acceleration
// -------------------------------------------------------------
std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    int n1 = a->ndim();
    int n2 = b->ndim();
    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    if (c1 != r2) throw std::runtime_error("cuda::matmul: inner dimensions mismatch");

    int batch_size = a->size() / (r1 * c1);
    std::vector<int64_t> out_shape = a->shape;
    out_shape[n1 - 2] = r1;
    out_shape[n1 - 1] = c2;

    auto out = std::make_shared<Tensor>(out_shape, Device::CUDA, a->dtype, a->requires_grad || b->requires_grad);

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C = A * B -> Col-major C^T = B^T * A^T
    if (a->dtype == DType::Float16) {
        if (batch_size == 1) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     c2, r1, c1,
                                     &alpha,
                                     b->data_ptr<void>(), CUDA_R_16F, c2,
                                     a->data_ptr<void>(), CUDA_R_16F, c1,
                                     &beta,
                                     out->data_ptr<void>(), CUDA_R_16F, c2,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            long long int stride_a = r1 * c1;
            long long int stride_b = r2 * c2;
            long long int stride_c = r1 * c2;
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   c2, r1, c1,
                                                   &alpha,
                                                   b->data_ptr<void>(), CUDA_R_16F, c2, stride_b,
                                                   a->data_ptr<void>(), CUDA_R_16F, c1, stride_a,
                                                   &beta,
                                                   out->data_ptr<void>(), CUDA_R_16F, c2, stride_c,
                                                   batch_size,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    } else if (a->dtype == DType::BFloat16) {
        if (batch_size == 1) {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     c2, r1, c1,
                                     &alpha,
                                     b->data_ptr<void>(), CUDA_R_16BF, c2,
                                     a->data_ptr<void>(), CUDA_R_16BF, c1,
                                     &beta,
                                     out->data_ptr<void>(), CUDA_R_16BF, c2,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            long long int stride_a = r1 * c1;
            long long int stride_b = r2 * c2;
            long long int stride_c = r1 * c2;
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   c2, r1, c1,
                                                   &alpha,
                                                   b->data_ptr<void>(), CUDA_R_16BF, c2, stride_b,
                                                   a->data_ptr<void>(), CUDA_R_16BF, c1, stride_a,
                                                   &beta,
                                                   out->data_ptr<void>(), CUDA_R_16BF, c2, stride_c,
                                                   batch_size,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    } else {
        // Float32 with TF32 Tensor Cores enabled via get_cublas_handle()
        if (batch_size == 1) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     c2, r1, c1,
                                     &alpha,
                                     b->data_ptr<float>(), c2,
                                     a->data_ptr<float>(), c1,
                                     &beta,
                                     out->data_ptr<float>(), c2));
        } else {
            long long int stride_a = r1 * c1;
            long long int stride_b = r2 * c2;
            long long int stride_c = r1 * c2;
            CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   c2, r1, c1,
                                                   &alpha,
                                                   b->data_ptr<float>(), c2, stride_b,
                                                   a->data_ptr<float>(), c1, stride_a,
                                                   &beta,
                                                   out->data_ptr<float>(), c2, stride_c,
                                                   batch_size));
        }
    }

    return out;
}

} // namespace cuda

#endif
