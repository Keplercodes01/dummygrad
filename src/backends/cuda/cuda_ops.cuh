#pragma once
#include "tensor.h"
#include <vector>

#ifdef USE_CUDA
// Headers that dispatch to these ops also call the runtime API directly (cudaMemcpy, cudaMemset)
#include <cuda_runtime.h>

namespace cuda {
    // Primitive tensor ops
    void fill(Tensor& tensor, float value);
    void add_inplace(Tensor& dst, const Tensor& src);
    void copy_from_vector(Tensor& tensor, const std::vector<float>& values);
    void make_contiguous(Tensor& dst, const Tensor& src);

    // Fast cuBLAS matrix multiplication (supports batched + broadcasting)
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
    void matmul_backward(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                         const std::shared_ptr<Tensor>& grad_out,
                         std::shared_ptr<Tensor>& grad_a, std::shared_ptr<Tensor>& grad_b);

    // Fused elementwise activations
    void relu_forward(const float* in, float* out, int64_t size);
    void relu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size);

    void gelu_forward(const float* in, float* out, int64_t size);
    void gelu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size);

    void sigmoid_forward(const float* in, float* out, int64_t size);
    void sigmoid_backward(const float* out, const float* grad_out, float* grad_in, int64_t size);

    void silu_forward(const float* in, float* out, int64_t size);
    void silu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size);

    void leaky_relu_forward(const float* in, float* out, int64_t size, float negative_slope);
    void leaky_relu_backward(const float* in, const float* grad_out, float* grad_in, int64_t size, float negative_slope);

    void tanh_forward(const float* in, float* out, int64_t size);
    void tanh_backward(const float* out, const float* grad_out, float* grad_in, int64_t size);

    // Normalizations & Masking
    void softmax_forward(const float* in, float* out, int rows, int cols);
    void softmax_backward(const float* out, const float* grad_out, float* grad_in, int rows, int cols);

    void layernorm_forward(const float* x, const float* gamma, const float* beta,
                           float* out, float* mean, float* rstd,
                           int rows, int cols, float eps);
    void layernorm_backward(const float* grad_out, const float* x, const float* gamma,
                            const float* mean, const float* rstd,
                            float* grad_x, float* grad_gamma, float* grad_beta,
                            int rows, int cols);

    void causal_mask(const float* in, float* out, int batch, int seq_len);
    void causal_mask_backward(const float* grad_out, float* grad_in, int batch, int seq_len);
    void sliding_window_mask(const float* in, float* out, int batch, int seq_len, int window_size);
    void sliding_window_mask_backward(const float* grad_out, float* grad_in, int batch, int seq_len, int window_size);
    void prefix_causal_mask(const float* in, float* out, int batch, int seq_len, int prefix_len);
    void prefix_causal_mask_backward(const float* grad_out, float* grad_in, int batch, int seq_len, int prefix_len);

    // ---- cuda_core_ops.cu: deterministic training-step kernels ----

    // Broadcast binary ops (op: 0 add, 1 sub, 2 mul, 3 div). x is [batch, r, c]; y is
    // [y_batch, yr, yc] with dummygrad's broadcast mapping (see broadcast_binary in functional.h).
    void broadcast_binary_forward(int op, const float* x, const float* y, float* out,
                                  int batch, int r, int c, int y_batch, int yr, int yc);
    void broadcast_binary_backward_x(int op, const float* grad, const float* y, float* gx,
                                     int batch, int r, int c, int y_batch, int yr, int yc);
    void broadcast_binary_backward_y(int op, const float* grad, const float* x, const float* y, float* gy,
                                     int batch, int r, int c, int y_batch, int yr, int yc);

    // Scalar ops (op: 0 add, 1 sub, 2 mul, 3 div, 4 rsub = s - a, 5 rdiv = s / a)
    void scalar_forward(int op, const float* a, float s, float* out, int64_t n);
    void scalar_backward(int op, const float* grad, const float* a, float s, float* ga, int64_t n);

    // RMSNorm over rows of `cols` elements; gx or ggamma may be null to skip that gradient
    void rmsnorm_forward(const float* x, const float* gamma, float* out, float* inv_rms, int64_t rows, int64_t cols, float eps);
    void rmsnorm_backward(const float* grad, const float* x, const float* gamma, const float* inv_rms,
                          float* gx, float* ggamma, int64_t rows, int64_t cols);

    // Fused cross entropy: logits [rows, vocab], targets = class ids as floats. loss and count are
    // single device floats (mean loss over valid rows, number of valid rows); lse is [rows].
    void cross_entropy_forward(const float* logits, const float* targets, float* loss, float* lse, float* count,
                               int64_t rows, int64_t vocab, int ignore_index);
    void cross_entropy_backward(const float* logits, const float* targets, const float* lse, const float* count,
                                const float* dloss, float* grad, int64_t rows, int64_t vocab, int ignore_index);

    // Embedding lookup; the backward takes positions grouped by token id (CSR, built on the host)
    void embedding_forward(const float* indices, const float* weight, float* out, int64_t n, int64_t vocab, int64_t dim);
    void embedding_backward(const float* grad, const std::vector<int>& positions, const std::vector<int>& unique_ids,
                            const std::vector<int>& starts, float* grad_weight, int64_t vocab, int64_t dim);

    // GQA: [B, kv_heads, S, D] -> [B, kv_heads * group, S, D], and its gradient
    void repeat_kv_forward(const float* in, float* out, int64_t batch, int64_t kv_heads, int64_t group, int64_t head_elems);
    void repeat_kv_backward(const float* grad, float* gin, int64_t batch, int64_t kv_heads, int64_t group, int64_t head_elems);

    // Sum of squares over several device buffers, in double precision (for gradient clipping)
    double sum_of_squares(const std::vector<const float*>& buffers, const std::vector<int64_t>& sizes);

    // Rotary Position Embeddings (RoPE)
    void rope_forward(const float* in, const float* cos_freqs, const float* sin_freqs,
                      float* out, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos);
    void rope_backward(const float* grad_out, const float* cos_freqs, const float* sin_freqs,
                       float* grad_in, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos);

    // Frontier Tiled FlashAttention (Online Softmax, O(N) memory)
    void flash_attention_forward(const float* q, const float* k, const float* v,
                                 float* out,
                                 int batch_size, int num_heads, int seq_len, int d_k,
                                 bool causal, float scale);
    void flash_attention_backward(const float* q, const float* k, const float* v,
                                  const float* out, const float* grad_out,
                                  float* grad_q, float* grad_k, float* grad_v,
                                  int batch_size, int num_heads, int seq_len, int d_k,
                                  bool causal, float scale);

    // Epilogue Fusion: Fused Bias + Activation
    void add_bias(float* data, const float* bias, int rows, int cols);
    void add_bias_and_gelu(const float* in, const float* bias, float* out, int rows, int cols);
    void add_bias_and_relu(const float* in, const float* bias, float* out, int rows, int cols);

    // Fused AdamW optimizer update
    void adamw_step(float* param, const float* grad, float* m, float* v,
                    float lr, float beta1, float beta2, float eps, float weight_decay,
                    int step, int64_t size);
}

#endif
