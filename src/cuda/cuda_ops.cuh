#pragma once
#include "tensor.h"
#include <vector>

#ifdef USE_CUDA

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
