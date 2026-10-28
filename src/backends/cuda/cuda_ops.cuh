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

    // cuBLAS matrix multiplication (batched, whole-operand broadcasting). Inputs may be Float32,
    // Float16 or BFloat16; compute is the GEMM input precision they are converted to (an input
    // already in it is used as is): Float32, or Float16/BFloat16 on tensor cores with fp32
    // accumulation (see AutocastGuard). Outputs and the gradients matmul_backward writes are Float32.
    // saved_a / saved_b, when given, receive the operands the GEMM consumed (contiguous, already
    // rounded to compute) so that matmul_backward needs neither a cast nor a densifying copy.
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                                   DType compute = DType::Float32,
                                   std::shared_ptr<Tensor>* saved_a = nullptr,
                                   std::shared_ptr<Tensor>* saved_b = nullptr);
    // a @ b^T for a 2-D b stored [N, K] (a tied LM head reading the [vocab, d] embedding), with no
    // transposed copy: the GEMM reads b transposed. Backward: matmul_backward with trans_b.
    std::shared_ptr<Tensor> matmul_nt(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                                      DType compute = DType::Float32,
                                      std::shared_ptr<Tensor>* saved_a = nullptr,
                                      std::shared_ptr<Tensor>* saved_b = nullptr);
    // out = x @ w + bias for a 2-D weight w [in, out] and a bias with one value per output feature.
    // The bias is added inside the GEMM (cuBLASLt epilogue) when cuBLASLt supports the case, else
    // by a separate kernel. saved_x / saved_w as for matmul; the gradients are matmul_backward's
    // plus bias_grad.
    std::shared_ptr<Tensor> linear(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& w,
                                   const std::shared_ptr<Tensor>& bias, DType compute = DType::Float32,
                                   std::shared_ptr<Tensor>* saved_x = nullptr,
                                   std::shared_ptr<Tensor>* saved_w = nullptr);
    // How many linear calls so far added the bias with a separate kernel because cuBLASLt could
    // not fuse it (no algorithm for the shape, or an unaligned pointer). Should stay 0 for
    // ordinary layers; a rising count means Linear is running slower than it should.
    int64_t linear_unfused_count();
    // Gradients of out = a @ b for inputs of shapes a_shape and b_shape (trans_b: out = a @ b^T, with
    // b_shape and grad_b in b's stored [N, K] layout). a and b are Float32 or
    // compute-dtype operands (normally matmul's saved ones). grad_a needs only b and grad_b only a:
    // a null grad_a / grad_b skips that GEMM, and the operand only it uses may then be null.
    void matmul_backward(const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
                         const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b,
                         const std::shared_ptr<Tensor>& grad_out,
                         std::shared_ptr<Tensor>* grad_a, std::shared_ptr<Tensor>* grad_b,
                         DType compute = DType::Float32, bool trans_b = false);

    // Rounds n device floats to Float16 or BFloat16 (nearest even) into out
    void cast_from_float(const float* in, void* out, int64_t n, DType dtype);
    // Converts n device elements between Float32, Float16 and BFloat16 (nearest even when narrowing)
    void convert(const void* in, DType in_dtype, void* out, DType out_dtype, int64_t n);

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

    // Normalizations & Masking. Masks act on [batch, rows, cols] scores: key j > query i is masked
    void softmax_forward(const float* in, float* out, int rows, int cols);
    void softmax_backward(const float* out, const float* grad_out, float* grad_in, int rows, int cols);

    void causal_mask(const float* in, float* out, int batch, int rows, int cols);
    void causal_mask_backward(const float* grad_out, float* grad_in, int batch, int rows, int cols);
    void sliding_window_mask(const float* in, float* out, int batch, int rows, int cols, int window_size);
    void sliding_window_mask_backward(const float* grad_out, float* grad_in, int batch, int rows, int cols, int window_size);
    void prefix_causal_mask(const float* in, float* out, int batch, int rows, int cols, int prefix_len);
    void prefix_causal_mask_backward(const float* grad_out, float* grad_in, int batch, int rows, int cols, int prefix_len);

    // ---- cuda_core_ops.cu: deterministic training-step kernels ----

    // Broadcast binary ops (op: 0 add, 1 sub, 2 mul, 3 div), NumPy-style: y_shape is aligned to
    // x_shape's rank and each y dimension is 1 or equal to x's. Output and gx have x's shape.
    void broadcast_binary_forward(int op, const float* x, const float* y, float* out,
                                  const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape);
    void broadcast_binary_backward_x(int op, const float* grad, const float* y, float* gx,
                                     const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape);
    void broadcast_binary_backward_y(int op, const float* grad, const float* x, const float* y, float* gy,
                                     const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape);

    // Scalar ops (op: 0 add, 1 sub, 2 mul, 3 div, 4 rsub = s - a, 5 rdiv = s / a)
    void scalar_forward(int op, const float* a, float s, float* out, int64_t n);
    void scalar_backward(int op, const float* grad, const float* a, float s, float* ga, int64_t n);

    // RMSNorm over rows of `cols` elements; gx or ggamma may be null to skip that gradient
    void rmsnorm_forward(const float* x, const float* gamma, float* out, float* inv_rms, int64_t rows, int64_t cols, float eps);
    void rmsnorm_backward(const float* grad, const float* x, const float* gamma, const float* inv_rms,
                          float* gx, float* ggamma, int64_t rows, int64_t cols);

    // LayerNorm over rows of `cols` elements, saving each row's mean and inverse std for backward.
    // gx, ggamma and gbeta may be null to skip that gradient.
    void layernorm_forward(const float* x, const float* gamma, const float* beta, float* out, float* mean,
                           float* rstd, int64_t rows, int64_t cols, float eps);
    void layernorm_backward(const float* grad, const float* x, const float* gamma, const float* mean, const float* rstd,
                            float* gx, float* ggamma, float* gbeta, int64_t rows, int64_t cols);

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

    // 2-D max pooling (no padding) over planes = B * C of [H, W]. argmax receives each window's
    // winning offset kh * kW + kw as a float (-1 if all NaN); backward gathers deterministically.
    void max_pool2d_forward(const float* x, float* out, float* argmax, int64_t planes, int H, int W, int H_out, int W_out,
                            int kH, int kW, int sH, int sW);
    void max_pool2d_backward(const float* grad, const float* argmax, float* gx, int64_t planes, int H, int W, int H_out,
                             int W_out, int kH, int kW, int sH, int sW);

    // Column sums of a row-major [rows, cols] gradient: a Linear's bias gradient. Deterministic, and
    // bit-identical to the broadcast-add backward it replaces.
    void bias_grad(const float* grad, float* db, int64_t rows, int64_t cols);

    // Reductions over one of the last two dims of [batch, rows, cols]: over_rows reduces dim -2
    // (out [batch, cols]), else dim -1 (out [batch, rows]). Deterministic. out = sum / divisor.
    void reduce_axis(const float* x, float* out, int64_t batch, int64_t rows, int64_t cols, bool over_rows, float divisor);
    // ga[i] = g[output of element i] / divisor (the gradient of reduce_axis)
    void reduce_axis_backward(const float* g, float* ga, int64_t batch, int64_t rows, int64_t cols, bool over_rows,
                              float divisor);
    // Population variance along the axis (take_sqrt: standard deviation); mean receives the means
    void moment_axis(const float* x, float* mean, float* out, int64_t batch, int64_t rows, int64_t cols, bool over_rows,
                     bool take_sqrt);
    // Its gradient given the means and the forward's variance or std (stat):
    // var: 2 (x - mean) / n * g;  std: (x - mean) / (n (std + 1e-8)) * g
    void moment_axis_backward(const float* x, const float* mean, const float* stat, const float* g, float* ga,
                              int64_t batch, int64_t rows, int64_t cols, bool over_rows, bool is_std);

    // Elementwise losses summed over n elements and divided by divisor into one device float.
    // kind: 0 squared error, 1 absolute error, 2 cross entropy on probabilities (-t log(p + 1e-8))
    void pointwise_loss_forward(int kind, const float* pred, const float* target, float* loss, int64_t n, float divisor);
    void pointwise_loss_backward(int kind, const float* pred, const float* target, const float* dloss, float* grad,
                                 int64_t n, float divisor);

    // Boolean mask: out = (m == 0) ? -1e9 : a; the gradient passes where m != 0
    void bool_mask_forward(const float* a, const float* m, float* out, int64_t n);
    void bool_mask_backward(const float* g, const float* m, float* ga, int64_t n);

    // im2col / col2im with dilation 1: col is [batch, C * kH * kW, H_out, W_out]. col2im writes
    // (does not accumulate into) x, summing deterministically.
    void im2col(const float* x, float* col, int64_t batch, int C, int H, int W, int kH, int kW, int pH, int pW, int sH,
                int sW, int H_out, int W_out);
    void col2im(const float* col, float* x, int64_t batch, int C, int H, int W, int kH, int kW, int pH, int pW, int sH,
                int sW, int H_out, int W_out);

    // Sum of squares over several device buffers, in double precision (gradient clipping): one
    // launch for all of them, deterministic, one host sync
    double sum_of_squares(const std::vector<const float*>& buffers, const std::vector<int64_t>& sizes);
    // Multiplies every buffer by scale in place, in one launch
    void scale_buffers(const std::vector<float*>& buffers, const std::vector<int64_t>& sizes, float scale);
    // Multiplies every buffer by inv_scale in place; returns whether any value was inf or NaN
    // (GradScaler). One host sync for all buffers.
    bool unscale_and_check(const std::vector<float*>& buffers, const std::vector<int64_t>& sizes, float inv_scale);

    // ---- cuda_attention.cu: attention (cuBLAS GEMM path or tiled kernels, deterministic backward) ----
    // q, k, v, out, dout, dq, dk, dv are [batch * heads, seq_len, head_dim]; lse is [batch * heads, seq_len].
    // seq_len <= attention_gemm_max_seq() runs as batched GEMMs (any head dim); longer sequences use the
    // tiled kernels (head dims up to 128, zero-padded to 16/32/64/128), or the GEMM path a chunk of
    // heads at a time for larger head dims. Every path is deterministic. Forward returns the
    // implementation it used; pass it to backward.
    // compute (see AutocastGuard) is the precision of the GEMM path's products: Float16/BFloat16 round
    // Q, K, V, dO and the probabilities P / score gradients dS for tensor cores, while scores, softmax,
    // lse and every output stay fp32. The tiled kernels always compute in fp32.
    enum class AttentionImpl { Gemm, Tiled };
    // Longest sequence that takes the GEMM path (default 2048). The GEMM path materializes [seq, seq]
    // scores for a chunk of heads (at most 256 MB); 0 forces the tiled kernels, whose memory is linear in seq.
    void set_attention_gemm_max_seq(int seq_len);
    int attention_gemm_max_seq();
    AttentionImpl attention_forward(const float* q, const float* k, const float* v, float* out, float* lse,
                                    int batch, int heads, int seq_len, int head_dim, bool causal, float scale,
                                    DType compute = DType::Float32);
    void attention_backward(const float* q, const float* k, const float* v, const float* out, const float* dout,
                            const float* lse, float* dq, float* dk, float* dv,
                            int batch, int heads, int seq_len, int head_dim, bool causal, float scale, AttentionImpl impl,
                            DType compute = DType::Float32);

    // Rotary Position Embeddings (RoPE)
    void rope_forward(const float* in, const float* cos_freqs, const float* sin_freqs,
                      float* out, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos);
    void rope_backward(const float* grad_out, const float* cos_freqs, const float* sin_freqs,
                       float* grad_in, int batch_size, int seq_len, int n_heads, int head_dim, int start_pos);

    // Fused AdamW optimizer update
    void adamw_step(float* param, const float* grad, float* m, float* v,
                    float lr, float beta1, float beta2, float eps, float weight_decay,
                    int step, int64_t size);
}

#endif
