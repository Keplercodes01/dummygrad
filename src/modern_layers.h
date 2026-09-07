#pragma once
#include "tensor.h"
#include "functional.h"
#include "linear.h"
#include "ops.h"
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>

// ============================================================================
// 1. RMSNorm (Root Mean Square Layer Normalization)
//    Used in LLaMA 1/2/3, Mistral, Gemma, DeepSeek, and modern LLMs.
//    Unlike standard LayerNorm, RMSNorm does not subtract the mean,
//    yielding higher training throughput with identical convergence.
// ============================================================================

struct RMSNormBackward : public Node {
    std::shared_ptr<Tensor> x, gamma, inv_rms;
    int r, c, batch_size;

    RMSNormBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> gamma,
                    std::shared_ptr<Tensor> inv_rms, int r, int c, int batch_size)
        : x(x), gamma(gamma), inv_rms(inv_rms), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto dout = grads[0];
        auto dx = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false); dx->fill_(0.0f);
        auto dg = std::make_shared<Tensor>(gamma->shape, gamma->device, gamma->dtype, false); dg->fill_(0.0f);

        const float* dout_ptr = dout->data_ptr<float>();
        const float* x_ptr = x->data_ptr<float>();
        const float* inv_rms_ptr = inv_rms->data_ptr<float>();
        const float* g_ptr = gamma->data_ptr<float>();

        float* dx_ptr = dx->data_ptr<float>();
        float* dg_ptr = dg->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                int offset = batch * r * c + i * c;
                float irms = inv_rms_ptr[batch * r + i];

                // Step 1: Accumulate dg and sum(dout * gamma * x)
                float sum_u_x = 0.0f;
                for (int j = 0; j < c; j++) {
                    float d = dout_ptr[offset + j];
                    float x_val = x_ptr[offset + j];
                    dg_ptr[j] += d * x_val * irms;

                    float u = d * g_ptr[j];
                    sum_u_x += u * x_val;
                }

                // Step 2: Compute dx_i = irms * (u_i - (x_i / c) * irms^2 * sum(u * x))
                float f_c = 1.0f / static_cast<float>(c);
                float factor = f_c * irms * irms * sum_u_x;
                for (int j = 0; j < c; j++) {
                    float u = dout_ptr[offset + j] * g_ptr[j];
                    float x_val = x_ptr[offset + j];
                    dx_ptr[offset + j] = irms * (u - x_val * factor);
                }
            }
        }
        return {dx, dg};
    }
};

class RMSNorm {
public:
    std::shared_ptr<Tensor> gamma;
    float eps;

    RMSNorm(int features, float eps = 1e-6f) : eps(eps) {
        gamma = std::make_shared<Tensor>(std::vector<int64_t>{features});
        gamma->fill_(1.0f);
        gamma->requires_grad = true;
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        int ndim = x->ndim();
        int c = x->shape[ndim - 1];
        int r = (ndim >= 2) ? x->shape[ndim - 2] : 1;
        int batch_size = x->size() / (r * c);

        auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, x->requires_grad);
        auto inv_rms = std::make_shared<Tensor>(std::vector<int64_t>{batch_size * r}, x->device, x->dtype, false);

        const float* x_ptr = x->data_ptr<float>();
        const float* g_ptr = gamma->data_ptr<float>();
        float* out_ptr = out->data_ptr<float>();
        float* irms_ptr = inv_rms->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                int offset = batch * r * c + i * c;
                float sum_sq = 0.0f;
                for (int j = 0; j < c; j++) {
                    float val = x_ptr[offset + j];
                    sum_sq += val * val;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(c) + eps);
                float irms = 1.0f / rms;
                irms_ptr[batch * r + i] = irms;

                for (int j = 0; j < c; j++) {
                    out_ptr[offset + j] = x_ptr[offset + j] * irms * g_ptr[j];
                }
            }
        }

        if (x->requires_grad) {
            auto grad_fn = std::make_shared<RMSNormBackward>(x, gamma, inv_rms, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(x).function, 0);
            grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
            out->grad_fn = grad_fn;
        }

        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {gamma};
    }
};

// ============================================================================
// 2. RoPE (Rotary Position Embeddings)
//    Used across modern architectures (LLaMA 3, Gemma, Mistral, Falcon, Qwen)
//    Applies continuous coordinate rotation to Query and Key representations.
// ============================================================================

// Precomputes the cosine and sine tables for RoPE
// Returns a pair of Tensors: {cos_freqs, sin_freqs}, each of shape [max_seq_len, dim / 2]
inline std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> precompute_freqs_cis(
    int dim, int max_seq_len, float theta = 10000.0f
) {
    int half_dim = dim / 2;
    auto cos_t = std::make_shared<Tensor>(std::vector<int64_t>{max_seq_len, half_dim});
    auto sin_t = std::make_shared<Tensor>(std::vector<int64_t>{max_seq_len, half_dim});

    float* cos_ptr = cos_t->data_ptr<float>();
    float* sin_ptr = sin_t->data_ptr<float>();

    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(dim));
        for (int m = 0; m < max_seq_len; ++m) {
            float val = static_cast<float>(m) * freq;
            cos_ptr[m * half_dim + i] = std::cos(val);
            sin_ptr[m * half_dim + i] = std::sin(val);
        }
    }
    return {cos_t, sin_t};
}

// Autograd Node for Rotary Position Embedding
struct RoPEBackward : public Node {
    std::shared_ptr<Tensor> x, cos_freqs, sin_freqs;
    int start_pos;
    int batch_size, seq_len, n_heads, head_dim;

    RoPEBackward(std::shared_ptr<Tensor> x,
                 std::shared_ptr<Tensor> cos_freqs,
                 std::shared_ptr<Tensor> sin_freqs,
                 int start_pos,
                 int batch_size, int seq_len, int n_heads, int head_dim)
        : x(x), cos_freqs(cos_freqs), sin_freqs(sin_freqs), start_pos(start_pos),
          batch_size(batch_size), seq_len(seq_len), n_heads(n_heads), head_dim(head_dim) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto dout = grads[0];
        auto dx = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);

        const float* dout_ptr = dout->data_ptr<float>();
        const float* cos_ptr = cos_freqs->data_ptr<float>();
        const float* sin_ptr = sin_freqs->data_ptr<float>();
        float* dx_ptr = dx->data_ptr<float>();

        int half_dim = head_dim / 2;

        // Backward of orthogonal 2D rotation matrix R(theta) is R(-theta) = R(theta)^T
        // dx_0 = dout_0 * cos + dout_1 * sin
        // dx_1 = -dout_0 * sin + dout_1 * cos
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                int pos = start_pos + s;
                int freq_offset = pos * half_dim;

                for (int h = 0; h < n_heads; ++h) {
                    int base_offset = ((b * seq_len + s) * n_heads + h) * head_dim;

                    for (int j = 0; j < half_dim; ++j) {
                        float c = cos_ptr[freq_offset + j];
                        float s_val = sin_ptr[freq_offset + j];

                        float d0 = dout_ptr[base_offset + 2 * j];
                        float d1 = dout_ptr[base_offset + 2 * j + 1];

                        dx_ptr[base_offset + 2 * j]     = d0 * c + d1 * s_val;
                        dx_ptr[base_offset + 2 * j + 1] = -d0 * s_val + d1 * c;
                    }
                }
            }
        }
        return {dx};
    }
};

// Applies Rotary Position Embedding to Query or Key tensor x
// Supports 4D [batch, seq_len, n_heads, head_dim], 3D [batch, seq_len, head_dim], and 2D [seq_len, head_dim]
inline std::shared_ptr<Tensor> apply_rotary_emb(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& cos_freqs,
    const std::shared_ptr<Tensor>& sin_freqs,
    int start_pos = 0
) {
    int ndim = x->ndim();
    int head_dim = x->shape[ndim - 1];
    int n_heads = (ndim == 4) ? x->shape[2] : 1;
    int seq_len = (ndim >= 2) ? x->shape[ndim - 2 - (ndim == 4 ? 1 : 0)] : 1;
    int batch_size = x->size() / (seq_len * n_heads * head_dim);

    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPE head dimension must be even, got: " + std::to_string(head_dim));
    }

    auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, x->requires_grad);

    const float* x_ptr = x->data_ptr<float>();
    const float* cos_ptr = cos_freqs->data_ptr<float>();
    const float* sin_ptr = sin_freqs->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    int half_dim = head_dim / 2;

    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int pos = start_pos + s;
            int freq_offset = pos * half_dim;

            for (int h = 0; h < n_heads; ++h) {
                int base_offset = ((b * seq_len + s) * n_heads + h) * head_dim;

                for (int j = 0; j < half_dim; ++j) {
                    float c = cos_ptr[freq_offset + j];
                    float s_val = sin_ptr[freq_offset + j];

                    float x0 = x_ptr[base_offset + 2 * j];
                    float x1 = x_ptr[base_offset + 2 * j + 1];

                    out_ptr[base_offset + 2 * j]     = x0 * c - x1 * s_val;
                    out_ptr[base_offset + 2 * j + 1] = x0 * s_val + x1 * c;
                }
            }
        }
    }

    if (x->requires_grad) {
        auto grad_fn = std::make_shared<RoPEBackward>(x, cos_freqs, sin_freqs, start_pos,
                                                      batch_size, seq_len, n_heads, head_dim);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ============================================================================
// 3. SwiGLU (Swish Gated Linear Unit)
//    Standard Feed-Forward Network in modern LLMs (LLaMA 3, PaLM, Mistral).
//    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
// ============================================================================

class SwiGLU {
public:
    Linear w_gate;
    Linear w_up;
    Linear w_down;

    // hidden_features is typically int(2.0f / 3.0f * 4.0f * in_features), rounded to multiple of 256
    SwiGLU(int in_features, int hidden_features = 0)
        : w_gate(in_features, hidden_features > 0 ? hidden_features : static_cast<int>(8.0f * in_features / 3.0f)),
          w_up(in_features,   hidden_features > 0 ? hidden_features : static_cast<int>(8.0f * in_features / 3.0f)),
          w_down(hidden_features > 0 ? hidden_features : static_cast<int>(8.0f * in_features / 3.0f), in_features) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        // 1. Gate projection with SiLU activation
        auto gate = silu(w_gate.forward(x));
        // 2. Up projection
        auto up = w_up.forward(x);
        // 3. Element-wise gated modulation
        auto h = mul(gate, up);
        // 4. Down projection back to in_features
        return w_down.forward(h);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = w_gate.parameters();
        auto p_up = w_up.parameters();
        auto p_down = w_down.parameters();
        p.insert(p.end(), p_up.begin(), p_up.end());
        p.insert(p.end(), p_down.begin(), p_down.end());
        return p;
    }
};

// ============================================================================
// 4. Gradient Clipping by Global L2 Norm
//    Prevents exploding gradients in deep LLM training.
//    Matches PyTorch's torch.nn.utils.clip_grad_norm_ semantics.
// ============================================================================

inline float clip_grad_norm_(
    const std::vector<std::shared_ptr<Tensor>>& parameters,
    float max_norm,
    float norm_type = 2.0f
) {
    if (parameters.empty()) return 0.0f;

    // 1. Compute total global L2 norm
    float total_norm_sq = 0.0f;
    for (const auto& p : parameters) {
        if (p && p->grad) {
            const float* g = p->grad->data_ptr<float>();
            int n = p->grad->size();
            for (int i = 0; i < n; i++) {
                total_norm_sq += g[i] * g[i];
            }
        }
    }

    float total_norm = std::sqrt(total_norm_sq);

    // 2. Rescale in-place if norm exceeds max_norm
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (const auto& p : parameters) {
            if (p && p->grad) {
                float* g = p->grad->data_ptr<float>();
                int n = p->grad->size();
                for (int i = 0; i < n; i++) {
                    g[i] *= scale;
                }
            }
        }
    }

    return total_norm;
}

// ============================================================================
// 5. CosineAnnealingLR (Cosine Annealing with Linear Warmup)
//    Gold-standard learning rate scheduler for LLM pretraining.
// ============================================================================

class CosineAnnealingLR {
public:
    float base_lr;
    float min_lr;
    int max_steps;
    int warmup_steps;
    int current_step;

    CosineAnnealingLR(float base_lr, int max_steps, int warmup_steps = 0, float min_lr = 0.0f)
        : base_lr(base_lr), min_lr(min_lr), max_steps(max_steps), warmup_steps(warmup_steps), current_step(0) {}

    float get_lr() const {
        if (current_step < warmup_steps) {
            return base_lr * static_cast<float>(current_step + 1) / static_cast<float>(std::max(1, warmup_steps));
        }
        if (current_step >= max_steps) {
            return min_lr;
        }
        float progress = static_cast<float>(current_step - warmup_steps) /
                         static_cast<float>(std::max(1, max_steps - warmup_steps));
        const float PI = 3.14159265358979323846f;
        return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + std::cos(progress * PI));
    }

    void step() {
        current_step++;
    }
};
