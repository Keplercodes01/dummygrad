#include <metal_stdlib>
using namespace metal;

// -------------------------------------------------------------
// Fused Elementwise: Add Bias + GELU Activation
// -------------------------------------------------------------
kernel void fused_add_bias_gelu_kernel(
    device const float* in       [[buffer(0)]],
    device const float* bias     [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant uint& total_elems   [[buffer(3)]],
    constant uint& cols          [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    uint c = id % cols;
    float val = in[id] + bias[c];

    // Fast GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;
    float cube = val * val * val;
    float inner = SQRT_2_OVER_PI * (val + COEFF * cube);
    out[id] = 0.5f * val * (1.0f + metal::tanh(inner));
}

// -------------------------------------------------------------
// Fused Elementwise: Add Bias + ReLU Activation
// -------------------------------------------------------------
kernel void fused_add_bias_relu_kernel(
    device const float* in       [[buffer(0)]],
    device const float* bias     [[buffer(1)]],
    device float* out            [[buffer(2)]],
    constant uint& total_elems   [[buffer(3)]],
    constant uint& cols          [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    uint c = id % cols;
    float val = in[id] + bias[c];
    out[id] = metal::max(val, 0.0f);
}

// -------------------------------------------------------------
// Fused LayerNorm: SIMD-group Shuffle Reduction
// -------------------------------------------------------------
kernel void fused_layernorm_kernel(
    device const float* in       [[buffer(0)]],
    device const float* gamma    [[buffer(1)]],
    device const float* beta     [[buffer(2)]],
    device float* out            [[buffer(3)]],
    constant uint& cols          [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint row                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint t_per_tg                [[threads_per_threadgroup]]
) {
    device const float* row_in = in + row * cols;
    device float* row_out = out + row * cols;

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (uint i = tid; i < cols; i += t_per_tg) {
        float x = row_in[i];
        sum += x;
        sq_sum += x * x;
    }

    // SIMD-group tree reduction
    sum = simd_sum(sum);
    sq_sum = simd_sum(sq_sum);

    threadgroup float s_mean[32];
    threadgroup float s_var[32];

    if (simd_lane == 0) {
        s_mean[simd_id] = sum;
        s_var[simd_id] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (t_per_tg + 31) / 32;
    float total_sum = 0.0f;
    float total_sq = 0.0f;
    if (tid < num_simds) {
        total_sum = s_mean[tid];
        total_sq = s_var[tid];
    }
    total_sum = simd_sum(total_sum);
    total_sq = simd_sum(total_sq);

    threadgroup float row_mean;
    threadgroup float row_rstd;
    if (tid == 0) {
        float mean = total_sum / float(cols);
        float variance = (total_sq / float(cols)) - (mean * mean);
        if (variance < 0.0f) variance = 0.0f;
        row_mean = mean;
        row_rstd = metal::rsqrt(variance + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = row_mean;
    float r = row_rstd;

    for (uint i = tid; i < cols; i += t_per_tg) {
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta ? beta[i] : 0.0f;
        row_out[i] = ((row_in[i] - m) * r) * g + b;
    }
}

// -------------------------------------------------------------
// Fused AdamW Optimizer Kernel
// -------------------------------------------------------------
kernel void fused_adamw_kernel(
    device float* p              [[buffer(0)]],
    device const float* g        [[buffer(1)]],
    device float* m              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    constant float& lr           [[buffer(4)]],
    constant float& beta1        [[buffer(5)]],
    constant float& beta2        [[buffer(6)]],
    constant float& eps          [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bc1          [[buffer(9)]],
    constant float& bc2          [[buffer(10)]],
    constant uint& total_params  [[buffer(11)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_params) return;

    float param = p[id];
    float grad = g[id];

    // Weight decay
    if (weight_decay != 0.0f) {
        param -= lr * weight_decay * param;
    }

    // Momentum and variance updates
    float mom = beta1 * m[id] + (1.0f - beta1) * grad;
    float var = beta2 * v[id] + (1.0f - beta2) * grad * grad;
    m[id] = mom;
    v[id] = var;

    float mom_hat = mom / bc1;
    float var_hat = var / bc2;

    param -= (lr * mom_hat) / (metal::sqrt(var_hat) + eps);
    p[id] = param;
}

// -------------------------------------------------------------
// Tiled FlashAttention Forward Pass (MSL implementation)
// -------------------------------------------------------------
kernel void tiled_flash_attention_kernel(
    device const float* Q        [[buffer(0)]],
    device const float* K        [[buffer(1)]],
    device const float* V        [[buffer(2)]],
    device float* Out            [[buffer(3)]],
    constant uint& seq_len       [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant float& scale        [[buffer(6)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]]
) {
    uint head_id = tg_pos.y;
    uint q_idx = tg_pos.x;
    if (q_idx >= seq_len) return;

    uint head_offset = head_id * seq_len * head_dim;
    device const float* q_vec = Q + head_offset + q_idx * head_dim;
    device float* out_vec = Out + head_offset + q_idx * head_dim;

    float max_score = -1e20f;
    float sum_exp = 0.0f;

    // First pass: compute causal attention scores and online softmax stats
    for (uint k_idx = 0; k_idx <= q_idx; ++k_idx) {
        device const float* k_vec = K + head_offset + k_idx * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        dot *= scale;

        if (dot > max_score) {
            sum_exp = sum_exp * metal::exp(max_score - dot) + 1.0f;
            max_score = dot;
        } else {
            sum_exp += metal::exp(dot - max_score);
        }
    }

    // Second pass: weighted sum into out_vec
    if (tid < head_dim) {
        float acc = 0.0f;
        for (uint k_idx = 0; k_idx <= q_idx; ++k_idx) {
            device const float* k_vec = K + head_offset + k_idx * head_dim;
            device const float* v_vec = V + head_offset + k_idx * head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            dot *= scale;
            float weight = metal::exp(dot - max_score) / sum_exp;
            acc += weight * v_vec[tid];
        }
        out_vec[tid] = acc;
    }
}

// -------------------------------------------------------------
// Sigmoid Forward & Backward
// -------------------------------------------------------------
kernel void sigmoid_kernel(
    device const float* in       [[buffer(0)]],
    device float* out            [[buffer(1)]],
    constant uint& total_elems   [[buffer(2)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    out[id] = 1.0f / (1.0f + metal::exp(-in[id]));
}

kernel void sigmoid_backward_kernel(
    device const float* out      [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in        [[buffer(2)]],
    constant uint& total_elems   [[buffer(3)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    float s = out[id];
    grad_in[id] = s * (1.0f - s) * grad_out[id];
}

// -------------------------------------------------------------
// SiLU (Swish) Forward & Backward
// -------------------------------------------------------------
kernel void silu_kernel(
    device const float* in       [[buffer(0)]],
    device float* out            [[buffer(1)]],
    constant uint& total_elems   [[buffer(2)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    float x = in[id];
    out[id] = x / (1.0f + metal::exp(-x));
}

kernel void silu_backward_kernel(
    device const float* in       [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in        [[buffer(2)]],
    constant uint& total_elems   [[buffer(3)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    float x = in[id];
    float s = 1.0f / (1.0f + metal::exp(-x));
    grad_in[id] = (s * (1.0f + x * (1.0f - s))) * grad_out[id];
}

// -------------------------------------------------------------
// LeakyReLU Forward & Backward
// -------------------------------------------------------------
kernel void leaky_relu_kernel(
    device const float* in          [[buffer(0)]],
    device float* out               [[buffer(1)]],
    constant float& negative_slope  [[buffer(2)]],
    constant uint& total_elems      [[buffer(3)]],
    uint id                         [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    float x = in[id];
    out[id] = x > 0.0f ? x : negative_slope * x;
}

kernel void leaky_relu_backward_kernel(
    device const float* in          [[buffer(0)]],
    device const float* grad_out    [[buffer(1)]],
    device float* grad_in           [[buffer(2)]],
    constant float& negative_slope  [[buffer(3)]],
    constant uint& total_elems      [[buffer(4)]],
    uint id                         [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    grad_in[id] = in[id] > 0.0f ? grad_out[id] : negative_slope * grad_out[id];
}

// -------------------------------------------------------------
// MSE & L1 Loss Backward
// -------------------------------------------------------------
kernel void mse_backward_kernel(
    device const float* pred     [[buffer(0)]],
    device const float* target   [[buffer(1)]],
    device float* grad_pred      [[buffer(2)]],
    constant float& scale        [[buffer(3)]],
    constant uint& total_elems   [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    grad_pred[id] = scale * (pred[id] - target[id]);
}

kernel void l1_loss_backward_kernel(
    device const float* pred     [[buffer(0)]],
    device const float* target   [[buffer(1)]],
    device float* grad_pred      [[buffer(2)]],
    constant float& scale        [[buffer(3)]],
    constant uint& total_elems   [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    float diff = pred[id] - target[id];
    float sgn = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
    grad_pred[id] = scale * sgn;
}

// -------------------------------------------------------------
// Causal Mask & Sliding Window Mask Forward & Backward
// -------------------------------------------------------------
kernel void causal_mask_kernel(
    device const float* in       [[buffer(0)]],
    device float* out            [[buffer(1)]],
    constant uint& total_elems   [[buffer(2)]],
    constant uint& seq_len       [[buffer(3)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    uint j = id % seq_len;
    uint i = (id / seq_len) % seq_len;
    out[id] = (j > i) ? -1e9f : in[id];
}

kernel void causal_mask_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device float* grad_in        [[buffer(1)]],
    constant uint& total_elems   [[buffer(2)]],
    constant uint& seq_len       [[buffer(3)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    uint j = id % seq_len;
    uint i = (id / seq_len) % seq_len;
    grad_in[id] = (j > i) ? 0.0f : grad_out[id];
}

kernel void sliding_window_mask_kernel(
    device const float* in       [[buffer(0)]],
    device float* out            [[buffer(1)]],
    constant uint& total_elems   [[buffer(2)]],
    constant uint& seq_len       [[buffer(3)]],
    constant uint& window_size   [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    uint j = id % seq_len;
    uint i = (id / seq_len) % seq_len;
    bool valid = (j <= i) && (i - j <= window_size);
    out[id] = valid ? in[id] : -1e9f;
}

kernel void sliding_window_mask_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device float* grad_in        [[buffer(1)]],
    constant uint& total_elems   [[buffer(2)]],
    constant uint& seq_len       [[buffer(3)]],
    constant uint& window_size   [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= total_elems) return;
    uint j = id % seq_len;
    uint i = (id / seq_len) % seq_len;
    bool valid = (j <= i) && (i - j <= window_size);
    grad_in[id] = valid ? grad_out[id] : 0.0f;
}
