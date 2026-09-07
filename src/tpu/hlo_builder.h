#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "../types.h"

// Pure C++ HLO (High-Level Optimizer) Graph Generator for TPU v5e
// Compiles into optimal systolic array execution (128x128 MXU) via OpenXLA / PJRT
namespace hlo {

inline std::string dtype_to_hlo(DType dt) {
    switch (dt) {
        case DType::Float32:  return "f32";
        case DType::Float16:  return "f16";
        case DType::BFloat16: return "bf16";
        case DType::Int8:     return "s8";
        default:              return "f32";
    }
}

inline std::string shape_to_hlo(const std::vector<int64_t>& shape, DType dt) {
    std::ostringstream ss;
    ss << dtype_to_hlo(dt) << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size()) ss << ",";
    }
    ss << "]";
    return ss.str();
}

// Emits an HLO Module for a high-performance Matrix Multiplication (GEMM)
// Hardware-accelerated directly onto TPU v5e's 128x128 systolic Matrix Multiply Unit (MXU)
inline std::string build_gemm_hlo(
    int64_t M, int64_t K, int64_t N,
    DType dt = DType::Float32
) {
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_gemm_" << M << "_" << K << "_" << N << "\n\n";
    ss << "ENTRY %main (x: " << t << "[" << M << "," << K << "], "
       << "w: " << t << "[" << K << "," << N << "]) -> " << t << "[" << M << "," << N << "] {\n";
    ss << "  %x = " << t << "[" << M << "," << K << "] parameter(0)\n";
    ss << "  %w = " << t << "[" << K << "," << N << "] parameter(1)\n";
    ss << "  ROOT %dot = " << t << "[" << M << "," << N << "] dot(%x, %w), "
       << "lhs_contracting_dims={1}, rhs_contracting_dims={0}\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for Fused Linear Layer (MatMul + Bias + GELU)
// XLA fuses the bias addition and Gaussian Error Linear Unit activation directly into the MXU writeback pipeline
inline std::string build_fused_linear_gelu_hlo(
    int64_t M, int64_t K, int64_t N,
    DType dt = DType::Float32
) {
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_fused_linear_gelu\n\n";
    ss << "ENTRY %main (x: " << t << "[" << M << "," << K << "], "
       << "w: " << t << "[" << K << "," << N << "], "
       << "bias: " << t << "[" << N << "]) -> " << t << "[" << M << "," << N << "] {\n";
    ss << "  %x = " << t << "[" << M << "," << K << "] parameter(0)\n";
    ss << "  %w = " << t << "[" << K << "," << N << "] parameter(1)\n";
    ss << "  %bias = " << t << "[" << N << "] parameter(2)\n";
    ss << "  %mm = " << t << "[" << M << "," << N << "] dot(%x, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n";
    ss << "  %bias_bcast = " << t << "[" << M << "," << N << "] broadcast(%bias), dimensions={1}\n";
    ss << "  %linear = " << t << "[" << M << "," << N << "] add(%mm, %bias_bcast)\n";
    
    // Exact GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ss << "  %const_half = " << t << "[] constant(0.5)\n";
    ss << "  %half = " << t << "[" << M << "," << N << "] broadcast(%const_half), dimensions={}\n";
    ss << "  %const_one = " << t << "[] constant(1.0)\n";
    ss << "  %one = " << t << "[" << M << "," << N << "] broadcast(%const_one), dimensions={}\n";
    ss << "  %const_sqrt_2_pi = " << t << "[] constant(0.79788456)\n";
    ss << "  %sqrt_2_pi = " << t << "[" << M << "," << N << "] broadcast(%const_sqrt_2_pi), dimensions={}\n";
    ss << "  %const_coeff = " << t << "[] constant(0.044715)\n";
    ss << "  %coeff = " << t << "[" << M << "," << N << "] broadcast(%const_coeff), dimensions={}\n";
    
    ss << "  %x2 = " << t << "[" << M << "," << N << "] multiply(%linear, %linear)\n";
    ss << "  %x3 = " << t << "[" << M << "," << N << "] multiply(%x2, %linear)\n";
    ss << "  %poly = " << t << "[" << M << "," << N << "] multiply(%coeff, %x3)\n";
    ss << "  %inner = " << t << "[" << M << "," << N << "] add(%linear, %poly)\n";
    ss << "  %scaled = " << t << "[" << M << "," << N << "] multiply(%sqrt_2_pi, %inner)\n";
    ss << "  %tanh = " << t << "[" << M << "," << N << "] tanh(%scaled)\n";
    ss << "  %one_plus = " << t << "[" << M << "," << N << "] add(%one, %tanh)\n";
    ss << "  %half_x = " << t << "[" << M << "," << N << "] multiply(%half, %linear)\n";
    ss << "  ROOT %gelu = " << t << "[" << M << "," << N << "] multiply(%half_x, %one_plus)\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for Multi-TPU Inter-Chip Interconnect (ICI) All-Reduce
// Dynamically forms the replica group for any arbitrary number of TPU chips (e.g. 2, 4, 8, 16, 32, 64...)
inline std::string build_ici_all_reduce_hlo(
    const std::vector<int64_t>& shape,
    size_t num_chips,
    DType dt = DType::Float32
) {
    std::string s = shape_to_hlo(shape, dt);
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_ici_all_reduce\n\n";
    ss << "%add_computation (lhs: " << t << "[], rhs: " << t << "[]) -> " << t << "[] {\n";
    ss << "  %lhs = " << t << "[] parameter(0)\n";
    ss << "  %rhs = " << t << "[] parameter(1)\n";
    ss << "  ROOT %res = " << t << "[] add(%lhs, %rhs)\n";
    ss << "}\n\n";

    ss << "ENTRY %main (grads: " << s << ") -> " << s << " {\n";
    ss << "  %grads = " << s << " parameter(0)\n";
    ss << "  ROOT %all_reduced = " << s << " all-reduce(%grads), to_apply=%add_computation, replica_groups={{";
    for (size_t i = 0; i < num_chips; ++i) {
        ss << i;
        if (i + 1 < num_chips) ss << ",";
    }
    ss << "}}\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for Fused Scaled Dot-Product Attention (FlashAttention equivalent on TPU)
// Q: [B, H, S, D], K: [B, H, S, D], V: [B, H, S, D] -> [B, H, S, D]
inline std::string build_tpu_attention_hlo(
    int64_t B, int64_t H, int64_t S, int64_t D,
    DType dt = DType::Float32
) {
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_flash_attention\n\n";
    ss << "%max_fn (lhs: " << t << "[], rhs: " << t << "[]) -> " << t << "[] {\n";
    ss << "  %lhs = " << t << "[] parameter(0)\n";
    ss << "  %rhs = " << t << "[] parameter(1)\n";
    ss << "  ROOT %res = " << t << "[] maximum(%lhs, %rhs)\n";
    ss << "}\n\n";
    ss << "%add_fn (lhs: " << t << "[], rhs: " << t << "[]) -> " << t << "[] {\n";
    ss << "  %lhs = " << t << "[] parameter(0)\n";
    ss << "  %rhs = " << t << "[] parameter(1)\n";
    ss << "  ROOT %res = " << t << "[] add(%lhs, %rhs)\n";
    ss << "}\n\n";

    ss << "ENTRY %main (q: " << t << "[" << B << "," << H << "," << S << "," << D << "], "
       << "k: " << t << "[" << B << "," << H << "," << S << "," << D << "], "
       << "v: " << t << "[" << B << "," << H << "," << S << "," << D << "]) -> "
       << t << "[" << B << "," << H << "," << S << "," << D << "] {\n";
    ss << "  %q = " << t << "[" << B << "," << H << "," << S << "," << D << "] parameter(0)\n";
    ss << "  %k = " << t << "[" << B << "," << H << "," << S << "," << D << "] parameter(1)\n";
    ss << "  %v = " << t << "[" << B << "," << H << "," << S << "," << D << "] parameter(2)\n";
    
    // Dot Q and K^T -> [B, H, S, S]
    ss << "  %scores = " << t << "[" << B << "," << H << "," << S << "," << S << "] dot(%q, %k), "
       << "lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={3}\n";
    
    // Scale by 1 / sqrt(D)
    float scale_val = 1.0f / std::sqrt(static_cast<float>(D));
    ss << "  %scale_const = " << t << "[] constant(" << scale_val << ")\n";
    ss << "  %scale = " << t << "[" << B << "," << H << "," << S << "," << S << "] broadcast(%scale_const), dimensions={}\n";
    ss << "  %scaled_scores = " << t << "[" << B << "," << H << "," << S << "," << S << "] multiply(%scores, %scale)\n";
    
    // Softmax over last dimension
    ss << "  %max_score = " << t << "[" << B << "," << H << "," << S << "] reduce(%scaled_scores, %scale_const), dimensions={3}, to_apply=%max_fn\n";
    ss << "  %max_bcast = " << t << "[" << B << "," << H << "," << S << "," << S << "] broadcast(%max_score), dimensions={0,1,2}\n";
    ss << "  %shifted = " << t << "[" << B << "," << H << "," << S << "," << S << "] subtract(%scaled_scores, %max_bcast)\n";
    ss << "  %exp = " << t << "[" << B << "," << H << "," << S << "," << S << "] exponential(%shifted)\n";
    ss << "  %sum_exp = " << t << "[" << B << "," << H << "," << S << "] reduce(%exp, %scale_const), dimensions={3}, to_apply=%add_fn\n";
    ss << "  %sum_bcast = " << t << "[" << B << "," << H << "," << S << "," << S << "] broadcast(%sum_exp), dimensions={0,1,2}\n";
    ss << "  %weights = " << t << "[" << B << "," << H << "," << S << "," << S << "] divide(%exp, %sum_bcast)\n";
    
    // Dot weights and V -> [B, H, S, D]
    ss << "  ROOT %out = " << t << "[" << B << "," << H << "," << S << "," << D << "] dot(%weights, %v), "
       << "lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for Sigmoid Activation
inline std::string build_sigmoid_hlo(const std::vector<int64_t>& shape, DType dt = DType::Float32) {
    std::string s = shape_to_hlo(shape, dt);
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_sigmoid\n\n";
    ss << "ENTRY %main (x: " << s << ") -> " << s << " {\n";
    ss << "  %x = " << s << " parameter(0)\n";
    ss << "  %neg = " << s << " negate(%x)\n";
    ss << "  %exp = " << s << " exponential(%neg)\n";
    ss << "  %one_c = " << t << "[] constant(1.0)\n";
    ss << "  %one = " << s << " broadcast(%one_c), dimensions={}\n";
    ss << "  %denom = " << s << " add(%one, %exp)\n";
    ss << "  ROOT %out = " << s << " divide(%one, %denom)\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for SiLU (Swish) Activation
inline std::string build_silu_hlo(const std::vector<int64_t>& shape, DType dt = DType::Float32) {
    std::string s = shape_to_hlo(shape, dt);
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_silu\n\n";
    ss << "ENTRY %main (x: " << s << ") -> " << s << " {\n";
    ss << "  %x = " << s << " parameter(0)\n";
    ss << "  %neg = " << s << " negate(%x)\n";
    ss << "  %exp = " << s << " exponential(%neg)\n";
    ss << "  %one_c = " << t << "[] constant(1.0)\n";
    ss << "  %one = " << s << " broadcast(%one_c), dimensions={}\n";
    ss << "  %denom = " << s << " add(%one, %exp)\n";
    ss << "  %sig = " << s << " divide(%one, %denom)\n";
    ss << "  ROOT %out = " << s << " multiply(%x, %sig)\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for LeakyReLU Activation
inline std::string build_leaky_relu_hlo(const std::vector<int64_t>& shape, float negative_slope = 0.01f, DType dt = DType::Float32) {
    std::string s = shape_to_hlo(shape, dt);
    std::string t = dtype_to_hlo(dt);
    std::ostringstream ss;
    ss << "HloModule tpu_leaky_relu\n\n";
    ss << "ENTRY %main (x: " << s << ") -> " << s << " {\n";
    ss << "  %x = " << s << " parameter(0)\n";
    ss << "  %slope_c = " << t << "[] constant(" << negative_slope << ")\n";
    ss << "  %slope = " << s << " broadcast(%slope_c), dimensions={}\n";
    ss << "  %scaled = " << s << " multiply(%x, %slope)\n";
    ss << "  ROOT %out = " << s << " maximum(%x, %scaled)\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for MSE Loss
inline std::string build_mse_hlo(const std::vector<int64_t>& shape, DType dt = DType::Float32) {
    std::string s = shape_to_hlo(shape, dt);
    std::string t = dtype_to_hlo(dt);
    int64_t total = 1;
    for (auto d : shape) total *= d;

    std::ostringstream ss;
    ss << "HloModule tpu_mse\n\n";
    ss << "%add_fn (lhs: " << t << "[], rhs: " << t << "[]) -> " << t << "[] {\n";
    ss << "  %lhs = " << t << "[] parameter(0)\n";
    ss << "  %rhs = " << t << "[] parameter(1)\n";
    ss << "  ROOT %res = " << t << "[] add(%lhs, %rhs)\n";
    ss << "}\n\n";

    ss << "ENTRY %main (pred: " << s << ", target: " << s << ") -> " << t << "[1] {\n";
    ss << "  %pred = " << s << " parameter(0)\n";
    ss << "  %target = " << s << " parameter(1)\n";
    ss << "  %diff = " << s << " subtract(%pred, %target)\n";
    ss << "  %sq = " << s << " multiply(%diff, %diff)\n";
    ss << "  %zero = " << t << "[] constant(0.0)\n";
    ss << "  %sum = " << t << "[] reduce(%sq, %zero), dimensions={";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << i;
        if (i + 1 < shape.size()) ss << ",";
    }
    ss << "}, to_apply=%add_fn\n";
    ss << "  %n_c = " << t << "[] constant(" << static_cast<float>(total) << ")\n";
    ss << "  %mean = " << t << "[] divide(%sum, %n_c)\n";
    ss << "  ROOT %out = " << t << "[1] reshape(%mean)\n";
    ss << "}\n";
    return ss.str();
}

// Emits an HLO Module for L1 (MAE) Loss
inline std::string build_l1_loss_hlo(const std::vector<int64_t>& shape, DType dt = DType::Float32) {
    std::string s = shape_to_hlo(shape, dt);
    std::string t = dtype_to_hlo(dt);
    int64_t total = 1;
    for (auto d : shape) total *= d;

    std::ostringstream ss;
    ss << "HloModule tpu_l1_loss\n\n";
    ss << "%add_fn (lhs: " << t << "[], rhs: " << t << "[]) -> " << t << "[] {\n";
    ss << "  %lhs = " << t << "[] parameter(0)\n";
    ss << "  %rhs = " << t << "[] parameter(1)\n";
    ss << "  ROOT %res = " << t << "[] add(%lhs, %rhs)\n";
    ss << "}\n\n";

    ss << "ENTRY %main (pred: " << s << ", target: " << s << ") -> " << t << "[1] {\n";
    ss << "  %pred = " << s << " parameter(0)\n";
    ss << "  %target = " << s << " parameter(1)\n";
    ss << "  %diff = " << s << " subtract(%pred, %target)\n";
    ss << "  %abs = " << s << " abs(%diff)\n";
    ss << "  %zero = " << t << "[] constant(0.0)\n";
    ss << "  %sum = " << t << "[] reduce(%abs, %zero), dimensions={";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << i;
        if (i + 1 < shape.size()) ss << ",";
    }
    ss << "}, to_apply=%add_fn\n";
    ss << "  %n_c = " << t << "[] constant(" << static_cast<float>(total) << ")\n";
    ss << "  %mean = " << t << "[] divide(%sum, %n_c)\n";
    ss << "  ROOT %out = " << t << "[1] reshape(%mean)\n";
    ss << "}\n";
    return ss.str();
}

} // namespace hlo
