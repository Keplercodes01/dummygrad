#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "../types.h"

// Pure C++ Interface for Apple Silicon Metal & MPS Acceleration
// Zero-overhead dispatch to Apple Matrix Coprocessor (AMX) and Apple GPUs
class MetalBackend {
public:
    static MetalBackend& get();

    virtual ~MetalBackend() = default;

    virtual bool is_available() const = 0;
    virtual std::string device_name() const = 0;

    // Zero-Copy Unified Memory Allocation (CPU and GPU share physical address space)
    virtual void* allocate_shared_buffer(size_t bytes) = 0;
    virtual void free_buffer(void* ptr, size_t bytes) = 0;

    // Apple MPS Matrix Multiplication (AMX Hardware Accelerated)
    virtual void matmul(
        const float* A, const float* B, float* C,
        int64_t M, int64_t K, int64_t N
    ) = 0;

    // Fused Bias + Activation Kernels
    virtual void fused_add_bias_gelu(
        const float* in, const float* bias, float* out,
        size_t total_elems, size_t cols
    ) = 0;

    virtual void fused_add_bias_relu(
        const float* in, const float* bias, float* out,
        size_t total_elems, size_t cols
    ) = 0;

    // Fused LayerNorm with SIMD-Group Reductions
    virtual void layernorm(
        const float* in, const float* gamma, const float* beta, float* out,
        size_t rows, size_t cols, float eps = 1e-5f
    ) = 0;

    // Fused AdamW Optimizer
    virtual void adamw(
        float* p, const float* g, float* m, float* v,
        size_t total_params, float lr, float beta1, float beta2,
        float eps, float weight_decay, int step
    ) = 0;

    // Tiled FlashAttention Forward Pass
    virtual void flash_attention(
        const float* Q, const float* K, const float* V, float* Out,
        size_t num_heads, size_t seq_len, size_t head_dim
    ) = 0;

    virtual void synchronize() = 0;
};
