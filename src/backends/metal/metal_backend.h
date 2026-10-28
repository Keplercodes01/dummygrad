#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "core/types.h"

// Metal kernels for the MPS device. MPS tensors live in unified memory shared with the CPU; a few
// elementwise ops, masks, loss gradients and AdamW run as Metal kernels here, and everything else
// (matmuls included) computes on the CPU over the same memory.
class MetalBackend {
public:
    static MetalBackend& get();

    virtual ~MetalBackend() = default;

    virtual bool is_available() const = 0;
    virtual std::string device_name() const = 0;

    // Zero-Copy Unified Memory Allocation (CPU and GPU share physical address space)
    virtual void* allocate_shared_buffer(size_t bytes) = 0;
    virtual void free_buffer(void* ptr, size_t bytes) = 0;

    // Fused AdamW Optimizer
    virtual void adamw(
        float* p, const float* g, float* m, float* v,
        size_t total_params, float lr, float beta1, float beta2,
        float eps, float weight_decay, int step
    ) = 0;

    // Activations & Backward Passes
    virtual void sigmoid_forward(const float* in, float* out, size_t size) = 0;
    virtual void sigmoid_backward(const float* out, const float* grad_out, float* grad_in, size_t size) = 0;

    virtual void silu_forward(const float* in, float* out, size_t size) = 0;
    virtual void silu_backward(const float* in, const float* grad_out, float* grad_in, size_t size) = 0;

    virtual void leaky_relu_forward(const float* in, float* out, size_t size, float negative_slope = 0.01f) = 0;
    virtual void leaky_relu_backward(const float* in, const float* grad_out, float* grad_in, size_t size, float negative_slope = 0.01f) = 0;

    // Loss Backward Passes
    virtual void mse_backward(const float* pred, const float* target, float* grad_pred, size_t size, float scale) = 0;
    virtual void l1_loss_backward(const float* pred, const float* target, float* grad_pred, size_t size, float scale) = 0;

    // Attention & Modern Masking
    virtual void causal_mask(const float* in, float* out, size_t batch, size_t seq_len) = 0;
    virtual void causal_mask_backward(const float* grad_out, float* grad_in, size_t batch, size_t seq_len) = 0;
    virtual void sliding_window_mask(const float* in, float* out, size_t batch, size_t seq_len, size_t window_size) = 0;
    virtual void sliding_window_mask_backward(const float* grad_out, float* grad_in, size_t batch, size_t seq_len, size_t window_size) = 0;

    virtual void synchronize() = 0;
};

#ifndef USE_METAL
inline MetalBackend& MetalBackend::get() {
    class NullMetalBackend : public MetalBackend {
    public:
        bool is_available() const override { return false; }
        std::string device_name() const override { return "Metal unavailable (built without USE_METAL)"; }
        void* allocate_shared_buffer(size_t) override { throw std::runtime_error("MetalBackend not supported on non-Apple platform"); }
        void free_buffer(void*, size_t) override {}
        void adamw(float*, const float*, float*, float*, size_t, float, float, float, float, float, int) override {}
        void sigmoid_forward(const float*, float*, size_t) override {}
        void sigmoid_backward(const float*, const float*, float*, size_t) override {}
        void silu_forward(const float*, float*, size_t) override {}
        void silu_backward(const float*, const float*, float*, size_t) override {}
        void leaky_relu_forward(const float*, float*, size_t, float) override {}
        void leaky_relu_backward(const float*, const float*, float*, size_t, float) override {}
        void mse_backward(const float*, const float*, float*, size_t, float) override {}
        void l1_loss_backward(const float*, const float*, float*, size_t, float) override {}
        void causal_mask(const float*, float*, size_t, size_t) override {}
        void causal_mask_backward(const float*, float*, size_t, size_t) override {}
        void sliding_window_mask(const float*, float*, size_t, size_t, size_t) override {}
        void sliding_window_mask_backward(const float*, float*, size_t, size_t, size_t) override {}
        void synchronize() override {}
    };
    static NullMetalBackend instance;
    return instance;
}
#endif
