#pragma once
#include "types.h"
#include <stdexcept>

// Mixed precision: while an AutocastGuard is alive, CUDA matmuls and the products inside attention
// (its cuBLAS path, sequences up to attention_gemm_max_seq()) round their fp32 inputs to fp16 (or
// bf16) and run on tensor cores, accumulating and writing fp32. The softmax, every other op, and all
// parameters, gradients and optimizer state stay fp32. Backward uses the dtype each op's forward
// ran with, so backward() may be called outside the guard. Thread-local, like grad mode.
//
// fp16 gradients underflow without loss scaling; pair Float16 with a GradScaler. BFloat16 has
// fp32's range and needs no scaler, but its tensor cores need an Ampere or newer GPU.
//
//   { AutocastGuard autocast(DType::Float16); loss = loss_fn(model.forward(x), y); }
//   scaler.scale(loss)->backward();
inline thread_local DType g_autocast_dtype = DType::Float32;

inline DType autocast_dtype() { return g_autocast_dtype; }

struct AutocastGuard {
    DType previous;
    explicit AutocastGuard(DType dtype) : previous(g_autocast_dtype) {
        if (dtype != DType::Float32 && dtype != DType::Float16 && dtype != DType::BFloat16) {
            throw std::runtime_error("AutocastGuard: compute dtype must be Float32, Float16 or BFloat16");
        }
        g_autocast_dtype = dtype;
    }
    ~AutocastGuard() { g_autocast_dtype = previous; }
    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;
};
