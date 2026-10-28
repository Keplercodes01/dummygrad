#pragma once
#include "types.h"
#include <atomic>
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

// TF32: on Ampere and newer GPUs, Float32 matmuls (and the Float32 attention GEMMs) can round their
// inputs to TF32 (10 mantissa bits) on tensor cores, several times faster. Off by default, as in
// PyTorch, so Float32 means Float32; turn it on for speed when that precision is enough. Applies to
// the GEMMs launched after the call, process-wide. No effect on GPUs without TF32 (T4) or on the CPU.
#define DUMMYGRAD_HAS_TF32_SWITCH 1  // lets tools built against several versions (bench.cpp) check
inline std::atomic<bool> g_allow_tf32{false};

inline void set_allow_tf32(bool allow) { g_allow_tf32 = allow; }
inline bool allow_tf32() { return g_allow_tf32.load(std::memory_order_relaxed); }
