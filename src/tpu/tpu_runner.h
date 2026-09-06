#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <functional>
#include "pjrt_client.h"
#include "hlo_builder.h"
#include "../tensor.h"

// High-performance TPU Execution Engine for Google Colab v5e-1 & Kaggle v5e-8
// Direct PJRT driver integration with zero Python and near-zero host latency
class TPUEngine {
private:
    size_t num_devices = 0;
    bool available = false;
    PJRT_LoadedExecutable* ici_all_reduce_exec = nullptr;

public:
    static TPUEngine& get() {
        static TPUEngine instance;
        return instance;
    }

    TPUEngine() {
        auto& mgr = PJRTTPUManager::get();
        if (mgr.is_available()) {
            num_devices = mgr.device_count();
            available = true;
            std::cout << "[dummygrad TPU] Detected " << num_devices << " TPU device(s) via OpenXLA PJRT.\n";
            if (num_devices >= 8) {
                std::cout << "[dummygrad TPU] Kaggle v5e-8 configuration active. Inter-Chip Interconnect (ICI) enabled.\n";
            } else if (num_devices == 1) {
                std::cout << "[dummygrad TPU] Google Colab v5e-1 single-chip configuration active.\n";
            }
        } else {
            available = false;
        }
    }

    bool is_available() const { return available; }
    size_t device_count() const { return num_devices; }

    // Synchronize gradients across all TPU chips via direct ICI ring reduction
    void all_reduce_gradients(
        std::vector<std::shared_ptr<TPUBufferHandle>>& per_device_gradients,
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32
    ) {
        if (!available || num_devices <= 1) return;

        auto& mgr = PJRTTPUManager::get();
        if (!ici_all_reduce_exec) {
            std::string hlo_src = hlo::build_ici_all_reduce_hlo(shape, num_devices, dtype);
            ici_all_reduce_exec = mgr.compile_hlo(hlo_src);
        }

        // Run ICI all-reduce on each device asynchronously
        for (size_t d = 0; d < num_devices; ++d) {
            auto reduced = mgr.execute(ici_all_reduce_exec, { per_device_gradients[d] }, { shape }, { dtype }, d);
            if (!reduced.empty()) {
                per_device_gradients[d] = reduced[0];
            }
        }
    }

    // High-performance GEMM executed directly on TPU v5e systolic MXU
    std::shared_ptr<TPUBufferHandle> matmul(
        const std::shared_ptr<TPUBufferHandle>& a,
        const std::shared_ptr<TPUBufferHandle>& b,
        int64_t M, int64_t K, int64_t N,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");

        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_gemm_hlo(M, K, N, dtype);
        PJRT_LoadedExecutable* exec = mgr.compile_hlo(hlo_src);
        auto out = mgr.execute(exec, { a, b }, { {M, N} }, { dtype }, device_id);
        mgr.destroy_executable(exec);

        if (out.empty()) throw std::runtime_error("TPUEngine::matmul execution failed");
        return out[0];
    }

    // Fused Linear + GELU executed in a single systolic pipeline pass
    std::shared_ptr<TPUBufferHandle> fused_linear_gelu(
        const std::shared_ptr<TPUBufferHandle>& x,
        const std::shared_ptr<TPUBufferHandle>& w,
        const std::shared_ptr<TPUBufferHandle>& bias,
        int64_t M, int64_t K, int64_t N,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");

        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_fused_linear_gelu_hlo(M, K, N, dtype);
        PJRT_LoadedExecutable* exec = mgr.compile_hlo(hlo_src);
        auto out = mgr.execute(exec, { x, w, bias }, { {M, N} }, { dtype }, device_id);
        mgr.destroy_executable(exec);

        if (out.empty()) throw std::runtime_error("TPUEngine::fused_linear_gelu execution failed");
        return out[0];
    }

    // FlashAttention executed on TPU VPU/MXU vector tiles
    std::shared_ptr<TPUBufferHandle> attention(
        const std::shared_ptr<TPUBufferHandle>& q,
        const std::shared_ptr<TPUBufferHandle>& k,
        const std::shared_ptr<TPUBufferHandle>& v,
        int64_t B, int64_t H, int64_t S, int64_t D,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");

        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_tpu_attention_hlo(B, H, S, D, dtype);
        PJRT_LoadedExecutable* exec = mgr.compile_hlo(hlo_src);
        auto out = mgr.execute(exec, { q, k, v }, { {B, H, S, D} }, { dtype }, device_id);
        mgr.destroy_executable(exec);

        if (out.empty()) throw std::runtime_error("TPUEngine::attention execution failed");
        return out[0];
    }

    ~TPUEngine() {
        if (ici_all_reduce_exec) {
            PJRTTPUManager::get().destroy_executable(ici_all_reduce_exec);
            ici_all_reduce_exec = nullptr;
        }
    }
};
