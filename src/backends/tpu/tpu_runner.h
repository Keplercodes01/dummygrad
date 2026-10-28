#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <thread>
#include <stdexcept>
#include "pjrt_client.h"
#include "hlo_builder.h"
#include "core/tensor.h"

// High-performance TPU Execution Engine for Single & Multi-TPU Pod Slices
// Direct OpenXLA PJRT driver integration with zero Python and near-zero host latency
class TPUEngine {
private:
    size_t num_devices = 0;
    bool available = false;
    // Compiled executables by HLO text: compiling is expensive and each program is shape-specific,
    // so every distinct program compiles once and is reused
    std::unordered_map<std::string, PJRT_LoadedExecutable*> executables;
    std::mutex executables_mutex;

    PJRT_LoadedExecutable* executable(const std::string& hlo_src) {
        std::lock_guard<std::mutex> lock(executables_mutex);
        auto it = executables.find(hlo_src);
        if (it != executables.end()) return it->second;
        PJRT_LoadedExecutable* exec = PJRTTPUManager::get().compile_hlo(hlo_src);
        executables[hlo_src] = exec;
        return exec;
    }

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
            if (num_devices > 1) {
                std::cout << "[dummygrad TPU] Multi-TPU Pod slice active (" << num_devices 
                          << " chips). Hardware Inter-Chip Interconnect (ICI) enabled.\n";
            } else if (num_devices == 1) {
                std::cout << "[dummygrad TPU] Single-chip TPU active.\n";
            }
        } else {
            available = false;
        }
    }

    bool is_available() const { return available; }
    size_t device_count() const { return num_devices; }

    // Synchronize gradients across any arbitrary number of TPU chips via direct ICI ring reduction
    void all_reduce_gradients(
        std::vector<std::shared_ptr<TPUBufferHandle>>& per_device_gradients,
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32
    ) {
        if (!available || num_devices <= 1) return;

        auto& mgr = PJRTTPUManager::get();
        // One program per gradient shape (it used to compile once, for the first shape seen)
        PJRT_LoadedExecutable* all_reduce = executable(hlo::build_ici_all_reduce_hlo(shape, num_devices, dtype));

        // Run ICI all-reduce on each device asynchronously
        for (size_t d = 0; d < num_devices; ++d) {
            auto reduced = mgr.execute(all_reduce, { per_device_gradients[d] }, { shape }, { dtype }, d);
            if (!reduced.empty()) {
                per_device_gradients[d] = reduced[0];
            }
        }
    }

    // High-performance GEMM executed directly on TPU systolic MXU (128x128)
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
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { a, b }, { {M, N} }, { dtype }, device_id);

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
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { x, w, bias }, { {M, N} }, { dtype }, device_id);

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
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { q, k, v }, { {B, H, S, D} }, { dtype }, device_id);

        if (out.empty()) throw std::runtime_error("TPUEngine::attention execution failed");
        return out[0];
    }

    // Sigmoid executed directly on TPU VPU vector units
    std::shared_ptr<TPUBufferHandle> sigmoid(
        const std::shared_ptr<TPUBufferHandle>& x,
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");
        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_sigmoid_hlo(shape, dtype);
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { x }, { shape }, { dtype }, device_id);
        if (out.empty()) throw std::runtime_error("TPUEngine::sigmoid execution failed");
        return out[0];
    }

    // SiLU (Swish) executed directly on TPU VPU vector units
    std::shared_ptr<TPUBufferHandle> silu(
        const std::shared_ptr<TPUBufferHandle>& x,
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");
        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_silu_hlo(shape, dtype);
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { x }, { shape }, { dtype }, device_id);
        if (out.empty()) throw std::runtime_error("TPUEngine::silu execution failed");
        return out[0];
    }

    // LeakyReLU executed directly on TPU VPU vector units
    std::shared_ptr<TPUBufferHandle> leaky_relu(
        const std::shared_ptr<TPUBufferHandle>& x,
        const std::vector<int64_t>& shape,
        float negative_slope = 0.01f,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");
        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_leaky_relu_hlo(shape, negative_slope, dtype);
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { x }, { shape }, { dtype }, device_id);
        if (out.empty()) throw std::runtime_error("TPUEngine::leaky_relu execution failed");
        return out[0];
    }

    // MSE Loss reduction executed on TPU VPU vector units
    std::shared_ptr<TPUBufferHandle> mse(
        const std::shared_ptr<TPUBufferHandle>& pred,
        const std::shared_ptr<TPUBufferHandle>& target,
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");
        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_mse_hlo(shape, dtype);
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { pred, target }, { {1} }, { dtype }, device_id);
        if (out.empty()) throw std::runtime_error("TPUEngine::mse execution failed");
        return out[0];
    }

    // L1 (MAE) Loss reduction executed on TPU VPU vector units
    std::shared_ptr<TPUBufferHandle> l1_loss(
        const std::shared_ptr<TPUBufferHandle>& pred,
        const std::shared_ptr<TPUBufferHandle>& target,
        const std::vector<int64_t>& shape,
        DType dtype = DType::Float32,
        size_t device_id = 0
    ) {
        if (!available) throw std::runtime_error("TPUEngine: TPU runtime not available");
        auto& mgr = PJRTTPUManager::get();
        std::string hlo_src = hlo::build_l1_loss_hlo(shape, dtype);
        PJRT_LoadedExecutable* exec = executable(hlo_src);
        auto out = mgr.execute(exec, { pred, target }, { {1} }, { dtype }, device_id);
        if (out.empty()) throw std::runtime_error("TPUEngine::l1_loss execution failed");
        return out[0];
    }

    ~TPUEngine() {
        for (auto& entry : executables) PJRTTPUManager::get().destroy_executable(entry.second);
        executables.clear();
    }
};

// Generic Multi-TPU Distributed Data-Parallel Runner
// Scales across ANY arbitrary number of TPU chips (2, 4, 8, 16, 32, 64, 128...) over hardware ICI
template <typename ModelType>
class MultiTPURunner {
private:
    size_t num_chips = 0;
    std::vector<ModelType> replicas;

public:
    explicit MultiTPURunner(size_t chips, std::function<ModelType(size_t rank)> factory)
        : num_chips(chips) {
        auto& engine = TPUEngine::get();
        if (!engine.is_available()) {
            throw std::runtime_error("MultiTPURunner: TPU runtime unavailable");
        }
        if (num_chips > engine.device_count()) {
            throw std::runtime_error("MultiTPURunner requested " + std::to_string(num_chips) +
                                     " chips but only " + std::to_string(engine.device_count()) + " detected");
        }

        replicas.reserve(num_chips);
        for (size_t rank = 0; rank < num_chips; ++rank) {
            replicas.push_back(factory(rank));
        }
    }

    // Default constructor: automatically uses all detected TPU chips on the host/pod
    explicit MultiTPURunner(std::function<ModelType(size_t rank)> factory)
        : MultiTPURunner(TPUEngine::get().device_count(), factory) {}

    size_t world_size() const { return num_chips; }
    ModelType& get_replica(size_t rank) { return replicas[rank]; }

    // One data-parallel step: step_fn(rank, replica) runs on every replica at once (one thread per
    // chip; the first exception is rethrown after all finish), then each parameter's gradient is
    // averaged across replicas. The average is computed on the host, which holds TPU tensors'
    // authoritative copy, and written back to every replica.
    void parallel_step(std::function<void(size_t rank, ModelType& model)> step_fn) {
        std::vector<std::exception_ptr> errors(num_chips);
        std::vector<std::thread> workers;
        workers.reserve(num_chips);
        for (size_t rank = 0; rank < num_chips; ++rank) {
            workers.emplace_back([this, rank, &step_fn, &errors]() {
                try {
                    step_fn(rank, replicas[rank]);
                } catch (...) {
                    errors[rank] = std::current_exception();
                }
            });
        }
        for (auto& w : workers) w.join();
        for (auto& e : errors) {
            if (e) std::rethrow_exception(e);
        }
        average_gradients();
    }

private:
    void average_gradients() {
        if (num_chips <= 1) return;
        std::vector<std::vector<std::shared_ptr<Tensor>>> params;
        for (auto& model : replicas) params.push_back(model.parameters());
        for (size_t i = 0; i < params[0].size(); ++i) {
            std::shared_ptr<Tensor> total;
            for (size_t r = 0; r < num_chips; ++r) {
                const auto& g = params[r][i]->grad;
                if (g) tensor_add_inplace(total, make_contiguous(g)->to(Device::CPU, DType::Float32));
            }
            if (!total) continue;
            float* t = total->data_ptr<float>();
            for (int64_t k = 0; k < total->size(); ++k) t[k] /= (float)num_chips;
            for (size_t r = 0; r < num_chips; ++r) {
                const auto& p = params[r][i];
                p->grad = total->to(p->device, DType::Float32, (int)r);
                p->grad->requires_grad = false;
            }
        }
    }
};
