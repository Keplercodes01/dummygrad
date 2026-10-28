#pragma once
// Single-process data parallelism: one model replica per GPU, each running its share of the batch
// on its own thread, with gradients averaged across replicas after every backward pass.

#include "tensor.h"
#include "ops.h"

#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_common.h"
#include "cuda/cuda_ops.cuh"
#endif

namespace multi_gpu_detail {

#ifdef USE_CUDA
// The GPU a device pointer lives on
inline int device_of(const void* ptr) {
    cudaPointerAttributes attr{};
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
    return attr.device;
}

// Restores the calling thread's current GPU when it goes out of scope
struct DeviceGuard {
    int previous = 0;
    DeviceGuard() { cudaGetDevice(&previous); }
    ~DeviceGuard() { cudaSetDevice(previous); }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};
#endif

}  // namespace multi_gpu_detail

// Averages (average = false: sums) every parameter's gradient across model replicas and writes
// the result into each replica's gradient. replicas[r] lists replica r's parameters in the same
// order, e.g. Module::parameters() of identically built models. Contributions are added in replica
// order, so the result is deterministic; averaging divides once at the end, which is exact for a
// power-of-two replica count. A replica without a gradient for a parameter contributes zero and
// still receives the result.
//
// Call it from one thread after every replica's backward has finished. CUDA replicas may live on
// different GPUs: gradients are copied device to device (directly when the GPUs have peer access,
// see DeviceMesh, else staged through the host by the driver) and summed on replica 0's GPU.
inline void all_reduce_gradients(const std::vector<std::vector<std::shared_ptr<Tensor>>>& replicas, bool average = true) {
    size_t world = replicas.size();
    if (world <= 1) return;
    size_t n_params = replicas[0].size();
    for (const auto& r : replicas) {
        if (r.size() != n_params) throw std::runtime_error("all_reduce_gradients: replicas have different numbers of parameters");
    }
#ifdef USE_CUDA
    multi_gpu_detail::DeviceGuard restore_device;
#endif
    for (size_t i = 0; i < n_params; ++i) {
        const auto& root = replicas[0][i];
        bool any_grad = false;
        for (size_t r = 0; r < world; ++r) {
            const auto& p = replicas[r][i];
            if (!p || !root || p->shape != root->shape || p->device != root->device || p->dtype != DType::Float32) {
                throw std::runtime_error("all_reduce_gradients: parameter " + std::to_string(i) +
                                         " differs between replicas (shape, device or dtype) or is not Float32");
            }
            any_grad = any_grad || (p->grad != nullptr);
        }
        if (!any_grad) continue;

        // Every replica gets a dense Float32 gradient, zeros where it had none, on its own device
        for (size_t r = 0; r < world; ++r) {
            const auto& p = replicas[r][i];
#ifdef USE_CUDA
            if (p->device == Device::CUDA) CUDA_CHECK(cudaSetDevice(multi_gpu_detail::device_of(p->data_ptr<void>())));
#endif
            if (!p->grad) {
                p->grad = std::make_shared<Tensor>(p->shape, p->device, DType::Float32, false);
                p->grad->fill_(0.0f);
            } else if (!p->grad->is_contiguous()) {
                p->grad = make_contiguous(p->grad);
            }
            if (p->grad->dtype != DType::Float32 || p->grad->shape != p->shape) {
                throw std::runtime_error("all_reduce_gradients: the gradient of parameter " + std::to_string(i) +
                                         " is not a Float32 tensor of the parameter's shape");
            }
        }

        const auto& total = root->grad;  // reduced in place
        int64_t n = total->size();
        size_t bytes = (size_t)n * sizeof(float);
#ifdef USE_CUDA
        if (root->device == Device::CUDA) {
            int dev0 = multi_gpu_detail::device_of(total->data_ptr<void>());
            CUDA_CHECK(cudaSetDevice(dev0));
            // cudaMemcpyPeer is ordered after earlier work on both GPUs, and the add after the copy
            auto incoming = std::make_shared<Tensor>(total->shape, Device::CUDA, DType::Float32, false);
            for (size_t r = 1; r < world; ++r) {
                const auto& g = replicas[r][i]->grad;
                CUDA_CHECK(cudaMemcpyPeer(incoming->data_ptr<void>(), dev0, g->data_ptr<void>(),
                                          multi_gpu_detail::device_of(g->data_ptr<void>()), bytes));
                ops::add_inplace(*total, *incoming);
            }
            if (average) cuda::scalar_forward(3 /* divide */, total->data_ptr<float>(), (float)world, total->data_ptr<float>(), n);
            for (size_t r = 1; r < world; ++r) {
                const auto& g = replicas[r][i]->grad;
                CUDA_CHECK(cudaMemcpyPeer(g->data_ptr<void>(), multi_gpu_detail::device_of(g->data_ptr<void>()),
                                          total->data_ptr<void>(), dev0, bytes));
            }
            continue;
        }
#endif
        if (root->device != Device::CPU) throw std::runtime_error("all_reduce_gradients: parameters must be on the CPU or a CUDA GPU");
        for (size_t r = 1; r < world; ++r) ops::add_inplace(*total, *replicas[r][i]->grad);
        if (average) {
            float* t = total->data_ptr<float>();
            for (int64_t k = 0; k < n; ++k) t[k] /= (float)world;
        }
        for (size_t r = 1; r < world; ++r) std::memcpy(replicas[r][i]->grad->data_ptr<void>(), total->data_ptr<void>(), bytes);
    }
}

#ifdef USE_CUDA
// Enables direct peer access between every pair of GPUs that supports it (NVLink, or PCIe P2P),
// once per process. Copies between GPUs work either way; without peer access the driver stages
// them through host memory.
class DeviceMesh {
public:
    static DeviceMesh& get() {
        static DeviceMesh instance;
        return instance;
    }

    void init() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return;
        multi_gpu_detail::DeviceGuard restore_device;
        CUDA_CHECK(cudaGetDeviceCount(&num_devices_));
        peer_.assign((size_t)num_devices_ * (size_t)num_devices_, false);
        for (int i = 0; i < num_devices_; ++i) {
            CUDA_CHECK(cudaSetDevice(i));
            for (int j = 0; j < num_devices_; ++j) {
                if (i == j) continue;
                int can_access = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (!can_access) continue;
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                    peer_[(size_t)i * num_devices_ + j] = true;
                }
                cudaGetLastError();  // clear a failed or redundant enable
            }
        }
        initialized_ = true;
    }

    int num_devices() const { return num_devices_; }
    // Whether GPU from reads and writes GPU to's memory directly (valid after init)
    bool peer_access(int from, int to) const {
        if (from < 0 || to < 0 || from >= num_devices_ || to >= num_devices_) return false;
        return peer_[(size_t)from * num_devices_ + to];
    }

private:
    DeviceMesh() = default;
    std::mutex mutex_;
    bool initialized_ = false;
    int num_devices_ = 0;
    std::vector<bool> peer_;
};

// One model replica per GPU (replica r on GPU r) in a single process. parallel_step runs your step
// function on every replica at once, one thread per GPU, then averages the gradients across
// replicas, so every replica sees the same full-batch gradient. Give each rank its own shard of the
// batch, and step each replica's own optimizer after the reduction:
//
//   MultiGPURunner<ModernTransformer> runner(2, [](int rank) {
//       ModernTransformer model(32000, 2048, 512, 8, 4, 2);
//       model.cuda();  // the factory runs with GPU `rank` current
//       return model;
//   });
//   std::vector<Adam> opts(2, Adam(3e-4f));
//   runner.parallel_step([&](int rank, ModernTransformer& model) {
//       model.zero_grad();
//       fused_cross_entropy(model.forward(inputs[rank]), targets[rank])->backward();
//   });
//   runner.for_each_replica([&](int rank, ModernTransformer& model) { opts[rank].step(model.parameters()); });
template <typename ModelType>
class MultiGPURunner {
public:
    int world_size;
    // false sums the gradients instead (the effective learning rate then grows with world_size)
    bool average_gradients = true;
    std::vector<std::unique_ptr<ModelType>> replicas;

    template <typename FactoryFn>
    MultiGPURunner(int num_gpus, FactoryFn&& factory) : world_size(num_gpus) {
        auto& mesh = DeviceMesh::get();
        mesh.init();
        if (num_gpus < 1 || num_gpus > mesh.num_devices()) {
            throw std::runtime_error("MultiGPURunner: asked for " + std::to_string(num_gpus) + " GPUs, " +
                                     std::to_string(mesh.num_devices()) + " available");
        }
        multi_gpu_detail::DeviceGuard restore_device;
        replicas.reserve(num_gpus);
        for (int rank = 0; rank < num_gpus; ++rank) {
            CUDA_CHECK(cudaSetDevice(rank));
            replicas.push_back(std::make_unique<ModelType>(factory(rank)));
        }
        // Identical starting weights, whatever each factory call initialized
        broadcast_weights();
    }

    // Each replica's parameters, in Module::parameters() order
    std::vector<std::vector<std::shared_ptr<Tensor>>> replica_parameters() {
        std::vector<std::vector<std::shared_ptr<Tensor>>> out;
        for (auto& model : replicas) out.push_back(model->parameters());
        return out;
    }

    // Copies replica 0's parameter values into every other replica
    void broadcast_weights() {
        auto params = replica_parameters();
        for (int rank = 1; rank < world_size; ++rank) {
            if (params[rank].size() != params[0].size()) throw std::runtime_error("MultiGPURunner: replicas have different parameters");
            for (size_t i = 0; i < params[0].size(); ++i) {
                const auto& src = params[0][i];
                const auto& dst = params[rank][i];
                if (src->shape != dst->shape || src->dtype != dst->dtype || !src->is_contiguous() || !dst->is_contiguous()) {
                    throw std::runtime_error("MultiGPURunner: parameter " + std::to_string(i) + " differs between replicas");
                }
                CUDA_CHECK(cudaMemcpyPeer(dst->data_ptr<void>(), multi_gpu_detail::device_of(dst->data_ptr<void>()),
                                          src->data_ptr<void>(), multi_gpu_detail::device_of(src->data_ptr<void>()),
                                          (size_t)src->size() * dtype_size(src->dtype)));
            }
        }
    }

    // Runs fn(rank, replica) for every replica at once, one thread per GPU with that GPU current.
    // Waits for all of them, then rethrows the first exception any of them threw.
    template <typename Fn>
    void for_each_replica(Fn&& fn) {
        std::vector<std::exception_ptr> errors(world_size);
        std::vector<std::thread> workers;
        workers.reserve(world_size);
        for (int rank = 0; rank < world_size; ++rank) {
            workers.emplace_back([&, rank]() {
                try {
                    CUDA_CHECK(cudaSetDevice(rank));
                    fn(rank, *replicas[rank]);
                    CUDA_CHECK(cudaDeviceSynchronize());
                } catch (...) {
                    errors[rank] = std::current_exception();
                }
            });
        }
        for (auto& w : workers) w.join();
        for (auto& e : errors) {
            if (e) std::rethrow_exception(e);
        }
    }

    // One data-parallel step: step_fn(rank, replica) runs forward and backward on every replica at
    // once, then the gradients are averaged across replicas (all_reduce_gradients)
    template <typename StepFn>
    void parallel_step(StepFn&& step_fn) {
        for_each_replica(step_fn);
        all_reduce_gradients(replica_parameters(), average_gradients);
    }
};
#endif
