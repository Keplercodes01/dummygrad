#pragma once
#include "tensor.h"
#include "autocast.h"
#include "profiler.h"
#include "ops.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.cuh"
// For the few runtime calls made directly from op headers
inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
#endif
#include "metal/metal_backend.h"
#include "tpu/tpu_runner.h"

#include "cpu/cpu_gemm.h"
#include "fn_node.h"
#include "kernels/kernels.h"
#include "parallel.h"
#include "simd_math.h"

// Elementwise host loop over n elements, split over threads. Capture by value ([=]) so pointers stay
// in registers and the loop vectorizes.
template <typename F>
inline void host_loop(int64_t n, const F& body) {
    parallel_for(0, n, 16384, [&](int64_t begin, int64_t end) {
        const F f = body;
        for (int64_t i = begin; i < end; ++i) f(i);
    });
}

// ========================================
// Contiguity
// ========================================
// --- CONTIGUOUS BACKWARD NODE ---
struct ContiguousBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        return {grads[0]};
    }
};

// Whether an op should record a graph edge for input t: false for null tensors, tensors that
// don't require grad, and everything inside a NoGradGuard.
inline bool needs_grad(const std::shared_ptr<Tensor>& t) {
    return grad_enabled() && t && t->requires_grad;
}

// What a backward node keeps of its op's output: the same memory and shape, without the grad_fn.
// Saving the output tensor itself would form a cycle (output -> grad_fn -> output) that keeps the
// node, and through its edges the whole graph, alive whenever backward never runs on it.
inline std::shared_ptr<Tensor> saved_output(const std::shared_ptr<Tensor>& out) {
    return std::make_shared<Tensor>(out->storage, out->shape, out->strides, out->global_offset, out->device, out->dtype,
                                    false);
}

// Dense copy of a strided view (transpose, slice) that stays connected to the autograd graph.
// Ops whose kernels index raw memory call this on every tensor input; it returns the input
// itself when it is already contiguous, so the common case costs nothing.
inline std::shared_ptr<Tensor> contiguous(const std::shared_ptr<Tensor>& a) {
    if (!a || a->is_contiguous()) return a;
    DG_PROFILE_OP("contiguous");
    auto out = make_contiguous(a);
    out->requires_grad = needs_grad(a);
    if (needs_grad(a)) {
        auto grad_fn = std::make_shared<ContiguousBackward>();
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

// ========================================
// Dtype conversion
// ========================================
inline bool is_float_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}

struct CastBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    DType input_dtype;
    explicit CastBackward(DType input_dtype) : input_dtype(input_dtype) {}
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        return {grads[0]->to(input_dtype)};
    }
};

// Converts between Float32, Float16 and BFloat16 (round to nearest even when narrowing), on the
// tensor's own device. Differentiable: the gradient is converted back to the input's dtype.
// Returns the input itself when it already has the dtype.
inline std::shared_ptr<Tensor> cast(const std::shared_ptr<Tensor>& a, DType dtype) {
    if (!a) throw std::invalid_argument("cast: null tensor");
    if (a->dtype == dtype) return a;
    if (!is_float_dtype(a->dtype) || !is_float_dtype(dtype)) {
        throw std::invalid_argument("cast: only Float32, Float16 and BFloat16 convert differentiably");
    }
    DG_PROFILE_OP("cast");
    auto out = a->to(a->device, dtype);
    out->requires_grad = needs_grad(a);
    if (needs_grad(a)) {
        auto grad_fn = std::make_shared<CastBackward>(a->dtype);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

// Ops without a native half-precision kernel compute in fp32: they pass their inputs through
// this, which converts a Float16 or BFloat16 tensor to Float32 (differentiably) and returns
// anything else untouched. Such an op returns fp32 for a half-precision input: an unsupported
// dtype costs a conversion, never precision or an error.
inline std::shared_ptr<Tensor> float32_input(const std::shared_ptr<Tensor>& a) {
    return (a && (a->dtype == DType::Float16 || a->dtype == DType::BFloat16)) ? cast(a, DType::Float32) : a;
}

// Under autocast, the ops with half-precision GPU kernels (relu, gelu, silu, the product of two equal
// shapes, RoPE, GQA repetition, attention) keep a tensor that is already in the autocast dtype in it.
// Outside autocast, or on any other device, a half-precision input computes in fp32 as before.
inline bool native_half(const std::shared_ptr<Tensor>& a) {
    return a && a->device == Device::CUDA && a->dtype != DType::Float32 && a->dtype == autocast_dtype();
}
inline std::shared_ptr<Tensor> native_input(const std::shared_ptr<Tensor>& a) { return native_half(a) ? a : float32_input(a); }

// A TPU op writes its result to a TPU buffer only; mirror it into the tensor's host memory, which is
// the authoritative copy every other op reads (see host_written in ops.cpp)
inline void sync_tpu_result(const std::shared_ptr<Tensor>& t) {
    PJRTTPUManager::get().copy_to_host(*t->storage->tpu_handle, t->data_ptr<void>(), (size_t)t->size() * dtype_size(t->dtype));
}

// ========================================
// From activations.h
// ========================================
// --- RELU BACKWARD NODE ---
struct ReluBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> a;
    explicit ReluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::relu_backward(a->data_ptr<void>(), grad->to(a->dtype)->data_ptr<void>(), da->data_ptr<void>(), a->size(), a->dtype);
            return {da};
        }
#endif
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int64_t size = a->size();

        host_loop(size, [=](int64_t i) {
            da_ptr[i] = (a_ptr[i] > 0.0f) ? g_ptr[i] : 0.0f;
        });
        return {da};
    }
};

// relu
inline std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("relu");
    auto a = contiguous(native_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::relu_forward(a->data_ptr<void>(), out->data_ptr<void>(), a->size(), a->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<ReluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int64_t size = a->size();

    host_loop(size, [=](int64_t i) {
        out_ptr[i] = a_ptr[i] > 0.0f ? a_ptr[i] : 0.0f;
    });

    if (req_grad) {
        auto grad_fn = std::make_shared<ReluBackward>(a);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- TANH BACKWARD NODE ---
struct TanhBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> out_val;
    explicit TanhBackward(std::shared_ptr<Tensor> out_val) : out_val(out_val) {}

    void release_variables() override {
        out_val.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !out_val) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
#ifdef USE_CUDA
        if (out_val->device == Device::CUDA) {
            auto da = std::make_shared<Tensor>(out_val->shape, out_val->device, out_val->dtype, false);
            cuda::tanh_backward(out_val->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), out_val->size());
            return {da};
        }
#endif
        Device dev = out_val->device;
        auto grad_cpu = (dev == Device::CPU) ? grad : grad->cpu();
        auto out_cpu = (dev == Device::CPU) ? out_val : out_val->cpu();
        auto da = std::make_shared<Tensor>(out_cpu->shape, Device::CPU, out_cpu->dtype, false);
        const float* g_ptr = grad_cpu->data_ptr<float>();
        const float* out_ptr = out_cpu->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int64_t size = out_cpu->size();

        host_loop(size, [=](int64_t i) {
            float t = out_ptr[i];
            da_ptr[i] = (1.0f - t * t) * g_ptr[i];
        });
        return {(dev == Device::CPU) ? da : da->to(dev)};
    }
};

// tanh
inline std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("tanh");
    auto a = contiguous(float32_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
        cuda::tanh_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<TanhBackward>(saved_output(out));
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    Device dev = a->device;
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto out_cpu = std::make_shared<Tensor>(a_cpu->shape, Device::CPU, a_cpu->dtype, false);
    const float* a_ptr = a_cpu->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();
    int64_t size = a_cpu->size();

    host_loop(size, [=](int64_t i) {
        out_ptr[i] = simd::tanh(a_ptr[i]);
    });

    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<TanhBackward>(saved_output(out));
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- GELU BACKWARD NODE ---
struct GeluBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> a;
    float sqrt_2_over_pi;
    float coeff;
    GeluBackward(std::shared_ptr<Tensor> a, float sqrt_2_over_pi, float coeff)
        : a(a), sqrt_2_over_pi(sqrt_2_over_pi), coeff(coeff) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::gelu_backward(a->data_ptr<void>(), grad->to(a->dtype)->data_ptr<void>(), da->data_ptr<void>(), a->size(), a->dtype);
            return {da};
        }
#endif
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int64_t size = a->size();
        const float sqrt_2_over_pi = this->sqrt_2_over_pi, coeff = this->coeff;

        host_loop(size, [=](int64_t i) {
            float x = a_ptr[i];
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float tanh_val = simd::tanh(inner);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
            float g = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
            da_ptr[i] = g * g_ptr[i];
        });
        return {da};
    }
};

// gelu
inline std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("gelu");
    auto a = contiguous(native_input(a_in));
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::gelu_forward(a->data_ptr<void>(), out->data_ptr<void>(), a->size(), a->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<GeluBackward>(a, sqrt_2_over_pi, coeff);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int64_t size = a->size();

    host_loop(size, [=](int64_t i) {
        float x = a_ptr[i];
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float tanh_val = simd::tanh(inner);
        out_ptr[i] = 0.5f * x * (1.0f + tanh_val);
    });

    if (req_grad) {
        auto grad_fn = std::make_shared<GeluBackward>(a, sqrt_2_over_pi, coeff);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SIGMOID BACKWARD NODE ---
struct SigmoidBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> out_val;
    explicit SigmoidBackward(std::shared_ptr<Tensor> out_val) : out_val(out_val) {}

    void release_variables() override {
        out_val.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !out_val) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(out_val->shape, out_val->device, out_val->dtype, false);
#ifdef USE_CUDA
        if (out_val->device == Device::CUDA) {
            cuda::sigmoid_backward(out_val->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), out_val->size());
            return {da};
        }
#endif
        if (out_val->device == Device::MPS) {
            MetalBackend::get().sigmoid_backward(out_val->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), out_val->size());
            return {da};
        }
        const float* g_ptr = grad->data_ptr<float>();
        const float* o_ptr = out_val->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int64_t size = out_val->size();

        host_loop(size, [=](int64_t i) {
            float s = o_ptr[i];
            da_ptr[i] = s * (1.0f - s) * g_ptr[i];
        });
        return {da};
    }
};

// sigmoid
inline std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("sigmoid");
    auto a = contiguous(float32_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::sigmoid_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(saved_output(out));
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().sigmoid_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(saved_output(out));
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    if (a->device == Device::TPU && a->storage->tpu_handle) {
        out->storage->tpu_handle = TPUEngine::get().sigmoid(a->storage->tpu_handle, a->shape, a->dtype);
        sync_tpu_result(out);
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(saved_output(out));
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int64_t size = a->size();

    host_loop(size, [=](int64_t i) {
        out_ptr[i] = 1.0f / (1.0f + simd::exp(-a_ptr[i]));
    });

    if (req_grad) {
        auto grad_fn = std::make_shared<SigmoidBackward>(saved_output(out));
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SILU (SWISH) BACKWARD NODE ---
struct SiluBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> a;
    explicit SiluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::silu_backward(a->data_ptr<void>(), grad->to(a->dtype)->data_ptr<void>(), da->data_ptr<void>(), a->size(), a->dtype);
            return {da};
        }
#endif
        if (a->device == Device::MPS) {
            MetalBackend::get().silu_backward(a->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), a->size());
            return {da};
        }
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int64_t size = a->size();

        host_loop(size, [=](int64_t i) {
            float x = a_ptr[i];
            float s = 1.0f / (1.0f + simd::exp(-x));
            da_ptr[i] = (s * (1.0f + x * (1.0f - s))) * g_ptr[i];
        });
        return {da};
    }
};

// silu (Swish)
inline std::shared_ptr<Tensor> silu(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("silu");
    auto a = contiguous(native_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::silu_forward(a->data_ptr<void>(), out->data_ptr<void>(), a->size(), a->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().silu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    if (a->device == Device::TPU && a->storage->tpu_handle) {
        out->storage->tpu_handle = TPUEngine::get().silu(a->storage->tpu_handle, a->shape, a->dtype);
        sync_tpu_result(out);
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int64_t size = a->size();

    host_loop(size, [=](int64_t i) {
        float x = a_ptr[i];
        out_ptr[i] = x / (1.0f + simd::exp(-x));
    });

    if (req_grad) {
        auto grad_fn = std::make_shared<SiluBackward>(a);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

inline std::shared_ptr<Tensor> swish(const std::shared_ptr<Tensor>& a) {
    return silu(a);
}

// --- LEAKY RELU BACKWARD NODE ---
struct LeakyReluBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> a;
    float negative_slope;
    LeakyReluBackward(std::shared_ptr<Tensor> a, float negative_slope)
        : a(a), negative_slope(negative_slope) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::leaky_relu_backward(a->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), a->size(), negative_slope);
            return {da};
        }
#endif
        if (a->device == Device::MPS) {
            MetalBackend::get().leaky_relu_backward(a->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), a->size(), negative_slope);
            return {da};
        }
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int64_t size = a->size();
        const float negative_slope = this->negative_slope;

        host_loop(size, [=](int64_t i) {
            da_ptr[i] = (a_ptr[i] > 0.0f) ? g_ptr[i] : (negative_slope * g_ptr[i]);
        });
        return {da};
    }
};

// leaky_relu
inline std::shared_ptr<Tensor> leaky_relu(const std::shared_ptr<Tensor>& a_in, float negative_slope = 0.01f) {
    DG_PROFILE_OP("leaky_relu");
    auto a = contiguous(float32_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::leaky_relu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size(), negative_slope);
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().leaky_relu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size(), negative_slope);
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    if (a->device == Device::TPU && a->storage->tpu_handle) {
        out->storage->tpu_handle = TPUEngine::get().leaky_relu(a->storage->tpu_handle, a->shape, negative_slope, a->dtype);
        sync_tpu_result(out);
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int64_t size = a->size();

    host_loop(size, [=](int64_t i) {
        float x = a_ptr[i];
        out_ptr[i] = (x > 0.0f) ? x : (negative_slope * x);
    });

    if (req_grad) {
        auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From broadcasting.h
// ========================================
// ========================================
// Broadcast binary ops: add, sub, mul, div
// ========================================
// NumPy-style broadcasting: shapes are aligned from the right, and dimensions must be equal or 1.
// The kernels broadcast y onto x; when x broadcasts too, it is expanded first. Padding masks [B, 1, 1, S], ALiBi biases
// [1, H, S, S] and Linear biases [1, out] all broadcast this way.
enum class BinaryOp : int { Add = 0, Sub = 1, Mul = 2, Div = 3 };

inline std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float s);
inline std::shared_ptr<Tensor> expand(const std::shared_ptr<Tensor>& x, const std::vector<int64_t>& shape);
inline std::shared_ptr<Tensor> unsqueeze(const std::shared_ptr<Tensor>& x, int dim);
inline std::shared_ptr<Tensor> squeeze(const std::shared_ptr<Tensor>& x, int dim);
namespace op_detail {
inline std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b, const char* who);
inline std::shared_ptr<Tensor> softmax_any(const std::shared_ptr<Tensor>& x, int dim, bool log);
}

inline const char* binary_op_name(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return "add";
        case BinaryOp::Sub: return "sub";
        case BinaryOp::Mul: return "mul";
        default: return "div";
    }
}

inline std::string shape_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

// Indexing plan for broadcasting y onto x. Shapes are simplified before use: dimensions of size 1
// are dropped and neighbouring dimensions with the same broadcast status are merged, so a Linear
// bias [B, T, O] + [1, O] becomes [B*T, O] + [1, O]. Both the CPU loops and the CUDA kernels index
// with these collapsed shapes.
struct BroadcastLayout {
    std::vector<int64_t> x_shape, y_shape;  // collapsed, same rank (>= 1)
    std::vector<int64_t> y_strides;         // 0 where y broadcasts
    int64_t numel = 1;

    // Calls f(x_index, y_index) for every element of x in order. The innermost dimension is a
    // plain loop and outer dimensions advance like an odometer, so no division per element.
    template <typename F>
    void for_each(F&& f) const {
        int nd = (int)x_shape.size();
        int64_t inner = x_shape[nd - 1];
        int64_t inner_y = y_strides[nd - 1];
        int64_t outer = numel / inner;
        std::vector<int64_t> idx(nd, 0);
        int64_t y_base = 0;
        for (int64_t o = 0; o < outer; ++o) {
            int64_t xb = o * inner;
            if (inner_y) {
                for (int64_t j = 0; j < inner; ++j) f(xb + j, y_base + j);
            } else {
                for (int64_t j = 0; j < inner; ++j) f(xb + j, y_base);
            }
            for (int d = nd - 2; d >= 0; --d) {
                y_base += y_strides[d];
                if (++idx[d] < x_shape[d]) break;
                y_base -= y_strides[d] * x_shape[d];
                idx[d] = 0;
            }
        }
    }
};

inline BroadcastLayout broadcast_layout(const std::vector<int64_t>& xs, const std::vector<int64_t>& ys, BinaryOp op) {
    std::vector<int64_t> x = xs.empty() ? std::vector<int64_t>{1} : xs;
    size_t rank = x.size();
    // Align y to x's rank; y may have extra leading dimensions only if they are 1
    size_t extra = ys.size() > rank ? ys.size() - rank : 0;
    bool ok = true;
    for (size_t i = 0; i < extra; ++i) ok &= (ys[i] == 1);
    std::vector<int64_t> y(rank, 1);
    for (size_t i = extra; i < ys.size(); ++i) y[rank - (ys.size() - i)] = ys[i];
    for (size_t d = 0; d < rank; ++d) ok &= (y[d] == 1 || y[d] == x[d]);
    if (!ok) {
        throw std::invalid_argument(std::string(binary_op_name(op)) + ": cannot broadcast shape " + shape_string(ys) +
                                    " onto " + shape_string(xs));
    }

    BroadcastLayout l;
    for (size_t d = 0; d < rank; ++d) {
        l.numel *= x[d];
        if (x[d] == 1) continue;  // size-1 dimensions don't affect indexing
        bool broadcast = (y[d] == 1);
        if (!l.x_shape.empty() && (l.y_shape.back() == 1) == broadcast) {
            l.x_shape.back() *= x[d];
            l.y_shape.back() *= y[d];
        } else {
            l.x_shape.push_back(x[d]);
            l.y_shape.push_back(y[d]);
        }
    }
    if (l.x_shape.empty()) {
        l.x_shape = {1};
        l.y_shape = {1};
    }
    l.y_strides.assign(l.x_shape.size(), 0);
    int64_t stride = 1;
    for (int d = (int)l.x_shape.size() - 1; d >= 0; --d) {
        if (l.y_shape[d] != 1) l.y_strides[d] = stride;
        stride *= l.y_shape[d];
    }
    return l;
}

namespace broadcast_detail {

// x dense, y with its broadcast strides, out dense
inline kern::Grid grid(const BroadcastLayout& l) {
    auto x_strides = make_strides(l.x_shape);
    return kern::make_grid(l.x_shape, {x_strides, l.y_strides, x_strides});
}

inline void forward_host(BinaryOp op, const float* x, const float* y, float* out, const BroadcastLayout& l) {
    static const kern::Binary ops[] = {kern::Binary::Add, kern::Binary::Sub, kern::Binary::Mul, kern::Binary::Div};
    kern::host::binary(ops[(int)op], x, y, out, grid(l));
}

// dL/dx for mul and div (for add and sub it is the incoming gradient itself)
inline void grad_x_host(BinaryOp op, const float* g, const float* y, float* gx, const BroadcastLayout& l) {
    kern::host::binary(op == BinaryOp::Mul ? kern::Binary::Mul : kern::Binary::Div, g, y, gx, grid(l));
}

// dL/dy accumulated in x order, so the result is deterministic. gy must start zeroed.
inline void grad_y_host(BinaryOp op, const float* g, const float* x, const float* y, float* gy, const BroadcastLayout& l) {
    if (l.x_shape == l.y_shape && (op == BinaryOp::Mul || op == BinaryOp::Div)) {
        if (op == BinaryOp::Mul) host_loop(l.numel, [=](int64_t i) { gy[i] = g[i] * x[i]; });
        else host_loop(l.numel, [=](int64_t i) { gy[i] = -(g[i] * x[i]) / (y[i] * y[i]); });
        return;
    }
    if (l.x_shape.size() == 2 && l.y_shape[0] == 1 && l.y_shape[1] != 1) {
        // One value per column (a bias or gain): whole rows at a time, split over columns
        int64_t rows = l.x_shape[0], cols = l.x_shape[1];
        parallel_for(0, cols, std::max<int64_t>(16, 32768 / std::max<int64_t>(1, rows)), [&](int64_t b, int64_t e) {
            for (int64_t r = 0; r < rows; ++r) {
                const float* gr = g + r * cols;
                const float* xr = x ? x + r * cols : nullptr;
                switch (op) {
                    case BinaryOp::Add: for (int64_t j = b; j < e; ++j) gy[j] += gr[j]; break;
                    case BinaryOp::Sub: for (int64_t j = b; j < e; ++j) gy[j] -= gr[j]; break;
                    case BinaryOp::Mul: for (int64_t j = b; j < e; ++j) gy[j] += gr[j] * xr[j]; break;
                    default: for (int64_t j = b; j < e; ++j) gy[j] -= gr[j] * xr[j] / (y[j] * y[j]); break;
                }
            }
        });
        return;
    }
    switch (op) {
        case BinaryOp::Add: l.for_each([&](int64_t i, int64_t yo) { gy[yo] += g[i]; }); break;
        case BinaryOp::Sub: l.for_each([&](int64_t i, int64_t yo) { gy[yo] -= g[i]; }); break;
        case BinaryOp::Mul: l.for_each([&](int64_t i, int64_t yo) { gy[yo] += g[i] * x[i]; }); break;
        default: l.for_each([&](int64_t i, int64_t yo) { gy[yo] -= g[i] * x[i] / (y[yo] * y[yo]); }); break;
    }
}

// Whether y broadcasts onto x without x changing shape
inline bool onto(const std::vector<int64_t>& x, const std::vector<int64_t>& y) {
    size_t extra = y.size() > x.size() ? y.size() - x.size() : 0;
    for (size_t i = 0; i < extra; ++i) if (y[i] != 1) return false;
    for (size_t i = extra; i < y.size(); ++i) {
        int64_t xd = x[x.size() - (y.size() - i)];
        if (y[i] != 1 && y[i] != xd) return false;
    }
    return true;
}

}  // namespace broadcast_detail

struct BroadcastBinaryBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    BinaryOp op;
    std::vector<int64_t> x_shape, y_shape;  // as given by the caller
    BroadcastLayout layout;
    std::shared_ptr<Tensor> x, y;           // saved only for mul and div
    bool same_shape = false;                // a CUDA product of equal shapes: both gradients in one kernel

    BroadcastBinaryBackward(BinaryOp op, std::vector<int64_t> x_shape, std::vector<int64_t> y_shape, BroadcastLayout layout,
                            std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y)
        : op(op), x_shape(std::move(x_shape)), y_shape(std::move(y_shape)), layout(std::move(layout)),
          x(std::move(x)), y(std::move(y)) {}

    void release_variables() override {
        x.reset();
        y.reset();
    }

    std::string name() const override { return std::string("BroadcastBinaryBackward(") + binary_op_name(op) + ")"; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr, nullptr};
        bool need_x = next_edges.size() > 0 && next_edges[0].function;
        bool need_y = next_edges.size() > 1 && next_edges[1].function;
        bool uses_values = (op == BinaryOp::Mul || op == BinaryOp::Div);
        if (uses_values && (!x || !y)) return {nullptr, nullptr};
        auto g = grads[0];
        Device dev = g->device;
        std::shared_ptr<Tensor> gx, gy;
#ifdef USE_CUDA
        if (same_shape) {
            g = g->to(x->dtype);
            if (need_x) gx = std::make_shared<Tensor>(x_shape, dev, x->dtype, false);
            if (need_y) gy = std::make_shared<Tensor>(y_shape, dev, x->dtype, false);
            cuda::mul_backward(g->data_ptr<void>(), x->data_ptr<void>(), y->data_ptr<void>(), gx ? gx->data_ptr<void>() : nullptr,
                               gy ? gy->data_ptr<void>() : nullptr, g->size(), x->dtype);
            return {gx, gy};
        }
#endif

        // add/sub: dL/dx is the incoming gradient itself (the output has x's shape)
        if (need_x && !uses_values) gx = g;
        // add/sub with equal shapes: dL/dy is the gradient (or its negation)
        if (need_y && !uses_values && layout.y_shape == layout.x_shape) {
            gy = (op == BinaryOp::Add) ? g->_reshape(y_shape) : mul_scalar(g, -1.0f)->_reshape(y_shape);
            need_y = false;
        }
        need_x = need_x && uses_values;
        if (!need_x && !need_y) return {gx, gy};

#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            if (need_x) {
                gx = std::make_shared<Tensor>(x_shape, dev, g->dtype, false);
                cuda::broadcast_binary_backward_x((int)op, g->data_ptr<float>(), y->data_ptr<float>(), gx->data_ptr<float>(),
                                                  layout.x_shape, layout.y_shape);
            }
            if (need_y) {
                gy = std::make_shared<Tensor>(y_shape, dev, g->dtype, false);
                cuda::broadcast_binary_backward_y((int)op, g->data_ptr<float>(), uses_values ? x->data_ptr<float>() : nullptr,
                                                  uses_values ? y->data_ptr<float>() : nullptr, gy->data_ptr<float>(),
                                                  layout.x_shape, layout.y_shape);
            }
            return {gx, gy};
        }
#endif
        // Host: CPU tensors directly; other devices through CPU copies
        auto host = [](const std::shared_ptr<Tensor>& t) { return (!t || t->device == Device::CPU) ? t : t->cpu(); };
        auto gh = host(g), xh = uses_values ? host(x) : nullptr, yh = uses_values ? host(y) : nullptr;
        if (need_x) {
            auto gxh = std::make_shared<Tensor>(x_shape, Device::CPU, g->dtype, false);
            broadcast_detail::grad_x_host(op, gh->data_ptr<float>(), yh->data_ptr<float>(), gxh->data_ptr<float>(), layout);
            gx = (dev == Device::CPU) ? gxh : gxh->to(dev);
        }
        if (need_y) {
            auto gyh = std::make_shared<Tensor>(y_shape, Device::CPU, g->dtype, false);
            gyh->fill_(0.0f);
            broadcast_detail::grad_y_host(op, gh->data_ptr<float>(), xh ? xh->data_ptr<float>() : nullptr,
                                          yh ? yh->data_ptr<float>() : nullptr, gyh->data_ptr<float>(), layout);
            gy = (dev == Device::CPU) ? gyh : gyh->to(dev);
        }
        return {gx, gy};
    }
};

inline std::shared_ptr<Tensor> broadcast_binary(BinaryOp op, const std::shared_ptr<Tensor>& x_in,
                                                const std::shared_ptr<Tensor>& y_in) {
    DG_PROFILE_OP(binary_op_name(op));
    if (!x_in || !y_in) throw std::invalid_argument(std::string(binary_op_name(op)) + ": null tensor");
    if (x_in->device != y_in->device) throw std::invalid_argument(std::string(binary_op_name(op)) + ": tensors are on different devices");
#ifdef USE_CUDA
    // The product of two tensors of one shape, in fp32 or (under autocast) kept in half precision
    if (op == BinaryOp::Mul && x_in->device == Device::CUDA && x_in->shape == y_in->shape && x_in->dtype == y_in->dtype &&
        (x_in->dtype == DType::Float32 || native_half(x_in))) {
        auto x = contiguous(x_in), y = contiguous(y_in);
        auto out = std::make_shared<Tensor>(x->shape, Device::CUDA, x->dtype, false);
        cuda::mul(x->data_ptr<void>(), y->data_ptr<void>(), out->data_ptr<void>(), x->size(), x->dtype);
        out->requires_grad = needs_grad(x) || needs_grad(y);
        if (out->requires_grad) {
            auto grad_fn = std::make_shared<BroadcastBinaryBackward>(op, x->shape, y->shape, BroadcastLayout{}, x, y);
            grad_fn->same_shape = true;
            grad_fn->add_next_edge(get_grad_edge(x).function);
            grad_fn->add_next_edge(get_grad_edge(y).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    auto x = contiguous(float32_input(x_in));
    auto y = contiguous(float32_input(y_in));
    if (x->dtype != DType::Float32 || y->dtype != DType::Float32) {
        throw std::invalid_argument(std::string(binary_op_name(op)) + ": tensors must be Float32, Float16 or BFloat16");
    }
    if (!broadcast_detail::onto(x->shape, y->shape)) {
        // x broadcasts too: give it the output's shape
        auto shape = op_detail::broadcast_shapes(x->shape, y->shape, binary_op_name(op));
        x = contiguous(expand(x, shape));
    }
    BroadcastLayout layout = broadcast_layout(x->shape, y->shape, op);
    Device dev = x->device;
    bool req_grad = needs_grad(x) || needs_grad(y);
    auto out = std::make_shared<Tensor>(x->shape, dev, x->dtype, false);

#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::broadcast_binary_forward((int)op, x->data_ptr<float>(), y->data_ptr<float>(), out->data_ptr<float>(),
                                       layout.x_shape, layout.y_shape);
    } else
#endif
    if (dev == Device::CPU) {
        broadcast_detail::forward_host(op, x->data_ptr<float>(), y->data_ptr<float>(), out->data_ptr<float>(), layout);
    } else {
        auto xh = x->cpu(), yh = y->cpu();
        auto oh = std::make_shared<Tensor>(x->shape, Device::CPU, x->dtype, false);
        broadcast_detail::forward_host(op, xh->data_ptr<float>(), yh->data_ptr<float>(), oh->data_ptr<float>(), layout);
        out = oh->to(dev);
    }

    out->requires_grad = req_grad;
    if (req_grad) {
        bool uses_values = (op == BinaryOp::Mul || op == BinaryOp::Div);
        auto grad_fn = std::make_shared<BroadcastBinaryBackward>(op, x->shape, y->shape, layout,
                                                                 uses_values ? x : nullptr, uses_values ? y : nullptr);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        grad_fn->add_next_edge(get_grad_edge(y).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    return broadcast_binary(BinaryOp::Add, x, y);
}
inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    return broadcast_binary(BinaryOp::Mul, x, y);
}

// Repeats a size-1 dimension n times: axis 0 the second-to-last dimension (rows), axis 1 the last
// (columns). It is a broadcast add onto zeros, so it runs wherever add does and its gradient sums the
// repeats back, deterministically.
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a_in, int axis, int n) {
    DG_PROFILE_OP("broadcast");
    auto a = float32_input(a_in);
    int ndim = a->ndim();
    if (ndim < 2) throw std::invalid_argument("broadcast requires tensor with ndim >= 2");
    if (axis != 0 && axis != 1) throw std::invalid_argument("broadcast: axis must be 0 (rows) or 1 (columns)");
    int dim = axis == 0 ? ndim - 2 : ndim - 1;
    if (a->shape[dim] != 1) throw std::runtime_error("broadcast: the dimension to be broadcast must have size 1");
    std::vector<int64_t> out_shape = a->shape;
    out_shape[dim] = n;
    auto base = std::make_shared<Tensor>(out_shape, a->device, DType::Float32, false);
    base->fill_(0.0f);
    return add(base, a);
}
inline std::shared_ptr<Tensor> cast_n_add(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    return broadcast_binary(BinaryOp::Add, x, y);
}
inline std::shared_ptr<Tensor> cast_n_sub(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    return broadcast_binary(BinaryOp::Sub, x, y);
}
inline std::shared_ptr<Tensor> cast_n_mul(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    return broadcast_binary(BinaryOp::Mul, x, y);
}
inline std::shared_ptr<Tensor> cast_n_div(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y) {
    return broadcast_binary(BinaryOp::Div, x, y);
}

// ========================================
// From concat.h
// ========================================
#pragma once 

#ifdef USE_CUDA
// Device-to-device copy of `rows` rows of `width` bytes between buffers with the given row pitches
inline void copy_rows_cuda(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t rows) {
    if (width == 0 || rows == 0) return;
    check_cuda(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, rows, cudaMemcpyDeviceToDevice), "concat");
}
#endif

// --- CONCAT BACKWARD NODE ---
struct ConcatBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::vector<int64_t> a_shape, b_shape;
    int axis, r_a, c_a, r_b, c_b, r_out, c_out, batch_size;

    ConcatBackward(const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
                   int axis, int r_a, int c_a, int r_b, int c_b, int r_out, int c_out, int batch_size)
        : a_shape(a_shape), b_shape(b_shape), axis(axis), r_a(r_a), c_a(c_a), r_b(r_b), c_b(c_b),
          r_out(r_out), c_out(c_out), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr, nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            // Each input's block of every output row (axis 0: per batch; axis 1: per row), copied back out
            auto ga = std::make_shared<Tensor>(a_shape, Device::CUDA, DType::Float32, false);
            auto gb = std::make_shared<Tensor>(b_shape, Device::CUDA, DType::Float32, false);
            const char* g = self_grad->data_ptr<char>();
            size_t f = sizeof(float);
            if (axis == 0) {
                size_t out_row = (size_t)r_out * c_out * f, a_row = (size_t)r_a * c_a * f, b_row = (size_t)r_b * c_b * f;
                copy_rows_cuda(ga->data_ptr<void>(), a_row, g, out_row, a_row, batch_size);
                copy_rows_cuda(gb->data_ptr<void>(), b_row, g + a_row, out_row, b_row, batch_size);
            } else {
                size_t out_row = (size_t)c_out * f, a_row = (size_t)c_a * f, b_row = (size_t)c_b * f;
                copy_rows_cuda(ga->data_ptr<void>(), a_row, g, out_row, a_row, (size_t)batch_size * r_a);
                copy_rows_cuda(gb->data_ptr<void>(), b_row, g + a_row, out_row, b_row, (size_t)batch_size * r_b);
            }
            return {ga, gb};
        }
#endif
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto ga = std::make_shared<Tensor>(a_shape, Device::CPU, sg_cpu->dtype, false);
        auto gb = std::make_shared<Tensor>(b_shape, Device::CPU, sg_cpu->dtype, false);

        const float* sg_ptr = sg_cpu->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();
        float* gb_ptr = gb->data_ptr<float>();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r_a; i++) {
                    for (int j = 0; j < c_a; j++) {
                        ga_ptr[batch * r_a * c_a + i * c_a + j] = sg_ptr[batch * r_out * c_out + i * c_out + j];
                    }
                }
                for (int i = 0; i < r_b; i++) {
                    for (int j = 0; j < c_b; j++) {
                        gb_ptr[batch * r_b * c_b + i * c_b + j] = sg_ptr[batch * r_out * c_out + (r_a + i) * c_out + j];
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r_a; i++) {
                    for (int j = 0; j < c_a; j++) {
                        ga_ptr[batch * r_a * c_a + i * c_a + j] = sg_ptr[batch * r_out * c_out + i * c_out + j];
                    }
                }
                for (int i = 0; i < r_b; i++) {
                    for (int j = 0; j < c_b; j++) {
                        gb_ptr[batch * r_b * c_b + i * c_b + j] = sg_ptr[batch * r_out * c_out + i * c_out + (c_a + j)];
                    }
                }
            }
        }
        return {(dev == Device::CPU) ? ga : ga->to(dev),
                (dev == Device::CPU) ? gb : gb->to(dev)};
    }
};

// concatenation
inline std::shared_ptr<Tensor> concat(const std::shared_ptr<Tensor>& a_in,
                                       const std::shared_ptr<Tensor>& b_in, int axis) {
    DG_PROFILE_OP("concat");
    auto a = contiguous(float32_input(a_in));
    auto b = contiguous(float32_input(b_in));
    if (!a || !b) {
        throw std::invalid_argument("concat: input tensors cannot be null");
    }
    int ndim = a->shape.size();
    if (ndim < 2 || (int)b->shape.size() != ndim) {
        throw std::invalid_argument("concat requires tensors with matching ndim >= 2");
    }
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("concat: axis must be 0 (row/seq) or 1 (column/feature)");
    }
    if (a->device != b->device) {
        throw std::runtime_error("concat: device mismatch");
    }
    int r_a = a->shape[ndim - 2];
    int c_a = a->shape[ndim - 1];
    int r_b = b->shape[ndim - 2];
    int c_b = b->shape[ndim - 1];

    for (int i = 0; i < ndim - 2; i++) {
        if (a->shape[i] != b->shape[i]) {
            throw std::invalid_argument("concat: batch dimension mismatch at dim " + std::to_string(i));
        }
    }
    if (axis == 0 && c_a != c_b) {
        throw std::invalid_argument("concat: column dimension mismatch for axis 0 concatenation");
    }
    if (axis == 1 && r_a != r_b) {
        throw std::invalid_argument("concat: row dimension mismatch for axis 1 concatenation");
    }

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = r_a + r_b : out_shape[ndim - 1] = c_a + c_b;

    bool req_grad = grad_enabled() && (a->requires_grad || b->requires_grad);
    Device dev = a->device;
    int r_out = out_shape[ndim - 2];
    int c_out = out_shape[ndim - 1];
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        auto out = std::make_shared<Tensor>(out_shape, Device::CUDA, DType::Float32, req_grad);
        char* o = out->data_ptr<char>();
        size_t f = sizeof(float);
        if (axis == 0) {
            size_t out_row = (size_t)r_out * c_out * f, a_row = (size_t)r_a * c_a * f, b_row = (size_t)r_b * c_b * f;
            copy_rows_cuda(o, out_row, a->data_ptr<void>(), a_row, a_row, batch_size);
            copy_rows_cuda(o + a_row, out_row, b->data_ptr<void>(), b_row, b_row, batch_size);
        } else {
            size_t out_row = (size_t)c_out * f, a_row = (size_t)c_a * f, b_row = (size_t)c_b * f;
            copy_rows_cuda(o, out_row, a->data_ptr<void>(), a_row, a_row, (size_t)batch_size * r_a);
            copy_rows_cuda(o + a_row, out_row, b->data_ptr<void>(), b_row, b_row, (size_t)batch_size * r_b);
        }
        if (req_grad) {
            auto grad_fn = std::make_shared<ConcatBackward>(a->shape, b->shape, axis, r_a, c_a, r_b, c_b, r_out, c_out, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            grad_fn->add_next_edge(get_grad_edge(b).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto b_cpu = (dev == Device::CPU) ? b : b->cpu();
    auto out_cpu = std::make_shared<Tensor>(out_shape, Device::CPU, a_cpu->dtype, false);

    const float* a_ptr = a_cpu->data_ptr<float>();
    const float* b_ptr = b_cpu->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r_a; i++) {
                for (int j = 0; j < c_a; j++) {
                    out_ptr[batch * r_out * c_out + i * c_out + j] = a_ptr[batch * r_a * c_a + i * c_a + j];
                }
            }
            for (int i = 0; i < r_b; i++) {
                for (int j = 0; j < c_b; j++) {
                    out_ptr[batch * r_out * c_out + (r_a + i) * c_out + j] = b_ptr[batch * r_b * c_b + i * c_b + j];
                }
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r_a; i++) {
                for (int j = 0; j < c_a; j++) {
                    out_ptr[batch * r_out * c_out + i * c_out + j] = a_ptr[batch * r_a * c_a + i * c_a + j];
                }
            }
            for (int i = 0; i < r_b; i++) {
                for (int j = 0; j < c_b; j++) {
                    out_ptr[batch * r_out * c_out + i * c_out + (c_a + j)] = b_ptr[batch * r_b * c_b + i * c_b + j];
                }
            }
        }
    }

    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<ConcatBackward>(a->shape, b->shape, axis, r_a, c_a, r_b, c_b, r_out, c_out, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        grad_fn->add_next_edge(get_grad_edge(b).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From loss.h
// ========================================
#pragma once 

// --- CROSS ENTROPY BACKWARD NODE ---
struct CrossEntropyBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> pred, target;
    int n, num_samples;
    CrossEntropyBackward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, int n, int num_samples = 0)
        : pred(pred), target(target), n(n), num_samples(num_samples > 0 ? num_samples : n) {}

    void release_variables() override {
        pred.reset();
        target.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !pred || !target) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
#ifdef USE_CUDA
        if (pred->device == Device::CUDA) {
            auto grad_pred = std::make_shared<Tensor>(pred->shape, Device::CUDA, DType::Float32, false);
            cuda::pointwise_loss_backward(2, pred->data_ptr<float>(), target->data_ptr<float>(), self_grad->data_ptr<float>(),
                                          grad_pred->data_ptr<float>(), n, (float)num_samples);
            return {grad_pred};
        }
#endif
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto p_cpu = (pred->device == Device::CPU) ? pred : pred->cpu();
        auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
        auto grad_pred = std::make_shared<Tensor>(p_cpu->shape, Device::CPU, p_cpu->dtype, false);
        grad_pred->fill_(0.0f);
        const float* p_ptr = p_cpu->data_ptr<float>();
        const float* t_ptr = t_cpu->data_ptr<float>();
        float* gp_ptr = grad_pred->data_ptr<float>();
        float g_val = sg_cpu->data_ptr<float>()[0];

        for (int i = 0; i < n; i++) {
            gp_ptr[i] = -(t_ptr[i] / ((p_ptr[i] + 1e-8f) * num_samples)) * g_val;
        }
        return {(dev == Device::CPU) ? grad_pred : grad_pred->to(dev)};
    }
};

// cross_entropy
inline std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor>& pred_in, const std::shared_ptr<Tensor>& target_in) {
    DG_PROFILE_OP("cross_entropy");
    auto pred = contiguous(float32_input(pred_in));
    auto target = contiguous(float32_input(target_in));
    if (pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    int n = static_cast<int>(pred->size());
    int V = pred->shape.empty() ? 1 : static_cast<int>(pred->shape.back());
    int num_samples = (V > 0 && pred->ndim() >= 2) ? (n / V) : n;
    if (num_samples <= 0) num_samples = 1;

    Device dev = pred->device;
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        if (target->device != Device::CUDA) throw std::invalid_argument("cross_entropy: prediction and target are on different devices");
        bool req_grad = needs_grad(pred);
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CUDA, DType::Float32, req_grad);
        cuda::pointwise_loss_forward(2, pred->data_ptr<float>(), target->data_ptr<float>(), out->data_ptr<float>(), n,
                                     (float)num_samples);
        if (req_grad) {
            auto grad_fn = std::make_shared<CrossEntropyBackward>(pred, target, n, num_samples);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    auto p_cpu = (dev == Device::CPU) ? pred : pred->cpu();
    auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
    const float* p_ptr = p_cpu->data_ptr<float>();
    const float* t_ptr = t_cpu->data_ptr<float>();

    float sum_loss = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_loss -= t_ptr[i] * std::log(p_ptr[i] + 1e-8f);
    }

    bool req_grad = grad_enabled() && (pred->requires_grad);
    auto out_cpu = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CPU, pred->dtype, false);
    out_cpu->data_ptr<float>()[0] = sum_loss / static_cast<float>(num_samples);
    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<CrossEntropyBackward>(pred, target, n, num_samples);
        grad_fn->add_next_edge(get_grad_edge(pred).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// Mean cross entropy between logits [..., vocab] and integer class ids (stored as floats), with
// a numerically stable log-sum-exp. Rows whose target equals ignore_index are excluded from the
// mean. Only the per-row log-sum-exp is saved; backward recomputes the softmax from it.
struct FusedCrossEntropyBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> logits, targets, lse, count;  // count: number of valid rows (1 element)
    int64_t rows, vocab;
    int ignore_index;

    FusedCrossEntropyBackward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> targets, std::shared_ptr<Tensor> lse,
                              std::shared_ptr<Tensor> count, int64_t rows, int64_t vocab, int ignore_index)
        : logits(std::move(logits)), targets(std::move(targets)), lse(std::move(lse)), count(std::move(count)),
          rows(rows), vocab(vocab), ignore_index(ignore_index) {}

    void release_variables() override {
        logits.reset();
        targets.reset();
        lse.reset();
        count.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !logits) return {nullptr};
        auto dloss = grads[0];
        Device dev = logits->device;
        auto grad = std::make_shared<Tensor>(logits->shape, dev, logits->dtype, false);
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            cuda::cross_entropy_backward(logits->data_ptr<float>(), targets->data_ptr<float>(), lse->data_ptr<float>(),
                                         count->data_ptr<float>(), dloss->data_ptr<float>(), grad->data_ptr<float>(),
                                         rows, vocab, ignore_index);
            return {grad};
        }
#endif
        auto host = [](const std::shared_ptr<Tensor>& t) { return t->device == Device::CPU ? t : t->cpu(); };
        auto lh = host(logits), th = host(targets), eh = host(lse), ch = host(count), dh = host(dloss);
        auto gh = (dev == Device::CPU) ? grad : std::make_shared<Tensor>(logits->shape, Device::CPU, logits->dtype, false);
        const float* l = lh->data_ptr<float>();
        const float* t = th->data_ptr<float>();
        const float* e = eh->data_ptr<float>();
        float* g = gh->data_ptr<float>();
        float n = ch->data_ptr<float>()[0];
        float scale = n > 0.0f ? dh->data_ptr<float>()[0] / n : 0.0f;
        const int64_t V = vocab;
        const int ignore = ignore_index;
        parallel_for(0, rows, std::max<int64_t>(1, 8192 / std::max<int64_t>(1, V)), [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                int target = (int)t[row];
                const float* lr = l + row * V;
                float* gr = g + row * V;
                if (target == ignore) {
                    std::fill(gr, gr + V, 0.0f);
                    continue;
                }
                const float er = e[row];
                for (int64_t c = 0; c < V; ++c) gr[c] = scale * simd::exp(lr[c] - er);
                gr[target] -= scale;
            }
        });
        return {(dev == Device::CPU) ? gh : gh->to(dev)};
    }
};

inline std::shared_ptr<Tensor> fused_cross_entropy(const std::shared_ptr<Tensor>& logits_in,
                                                   const std::shared_ptr<Tensor>& targets_in, int ignore_index = -100) {
    DG_PROFILE_OP("fused_cross_entropy");
    auto logits = contiguous(float32_input(logits_in));
    if (logits->ndim() < 1) throw std::invalid_argument("fused_cross_entropy: logits must have a class dimension");
    int64_t vocab = logits->shape.back();
    int64_t rows = logits->size() / vocab;
    if (targets_in->size() != rows) {
        throw std::invalid_argument("fused_cross_entropy: target size (" + std::to_string(targets_in->size()) +
                                    ") must match total logit positions (" + std::to_string(rows) + ")");
    }
    Device dev = logits->device;
    auto targets = contiguous(targets_in->device == dev ? targets_in : targets_in->to(dev));

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, dev, logits->dtype, false);
    auto lse = std::make_shared<Tensor>(std::vector<int64_t>{rows}, dev, logits->dtype, false);
    auto count = std::make_shared<Tensor>(std::vector<int64_t>{1}, dev, logits->dtype, false);

#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::cross_entropy_forward(logits->data_ptr<float>(), targets->data_ptr<float>(), out->data_ptr<float>(),
                                    lse->data_ptr<float>(), count->data_ptr<float>(), rows, vocab, ignore_index);
    } else
#endif
    {
        auto lh = (dev == Device::CPU) ? logits : logits->cpu();
        auto th = (dev == Device::CPU) ? targets : targets->cpu();
        auto eh = (dev == Device::CPU) ? lse : std::make_shared<Tensor>(std::vector<int64_t>{rows}, Device::CPU, logits->dtype, false);
        const float* l = lh->data_ptr<float>();
        const float* t = th->data_ptr<float>();
        float* e = eh->data_ptr<float>();
        parallel_for(0, rows, std::max<int64_t>(1, 8192 / std::max<int64_t>(1, vocab)), [&](int64_t begin, int64_t end) {
            // Sums in 16 fixed lanes, so they vectorize
            for (int64_t row = begin; row < end; ++row) {
                const float* lr = l + row * vocab;
                float mx = lr[0];
                for (int64_t c = 1; c < vocab; ++c) mx = lr[c] > mx ? lr[c] : mx;
                float lane[16] = {};
                int64_t c = 0;
                for (; c + 16 <= vocab; c += 16)
                    for (int k = 0; k < 16; ++k) lane[k] += simd::exp(lr[c + k] - mx);
                float total_exp = 0.0f;
                for (int k = 0; k < 16; ++k) total_exp += lane[k];
                for (; c < vocab; ++c) total_exp += simd::exp(lr[c] - mx);
                e[row] = std::log(total_exp) + mx;
            }
        });
        double total = 0.0;
        int64_t valid = 0;
        for (int64_t row = 0; row < rows; ++row) {
            const float* lr = l + row * vocab;
            int target = (int)t[row];
            if (target == ignore_index) continue;
            if (target < 0 || target >= vocab) {
                throw std::out_of_range("fused_cross_entropy: target index " + std::to_string(target) +
                                        " is out of bounds for vocab size " + std::to_string(vocab));
            }
            total += e[row] - lr[target];
            ++valid;
        }
        float loss = valid > 0 ? (float)(total / (double)valid) : 0.0f;
        if (dev == Device::CPU) {
            out->data_ptr<float>()[0] = loss;
            count->data_ptr<float>()[0] = (float)valid;
        } else {
            out->copy_from_vector({loss});
            count->copy_from_vector({(float)valid});
            lse = eh->to(dev);
        }
    }

    out->requires_grad = needs_grad(logits);
    if (needs_grad(logits)) {
        auto grad_fn = std::make_shared<FusedCrossEntropyBackward>(logits, targets, lse, count, rows, vocab, ignore_index);
        grad_fn->add_next_edge(get_grad_edge(logits).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

// --- MSE BACKWARD NODE ---
struct MseBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> pred, target;
    int n;
    MseBackward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, int n)
        : pred(pred), target(target), n(n) {}

    void release_variables() override {
        pred.reset();
        target.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !pred || !target) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
#ifdef USE_CUDA
        if (pred->device == Device::CUDA) {
            auto grad_pred = std::make_shared<Tensor>(pred->shape, Device::CUDA, DType::Float32, false);
            cuda::pointwise_loss_backward(0, pred->data_ptr<float>(), target->data_ptr<float>(), self_grad->data_ptr<float>(),
                                          grad_pred->data_ptr<float>(), pred->size(), (float)n);
            return {grad_pred};
        }
#endif
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        float g_val = sg_cpu->data_ptr<float>()[0];

        if (pred->device == Device::MPS) {
            auto grad_pred = std::make_shared<Tensor>(pred->shape, pred->device, pred->dtype, false);
            grad_pred->fill_(0.0f);
            float scale = (2.0f / static_cast<float>(n)) * g_val;
            MetalBackend::get().mse_backward(pred->data_ptr<float>(), target->data_ptr<float>(), grad_pred->data_ptr<float>(), n, scale);
            return {grad_pred};
        }

        auto p_cpu = (pred->device == Device::CPU) ? pred : pred->cpu();
        auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
        auto grad_pred = std::make_shared<Tensor>(p_cpu->shape, Device::CPU, p_cpu->dtype, false);
        grad_pred->fill_(0.0f);

        const float* p_ptr = p_cpu->data_ptr<float>();
        const float* t_ptr = t_cpu->data_ptr<float>();
        float* gp_ptr = grad_pred->data_ptr<float>();

        const int count = n;
        host_loop(count, [=](int64_t i) { gp_ptr[i] = (2.0f * (p_ptr[i] - t_ptr[i]) / count) * g_val; });
        return {(dev == Device::CPU) ? grad_pred : grad_pred->to(dev)};
    }
};

// mse
inline std::shared_ptr<Tensor> mse(const std::shared_ptr<Tensor>& pred_in, const std::shared_ptr<Tensor>& target_in) {
    DG_PROFILE_OP("mse");
    auto pred = contiguous(float32_input(pred_in));
    auto target = contiguous(float32_input(target_in));
    if (pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    int n = static_cast<int>(pred->size());

    if (pred->device == Device::TPU && pred->storage->tpu_handle && target->storage->tpu_handle) {
        bool req_grad = grad_enabled() && (pred->requires_grad);
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::TPU, pred->dtype, req_grad);
        out->storage->tpu_handle = TPUEngine::get().mse(pred->storage->tpu_handle, target->storage->tpu_handle, pred->shape, pred->dtype);
        sync_tpu_result(out);
        if (req_grad) {
            auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#ifdef USE_CUDA
    if (pred->device == Device::CUDA) {
        if (target->device != Device::CUDA) throw std::invalid_argument("mse: prediction and target are on different devices");
        bool req_grad = needs_grad(pred);
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CUDA, DType::Float32, req_grad);
        cuda::pointwise_loss_forward(0, pred->data_ptr<float>(), target->data_ptr<float>(), out->data_ptr<float>(),
                                     pred->size(), (float)n);
        if (req_grad) {
            auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif

    Device dev = pred->device;
    auto p_cpu = (dev == Device::CPU) ? pred : pred->cpu();
    auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
    const float* p_ptr = p_cpu->data_ptr<float>();
    const float* t_ptr = t_cpu->data_ptr<float>();

    float sq_sum = (float)parallel_sum(n, [&](int64_t b, int64_t e) {
        float acc = 0.0f;
        for (int64_t i = b; i < e; ++i) acc += (p_ptr[i] - t_ptr[i]) * (p_ptr[i] - t_ptr[i]);
        return acc;
    });

    bool req_grad = grad_enabled() && (pred->requires_grad);
    auto out_cpu = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CPU, pred->dtype, false);
    out_cpu->data_ptr<float>()[0] = sq_sum / n;
    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- L1 LOSS (MAE) BACKWARD NODE ---
struct L1LossBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> pred, target;
    int n;
    L1LossBackward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, int n)
        : pred(pred), target(target), n(n) {}

    void release_variables() override {
        pred.reset();
        target.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !pred || !target) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
#ifdef USE_CUDA
        if (pred->device == Device::CUDA) {
            auto grad_pred = std::make_shared<Tensor>(pred->shape, Device::CUDA, DType::Float32, false);
            cuda::pointwise_loss_backward(1, pred->data_ptr<float>(), target->data_ptr<float>(), self_grad->data_ptr<float>(),
                                          grad_pred->data_ptr<float>(), pred->size(), (float)n);
            return {grad_pred};
        }
#endif
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        float g_val = sg_cpu->data_ptr<float>()[0];

        if (pred->device == Device::MPS) {
            auto grad_pred = std::make_shared<Tensor>(pred->shape, pred->device, pred->dtype, false);
            grad_pred->fill_(0.0f);
            float scale = (1.0f / static_cast<float>(n)) * g_val;
            MetalBackend::get().l1_loss_backward(pred->data_ptr<float>(), target->data_ptr<float>(), grad_pred->data_ptr<float>(), n, scale);
            return {grad_pred};
        }

        auto p_cpu = (pred->device == Device::CPU) ? pred : pred->cpu();
        auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
        auto grad_pred = std::make_shared<Tensor>(p_cpu->shape, Device::CPU, p_cpu->dtype, false);
        grad_pred->fill_(0.0f);

        const float* p_ptr = p_cpu->data_ptr<float>();
        const float* t_ptr = t_cpu->data_ptr<float>();
        float* gp_ptr = grad_pred->data_ptr<float>();

        for (int i = 0; i < n; i++) {
            float diff = p_ptr[i] - t_ptr[i];
            float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
            gp_ptr[i] = (sgn / static_cast<float>(n)) * g_val;
        }
        return {(dev == Device::CPU) ? grad_pred : grad_pred->to(dev)};
    }
};

// l1_loss (MAE - Mean Absolute Error)
inline std::shared_ptr<Tensor> l1_loss(const std::shared_ptr<Tensor>& pred_in, const std::shared_ptr<Tensor>& target_in) {
    DG_PROFILE_OP("l1_loss");
    auto pred = contiguous(float32_input(pred_in));
    auto target = contiguous(float32_input(target_in));
    if (pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    int n = static_cast<int>(pred->size());

    if (pred->device == Device::TPU && pred->storage->tpu_handle && target->storage->tpu_handle) {
        bool req_grad = grad_enabled() && (pred->requires_grad);
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::TPU, pred->dtype, req_grad);
        out->storage->tpu_handle = TPUEngine::get().l1_loss(pred->storage->tpu_handle, target->storage->tpu_handle, pred->shape, pred->dtype);
        sync_tpu_result(out);
        if (req_grad) {
            auto grad_fn = std::make_shared<L1LossBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#ifdef USE_CUDA
    if (pred->device == Device::CUDA) {
        if (target->device != Device::CUDA) throw std::invalid_argument("l1_loss: prediction and target are on different devices");
        bool req_grad = needs_grad(pred);
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CUDA, DType::Float32, req_grad);
        cuda::pointwise_loss_forward(1, pred->data_ptr<float>(), target->data_ptr<float>(), out->data_ptr<float>(),
                                     pred->size(), (float)n);
        if (req_grad) {
            auto grad_fn = std::make_shared<L1LossBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif

    Device dev = pred->device;
    auto p_cpu = (dev == Device::CPU) ? pred : pred->cpu();
    auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
    const float* p_ptr = p_cpu->data_ptr<float>();
    const float* t_ptr = t_cpu->data_ptr<float>();

    float abs_sum = (float)parallel_sum(n, [&](int64_t b, int64_t e) {
        float acc = 0.0f;
        for (int64_t i = b; i < e; ++i) acc += std::fabs(p_ptr[i] - t_ptr[i]);
        return acc;
    });

    bool req_grad = grad_enabled() && (pred->requires_grad);
    auto out_cpu = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CPU, pred->dtype, false);
    out_cpu->data_ptr<float>()[0] = abs_sum / static_cast<float>(n);
    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<L1LossBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

inline std::shared_ptr<Tensor> l1loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    return l1_loss(pred, target);
}

inline std::shared_ptr<Tensor> mae(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    return l1_loss(pred, target);
}

// ========================================
// From masking.h
// ========================================
// --- BOOL MASK BACKWARD NODE ---
struct BoolMaskBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> m;
    int r, c, batch_size;
    BoolMaskBackward(std::shared_ptr<Tensor> m, int r, int c, int batch_size)
        : m(m), r(r), c(c), batch_size(batch_size) {}

    void release_variables() override {
        m.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !m) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            auto ga = std::make_shared<Tensor>(self_grad->shape, Device::CUDA, DType::Float32, false);
            cuda::bool_mask_backward(self_grad->data_ptr<float>(), m->data_ptr<float>(), ga->data_ptr<float>(), ga->size());
            return {ga};
        }
#endif
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto m_cpu = (m->device == Device::CPU) ? m : m->cpu();
        auto ga = std::make_shared<Tensor>(sg_cpu->shape, Device::CPU, sg_cpu->dtype, false);
        ga->fill_(0.0f);

        const float* sg_ptr = sg_cpu->data_ptr<float>();
        const float* m_ptr = m_cpu->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    ga_ptr[idx] = (m_ptr[idx] != 0.0f) ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {(dev == Device::CPU) ? ga : ga->to(dev)};
    }
};

// boolean mask
inline std::shared_ptr<Tensor> bool_mask(const std::shared_ptr<Tensor>& a_in, const std::shared_ptr<Tensor>& m_in) {
    DG_PROFILE_OP("bool_mask");
    auto a = contiguous(float32_input(a_in));
    auto m = make_contiguous(float32_input(m_in));
    int ndim = a->shape.size();
    if (ndim < 2) throw std::invalid_argument("bool_mask requires tensor with ndim >= 2");
    if (m->shape != a->shape) throw std::invalid_argument("bool_mask: mask " + shape_string(m->shape) + " must match " + shape_string(a->shape));
    if (m->device != a->device) throw std::invalid_argument("bool_mask: tensor and mask are on different devices");
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = grad_enabled() && (a->requires_grad);
    Device dev = a->device;
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        auto out = std::make_shared<Tensor>(a->shape, Device::CUDA, DType::Float32, req_grad);
        cuda::bool_mask_forward(a->data_ptr<float>(), m->data_ptr<float>(), out->data_ptr<float>(), out->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<BoolMaskBackward>(m, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto m_cpu = (m->device == Device::CPU) ? m : m->cpu();
    auto out_cpu = std::make_shared<Tensor>(a_cpu->shape, Device::CPU, a_cpu->dtype, false);
    const float* a_ptr = a_cpu->data_ptr<float>();
    const float* m_ptr = m_cpu->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int idx = batch * r * c + i * c + j;
                out_ptr[idx] = m_ptr[idx] == 0.0f
                              ? -1e9f
                              : a_ptr[idx];
            }
        }
    }

    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<BoolMaskBackward>(m, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAUSAL MASK BACKWARD NODE ---
struct CausalMaskBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    int r, c, batch_size;
    CausalMaskBackward(int r, int c, int batch_size) : r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];

#ifdef USE_CUDA
        if (self_grad->device == Device::CUDA) {
            auto ga = std::make_shared<Tensor>(self_grad->shape, self_grad->device, self_grad->dtype, false);
            cuda::causal_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(), batch_size, r, c);
            return {ga};
        }
#endif
        // The Metal kernel handles square scores only
        if (self_grad->device == Device::MPS && r == c) {
            auto ga = std::make_shared<Tensor>(self_grad->shape, self_grad->device, self_grad->dtype, false);
            ga->fill_(0.0f);
            MetalBackend::get().causal_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(), batch_size, r);
            return {ga};
        }

        Device dev = self_grad->device;
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto ga_cpu = std::make_shared<Tensor>(sg_cpu->shape, Device::CPU, sg_cpu->dtype, false);

        const float* sg_ptr = sg_cpu->data_ptr<float>();
        float* ga_ptr = ga_cpu->data_ptr<float>();

        const int64_t rows = r, cols = c;
        parallel_for(0, (int64_t)batch_size * rows, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, cols)), [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                int64_t keep = std::min(cols, row % rows + 1);
                std::copy(sg_ptr + row * cols, sg_ptr + row * cols + keep, ga_ptr + row * cols);
                std::fill(ga_ptr + row * cols + keep, ga_ptr + (row + 1) * cols, 0.0f);
            }
        });
        return {(dev == Device::CPU) ? ga_cpu : ga_cpu->to(dev)};
    }
};

// causal mask
inline std::shared_ptr<Tensor> causal_mask(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("causal_mask");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 2) throw std::invalid_argument("causal_mask requires tensor with ndim >= 2");
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);

#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::causal_mask(a->data_ptr<float>(), out->data_ptr<float>(), batch_size, r, c);
        if (req_grad) {
            auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS && r == c) {
        MetalBackend::get().causal_mask(a->data_ptr<float>(), out->data_ptr<float>(), batch_size, r);
        if (req_grad) {
            auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    Device dev = a->device;
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto out_cpu = std::make_shared<Tensor>(a_cpu->shape, Device::CPU, a_cpu->dtype, false);
    const float* a_ptr = a_cpu->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();

    const int64_t rows = r, cols = c;
    parallel_for(0, (int64_t)batch_size * rows, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, cols)), [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            int64_t keep = std::min(cols, row % rows + 1);
            std::copy(a_ptr + row * cols, a_ptr + row * cols + keep, out_ptr + row * cols);
            std::fill(out_ptr + row * cols + keep, out_ptr + (row + 1) * cols, -1e9f);
        }
    });

    out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From scalar_ops.h
// ========================================
// Elementwise ops with a scalar: add, sub, mul, div, and the reversed s - a and s / a.
enum class ScalarOp : int { Add = 0, Sub = 1, Mul = 2, Div = 3, RSub = 4, RDiv = 5 };

namespace scalar_detail {

inline float forward(ScalarOp op, float a, float s) {
    switch (op) {
        case ScalarOp::Add: return a + s;
        case ScalarOp::Sub: return a - s;
        case ScalarOp::Mul: return a * s;
        case ScalarOp::Div: return a / s;
        case ScalarOp::RSub: return s - a;
        default: return s / a;
    }
}

inline float backward(ScalarOp op, float g, float a, float s) {
    switch (op) {
        case ScalarOp::Add: case ScalarOp::Sub: return g;
        case ScalarOp::Mul: return g * s;
        case ScalarOp::Div: return g / s;
        case ScalarOp::RSub: return -g;
        default: return -s * g / (a * a);
    }
}

}  // namespace scalar_detail

struct ScalarOpBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    ScalarOp op;
    float scalar;
    std::shared_ptr<Tensor> a;  // saved only for rdiv

    ScalarOpBackward(ScalarOp op, float scalar, std::shared_ptr<Tensor> a) : op(op), scalar(scalar), a(std::move(a)) {}

    std::string name() const override {
        static const char* names[] = {"add", "sub", "mul", "div", "rsub", "rdiv"};
        return std::string("ScalarOpBackward(") + names[(int)op] + ")";
    }

    void release_variables() override { a.reset(); }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        auto g = grads[0];
        if (op == ScalarOp::Add || op == ScalarOp::Sub) return {g};
        if (op == ScalarOp::RDiv && !a) return {nullptr};
        Device dev = g->device;
        auto ga = std::make_shared<Tensor>(g->shape, dev, g->dtype, false);
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            cuda::scalar_backward((int)op, g->data_ptr<float>(), a ? a->data_ptr<float>() : nullptr, scalar,
                                  ga->data_ptr<float>(), g->size());
            return {ga};
        }
#endif
        auto gh = (dev == Device::CPU) ? g : g->cpu();
        auto ah = !a ? nullptr : (a->device == Device::CPU ? a : a->cpu());
        auto out = (dev == Device::CPU) ? ga : std::make_shared<Tensor>(g->shape, Device::CPU, g->dtype, false);
        const float* gp = gh->data_ptr<float>();
        const float* ap = ah ? ah->data_ptr<float>() : nullptr;
        float* op_ = out->data_ptr<float>();
        const ScalarOp kind = op;
        const float sv = scalar;
        host_loop(g->size(), [=](int64_t i) { op_[i] = scalar_detail::backward(kind, gp[i], ap ? ap[i] : 0.0f, sv); });
        return {(dev == Device::CPU) ? out : out->to(dev)};
    }
};

inline std::shared_ptr<Tensor> scalar_op(ScalarOp op, const std::shared_ptr<Tensor>& a_in, float s) {
    static const char* names[] = {"add_scalar", "sub_scalar", "mul_scalar", "div_scalar", "rsub_scalar", "rdiv_scalar"};
    DG_PROFILE_OP(names[(int)op]);
    auto a = contiguous(float32_input(a_in));
    if (a->dtype != DType::Float32) throw std::invalid_argument("scalar ops: tensors must be Float32, Float16 or BFloat16");
    Device dev = a->device;
    auto out = std::make_shared<Tensor>(a->shape, dev, a->dtype, false);
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::scalar_forward((int)op, a->data_ptr<float>(), s, out->data_ptr<float>(), a->size());
    } else
#endif
    {
        auto ah = (dev == Device::CPU) ? a : a->cpu();
        auto oh = (dev == Device::CPU) ? out : std::make_shared<Tensor>(a->shape, Device::CPU, a->dtype, false);
        const float* ap = ah->data_ptr<float>();
        float* op_ = oh->data_ptr<float>();
        host_loop(a->size(), [=](int64_t i) { op_[i] = scalar_detail::forward(op, ap[i], s); });
        if (dev != Device::CPU) out = oh->to(dev);
    }
    out->requires_grad = needs_grad(a);
    if (needs_grad(a)) {
        auto grad_fn = std::make_shared<ScalarOpBackward>(op, s, op == ScalarOp::RDiv ? a : nullptr);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

inline std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float s) { return scalar_op(ScalarOp::Add, a, s); }
inline std::shared_ptr<Tensor> sub_scalar(const std::shared_ptr<Tensor>& a, float s) { return scalar_op(ScalarOp::Sub, a, s); }
inline std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float s) { return scalar_op(ScalarOp::Mul, a, s); }
inline std::shared_ptr<Tensor> div_scalar(const std::shared_ptr<Tensor>& a, float s) { return scalar_op(ScalarOp::Div, a, s); }
inline std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& a) { return scalar_op(ScalarOp::Mul, a, -1.0f); }
inline std::shared_ptr<Tensor> rsub_scalar(float s, const std::shared_ptr<Tensor>& a) { return scalar_op(ScalarOp::RSub, a, s); }
inline std::shared_ptr<Tensor> rdiv_scalar(float s, const std::shared_ptr<Tensor>& a) { return scalar_op(ScalarOp::RDiv, a, s); }

// ========================================
// From slice.h
// ========================================
#include <stdexcept>

struct SliceBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::vector<int64_t> in_shape;  // the sliced tensor's shape (not the tensor: that would keep its memory alive)
    int dim;
    int start;
    int end;

    SliceBackward(std::vector<int64_t> in_shape, int dim, int start, int end)
        : in_shape(std::move(in_shape)), dim(dim), start(start), end(end) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        const auto& grad_out = grads[0];  // dense: the engine densifies grads
        Device dev = grad_out->device;

        // Zero gradient for the full input; the sliced window receives grad_out. grad_in is dense,
        // so the window is `len` consecutive rows of `inner` elements inside each of `outer` blocks:
        // one strided 2-D copy of `outer` rows, in any dtype.
        const auto& shape = in_shape;
        auto grad_in = std::make_shared<Tensor>(shape, dev, grad_out->dtype, false);
        int64_t outer = 1, inner = 1;
        for (int d = 0; d < dim; ++d) outer *= shape[d];
        for (int d = dim + 1; d < (int)shape.size(); ++d) inner *= shape[d];
        int64_t full = shape[dim], len = end - start;
        size_t elem = dtype_size(grad_out->dtype);
        size_t row_bytes = (size_t)(len * inner) * elem, dst_pitch = (size_t)(full * inner) * elem;
        const char* src = grad_out->data_ptr<char>();
        char* dst = grad_in->data_ptr<char>() + (size_t)(start * inner) * elem;

#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            check_cuda(cudaMemsetAsync(grad_in->data_ptr<void>(), 0, grad_in->size() * elem), "slice backward");
            if (row_bytes > 0 && outer > 0) {
                check_cuda(cudaMemcpy2DAsync(dst, dst_pitch, src, row_bytes, row_bytes, (size_t)outer,
                                             cudaMemcpyDeviceToDevice), "slice backward");
            }
            return {grad_in};
        }
#endif
        if (dev != Device::CPU) {
            // Other accelerators: assemble on the host
            auto host = SliceBackward(in_shape, dim, start, end).apply({grad_out->cpu()});
            return {host[0]->to(dev)};
        }
        std::memset(grad_in->data_ptr<void>(), 0, grad_in->size() * elem);
        for (int64_t o = 0; o < outer; ++o) std::memcpy(dst + o * dst_pitch, src + o * row_bytes, row_bytes);
        return {grad_in};
    }
};

// O(1) Tensor Slicing (Memory Sharing)
inline std::shared_ptr<Tensor> slice(const std::shared_ptr<Tensor>& x, int dim, int start, int end) {
    DG_PROFILE_OP("slice");
    if (dim < 0 || dim >= x->ndim()) throw std::runtime_error("slice: dim out of bounds");
    if (start < 0) start += x->shape[dim];
    if (end < 0) end += x->shape[dim];
    if (start < 0 || end > x->shape[dim] || start >= end) throw std::runtime_error("slice: invalid bounds");

    // Inherit memory pointers with zero re-allocation using storage sharing constructor
    std::vector<int64_t> out_shape = x->shape;
    out_shape[dim] = end - start;
    size_t new_offset = x->global_offset + start * x->strides[dim];
    auto out = std::make_shared<Tensor>(x->storage, out_shape, x->strides, new_offset, x->device, x->dtype, needs_grad(x));

    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<SliceBackward>(x->shape, dim, start, end);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From softmax.h
// ========================================
struct SoftmaxBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> s;
    int r, c, batch_size, ndim;

    SoftmaxBackward(std::shared_ptr<Tensor> s, int r, int c, int batch_size, int ndim)
        : s(s), r(r), c(c), batch_size(batch_size), ndim(ndim) {}

    void release_variables() override {
        s.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !s) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];

#ifdef USE_CUDA
        if (s->device == Device::CUDA) {
            auto ga = std::make_shared<Tensor>(s->shape, s->device, s->dtype, false);
            cuda::softmax_backward(s->data_ptr<float>(), self_grad->data_ptr<float>(), ga->data_ptr<float>(), batch_size * r, c);
            return {ga};
        }
#endif

        if (s->device != Device::CPU) {
            auto s_cpu = s->to(Device::CPU);
            auto sg_cpu = self_grad->to(Device::CPU);
            SoftmaxBackward cpu_node(s_cpu, r, c, batch_size, ndim);
            auto cpu_grads = cpu_node.apply({sg_cpu});
            if (cpu_grads.empty() || !cpu_grads[0]) return {nullptr};
            return {cpu_grads[0]->to(s->device)};
        }

        auto ga = std::make_shared<Tensor>(s->shape, s->device, s->dtype, false);

        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* s_ptr = s->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();
        const int64_t cols = c;
        parallel_for(0, (int64_t)batch_size * r, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, cols)), [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                const float* gr = sg_ptr + row * cols;
                const float* sr = s_ptr + row * cols;
                float* out = ga_ptr + row * cols;
                float lane[16] = {};
                int64_t j = 0;
                for (; j + 16 <= cols; j += 16)
                    for (int k = 0; k < 16; ++k) lane[k] += gr[j + k] * sr[j + k];
                float dot = 0.0f;
                for (int k = 0; k < 16; ++k) dot += lane[k];
                for (; j < cols; ++j) dot += gr[j] * sr[j];
                for (j = 0; j < cols; ++j) out[j] = sr[j] * (gr[j] - dot);
            }
        });
        return {ga};
    }
};

inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a_in, int axis = -1) {
    DG_PROFILE_OP("softmax");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 1) throw std::runtime_error("softmax: tensor must be at least 1D");
    if (axis < 0) axis = ndim + axis;
    if (axis < 0 || axis >= ndim) throw std::invalid_argument("softmax: axis out of range");
    if (axis != ndim - 1) return op_detail::softmax_any(a, axis, false);

    int r = (ndim >= 2) ? a->shape[ndim - 2] : 1;
    int c = a->shape[ndim - 1];
    int batch_size = a->size() / (r * c);

    bool req_grad = grad_enabled() && (a->requires_grad);

#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
        cuda::softmax_forward(a->data_ptr<float>(), out->data_ptr<float>(), batch_size * r, c);
        if (req_grad) {
            auto grad_fn = std::make_shared<SoftmaxBackward>(saved_output(out), r, c, batch_size, ndim);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif

    if (a->device != Device::CPU) {
        auto a_cpu = a->to(Device::CPU);
        auto out_cpu = softmax(a_cpu, axis);
        auto out_res = out_cpu->to(a->device);
        if (req_grad) {
            auto grad_fn = std::make_shared<SoftmaxBackward>(saved_output(out_res), r, c, batch_size, ndim);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }

    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    kern::host::softmax(a_ptr, out_ptr, (int64_t)batch_size * r, c, 1, /*log=*/false);

    if (req_grad) {
        auto grad_fn = std::make_shared<SoftmaxBackward>(saved_output(out), r, c, batch_size, ndim);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

// ========================================
// From transpose.h
// ========================================
#pragma once 
#include "tensor.h" 

// Forward declaration with default arguments for swapping last 2 dimensions
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a, int dim0 = -2, int dim1 = -1);

// --- TRANSPOSE BACKWARD NODE ---
struct TransposeBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    int dim0, dim1;
    TransposeBackward(int dim0, int dim1) : dim0(dim0), dim1(dim1) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        return {transpose(grads[0], dim0, dim1)};
    }
};

// Functional transpose with autograd support
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a, int dim0, int dim1) {
    DG_PROFILE_OP("transpose");
    int n = a->ndim();
    if (n < 2) {
        throw std::runtime_error("transpose: tensor must be at least 2D");
    }

    int d0 = (dim0 < 0) ? dim0 + n : dim0;
    int d1 = (dim1 < 0) ? dim1 + n : dim1;

    if (d0 < 0 || d0 >= n || d1 < 0 || d1 >= n) {
        throw std::runtime_error("transpose: dimension out of range");
    }

    auto out = a->_transpose(d0, d1);
    out->requires_grad = needs_grad(a);

    if (needs_grad(a)) {
        auto grad_fn = std::make_shared<TransposeBackward>(d0, d1);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From view.h
// ========================================
#pragma once 

// --- VIEW / RESHAPE BACKWARD NODE ---
struct ViewBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::vector<int64_t> in_shape;
    explicit ViewBackward(const std::vector<int64_t>& in_shape) : in_shape(in_shape) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        // Reshape incoming gradient back to input shape
        return {grads[0]->_reshape(in_shape)};
    }
};

// Fast zero-copy functional view (throws error if non-contiguous)
inline std::shared_ptr<Tensor> view(const std::shared_ptr<Tensor>& a, const std::vector<int64_t>& new_shape) {
    DG_PROFILE_OP("view");
    auto out = a->_view(new_shape);
    out->requires_grad = needs_grad(a);

    if (needs_grad(a)) {
        auto grad_fn = std::make_shared<ViewBackward>(a->shape);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// General functional reshape (handles any layout)
inline std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& a, const std::vector<int64_t>& new_shape) {
    DG_PROFILE_OP("reshape");
    auto out = a->_reshape(new_shape);
    out->requires_grad = needs_grad(a);

    if (needs_grad(a)) {
        auto grad_fn = std::make_shared<ViewBackward>(a->shape);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From im2col.h
// ========================================
#include <vector>
#include <stdexcept>

inline void im2col_cpu(const float* data_im, int channels, int height, int width,
                       int ksize_h, int ksize_w, int pad_h, int pad_w,
                       int stride_h, int stride_w, int dilation_h, int dilation_w,
                       float* data_col) {
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int channels_col = channels * ksize_h * ksize_w;

    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilation_h + h * stride_h - pad_h;
                int im_col = w_offset * dilation_w + w * stride_w - pad_w;
                int col_index = (c * height_col + h) * width_col + w;
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0.0f;
                }
            }
        }
    }
}

inline void col2im_cpu(const float* data_col, int channels, int height, int width,
                       int ksize_h, int ksize_w, int pad_h, int pad_w,
                       int stride_h, int stride_w, int dilation_h, int dilation_w,
                       float* data_im) {
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int channels_col = channels * ksize_h * ksize_w;

    for (int i = 0; i < channels * height * width; i++) data_im[i] = 0.0f;

    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int c_im = c / ksize_h / ksize_w;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset * dilation_h + h * stride_h - pad_h;
                int im_col = w_offset * dilation_w + w * stride_w - pad_w;
                int col_index = (c * height_col + h) * width_col + w;
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    data_im[(c_im * height + im_row) * width + im_col] += data_col[col_index];
                }
            }
        }
    }
}

// Gradient of im2col: col2im of the column gradient. Needs only the input's shape.
struct Im2ColBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::vector<int64_t> x_shape;
    int kH, kW, padH, padW, strideH, strideW;

    Im2ColBackward(std::vector<int64_t> x_shape, int kH, int kW, int padH, int padW, int strideH, int strideW)
        : x_shape(std::move(x_shape)), kH(kH), kW(kW), padH(padH), padW(padW), strideH(strideH), strideW(strideW) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        auto grad_out = grads[0];  // [B, C * kH * kW, H_col, W_col], dense
        int B = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
        int H_col = grad_out->shape[2], W_col = grad_out->shape[3];
        Device dev = grad_out->device;
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            auto grad_x = std::make_shared<Tensor>(x_shape, Device::CUDA, DType::Float32, false);
            cuda::col2im(grad_out->data_ptr<float>(), grad_x->data_ptr<float>(), B, C, H, W, kH, kW, padH, padW, strideH,
                         strideW, H_col, W_col);
            return {grad_x};
        }
#endif
        auto go = (dev == Device::CPU) ? grad_out : grad_out->cpu();
        auto grad_x = std::make_shared<Tensor>(x_shape, Device::CPU, DType::Float32, false);
        const float* go_data = go->data_ptr<float>();
        float* gx_data = grad_x->data_ptr<float>();
        int64_t x_batch_stride = (int64_t)C * H * W;
        int64_t go_batch_stride = (int64_t)C * kH * kW * H_col * W_col;
        for (int b = 0; b < B; ++b) {
            col2im_cpu(go_data + b * go_batch_stride, C, H, W, kH, kW, padH, padW, strideH, strideW, 1, 1, gx_data + b * x_batch_stride);
        }
        return {(dev == Device::CPU) ? grad_x : grad_x->to(dev)};
    }
};

// Unfolds every kH x kW patch of x [B, C, H, W] into a column: [B, C * kH * kW, H_out, W_out]
inline std::shared_ptr<Tensor> im2col(const std::shared_ptr<Tensor>& x_in, int kH, int kW, int padH, int padW, int strideH, int strideW) {
    DG_PROFILE_OP("im2col");
    auto x = contiguous(float32_input(x_in));
    if (x->ndim() != 4) throw std::runtime_error("im2col requires 4D tensor (B, C, H, W)");
    int B = x->shape[0];
    int C = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];
    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;
    std::vector<int64_t> out_shape{B, (int64_t)C * kH * kW, H_out, W_out};
    Device dev = x->device;
    auto out = std::make_shared<Tensor>(out_shape, dev, DType::Float32, false);
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::im2col(x->data_ptr<float>(), out->data_ptr<float>(), B, C, H, W, kH, kW, padH, padW, strideH, strideW, H_out, W_out);
    } else
#endif
    {
        auto xh = (dev == Device::CPU) ? x : x->cpu();
        auto oh = (dev == Device::CPU) ? out : std::make_shared<Tensor>(out_shape, Device::CPU, DType::Float32, false);
        const float* x_data = xh->data_ptr<float>();
        float* out_data = oh->data_ptr<float>();
        int64_t x_batch_stride = (int64_t)C * H * W;
        int64_t out_batch_stride = (int64_t)C * kH * kW * H_out * W_out;
        for (int b = 0; b < B; ++b) {
            im2col_cpu(x_data + b * x_batch_stride, C, H, W, kH, kW, padH, padW, strideH, strideW, 1, 1, out_data + b * out_batch_stride);
        }
        if (dev != Device::CPU) out = oh->to(dev);
    }
    out->requires_grad = needs_grad(x);
    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<Im2ColBackward>(x->shape, kH, kW, padH, padW, strideH, strideW);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

// ========================================
// Matmul
// ========================================
#include <stdexcept>

namespace matmul_detail {

// A host matmul C = A B (trans_b: A B^T for a 2-D B) as a list of GEMMs
struct Plan {
    int64_t rows = 0, inner = 0, cols = 0;      // of one GEMM
    std::vector<int64_t> a_offset, b_offset;    // each GEMM's operands, in elements
    std::vector<int64_t> out_shape;
    bool a_repeats = false, b_repeats = false;  // an operand is used by several GEMMs
    int64_t batch() const { return (int64_t)a_offset.size(); }
};

inline Plan make_plan(const Tensor& a, const Tensor& b, bool trans_b) {
    int n1 = (int)a.ndim(), n2 = (int)b.ndim();
    if (n1 < 2 || n2 < 2) throw std::runtime_error("matmul: Tensors must be at least 2D...");
    Plan p;
    p.rows = a.shape[n1 - 2];
    p.inner = a.shape[n1 - 1];
    p.cols = trans_b ? b.shape[0] : b.shape[n2 - 1];
    if (p.inner != (trans_b ? b.shape[1] : b.shape[n2 - 2])) {
        throw std::runtime_error("matmul: inner dimensions mismatch: " + shape_string(a.shape) + (trans_b ? " and transposed " : " and ") + shape_string(b.shape));
    }
    std::vector<int64_t> ba(a.shape.begin(), a.shape.end() - 2), bb(b.shape.begin(), b.shape.end() - 2);
    size_t nb = std::max(ba.size(), bb.size());
    std::vector<int64_t> batch(nb), sa(nb, 0), sb(nb, 0);
    int64_t na = 1, nbt = 1, total = 1;
    for (size_t i = 0; i < nb; ++i) {
        int64_t da = i + ba.size() >= nb ? ba[i - (nb - ba.size())] : 1;
        int64_t db = i + bb.size() >= nb ? bb[i - (nb - bb.size())] : 1;
        if (da != db && da != 1 && db != 1) throw std::runtime_error("matmul: batch dimensions mismatch for broadcasting...");
        batch[i] = da == 1 ? db : da;
        if (da != 1) sa[i] = a.strides[i - (nb - ba.size())];
        if (db != 1) sb[i] = b.strides[i - (nb - bb.size())];
        na *= da;
        nbt *= db;
        total *= batch[i];
    }
    p.out_shape = batch;
    p.out_shape.push_back(p.rows);
    p.out_shape.push_back(p.cols);
    if (nbt == 1 && na == total) {
        // One matrix on the right: a's batch is just more rows of one GEMM
        p.rows *= total;
        total = 1;
        batch.clear();
        na = 1;
    }
    p.a_repeats = na != total;
    p.b_repeats = nbt != total;
    std::vector<int64_t> idx(batch.size(), 0);
    for (int64_t t = 0; t < total; ++t) {
        int64_t oa = 0, ob = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            oa += idx[i] * sa[i];
            ob += idx[i] * sb[i];
        }
        p.a_offset.push_back(oa);
        p.b_offset.push_back(ob);
        for (int i = (int)batch.size() - 1; i >= 0; --i) {
            if (++idx[(size_t)i] < batch[(size_t)i]) break;
            idx[(size_t)i] = 0;
        }
    }
    return p;
}

// Runs body(t) for every GEMM of the plan. Small independent ones are spread over threads; big ones run
// one after another, each threaded inside.
template <typename F>
inline void for_each_gemm(const Plan& p, bool independent, const F& body) {
    int64_t flops = p.rows * p.inner * p.cols;
    if (independent && flops <= 262144) {
        parallel_for(0, p.batch(), std::max<int64_t>(1, 131072 / std::max<int64_t>(1, flops)), [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; ++t) body(t);
        });
    } else {
        for (int64_t t = 0; t < p.batch(); ++t) body(t);
    }
}

}  // namespace matmul_detail

// Backward of a host matmul, matmul_nt or linear (with a bias: a third input)
struct MatmulBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> a, b;  // dense operands; null when no gradient needs them
    std::vector<int64_t> a_shape, b_shape, bias_shape;
    matmul_detail::Plan plan;
    bool trans_b;
    bool has_bias;

    MatmulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, matmul_detail::Plan plan, bool trans_b,
                   const std::shared_ptr<Tensor>& bias)
        : a(std::move(a)), b(std::move(b)), plan(std::move(plan)), trans_b(trans_b), has_bias(bias != nullptr) {
        a_shape = this->a->shape;
        b_shape = this->b->shape;
        if (bias) bias_shape = bias->shape;
    }

    std::string name() const override { return has_bias ? "LinearBackward" : "MatmulBackward"; }

    void release_variables() override {
        a = nullptr;
        b = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::vector<std::shared_ptr<Tensor>> out(next_edges.size());
        if (grads.empty() || !grads[0] || !a || !b) return out;
        auto needs = [&](size_t i) { return i < next_edges.size() && next_edges[i].function != nullptr; };
        const auto& g_in = grads[0];
        auto g = g_in->dtype == DType::Float32 ? g_in : g_in->to(DType::Float32);
        const float* gp = g->data_ptr<float>();
        const float* ap = a->data_ptr<float>();
        const float* bp = b->data_ptr<float>();
        int64_t rows = plan.rows, inner = plan.inner, cols = plan.cols, gemm_size = rows * cols;
        // A buffer several GEMMs add into starts zeroed; otherwise each GEMM overwrites its part
        if (needs(0)) {
            out[0] = std::make_shared<Tensor>(a_shape, a->device, DType::Float32, false);
            if (plan.a_repeats) out[0]->fill_(0.0f);
            float* ga = out[0]->data_ptr<float>();
            float beta = plan.a_repeats ? 1.0f : 0.0f;
            matmul_detail::for_each_gemm(plan, !plan.a_repeats, [&](int64_t t) {
                // dA = G B^T (trans_b: G B)
                cpu::gemm(false, !trans_b, rows, inner, cols, gp + t * gemm_size, cols, bp + plan.b_offset[(size_t)t],
                          trans_b ? inner : cols, beta, ga + plan.a_offset[(size_t)t], inner);
            });
        }
        if (needs(1)) {
            out[1] = std::make_shared<Tensor>(b_shape, b->device, DType::Float32, false);
            if (plan.b_repeats) out[1]->fill_(0.0f);
            float* gb = out[1]->data_ptr<float>();
            float beta = plan.b_repeats ? 1.0f : 0.0f;
            matmul_detail::for_each_gemm(plan, !plan.b_repeats, [&](int64_t t) {
                const float* gt = gp + t * gemm_size;
                const float* at = ap + plan.a_offset[(size_t)t];
                float* bt = gb + plan.b_offset[(size_t)t];
                if (trans_b) cpu::gemm(true, false, cols, inner, rows, gt, cols, at, inner, beta, bt, inner);  // dB = G^T A
                else cpu::gemm(true, false, inner, cols, rows, at, inner, gt, cols, beta, bt, cols);           // dB = A^T G
            });
        }
        if (has_bias && needs(2)) {
            out[2] = std::make_shared<Tensor>(bias_shape, a->device, DType::Float32, false);
            kern::host::reduce(kern::Reduce::Sum, gp, out[2]->data_ptr<float>(), 1, g->size() / cols, cols);
        }
        return out;
    }
};

namespace matmul_detail {

// a @ b, a @ b^T (trans_b) or a @ b + bias, on tensors whose memory the host can read
inline std::shared_ptr<Tensor> host_matmul(const std::shared_ptr<Tensor>& a_in, const std::shared_ptr<Tensor>& b_in,
                                           const std::shared_ptr<Tensor>& bias_in, bool trans_b) {
    if (a_in->dtype != DType::Float32 || b_in->dtype != DType::Float32) throw std::invalid_argument("matmul: tensors must be Float32, Float16 or BFloat16");
    auto a = make_contiguous(a_in), b = make_contiguous(b_in);
    auto bias = bias_in ? make_contiguous(bias_in) : nullptr;
    auto plan = make_plan(*a, *b, trans_b);
    auto out = std::make_shared<Tensor>(plan.out_shape, a->device, DType::Float32, false);
    const float* ap = a->data_ptr<float>();
    const float* bp = b->data_ptr<float>();
    float* op = out->data_ptr<float>();
    int64_t size = plan.rows * plan.cols;
    for_each_gemm(plan, true, [&](int64_t t) {
        cpu::gemm(false, trans_b, plan.rows, plan.cols, plan.inner, ap + plan.a_offset[(size_t)t], plan.inner,
                  bp + plan.b_offset[(size_t)t], trans_b ? plan.inner : plan.cols, 0.0f, op + t * size, plan.cols);
    });
    if (bias) {
        const float* bias_p = bias->data_ptr<float>();
        const int64_t cols = plan.cols;
        parallel_for(0, out->size() / std::max<int64_t>(1, cols), std::max<int64_t>(1, 32768 / std::max<int64_t>(1, cols)), [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r)
                for (int64_t j = 0; j < cols; ++j) op[r * cols + j] += bias_p[j];
        });
    }
    bool req_grad = needs_grad(a_in) || needs_grad(b_in) || (bias_in && needs_grad(bias_in));
    out->requires_grad = req_grad;
    if (req_grad) {
        // saved_output: the node keeps the operands' values, not their graphs
        auto grad_fn = std::make_shared<MatmulBackward>(saved_output(a), saved_output(b), std::move(plan), trans_b, bias);
        grad_fn->add_next_edge(get_grad_edge(a_in).function);
        grad_fn->add_next_edge(get_grad_edge(b_in).function);
        if (bias_in) grad_fn->add_next_edge(get_grad_edge(bias_in).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

}  // namespace matmul_detail

#ifdef USE_CUDA
// Backward of a CUDA matmul or linear (matmul plus bias). Holds the operands exactly as the
// forward GEMM consumed them (contiguous, already rounded to the autocast dtype), so backward
// neither re-casts nor re-densifies them, and only those it needs: B for dA, A for dB. Under fp16
// autocast a saved activation is a half-size copy, and the fp32 original can be freed once forward
// moves on. With a bias (bias_shape non-empty) it has a third input, whose gradient is the column
// sum of the output gradient.
struct CudaMatmulBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::vector<int64_t> a_shape, b_shape, bias_shape;
    DType a_dtype, b_dtype;        // the inputs' dtypes, which their gradients take
    std::shared_ptr<Tensor> a, b;  // saved GEMM operands; null when no gradient needs them
    bool grad_a, grad_b, grad_bias;
    DType compute;  // GEMM precision the forward ran with (autocast)

    bool trans_b = false;  // matmul_nt: b is stored [N, K] and the product is a @ b^T

    CudaMatmulBackward(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, DType a_dtype, DType b_dtype,
                       std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                       bool grad_a, bool grad_b, DType compute,
                       std::vector<int64_t> bias_shape = {}, bool grad_bias = false, bool trans_b = false)
        : a_shape(std::move(a_shape)), b_shape(std::move(b_shape)), bias_shape(std::move(bias_shape)),
          a_dtype(a_dtype), b_dtype(b_dtype), a(std::move(a)), b(std::move(b)),
          grad_a(grad_a), grad_b(grad_b), grad_bias(grad_bias), compute(compute), trans_b(trans_b) {}

    // MatmulBackward is the CPU node's label too, so profiles stay comparable across devices
    std::string name() const override { return bias_shape.empty() ? "MatmulBackward" : "LinearBackward"; }

    void release_variables() override {
        a = nullptr;
        b = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr, nullptr};
        if ((grad_a && !b) || (grad_b && !a)) {
            throw std::runtime_error("MatmulBackward: saved operands were released; pass retain_graph=true "
                                     "to call backward through this graph more than once");
        }
        std::shared_ptr<Tensor> da, db;
        cuda::matmul_backward(a_shape, b_shape, a, b, grads[0], grad_a ? &da : nullptr, grad_b ? &db : nullptr,
                              compute, trans_b, a_dtype);
        // Each input gets its gradient in its own dtype (no-ops where the GEMM already wrote it)
        if (da) da = da->to(Device::CUDA, a_dtype);
        if (db) db = db->to(Device::CUDA, b_dtype);
        if (bias_shape.empty()) return {da, db};
        std::shared_ptr<Tensor> dbias;
        if (grad_bias) {
            int64_t cols = b_shape.back();
            dbias = std::make_shared<Tensor>(bias_shape, Device::CUDA, DType::Float32, false);
            const auto& g = grads[0];
            cuda::bias_grad(g->data_ptr<void>(), dbias->data_ptr<float>(), g->size() / cols, cols, g->dtype);
        }
        return {da, db, dbias};
    }
};
#endif

inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    DG_PROFILE_OP("matmul");
    if (a->device != b->device) {
        throw std::runtime_error("matmul: devices mismatch between tensor a and tensor b");
    }
    if (a->ndim() < 1 || b->ndim() < 1) throw std::runtime_error("matmul: tensors must have at least one dimension");
    // A vector is a row on the left and a column on the right; its dimension is dropped afterwards
    if (a->ndim() == 1 && b->ndim() == 1) return squeeze(squeeze(matmul(unsqueeze(a, 0), unsqueeze(b, 1)), 1), 0);
    if (a->ndim() == 1) return squeeze(matmul(unsqueeze(a, 0), b), -2);
    if (b->ndim() == 1) return squeeze(matmul(a, unsqueeze(b, 1)), -1);
#ifdef USE_CUDA
    if (a->device == Device::CUDA || b->device == Device::CUDA) {
        // The GEMMs broadcast a whole unbatched operand; batch dimensions that broadcast in part are expanded
        auto batch_dims = [](const Tensor& t) {
            std::vector<int64_t> d(t.shape.begin(), t.shape.end() - 2);
            while (!d.empty() && d.front() == 1) d.erase(d.begin());
            return d;
        };
        auto da = batch_dims(*a), db = batch_dims(*b);
        if (!da.empty() && !db.empty() && da != db) {
            auto batch = op_detail::broadcast_shapes(da, db, "matmul");
            auto with_batch = [&](const std::shared_ptr<Tensor>& t) {
                auto shape = batch;
                shape.insert(shape.end(), t->shape.end() - 2, t->shape.end());
                while (shape.size() < t->shape.size()) shape.insert(shape.begin(), 1);
                return contiguous(expand(t, shape));
            };
            return matmul(with_batch(a), with_batch(b));
        }
        DType compute = autocast_dtype();
        bool grad_a = needs_grad(a), grad_b = needs_grad(b);
        // dA = dC B^T needs only B and dB = A^T dC only A: keep just the operands backward will use
        std::shared_ptr<Tensor> saved_a, saved_b;
        auto out = cuda::matmul(a, b, compute, grad_b ? &saved_a : nullptr, grad_a ? &saved_b : nullptr);
        out->requires_grad = grad_a || grad_b;
        if (grad_a || grad_b) {
            auto grad_fn = std::make_shared<CudaMatmulBackward>(a->shape, b->shape, a->dtype, b->dtype, std::move(saved_a),
                                                                std::move(saved_b), grad_a, grad_b, compute);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            grad_fn->add_next_edge(get_grad_edge(b).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    return matmul_detail::host_matmul(float32_input(a), float32_input(b), nullptr, false);
}

// a @ b^T for a 2-D b [N, K]: a [..., K] -> [..., N]. A tied LM head multiplies by the [vocab, d]
// embedding this way. The GEMM reads b transposed in place and b's gradient is produced in b's own
// layout, so neither direction materializes a transpose.
inline std::shared_ptr<Tensor> matmul_nt(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    DG_PROFILE_OP("matmul_nt");
    if (!a || !b) throw std::invalid_argument("matmul_nt: null tensor");
    if (b->ndim() != 2) throw std::invalid_argument("matmul_nt: b must be 2-D [N, K], got " + shape_string(b->shape));
    if (a->device != b->device) throw std::invalid_argument("matmul_nt: tensors are on different devices");
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        DType compute = autocast_dtype();
        bool grad_a = needs_grad(a), grad_b = needs_grad(b);
        std::shared_ptr<Tensor> saved_a, saved_b;
        auto out = cuda::matmul_nt(a, b, compute, grad_b ? &saved_a : nullptr, grad_a ? &saved_b : nullptr);
        out->requires_grad = grad_a || grad_b;
        if (grad_a || grad_b) {
            auto grad_fn = std::make_shared<CudaMatmulBackward>(a->shape, b->shape, a->dtype, b->dtype, std::move(saved_a),
                                                                std::move(saved_b), grad_a, grad_b, compute,
                                                                std::vector<int64_t>{}, false, /*trans_b=*/true);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            grad_fn->add_next_edge(get_grad_edge(b).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    return matmul_detail::host_matmul(float32_input(a), float32_input(b), nullptr, true);
}

// y = x @ w + b: x [..., in], w [in, out], b [out] or [1, out], or null for no bias. On CUDA the
// bias is added inside the GEMM (cuBLASLt epilogue) instead of by a separate pass over the output,
// and one LinearBackward node returns dx, dw and db. On the host it is added after the GEMM, in place.
// The output is fp32. Under autocast, out_dtype = autocast_dtype() keeps it in that dtype instead, for
// ops with half-precision kernels to read (see native_half).
inline std::shared_ptr<Tensor> linear(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& w,
                                      const std::shared_ptr<Tensor>& b = nullptr, DType out_dtype = DType::Float32) {
    (void)out_dtype;  // CUDA only
    DG_PROFILE_OP("linear");
    if (!x || !w) throw std::invalid_argument("linear: null input or weight");
    if (!b) return matmul(x, w);
    if (w->ndim() != 2) throw std::invalid_argument("linear: weight must be 2-D [in, out], got " + shape_string(w->shape));
    if (b->size() != w->shape[1]) {
        throw std::invalid_argument("linear: bias " + shape_string(b->shape) + " does not match " +
                                    std::to_string(w->shape[1]) + " output features");
    }
    if (x->device != w->device || b->device != w->device) {
        throw std::invalid_argument("linear: input, weight and bias are on different devices");
    }
#ifdef USE_CUDA
    if (x->device == Device::CUDA) {
        // The bias epilogue adds an fp32 bias to the fp32 output
        auto bias = float32_input(b);
        DType compute = autocast_dtype();
        bool grad_x = needs_grad(x), grad_w = needs_grad(w), grad_b = needs_grad(bias);
        // As in matmul: dx needs only w and dw only x
        std::shared_ptr<Tensor> saved_x, saved_w;
        auto out = cuda::linear(x, w, bias, compute, grad_w ? &saved_x : nullptr, grad_x ? &saved_w : nullptr,
                                out_dtype == compute ? compute : DType::Float32);
        out->requires_grad = grad_x || grad_w || grad_b;
        if (out->requires_grad) {
            auto grad_fn = std::make_shared<CudaMatmulBackward>(x->shape, w->shape, x->dtype, w->dtype, std::move(saved_x),
                                                                std::move(saved_w), grad_x, grad_w, compute, bias->shape, grad_b);
            grad_fn->add_next_edge(get_grad_edge(x).function);
            grad_fn->add_next_edge(get_grad_edge(w).function);
            grad_fn->add_next_edge(get_grad_edge(bias).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (x->ndim() < 2) return add(matmul(x, w), reshape(b, {b->size()}));
    return matmul_detail::host_matmul(float32_input(x), float32_input(w), float32_input(b), false);
}

// The rest of the op library, in dependency order, then create_graph support for the nodes above
#include "elementwise.h"
#include "shape_ops.h"
#include "reduce_ops.h"
#include "index_ops.h"
#include "random_ops.h"
#include "contract_ops.h"
#include "loss_ops.h"
#include "conv_ops.h"
#include "graph_backward.h"
