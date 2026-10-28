#pragma once
#include "tensor.h"
#include "autocast.h"
#include "profiler.h"
#include "functional.h"
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

#if defined(USE_BLAS) || defined(USE_OPENBLAS) || defined(USE_MKL)
extern "C" {
    #ifndef CBLAS_ENUM_DEFINED
    #define CBLAS_ENUM_DEFINED
    typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
    typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
    #endif
    void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const int lda, const float *B, const int ldb,
                     const float beta, float *C, const int ldc);
}
#define HAS_BLAS 1
#else
#define HAS_BLAS 0
#endif

// ========================================
// Contiguity
// ========================================
// --- CONTIGUOUS BACKWARD NODE ---
struct ContiguousBackward : public Node {
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

// ========================================
// From activations.h
// ========================================
// --- RELU BACKWARD NODE ---
struct ReluBackward : public Node {
    std::shared_ptr<Tensor> a;
    explicit ReluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
        da->fill_(0.0f);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::relu_backward(a->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), a->size());
            return {da};
        }
#endif
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = (a_ptr[i] > 0.0f) ? g_ptr[i] : 0.0f;
        }
        return {da};
    }
};

// relu
inline std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("relu");
    auto a = contiguous(float32_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::relu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
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
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::max(0.0f, a_ptr[i]);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<ReluBackward>(a);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- TANH BACKWARD NODE ---
struct TanhBackward : public Node {
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
        da->fill_(0.0f);
        const float* g_ptr = grad_cpu->data_ptr<float>();
        const float* out_ptr = out_cpu->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = out_cpu->size();

        for (int i = 0; i < size; i++) {
            float t = out_ptr[i];
            da_ptr[i] = (1.0f - t * t) * g_ptr[i];
        }
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
    int size = a_cpu->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::tanh(a_ptr[i]);
    }

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
        da->fill_(0.0f);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::gelu_backward(a->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), a->size());
            return {da};
        }
#endif
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            float x = a_ptr[i];
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            float tanh_val = std::tanh(inner);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
            float g = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
            da_ptr[i] = g * g_ptr[i];
        }
        return {da};
    }
};

// gelu
inline std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("gelu");
    auto a = contiguous(float32_input(a_in));
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::gelu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
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
    int size = a->size();

    for (int i = 0; i < size; i++) {
        float x = a_ptr[i];
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        float tanh_val = std::tanh(inner);
        out_ptr[i] = 0.5f * x * (1.0f + tanh_val);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<GeluBackward>(a, sqrt_2_over_pi, coeff);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SIGMOID BACKWARD NODE ---
struct SigmoidBackward : public Node {
    std::shared_ptr<Tensor> out_val;
    explicit SigmoidBackward(std::shared_ptr<Tensor> out_val) : out_val(out_val) {}

    void release_variables() override {
        out_val.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !out_val) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(out_val->shape, out_val->device, out_val->dtype, false);
        da->fill_(0.0f);
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
        int size = static_cast<int>(out_val->size());

        for (int i = 0; i < size; i++) {
            float s = o_ptr[i];
            da_ptr[i] = s * (1.0f - s) * g_ptr[i];
        }
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
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(saved_output(out));
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = static_cast<int>(a->size());

    for (int i = 0; i < size; i++) {
        out_ptr[i] = 1.0f / (1.0f + std::exp(-a_ptr[i]));
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SigmoidBackward>(saved_output(out));
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SILU (SWISH) BACKWARD NODE ---
struct SiluBackward : public Node {
    std::shared_ptr<Tensor> a;
    explicit SiluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
        da->fill_(0.0f);
#ifdef USE_CUDA
        if (a->device == Device::CUDA) {
            cuda::silu_backward(a->data_ptr<float>(), grad->data_ptr<float>(), da->data_ptr<float>(), a->size());
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
        int size = static_cast<int>(a->size());

        for (int i = 0; i < size; i++) {
            float x = a_ptr[i];
            float s = 1.0f / (1.0f + std::exp(-x));
            da_ptr[i] = (s * (1.0f + x * (1.0f - s))) * g_ptr[i];
        }
        return {da};
    }
};

// silu (Swish)
inline std::shared_ptr<Tensor> silu(const std::shared_ptr<Tensor>& a_in) {
    DG_PROFILE_OP("silu");
    auto a = contiguous(float32_input(a_in));
    bool req_grad = grad_enabled() && (a->requires_grad);
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::silu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
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
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = static_cast<int>(a->size());

    for (int i = 0; i < size; i++) {
        float x = a_ptr[i];
        out_ptr[i] = x / (1.0f + std::exp(-x));
    }

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
        da->fill_(0.0f);
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
        int size = static_cast<int>(a->size());

        for (int i = 0; i < size; i++) {
            da_ptr[i] = (a_ptr[i] > 0.0f) ? g_ptr[i] : (negative_slope * g_ptr[i]);
        }
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
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = static_cast<int>(a->size());

    for (int i = 0; i < size; i++) {
        float x = a_ptr[i];
        out_ptr[i] = (x > 0.0f) ? x : (negative_slope * x);
    }

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
// --- BROADCAST BACKWARD NODE ---
struct BroadcastBackward : public Node {
    std::vector<int64_t> in_shape;
    int axis, n, r, c, batch_size;

    BroadcastBackward(const std::vector<int64_t>& in_shape, int axis, int n, int r, int c, int batch_size)
        : in_shape(in_shape), axis(axis), n(n), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto grad_a = std::make_shared<Tensor>(in_shape, Device::CPU, sg_cpu->dtype, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = sg_cpu->data_ptr<float>();
        float* ga_ptr = grad_a->data_ptr<float>();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < c; j++) {
                        ga_ptr[j + batch * c] += sg_ptr[i * c + j + batch * n * c];
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < n; j++) {
                        ga_ptr[i + batch * r] += sg_ptr[i * n + j + batch * r * n];
                    }
                }
            }
        }
        return {(dev == Device::CPU) ? grad_a : grad_a->to(dev)};
    }
};

// broadcast
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a_in, int axis, int n) {
    DG_PROFILE_OP("broadcast");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 2) {
        throw std::invalid_argument("broadcast requires tensor with ndim >= 2");
    }
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    if (axis == 0 && r != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }
    if (axis == 1 && c != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; } 

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = n : out_shape[ndim - 1] = n;

    bool req_grad = grad_enabled() && (a->requires_grad);
    Device dev = a->device;
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto out_cpu = std::make_shared<Tensor>(out_shape, Device::CPU, a_cpu->dtype, false);
    const float* a_ptr = a_cpu->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < c; j++) {
                    out_ptr[i * c + j + batch * n * c] = a_ptr[j + batch * c];
                }
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < n; j++) {
                    out_ptr[i * n + j + batch * r * n] = a_ptr[i + batch * r];
                }
            }
        }
    }

    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<BroadcastBackward>(a->shape, axis, n, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// Broadcast binary ops: add, sub, mul, div
// ========================================
// NumPy-style broadcasting of y onto x: shapes are aligned from the right, and every y dimension
// must be 1 or equal to x's. The output has x's shape. Padding masks [B, 1, 1, S], ALiBi biases
// [1, H, S, S] and Linear biases [1, out] all broadcast this way.
enum class BinaryOp : int { Add = 0, Sub = 1, Mul = 2, Div = 3 };

inline std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float s);

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

inline void forward_host(BinaryOp op, const float* x, const float* y, float* out, const BroadcastLayout& l) {
    switch (op) {
        case BinaryOp::Add: l.for_each([&](int64_t i, int64_t yo) { out[i] = x[i] + y[yo]; }); break;
        case BinaryOp::Sub: l.for_each([&](int64_t i, int64_t yo) { out[i] = x[i] - y[yo]; }); break;
        case BinaryOp::Mul: l.for_each([&](int64_t i, int64_t yo) { out[i] = x[i] * y[yo]; }); break;
        default: l.for_each([&](int64_t i, int64_t yo) { out[i] = x[i] / y[yo]; }); break;
    }
}

// dL/dx for mul and div (for add and sub it is the incoming gradient itself)
inline void grad_x_host(BinaryOp op, const float* g, const float* y, float* gx, const BroadcastLayout& l) {
    if (op == BinaryOp::Mul) l.for_each([&](int64_t i, int64_t yo) { gx[i] = g[i] * y[yo]; });
    else l.for_each([&](int64_t i, int64_t yo) { gx[i] = g[i] / y[yo]; });
}

// dL/dy accumulated in x order, so the result is deterministic. gy must start zeroed.
inline void grad_y_host(BinaryOp op, const float* g, const float* x, const float* y, float* gy, const BroadcastLayout& l) {
    switch (op) {
        case BinaryOp::Add: l.for_each([&](int64_t i, int64_t yo) { gy[yo] += g[i]; }); break;
        case BinaryOp::Sub: l.for_each([&](int64_t i, int64_t yo) { gy[yo] -= g[i]; }); break;
        case BinaryOp::Mul: l.for_each([&](int64_t i, int64_t yo) { gy[yo] += g[i] * x[i]; }); break;
        default: l.for_each([&](int64_t i, int64_t yo) { gy[yo] -= g[i] * x[i] / (y[yo] * y[yo]); }); break;
    }
}

}  // namespace broadcast_detail

struct BroadcastBinaryBackward : public Node {
    BinaryOp op;
    std::vector<int64_t> x_shape, y_shape;  // as given by the caller
    BroadcastLayout layout;
    std::shared_ptr<Tensor> x, y;           // saved only for mul and div

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
    auto x = contiguous(float32_input(x_in));
    auto y = contiguous(float32_input(y_in));
    if (x->dtype != DType::Float32 || y->dtype != DType::Float32) {
        throw std::invalid_argument(std::string(binary_op_name(op)) + ": tensors must be Float32, Float16 or BFloat16");
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

// --- CONCAT BACKWARD NODE ---
struct ConcatBackward : public Node {
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
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto ga = std::make_shared<Tensor>(a_shape, Device::CPU, sg_cpu->dtype, false);
        ga->fill_(0.0f);
        auto gb = std::make_shared<Tensor>(b_shape, Device::CPU, sg_cpu->dtype, false);
        gb->fill_(0.0f);

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
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto b_cpu = (dev == Device::CPU) ? b : b->cpu();
    auto out_cpu = std::make_shared<Tensor>(out_shape, Device::CPU, a_cpu->dtype, false);
    int r_out = out_shape[ndim - 2];
    int c_out = out_shape[ndim - 1];

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
        for (int64_t row = 0; row < rows; ++row) {
            int target = (int)t[row];
            for (int64_t c = 0; c < vocab; ++c) {
                int64_t i = row * vocab + c;
                g[i] = (target == ignore_index) ? 0.0f : scale * (std::exp(l[i] - e[row]) - (c == target ? 1.0f : 0.0f));
            }
        }
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
        double total = 0.0;
        int64_t valid = 0;
        for (int64_t row = 0; row < rows; ++row) {
            const float* lr = l + row * vocab;
            float mx = -INFINITY;
            for (int64_t c = 0; c < vocab; ++c) mx = std::max(mx, lr[c]);
            float sum = 0.0f;
            for (int64_t c = 0; c < vocab; ++c) sum += std::exp(lr[c] - mx);
            e[row] = std::log(sum) + mx;
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

        for (int i = 0; i < n; i++) {
            gp_ptr[i] = (2.0f * (p_ptr[i] - t_ptr[i]) / n) * g_val;
        }
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
        if (req_grad) {
            auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    Device dev = pred->device;
    auto p_cpu = (dev == Device::CPU) ? pred : pred->cpu();
    auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
    const float* p_ptr = p_cpu->data_ptr<float>();
    const float* t_ptr = t_cpu->data_ptr<float>();

    float sq_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = p_ptr[i] - t_ptr[i];
        sq_sum += diff * diff;
    }

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
        if (req_grad) {
            auto grad_fn = std::make_shared<L1LossBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    Device dev = pred->device;
    auto p_cpu = (dev == Device::CPU) ? pred : pred->cpu();
    auto t_cpu = (target->device == Device::CPU) ? target : target->cpu();
    const float* p_ptr = p_cpu->data_ptr<float>();
    const float* t_ptr = t_cpu->data_ptr<float>();

    float abs_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        abs_sum += std::abs(p_ptr[i] - t_ptr[i]);
    }

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
    auto m = contiguous(float32_input(m_in));
    int ndim = a->shape.size();
    if (ndim < 2) throw std::invalid_argument("bool_mask requires tensor with ndim >= 2");
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = grad_enabled() && (a->requires_grad);
    Device dev = a->device;
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

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    ga_ptr[idx] = (j <= i) ? sg_ptr[idx] : 0.0f;
                }
            }
        }
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

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int idx = batch * r * c + i * c + j;
                out_ptr[idx] = j > i
                              ? -1e9f
                              : a_ptr[idx];
            }
        }
    }

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
// From mean.h
// ========================================
#pragma once 

// --- MEAN BACKWARD NODE ---
struct MeanBackward : public Node {
    std::vector<int64_t> in_shape;
    std::vector<int64_t> in_strides;
    int axis;
    int r, c, batch_size, ndim, nout;

    MeanBackward(const std::vector<int64_t>& in_shape, const std::vector<int64_t>& in_strides,
                 int axis, int r, int c, int batch_size, int ndim, int nout)
        : in_shape(in_shape), in_strides(in_strides), axis(axis), r(r), c(c),
          batch_size(batch_size), ndim(ndim), nout(nout) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto grad_a = std::make_shared<Tensor>(in_shape, Device::CPU, sg_cpu->dtype, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = sg_cpu->data_ptr<float>();
        float* ga_ptr = grad_a->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int64_t> batch_idx = unravel(batch, std::vector<int64_t>(in_shape.begin(), in_shape.end() - 2));

            int batch_off_a = 0;
            int batch_off_out = 0;
            for (int i = 0; i < ndim - 2; i++) {
                batch_off_a   += batch_idx[i] * in_strides[i];
                batch_off_out += batch_idx[i] * sg_cpu->strides[i];
            }

            if (axis == 0) {
                for (int i = 0; i < c; i++) {
                    int flat_out = batch_off_out + sg_cpu->strides[nout - 2] * 0 + sg_cpu->strides[nout - 1] * i;
                    float val = sg_ptr[flat_out] / r;
                    for (int j = 0; j < r; j++) {
                        int flat_a = batch_off_a + in_strides[ndim - 2] * j + in_strides[ndim - 1] * i;
                        ga_ptr[flat_a] = val;
                    }
                }
            } else {
                for (int i = 0; i < r; i++) {
                    int flat_out = batch_off_out + sg_cpu->strides[nout - 2] * i + sg_cpu->strides[nout - 1] * 0;
                    float val = sg_ptr[flat_out] / c;
                    for (int j = 0; j < c; j++) {
                        int flat_a = batch_off_a + in_strides[ndim - 2] * i + in_strides[ndim - 1] * j;
                        ga_ptr[flat_a] = val;
                    }
                }
            }
        }
        return {(dev == Device::CPU) ? grad_a : grad_a->to(dev)};
    }
};

inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a_in, int axis) {
    DG_PROFILE_OP("mean");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->ndim();
    if (ndim < 2) throw std::invalid_argument("mean requires tensor with ndim >= 2");
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;
    
    bool req_grad = grad_enabled() && (a->requires_grad);
    Device dev = a->device;
    auto a_cpu = (dev == Device::CPU) ? a : a->cpu();
    auto out_cpu = std::make_shared<Tensor>(out_shape, Device::CPU, a_cpu->dtype, false);
    int nout = out_cpu->ndim();

    const float* a_ptr = a_cpu->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();

    // forward
    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int64_t> batch_idx = unravel(batch, std::vector<int64_t>(a_cpu->shape.begin(), a_cpu->shape.end() - 2));

        int batch_off_a = 0;
        int batch_off_out = 0;
        for (int i = 0; i < ndim - 2; i++) {
            batch_off_a   += batch_idx[i] * a_cpu->strides[i];
            batch_off_out += batch_idx[i] * out_cpu->strides[i];
        }

        if (axis == 0) {
            for (int i = 0; i < c; i++) {
                float total = 0.0f;
                for (int j = 0; j < r; j++) {
                    int flat_a = batch_off_a + a_cpu->strides[ndim - 2] * j + a_cpu->strides[ndim - 1] * i;
                    total += a_ptr[flat_a];
                }
                int flat_out = batch_off_out + out_cpu->strides[nout - 2] * 0 + out_cpu->strides[nout - 1] * i;
                out_ptr[flat_out] = total / r;
            }
        } else {
            for (int i = 0; i < r; i++) {
                float total = 0.0f;
                for (int j = 0; j < c; j++) {
                    int flat_a = batch_off_a + a_cpu->strides[ndim - 2] * i + a_cpu->strides[ndim - 1] * j;
                    total += a_ptr[flat_a];
                }
                int flat_out = batch_off_out + out_cpu->strides[nout - 2] * i + out_cpu->strides[nout - 1] * 0;
                out_ptr[flat_out] = total / c;
            }
        }
    }

    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<MeanBackward>(a->shape, a->strides, axis, r, c, batch_size, ndim, nout);
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
        for (int64_t i = 0; i < g->size(); ++i) op_[i] = scalar_detail::backward(op, gp[i], ap ? ap[i] : 0.0f, scalar);
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
        for (int64_t i = 0; i < a->size(); ++i) op_[i] = scalar_detail::forward(op, ap[i], s);
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
    std::shared_ptr<Tensor> original_tensor;
    int dim;
    int start;
    int end;
    
    SliceBackward(std::shared_ptr<Tensor> original, int dim, int start, int end)
        : original_tensor(original), dim(dim), start(start), end(end) {}

    void release_variables() override {
        original_tensor = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !original_tensor) return {nullptr};
        const auto& grad_out = grads[0];  // dense: the engine densifies grads
        Device dev = grad_out->device;

        // Zero gradient for the full input; the sliced window receives grad_out. grad_in is dense,
        // so the window is `len` consecutive rows of `inner` elements inside each of `outer` blocks:
        // one strided 2-D copy of `outer` rows, in any dtype.
        const auto& shape = original_tensor->shape;
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
            auto host = SliceBackward(original_tensor, dim, start, end).apply({grad_out->cpu()});
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
        auto grad_fn = std::make_shared<SliceBackward>(x, dim, start, end);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From softmax.h
// ========================================
struct SoftmaxBackward : public Node {
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
        ga->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* s_ptr = s->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();

        // Optimized backward pass
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                int offset = batch * r * c + i * c;
                float sum_dot = 0.0f;
                
                for (int j = 0; j < c; j++) sum_dot += sg_ptr[offset + j] * s_ptr[offset + j];
                for (int j = 0; j < c; j++) {
                    ga_ptr[offset + j] = s_ptr[offset + j] * (sg_ptr[offset + j] - sum_dot);
                }
            }
        }
        return {ga};
    }
};

inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a_in, int axis = -1) {
    DG_PROFILE_OP("softmax");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 1) throw std::runtime_error("softmax: tensor must be at least 1D");
    if (axis < 0) axis = ndim + axis;
    if (axis != ndim - 1) throw std::runtime_error("softmax currently optimized only for last dimension (axis=-1)");

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

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            int offset = batch * r * c + i * c;
            
            float max_val = -1e9f;
            for (int j = 0; j < c; j++) if (a_ptr[offset + j] > max_val) max_val = a_ptr[offset + j];
            float sum = 0.0f;
            for (int j = 0; j < c; j++) {
                out_ptr[offset + j] = std::exp(a_ptr[offset + j] - max_val);
                sum += out_ptr[offset + j];
            }
            float inv_sum = 1.0f / (sum + 1e-12f);
            for (int j = 0; j < c; j++) out_ptr[offset + j] *= inv_sum;
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SoftmaxBackward>(saved_output(out), r, c, batch_size, ndim);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

// ========================================
// From std.h
// ========================================
// --- STD DEV BACKWARD NODE ---
struct StdDevBackward : public Node {
    std::shared_ptr<Tensor> a;
    int axis, r, c, batch_size;

    StdDevBackward(std::shared_ptr<Tensor> a, int axis, int r, int c, int batch_size)
        : a(a), axis(axis), r(r), c(c), batch_size(batch_size) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];

        if (a->device != Device::CPU) {
            auto a_cpu = a->to(Device::CPU);
            auto sg_cpu = self_grad->to(Device::CPU);
            StdDevBackward cpu_node(a_cpu, axis, r, c, batch_size);
            auto cpu_grads = cpu_node.apply({sg_cpu});
            if (cpu_grads.empty() || !cpu_grads[0]) return {nullptr};
            return {cpu_grads[0]->to(a->device)};
        }

        auto grad_a = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* ga_ptr = grad_a->data_ptr<float>();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < c; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < r; j++) {
                        sum += a_ptr[j * c + i + batch * r * c];
                    }
                    float mean_val = sum / r;
                    float x = 0.0f;
                    for (int j = 0; j < r; j++) {
                        float diff = a_ptr[j * c + i + batch * r * c] - mean_val;
                        x += diff * diff;
                    }
                    float std_val = std::sqrt(x / r) + 1e-8f;
                    float dout = sg_ptr[i + batch * c];
                    for (int j = 0; j < r; j++) {
                        ga_ptr[j * c + i + batch * r * c] = ((a_ptr[j * c + i + batch * r * c] - mean_val) / (r * std_val)) * dout;
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < c; j++) {
                        sum += a_ptr[i * c + j + batch * r * c];
                    }
                    float mean_val = sum / c;
                    float x = 0.0f;
                    for (int j = 0; j < c; j++) {
                        float diff = a_ptr[i * c + j + batch * r * c] - mean_val;
                        x += diff * diff;
                    }
                    float std_val = std::sqrt(x / c) + 1e-8f;
                    float dout = sg_ptr[i + batch * r];
                    for (int j = 0; j < c; j++) {
                        ga_ptr[i * c + j + batch * r * c] = ((a_ptr[i * c + j + batch * r * c] - mean_val) / (c * std_val)) * dout;
                    }
                }
            }
        }
        return {grad_a};
    }
};

// standard deviation
inline std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a_in, int axis) {
    DG_PROFILE_OP("std_dev");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 2) throw std::runtime_error("std_dev: tensor must be at least 2D");
    if (axis != 0 && axis != 1) throw std::runtime_error("std_dev: axis must be 0 or 1");

    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = grad_enabled() && (a->requires_grad);

    if (a->device != Device::CPU) {
        auto a_cpu = a->to(Device::CPU);
        auto out_cpu = std_dev(a_cpu, axis);
        auto out_res = out_cpu->to(a->device);
        if (req_grad) {
            auto grad_fn = std::make_shared<StdDevBackward>(a, axis, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }

    auto out = std::make_shared<Tensor>(out_shape, a->device, a->dtype, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < c; i++) {
                float sum = 0.0f;
                for (int j = 0; j < r; j++) {
                    sum += a_ptr[j * c + i + batch * r * c];
                }
                float mean_val = sum / r;
                float x = 0.0f;
                for (int j = 0; j < r; j++) {
                    float diff = a_ptr[j * c + i + batch * r * c] - mean_val;
                    x += diff * diff;
                }
                out_ptr[i + batch * c] = std::sqrt(x / r);
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                float sum = 0.0f;
                for (int j = 0; j < c; j++) {
                    sum += a_ptr[i * c + j + batch * r * c];
                }
                float mean_val = sum / c;
                float x = 0.0f;
                for (int j = 0; j < c; j++) {
                    float diff = a_ptr[i * c + j + batch * r * c] - mean_val;
                    x += diff * diff;
                }
                out_ptr[i + batch * r] = std::sqrt(x / c);
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<StdDevBackward>(a, axis, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From sum.h
// ========================================
#pragma once 

// --- SUM BACKWARD NODE ---
struct SumBackward : public Node {
    std::vector<int64_t> in_shape;
    int axis;
    int r, c, batch_size, ndim;

    SumBackward(const std::vector<int64_t>& in_shape, int axis, int r, int c, int batch_size, int ndim)
        : in_shape(in_shape), axis(axis), r(r), c(c), batch_size(batch_size), ndim(ndim) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];

        if (self_grad->device != Device::CPU) {
            auto sg_cpu = self_grad->to(Device::CPU);
            SumBackward cpu_node(in_shape, axis, r, c, batch_size, ndim);
            auto cpu_grads = cpu_node.apply({sg_cpu});
            if (cpu_grads.empty() || !cpu_grads[0]) return {nullptr};
            return {cpu_grads[0]->to(self_grad->device)};
        }

        auto grad_a = std::make_shared<Tensor>(in_shape, self_grad->device, self_grad->dtype, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = self_grad->data_ptr<float>();
        float* ga_ptr = grad_a->data_ptr<float>();

        if (axis == ndim - 2) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < c; i++) {
                    float g_val = sg_ptr[i + batch * c];
                    for (int j = 0; j < r; j++) {
                        ga_ptr[j * c + i + batch * r * c] = g_val;
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    float g_val = sg_ptr[i + batch * r];
                    for (int j = 0; j < c; j++) {
                        ga_ptr[i * c + j + batch * r * c] = g_val;
                    }
                }
            }
        }
        return {grad_a};
    }
};

// sum
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a_in, int axis) {
    DG_PROFILE_OP("sum");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 2) throw std::runtime_error("sum: tensor must be at least 2D");
    if (axis < 0) axis = ndim + axis;
    if (axis != ndim - 2 && axis != ndim - 1) {
        throw std::runtime_error("only supported for summing along the last two dimensions");
    }
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int64_t> out_shape = a->shape;
    out_shape[axis] = 1;

    bool req_grad = grad_enabled() && (a->requires_grad);

    if (a->device != Device::CPU) {
        auto a_cpu = a->to(Device::CPU);
        auto out_cpu = sum(a_cpu, axis);
        auto out_res = out_cpu->to(a->device);
        if (req_grad) {
            auto grad_fn = std::make_shared<SumBackward>(a->shape, axis, r, c, batch_size, ndim);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }

    auto out = std::make_shared<Tensor>(out_shape, a->device, a->dtype, req_grad);

    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    if (axis == ndim - 2) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < c; i++) {
                float total = 0.0f;
                for (int j = 0; j < r; j++) {
                    total += a_ptr[j * c + i + batch * r * c];
                }
                out_ptr[i + batch * c] = total;
            }
        }
    } else if (axis == ndim - 1) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                float total = 0.0f;
                for (int j = 0; j < c; j++) {
                    total += a_ptr[i * c + j + batch * r * c];
                }
                out_ptr[i + batch * r] = total;
            }
        }
    } else {
        throw std::runtime_error("only supported for summing along the last two dimensions");
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SumBackward>(a->shape, axis, r, c, batch_size, ndim);
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
// From var.h
// ========================================
#pragma once 

// --- VAR BACKWARD NODE ---
struct VarBackward : public Node {
    std::shared_ptr<Tensor> a;
    int axis, r, c, batch_size;

    VarBackward(std::shared_ptr<Tensor> a, int axis, int r, int c, int batch_size)
        : a(a), axis(axis), r(r), c(c), batch_size(batch_size) {}

    void release_variables() override {
        a.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a) return {nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];

        if (a->device != Device::CPU) {
            auto a_cpu = a->to(Device::CPU);
            auto sg_cpu = self_grad->to(Device::CPU);
            VarBackward cpu_node(a_cpu, axis, r, c, batch_size);
            auto cpu_grads = cpu_node.apply({sg_cpu});
            if (cpu_grads.empty() || !cpu_grads[0]) return {nullptr};
            return {cpu_grads[0]->to(a->device)};
        }

        auto grad_a = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* ga_ptr = grad_a->data_ptr<float>();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < c; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < r; j++) {
                        sum += a_ptr[j * c + i + batch * r * c];
                    }
                    float mean_val = sum / r;
                    float dout = sg_ptr[i + batch * c];
                    for (int j = 0; j < r; j++) {
                        ga_ptr[j * c + i + batch * r * c] = (2.0f * (a_ptr[j * c + i + batch * r * c] - mean_val) / r) * dout;
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < c; j++) {
                        sum += a_ptr[i * c + j + batch * r * c];
                    }
                    float mean_val = sum / c;
                    float dout = sg_ptr[i + batch * r];
                    for (int j = 0; j < c; j++) {
                        ga_ptr[i * c + j + batch * r * c] = (2.0f * (a_ptr[i * c + j + batch * r * c] - mean_val) / c) * dout;
                    }
                }
            }
        }
        return {grad_a};
    }
};

// variance
inline std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& a_in, int axis) {
    DG_PROFILE_OP("var");
    auto a = contiguous(float32_input(a_in));
    int ndim = a->shape.size();
    if (ndim < 2) throw std::runtime_error("var: tensor must be at least 2D");
    if (axis != 0 && axis != 1) throw std::runtime_error("var: axis must be 0 or 1");

    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = grad_enabled() && (a->requires_grad);

    if (a->device != Device::CPU) {
        auto a_cpu = a->to(Device::CPU);
        auto out_cpu = var(a_cpu, axis);
        auto out_res = out_cpu->to(a->device);
        if (req_grad) {
            auto grad_fn = std::make_shared<VarBackward>(a, axis, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }

    auto out = std::make_shared<Tensor>(out_shape, a->device, a->dtype, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < c; i++) {
                float sum = 0.0f;
                for (int j = 0; j < r; j++) {
                    sum += a_ptr[j * c + i + batch * r * c];
                }
                float mean_val = sum / r;
                float x = 0.0f;
                for (int j = 0; j < r; j++) {
                    float diff = a_ptr[j * c + i + batch * r * c] - mean_val;
                    x += diff * diff;
                }
                out_ptr[i + batch * c] = x / r;
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                float sum = 0.0f;
                for (int j = 0; j < c; j++) {
                    sum += a_ptr[i * c + j + batch * r * c];
                }
                float mean_val = sum / c;
                float x = 0.0f;
                for (int j = 0; j < c; j++) {
                    float diff = a_ptr[i * c + j + batch * r * c] - mean_val;
                    x += diff * diff;
                }
                out_ptr[i + batch * r] = x / c;
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<VarBackward>(a, axis, r, c, batch_size);
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

struct Im2ColBackward : public Node {
    std::shared_ptr<Tensor> x;
    int kH, kW, padH, padW, strideH, strideW;
    
    Im2ColBackward(std::shared_ptr<Tensor> x, int kH, int kW, int padH, int padW, int strideH, int strideW)
        : x(x), kH(kH), kW(kW), padH(padH), padW(padW), strideH(strideH), strideW(strideW) {}

    void release_variables() override { x = nullptr; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !x) return {nullptr};
        auto grad_out = grads[0];

        if (x->device != Device::CPU) {
            auto x_cpu = x->to(Device::CPU);
            auto go_cpu = grad_out->to(Device::CPU);
            Im2ColBackward cpu_node(x_cpu, kH, kW, padH, padW, strideH, strideW);
            auto cpu_grads = cpu_node.apply({go_cpu});
            if (cpu_grads.empty() || !cpu_grads[0]) return {nullptr};
            return {cpu_grads[0]->to(x->device)};
        }

        auto grad_x = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);
        grad_x->fill_(0.0f);
        
        int B = x->shape[0];
        int C = x->shape[1];
        int H = x->shape[2];
        int W = x->shape[3];

        int H_col = grad_out->shape[2];
        int W_col = grad_out->shape[3];

        const float* go_data = grad_out->data_ptr<float>();
        float* gx_data = grad_x->data_ptr<float>();

        int x_batch_stride = C * H * W;
        int go_batch_stride = (C * kH * kW) * (H_col * W_col);

        for (int b = 0; b < B; ++b) {
            col2im_cpu(go_data + b * go_batch_stride, C, H, W, kH, kW, padH, padW, strideH, strideW, 1, 1, gx_data + b * x_batch_stride);
        }
        
        return {grad_x};
    }
};

inline std::shared_ptr<Tensor> im2col(const std::shared_ptr<Tensor>& x_in, int kH, int kW, int padH, int padW, int strideH, int strideW) {
    DG_PROFILE_OP("im2col");
    auto x = contiguous(float32_input(x_in));
    if (x->ndim() != 4) throw std::runtime_error("im2col requires 4D tensor (B, C, H, W)");

    if (x->device != Device::CPU) {
        auto x_cpu = x->to(Device::CPU);
        auto out_cpu = im2col(x_cpu, kH, kW, padH, padW, strideH, strideW);
        auto out_res = out_cpu->to(x->device);
        if (needs_grad(x)) {
            auto grad_fn = std::make_shared<Im2ColBackward>(x, kH, kW, padH, padW, strideH, strideW);
            grad_fn->add_next_edge(get_grad_edge(x).function);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }
    
    auto xc = make_contiguous(x);
    int B = xc->shape[0];
    int C = xc->shape[1];
    int H = xc->shape[2];
    int W = xc->shape[3];

    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{B, C * kH * kW, H_out, W_out}, xc->device, xc->dtype, needs_grad(xc));
    
    const float* x_data = xc->data_ptr<float>();
    float* out_data = out->data_ptr<float>();

    int x_batch_stride = C * H * W;
    int out_batch_stride = (C * kH * kW) * (H_out * W_out);

    for (int b = 0; b < B; ++b) {
        im2col_cpu(x_data + b * x_batch_stride, C, H, W, kH, kW, padH, padW, strideH, strideW, 1, 1, out_data + b * out_batch_stride);
    }

    if (needs_grad(xc)) {
        auto grad_fn = std::make_shared<Im2ColBackward>(xc, kH, kW, padH, padW, strideH, strideW);
        grad_fn->add_next_edge(get_grad_edge(xc).function);
        out->grad_fn = grad_fn;
    }
    
    return out;
}

// ========================================
// From matmul.h
// ========================================
#include <stdexcept>

// --- MATMUL BACKWARD NODE ---
struct MatmulBackward : public Node {
    std::shared_ptr<Tensor> a, b;
    int r1, c1, r2, c2, batch_size, n1, n2, nout, nout_batch;
    std::vector<int64_t> strides_a_broad, strides_b_broad, out_shape;

    MatmulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                   int r1, int c1, int r2, int c2, int batch_size,
                   int n1, int n2, int nout, int nout_batch,
                   std::vector<int64_t> strides_a_broad, std::vector<int64_t> strides_b_broad,
                   std::vector<int64_t> out_shape)
        : a(a), b(b), r1(r1), c1(c1), r2(r2), c2(c2), batch_size(batch_size),
          n1(n1), n2(n2), nout(nout), nout_batch(nout_batch),
          strides_a_broad(std::move(strides_a_broad)),
          strides_b_broad(std::move(strides_b_broad)),
          out_shape(std::move(out_shape)) {}

    void release_variables() override {
        a = nullptr;
        b = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !a || !b) return {nullptr, nullptr};
        std::shared_ptr<Tensor> self_grad = grads[0];

#ifdef USE_CUDA
        if (self_grad->device == Device::CUDA || a->device == Device::CUDA || b->device == Device::CUDA) {
            std::shared_ptr<Tensor> grad_a, grad_b;
            cuda::matmul_backward(a->shape, b->shape, a, b, self_grad, &grad_a, &grad_b);
            return {grad_a, grad_b};
        }
#endif

        if (self_grad->device != Device::CPU || a->device != Device::CPU || b->device != Device::CPU) {
            auto a_cpu = a->to(Device::CPU);
            auto b_cpu = b->to(Device::CPU);
            auto sg_cpu = self_grad->to(Device::CPU);
            MatmulBackward cpu_node(a_cpu, b_cpu, r1, c1, r2, c2, batch_size, n1, n2, nout, nout_batch,
                                    strides_a_broad, strides_b_broad, out_shape);
            auto cpu_grads = cpu_node.apply({sg_cpu});
            if (cpu_grads.size() < 2) return {nullptr, nullptr};
            return {cpu_grads[0] ? cpu_grads[0]->to(a->device) : nullptr,
                    cpu_grads[1] ? cpu_grads[1]->to(b->device) : nullptr};
        }

        auto grad_a = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
        grad_a->fill_(0.0f);
        auto grad_b = std::make_shared<Tensor>(b->shape, b->device, b->dtype, false);
        grad_b->fill_(0.0f);

        const float* b_data = b->data_ptr<float>();
        const float* a_data = a->data_ptr<float>();
        const float* out_grad_data = self_grad->data_ptr<float>();

        float* ga_data = grad_a->data_ptr<float>();
        float* gb_data = grad_b->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int64_t> out_shape_64(out_shape.begin(), out_shape.begin() + nout_batch);
            std::vector<int64_t> batch_idx_64 = unravel(batch, out_shape_64); 
            
            int batch_off_a = 0;
            int batch_off_b = 0;
            int batch_off_out = 0;

            for (int i = 0; i < nout_batch; i++) {
                batch_off_a += batch_idx_64[i] * strides_a_broad[i];
                batch_off_b += batch_idx_64[i] * strides_b_broad[i];
                batch_off_out += batch_idx_64[i] * self_grad->strides[i];
            }

#if HAS_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                        r1, c1, c2, 1.0f, 
                        out_grad_data + batch_off_out, c2, 
                        b_data + batch_off_b, c2, 1.0f, 
                        ga_data + batch_off_a, c1);
#else
            for (int i = 0; i < r1; i++) {
                for (int j = 0; j < c1; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < c2; k++) {
                        sum += out_grad_data[batch_off_out + i*c2 + k] * b_data[batch_off_b + j*c2 + k];
                    }
                    ga_data[batch_off_a + i*c1 + j] += sum;
                }
            }
#endif

#if HAS_BLAS
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                        c1, c2, r1, 1.0f, 
                        a_data + batch_off_a, c1, 
                        out_grad_data + batch_off_out, c2, 1.0f, 
                        gb_data + batch_off_b, c2);
#else
            for (int i = 0; i < c1; i++) {
                for (int j = 0; j < c2; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < r1; k++) {
                        sum += a_data[batch_off_a + k*c1 + i] * out_grad_data[batch_off_out + k*c2 + j];
                    }
                    gb_data[batch_off_b + i*c2 + j] += sum;
                }
            }
#endif
        }

        return {grad_a, grad_b};
    }
};

inline std::shared_ptr<Tensor> manual_matmul(const std::shared_ptr<Tensor>& _a, const std::shared_ptr<Tensor>& _b) {
    if (_a->device != _b->device) {
        throw std::runtime_error("matmul: devices mismatch between tensor a and tensor b");
    }

    auto a = make_contiguous(_a);
    auto b = make_contiguous(_b);

    int n1 = a->ndim();
    int n2 = b->ndim();

    if (n1 < 2 || n2 < 2) throw std::runtime_error("matmul: Tensors must be at least 2D...");

    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    if (c1 != r2) throw std::runtime_error("matmul: inner dimensions mismatch...");  

    int b1 = n1 - 2;
    int b2 = n2 - 2;
    int nout_batch = std::max(b1, b2);

    std::vector<int64_t> out_shape;
    std::vector<int64_t> strides_a_broad(nout_batch, 0);
    std::vector<int64_t> strides_b_broad(nout_batch, 0);

    for (int i = 0; i < nout_batch; ++i) {
        int idx_a = i - (nout_batch - b1);
        int idx_b = i - (nout_batch - b2);

        int dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("matmul: batch dimensions mismatch for broadcasting...");
        }

        int dim_out = std::max(dim_a, dim_b);
        out_shape.push_back(dim_out);

        strides_a_broad[i] = (idx_a >= 0 && dim_a != 1) ? a->strides[idx_a] : 0;
        strides_b_broad[i] = (idx_b >= 0 && dim_b != 1) ? b->strides[idx_b] : 0;
    }

    out_shape.push_back(r1);
    out_shape.push_back(c2);

    bool req_grad = grad_enabled() && (_a->requires_grad || _b->requires_grad);

    if (a->device != Device::CPU || b->device != Device::CPU) {
        auto a_cpu = a->to(Device::CPU);
        auto b_cpu = b->to(Device::CPU);
        auto out_cpu = manual_matmul(a_cpu, b_cpu);
        auto out_res = out_cpu->to(a->device);
        if (req_grad) {
            int batch_size = 1;
            for (int i = 0; i < nout_batch; i++) batch_size *= out_shape[i];
            int nout = out_res->ndim();
            auto grad_fn = std::make_shared<MatmulBackward>(
                _a, _b, r1, c1, r2, c2, batch_size, n1, n2, nout, nout_batch,
                strides_a_broad, strides_b_broad, out_shape
            );
            grad_fn->add_next_edge(get_grad_edge(_a).function);
            grad_fn->add_next_edge(get_grad_edge(_b).function);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }
    
    std::vector<int64_t> out_shape_64(out_shape.begin(), out_shape.end());
    auto out = std::make_shared<Tensor>(out_shape_64, a->device, a->dtype, req_grad);
    int nout = out->ndim();

    int batch_size = 1;
    for (int i = 0; i < nout_batch; i++) batch_size *= out_shape[i];

    const float* a_data = a->data_ptr<float>();
    const float* b_data = b->data_ptr<float>();
    float* out_data = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int64_t> batch_shape_64(out_shape.begin(), out_shape.begin() + nout_batch);
        std::vector<int64_t> batch_idx = unravel(batch, batch_shape_64); 

        int batch_off_a = 0;
        int batch_off_b = 0;
        int batch_off_out = 0;

        for (int i = 0; i < nout_batch; i++) {
            batch_off_a += batch_idx[i] * strides_a_broad[i];
            batch_off_b += batch_idx[i] * strides_b_broad[i];
            batch_off_out += batch_idx[i] * out->strides[i];
        }

#if HAS_BLAS
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    r1, c2, c1, 
                    1.0f, 
                    a_data + batch_off_a, c1, 
                    b_data + batch_off_b, c2, 
                    0.0f, 
                    out_data + batch_off_out, c2);
#else
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                float sum = 0.0f;
                for (int k = 0; k < c1; k++) {
                    sum += a_data[batch_off_a + i*c1 + k] * b_data[batch_off_b + k*c2 + j];
                }
                out_data[batch_off_out + i*c2 + j] = sum;
            }
        }
#endif
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<MatmulBackward>(
            a, b, r1, c1, r2, c2, batch_size, n1, n2, nout, nout_batch,
            strides_a_broad, strides_b_broad, out_shape
        );
        grad_fn->add_next_edge(get_grad_edge(_a).function);
        grad_fn->add_next_edge(get_grad_edge(_b).function);
        out->grad_fn = grad_fn;
    }

    return out;
}

#ifdef USE_CUDA
// Backward of a CUDA matmul or linear (matmul plus bias). Holds the operands exactly as the
// forward GEMM consumed them (contiguous, already rounded to the autocast dtype), so backward
// neither re-casts nor re-densifies them, and only those it needs: B for dA, A for dB. Under fp16
// autocast a saved activation is a half-size copy, and the fp32 original can be freed once forward
// moves on. With a bias (bias_shape non-empty) it has a third input, whose gradient is the column
// sum of the output gradient.
struct CudaMatmulBackward : public Node {
    std::vector<int64_t> a_shape, b_shape, bias_shape;
    DType a_dtype, b_dtype;        // the inputs' dtypes, which their gradients take
    std::shared_ptr<Tensor> a, b;  // saved GEMM operands; null when no gradient needs them
    bool grad_a, grad_b, grad_bias;
    DType compute;  // GEMM precision the forward ran with (autocast)

    CudaMatmulBackward(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, DType a_dtype, DType b_dtype,
                       std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                       bool grad_a, bool grad_b, DType compute,
                       std::vector<int64_t> bias_shape = {}, bool grad_bias = false)
        : a_shape(std::move(a_shape)), b_shape(std::move(b_shape)), bias_shape(std::move(bias_shape)),
          a_dtype(a_dtype), b_dtype(b_dtype), a(std::move(a)), b(std::move(b)),
          grad_a(grad_a), grad_b(grad_b), grad_bias(grad_bias), compute(compute) {}

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
                              compute);
        // The GEMMs write fp32; a half-precision input gets its gradient in its own dtype
        if (da) da = da->to(Device::CUDA, a_dtype);
        if (db) db = db->to(Device::CUDA, b_dtype);
        if (bias_shape.empty()) return {da, db};
        std::shared_ptr<Tensor> dbias;
        if (grad_bias) {
            int64_t cols = b_shape.back();
            dbias = std::make_shared<Tensor>(bias_shape, Device::CUDA, DType::Float32, false);
            auto g = grads[0]->to(Device::CUDA, DType::Float32);
            cuda::bias_grad(g->data_ptr<float>(), dbias->data_ptr<float>(), g->size() / cols, cols);
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
#ifdef USE_CUDA
    if (a->device == Device::CUDA || b->device == Device::CUDA) {
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
    return manual_matmul(float32_input(a), float32_input(b));
}

// y = x @ w + b: x [..., in], w [in, out], b [out] or [1, out], or null for no bias. On CUDA the
// bias is added inside the GEMM (cuBLASLt epilogue) instead of by a separate pass over the output,
// and one LinearBackward node returns dx, dw and db. Elsewhere it is matmul followed by add.
inline std::shared_ptr<Tensor> linear(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& w,
                                      const std::shared_ptr<Tensor>& b = nullptr) {
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
        auto out = cuda::linear(x, w, bias, compute, grad_w ? &saved_x : nullptr, grad_x ? &saved_w : nullptr);
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
    return add(matmul(x, w), b);
}

