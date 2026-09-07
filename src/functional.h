#pragma once
#include "tensor.h"
#include "functional.h"
#include "ops.h"
#include <cmath>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda/cuda_ops.cuh"
#endif
#include "metal/metal_backend.h"
#include "tpu/tpu_runner.h"

#if defined(USE_BLAS) || defined(USE_OPENBLAS) || defined(USE_MKL)
    #include <cblas.h>
    #define HAS_BLAS 1
#else
    #define HAS_BLAS 0
#endif

// ========================================
// From activations.h
// ========================================
// --- RELU BACKWARD NODE ---
struct ReluBackward : public Node {
    std::shared_ptr<Tensor> a;
    explicit ReluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
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
inline std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::relu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<ReluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- TANH BACKWARD NODE ---
struct TanhBackward : public Node {
    std::shared_ptr<Tensor> out_val;
    explicit TanhBackward(std::shared_ptr<Tensor> out_val) : out_val(out_val) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(out_val->shape, out_val->device, out_val->dtype, false);
        da->fill_(0.0f);
        const float* g_ptr = grad->data_ptr<float>();
        const float* out_ptr = out_val->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = out_val->size();

        for (int i = 0; i < size; i++) {
            float t = out_ptr[i];
            da_ptr[i] = (1.0f - t * t) * g_ptr[i];
        }
        return {da};
    }
};

// tanh
inline std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::tanh(a_ptr[i]);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<TanhBackward>(out);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
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
inline std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor>& a) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::gelu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<GeluBackward>(a, sqrt_2_over_pi, coeff);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SIGMOID BACKWARD NODE ---
struct SigmoidBackward : public Node {
    std::shared_ptr<Tensor> out_val;
    explicit SigmoidBackward(std::shared_ptr<Tensor> out_val) : out_val(out_val) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
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
inline std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::sigmoid_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(out);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().sigmoid_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(out);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    if (a->device == Device::TPU && a->storage->tpu_handle) {
        out->storage->tpu_handle = TPUEngine::get().sigmoid(a->storage->tpu_handle, a->shape, a->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<SigmoidBackward>(out);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        auto grad_fn = std::make_shared<SigmoidBackward>(out);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SILU (SWISH) BACKWARD NODE ---
struct SiluBackward : public Node {
    std::shared_ptr<Tensor> a;
    explicit SiluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
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
inline std::shared_ptr<Tensor> silu(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::silu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().silu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size());
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    if (a->device == Device::TPU && a->storage->tpu_handle) {
        out->storage->tpu_handle = TPUEngine::get().silu(a->storage->tpu_handle, a->shape, a->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<SiluBackward>(a);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
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
inline std::shared_ptr<Tensor> leaky_relu(const std::shared_ptr<Tensor>& a, float negative_slope = 0.01f) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);
#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::leaky_relu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size(), negative_slope);
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().leaky_relu_forward(a->data_ptr<float>(), out->data_ptr<float>(), a->size(), negative_slope);
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
    if (a->device == Device::TPU && a->storage->tpu_handle) {
        out->storage->tpu_handle = TPUEngine::get().leaky_relu(a->storage->tpu_handle, a->shape, negative_slope, a->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<LeakyReluBackward>(a, negative_slope);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(in_shape, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = self_grad->data_ptr<float>();
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
        return {grad_a};
    }
};

// broadcast
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a, int axis, int n) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    if (axis == 0 && r != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }
    if (axis == 1 && c != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; } 

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = n : out_shape[ndim - 1] = n;

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

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

    if (req_grad) {
        auto grad_fn = std::make_shared<BroadcastBackward>(a->shape, axis, n, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- ADD BACKWARD NODE ---
struct AddBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        return {grads[0], grads[0]};
    }
};

// --- CAST ADD BACKWARD NODE ---
struct CastAddBackward : public Node {
    std::vector<int64_t> x_shape, y_shape;
    int r, c, batch_size;

    CastAddBackward(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, int r, int c, int batch_size)
        : x_shape(x_shape), y_shape(y_shape), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x_shape, self_grad->device, self_grad->dtype, false);
        gx->fill_(0.0f);
        auto gy = std::make_shared<Tensor>(y_shape, self_grad->device, self_grad->dtype, false);
        gy->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
        float* gx_ptr = gx->data_ptr<float>();
        float* gy_ptr = gy->data_ptr<float>();

        int yr = y_shape.size() > 1 ? y_shape[y_shape.size() - 2] : 1;
        int yc = y_shape.size() > 0 ? y_shape[y_shape.size() - 1] : 1;

        int y_batch_size = 1;
        for (int i = 0; i < (int)y_shape.size() - 2; i++) { y_batch_size *= y_shape[i]; }

        for (int batch = 0; batch < batch_size; batch++) {
            int y_b = (y_batch_size == batch_size) ? batch : 0;
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = y_b * yr * yc + yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    gx_ptr[x_idx] += val;
                    gy_ptr[y_idx] += val;
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and add
inline std::shared_ptr<Tensor> cast_n_add(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = (ndim >= 2) ? x->shape[ndim - 2] : 1;
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    int y_batch_size = 1;
    for (int i = 0; i < (int)y->shape.size() - 2; i++) { y_batch_size *= y->shape[i]; }

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, req_grad);

#ifdef USE_CUDA
    if (x->device == Device::CUDA) {
        auto xc = make_contiguous(x);
        auto yc_t = make_contiguous(y);
        cuda::make_contiguous(*out, *xc);
        if (yc_t->size() == c) {
            cuda::add_bias(out->data_ptr<float>(), yc_t->data_ptr<float>(), batch_size * r, c);
        } else {
            cuda::add_inplace(*out, *yc_t);
        }

        if (req_grad) {
            auto grad_fn = std::make_shared<CastAddBackward>(x->shape, y->shape, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(x).function, 0);
            grad_fn->add_next_edge(get_grad_edge(y).function, 1);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif

    const float* x_ptr = x->data_ptr<float>();
    const float* y_ptr = y->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        int y_b = (y_batch_size == batch_size) ? batch : 0;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = y_b * yr * yc + yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] + y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastAddBackward>(x->shape, y->shape, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// Intelligent elementwise / broadcast addition
inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& x,
                                   const std::shared_ptr<Tensor>& y) {
    if (x->shape == y->shape) {
        bool req_grad = x->requires_grad || y->requires_grad;
        auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, req_grad);

#ifdef USE_CUDA
        if (x->device == Device::CUDA) {
            auto xc = make_contiguous(x);
            auto yc_t = make_contiguous(y);
            cuda::make_contiguous(*out, *xc);
            cuda::add_inplace(*out, *yc_t);

            if (req_grad) {
                auto grad_fn = std::make_shared<AddBackward>();
                grad_fn->add_next_edge(get_grad_edge(x).function, 0);
                grad_fn->add_next_edge(get_grad_edge(y).function, 1);
                out->grad_fn = grad_fn;
            }
            return out;
        }
#endif

        auto xc = make_contiguous(x);
        auto yc = make_contiguous(y);
        const float* xp = xc->data_ptr<float>();
        const float* yp = yc->data_ptr<float>();
        float* outp = out->data_ptr<float>();
        int64_t n = x->size();

        for (int64_t i = 0; i < n; i++) {
            outp[i] = xp[i] + yp[i];
        }

        if (req_grad) {
            auto grad_fn = std::make_shared<AddBackward>();
            grad_fn->add_next_edge(get_grad_edge(x).function, 0);
            grad_fn->add_next_edge(get_grad_edge(y).function, 1);
            out->grad_fn = grad_fn;
        }

        return out;
    }

    return cast_n_add(x, y);
}

// --- CAST SUB BACKWARD NODE ---
struct CastSubBackward : public Node {
    std::vector<int64_t> x_shape, y_shape;
    int r, c, batch_size;

    CastSubBackward(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, int r, int c, int batch_size)
        : x_shape(x_shape), y_shape(y_shape), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x_shape, false);
        gx->fill_(0.0f);
        auto gy = std::make_shared<Tensor>(y_shape, false);
        gy->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
        float* gx_ptr = gx->data_ptr<float>();
        float* gy_ptr = gy->data_ptr<float>();

        int yr = y_shape.size() > 1 ? y_shape[y_shape.size() - 2] : 1;
        int yc = y_shape.size() > 0 ? y_shape[y_shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    gx_ptr[x_idx] += val;
                    gy_ptr[y_idx] -= val;
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and subtract
inline std::shared_ptr<Tensor> cast_n_sub(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr<float>();
    const float* y_ptr = y->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] - y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastSubBackward>(x->shape, y->shape, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAST MUL BACKWARD NODE ---
struct CastMulBackward : public Node {
    std::shared_ptr<Tensor> x, y;
    int r, c, batch_size;

    CastMulBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y, int r, int c, int batch_size)
        : x(x), y(y), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x->shape, false);
        gx->fill_(0.0f);
        auto gy = std::make_shared<Tensor>(y->shape, false);
        gy->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* x_ptr = x->data_ptr<float>();
        const float* y_ptr = y->data_ptr<float>();
        float* gx_ptr = gx->data_ptr<float>();
        float* gy_ptr = gy->data_ptr<float>();

        int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
        int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    gx_ptr[x_idx] += y_ptr[y_idx] * val;
                    gy_ptr[y_idx] += x_ptr[x_idx] * val;
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and multiply
inline std::shared_ptr<Tensor> cast_n_mul(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr<float>();
    const float* y_ptr = y->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] * y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastMulBackward>(x, y, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAST DIV BACKWARD NODE ---
struct CastDivBackward : public Node {
    std::shared_ptr<Tensor> x, y;
    int r, c, batch_size;

    CastDivBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y, int r, int c, int batch_size)
        : x(x), y(y), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x->shape, false);
        gx->fill_(0.0f);
        auto gy = std::make_shared<Tensor>(y->shape, false);
        gy->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* x_ptr = x->data_ptr<float>();
        const float* y_ptr = y->data_ptr<float>();
        float* gx_ptr = gx->data_ptr<float>();
        float* gy_ptr = gy->data_ptr<float>();

        int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
        int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    float y_val = y_ptr[y_idx];
                    gx_ptr[x_idx] += val / y_val;
                    gy_ptr[y_idx] -= (x_ptr[x_idx] * val) / (y_val * y_val);
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and divide
inline std::shared_ptr<Tensor> cast_n_div(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr<float>();
    const float* y_ptr = y->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] / y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastDivBackward>(x, y, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
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
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(a_shape, false);
        ga->fill_(0.0f);
        auto gb = std::make_shared<Tensor>(b_shape, false);
        gb->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
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
        return {ga, gb};
    }
};

// concatenation
inline std::shared_ptr<Tensor> concat(const std::shared_ptr<Tensor>& a,
                                       const std::shared_ptr<Tensor>& b, int axis) {
    int ndim = a->shape.size();
    int r_a = a->shape[ndim - 2];
    int c_a = a->shape[ndim - 1];
    int r_b = b->shape[ndim - 2];
    int c_b = b->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = r_a + r_b : out_shape[ndim - 1] = c_a + c_b;

    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    int r_out = out_shape[ndim - 2];
    int c_out = out_shape[ndim - 1];

    const float* a_ptr = a->data_ptr<float>();
    const float* b_ptr = b->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

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

    if (req_grad) {
        auto grad_fn = std::make_shared<ConcatBackward>(a->shape, b->shape, axis, r_a, c_a, r_b, c_b, r_out, c_out, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(b).function, 1);
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
    int n;
    CrossEntropyBackward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, int n)
        : pred(pred), target(target), n(n) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_pred = std::make_shared<Tensor>(pred->shape, false);
        grad_pred->fill_(0.0f);
        const float* p_ptr = pred->data_ptr<float>();
        const float* t_ptr = target->data_ptr<float>();
        float* gp_ptr = grad_pred->data_ptr<float>();
        float g_val = self_grad->data_ptr<float>()[0];

        for (int i = 0; i < n; i++) {
            gp_ptr[i] = -(t_ptr[i] / (p_ptr[i] * n)) * g_val;
        }
        return {grad_pred};
    }
};

// cross_entropy
inline std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if (pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    int n = static_cast<int>(pred->size());
    const float* p_ptr = pred->data_ptr<float>();
    const float* t_ptr = target->data_ptr<float>();

    float sum_loss = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_loss -= t_ptr[i] * std::log(p_ptr[i] + 1e-8f);
    }

    bool req_grad = pred->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, req_grad);
    out->data_ptr<float>()[0] = sum_loss / n;

    if (req_grad) {
        auto grad_fn = std::make_shared<CrossEntropyBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- FUSED SPARSE CROSS-ENTROPY BACKWARD NODE ---
struct FusedCrossEntropyBackward : public Node {
    std::shared_ptr<Tensor> logits;
    std::shared_ptr<Tensor> targets;
    std::shared_ptr<Tensor> probs;
    int N, V;
    int valid_count;
    int ignore_index;

    FusedCrossEntropyBackward(std::shared_ptr<Tensor> logits,
                              std::shared_ptr<Tensor> targets,
                              std::shared_ptr<Tensor> probs,
                              int N, int V, int valid_count, int ignore_index)
        : logits(logits), targets(targets), probs(probs),
          N(N), V(V), valid_count(valid_count), ignore_index(ignore_index) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto dloss = grads[0];
        float dloss_val = dloss->data_ptr<float>()[0];
        auto grad_logits = std::make_shared<Tensor>(logits->shape, logits->device, logits->dtype, false);
        grad_logits->fill_(0.0f);

        const float* p_ptr = probs->data_ptr<float>();
        const float* t_ptr = targets->data_ptr<float>();
        float* g_ptr = grad_logits->data_ptr<float>();

        float scale = (valid_count > 0) ? (dloss_val / static_cast<float>(valid_count)) : 0.0f;

        for (int i = 0; i < N; ++i) {
            int target_idx = static_cast<int>(t_ptr[i]);
            if (target_idx == ignore_index) continue;

            int offset = i * V;
            for (int c = 0; c < V; ++c) {
                float prob = p_ptr[offset + c];
                float grad_c = (c == target_idx) ? (prob - 1.0f) : prob;
                g_ptr[offset + c] = scale * grad_c;
            }
        }

        return {grad_logits};
    }
};

// Fused Logits Cross Entropy with Integer Class Targets and LogSumExp Stability
// logits: [..., vocab_size] (unnormalized scores)
// targets: [...] (integer token/class IDs, e.g. [batch, seq_len] or [N])
// ignore_index: tokens with this target ID are ignored in loss computation (default: -100)
inline std::shared_ptr<Tensor> fused_cross_entropy(
    const std::shared_ptr<Tensor>& logits,
    const std::shared_ptr<Tensor>& targets,
    int ignore_index = -100
) {
    int ndim = logits->ndim();
    int V = logits->shape[ndim - 1];
    int N = logits->size() / V;

    if (targets->size() != N) {
        throw std::invalid_argument(
            "fused_cross_entropy: target size (" + std::to_string(targets->size()) +
            ") must match total logit positions (" + std::to_string(N) + ")"
        );
    }

    const float* l_ptr = logits->data_ptr<float>();
    const float* t_ptr = targets->data_ptr<float>();

    auto probs = std::make_shared<Tensor>(std::vector<int64_t>{N, V}, logits->device, logits->dtype, false);
    float* p_ptr = probs->data_ptr<float>();

    float total_loss = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < N; ++i) {
        int target_idx = static_cast<int>(t_ptr[i]);
        int offset = i * V;

        // 1. Numerically stable row-max subtraction
        float max_val = -1e20f;
        for (int c = 0; c < V; ++c) {
            float val = l_ptr[offset + c];
            if (val > max_val) max_val = val;
        }

        // 2. Compute sum of exps
        float sum_exp = 0.0f;
        for (int c = 0; c < V; ++c) {
            float exp_val = std::exp(l_ptr[offset + c] - max_val);
            p_ptr[offset + c] = exp_val;
            sum_exp += exp_val;
        }

        // 3. Normalize into probabilities
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < V; ++c) {
            p_ptr[offset + c] *= inv_sum;
        }

        // 4. LogSumExp loss: (log(sum_exp) + max_val) - logits[target_idx]
        if (target_idx != ignore_index) {
            if (target_idx < 0 || target_idx >= V) {
                throw std::out_of_range(
                    "fused_cross_entropy: target index " + std::to_string(target_idx) +
                    " is out of bounds for vocab size " + std::to_string(V)
                );
            }
            float lse = std::log(sum_exp) + max_val;
            total_loss += (lse - l_ptr[offset + target_idx]);
            valid_count++;
        }
    }

    float mean_loss = (valid_count > 0) ? (total_loss / static_cast<float>(valid_count)) : 0.0f;

    bool req_grad = logits->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, logits->device, logits->dtype, req_grad);
    out->data_ptr<float>()[0] = mean_loss;

    if (req_grad) {
        auto grad_fn = std::make_shared<FusedCrossEntropyBackward>(
            logits, targets, probs, N, V, valid_count, ignore_index
        );
        grad_fn->add_next_edge(get_grad_edge(logits).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_pred = std::make_shared<Tensor>(pred->shape, pred->device, pred->dtype, false);
        grad_pred->fill_(0.0f);
        float g_val = self_grad->data_ptr<float>()[0];

        if (pred->device == Device::MPS) {
            float scale = (2.0f / static_cast<float>(n)) * g_val;
            MetalBackend::get().mse_backward(pred->data_ptr<float>(), target->data_ptr<float>(), grad_pred->data_ptr<float>(), n, scale);
            return {grad_pred};
        }

        const float* p_ptr = pred->data_ptr<float>();
        const float* t_ptr = target->data_ptr<float>();
        float* gp_ptr = grad_pred->data_ptr<float>();

        for (int i = 0; i < n; i++) {
            gp_ptr[i] = (2.0f * (p_ptr[i] - t_ptr[i]) / n) * g_val;
        }
        return {grad_pred};
    }
};

// mse
inline std::shared_ptr<Tensor> mse(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if (pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    int n = static_cast<int>(pred->size());

    if (pred->device == Device::TPU && pred->storage->tpu_handle && target->storage->tpu_handle) {
        bool req_grad = pred->requires_grad;
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::TPU, pred->dtype, req_grad);
        out->storage->tpu_handle = TPUEngine::get().mse(pred->storage->tpu_handle, target->storage->tpu_handle, pred->shape, pred->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    const float* p_ptr = pred->data_ptr<float>();
    const float* t_ptr = target->data_ptr<float>();

    float sq_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = p_ptr[i] - t_ptr[i];
        sq_sum += diff * diff;
    }

    bool req_grad = pred->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, pred->device, pred->dtype, req_grad);
    out->data_ptr<float>()[0] = sq_sum / n;

    if (req_grad) {
        auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_pred = std::make_shared<Tensor>(pred->shape, pred->device, pred->dtype, false);
        grad_pred->fill_(0.0f);
        float g_val = self_grad->data_ptr<float>()[0];

        if (pred->device == Device::MPS) {
            float scale = (1.0f / static_cast<float>(n)) * g_val;
            MetalBackend::get().l1_loss_backward(pred->data_ptr<float>(), target->data_ptr<float>(), grad_pred->data_ptr<float>(), n, scale);
            return {grad_pred};
        }

        const float* p_ptr = pred->data_ptr<float>();
        const float* t_ptr = target->data_ptr<float>();
        float* gp_ptr = grad_pred->data_ptr<float>();

        for (int i = 0; i < n; i++) {
            float diff = p_ptr[i] - t_ptr[i];
            float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
            gp_ptr[i] = (sgn / static_cast<float>(n)) * g_val;
        }
        return {grad_pred};
    }
};

// l1_loss (MAE - Mean Absolute Error)
inline std::shared_ptr<Tensor> l1_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if (pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    int n = static_cast<int>(pred->size());

    if (pred->device == Device::TPU && pred->storage->tpu_handle && target->storage->tpu_handle) {
        bool req_grad = pred->requires_grad;
        auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::TPU, pred->dtype, req_grad);
        out->storage->tpu_handle = TPUEngine::get().l1_loss(pred->storage->tpu_handle, target->storage->tpu_handle, pred->shape, pred->dtype);
        if (req_grad) {
            auto grad_fn = std::make_shared<L1LossBackward>(pred, target, n);
            grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    const float* p_ptr = pred->data_ptr<float>();
    const float* t_ptr = target->data_ptr<float>();

    float abs_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        abs_sum += std::abs(p_ptr[i] - t_ptr[i]);
    }

    bool req_grad = pred->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, pred->device, pred->dtype, req_grad);
    out->data_ptr<float>()[0] = abs_sum / static_cast<float>(n);

    if (req_grad) {
        auto grad_fn = std::make_shared<L1LossBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(self_grad->shape, false);
        ga->fill_(0.0f);

        const float* sg_ptr = self_grad->data_ptr<float>();
        const float* m_ptr = m->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    ga_ptr[idx] = (m_ptr[idx] != 0.0f) ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {ga};
    }
};

// boolean mask
inline std::shared_ptr<Tensor> bool_mask(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& m) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    const float* m_ptr = m->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

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

    if (req_grad) {
        auto grad_fn = std::make_shared<BoolMaskBackward>(m, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAUSAL MASK BACKWARD NODE ---
struct CausalMaskBackward : public Node {
    int r, c, batch_size;
    CausalMaskBackward(int r, int c, int batch_size) : r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(self_grad->shape, self_grad->device, self_grad->dtype, false);
        ga->fill_(0.0f);

#ifdef USE_CUDA
        if (self_grad->device == Device::CUDA) {
            cuda::causal_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(), batch_size, r);
            return {ga};
        }
#endif
        if (self_grad->device == Device::MPS) {
            MetalBackend::get().causal_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(), batch_size, r);
            return {ga};
        }

        const float* sg_ptr = self_grad->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    ga_ptr[idx] = (j <= i) ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {ga};
    }
};

// causal mask
inline std::shared_ptr<Tensor> causal_mask(const std::shared_ptr<Tensor>& a) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);

#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::causal_mask(a->data_ptr<float>(), out->data_ptr<float>(), batch_size, r);
        if (req_grad) {
            auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (a->device == Device::MPS) {
        MetalBackend::get().causal_mask(a->data_ptr<float>(), out->data_ptr<float>(), batch_size, r);
        if (req_grad) {
            auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

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

    if (req_grad) {
        auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(in_shape, false);
        grad_a->fill_(0.0f);
        const float* sg_ptr = self_grad->data_ptr<float>();
        float* ga_ptr = grad_a->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int64_t> batch_idx = unravel(batch, std::vector<int64_t>(in_shape.begin(), in_shape.end() - 2));

            int batch_off_a = 0;
            int batch_off_out = 0;
            for (int i = 0; i < ndim - 2; i++) {
                batch_off_a   += batch_idx[i] * in_strides[i];
                batch_off_out += batch_idx[i] * self_grad->strides[i];
            }

            if (axis == 0) {
                for (int i = 0; i < c; i++) {
                    int flat_out = batch_off_out + self_grad->strides[nout - 2] * 0 + self_grad->strides[nout - 1] * i;
                    float val = sg_ptr[flat_out] / r;
                    for (int j = 0; j < r; j++) {
                        int flat_a = batch_off_a + in_strides[ndim - 2] * j + in_strides[ndim - 1] * i;
                        ga_ptr[flat_a] = val;
                    }
                }
            } else {
                for (int i = 0; i < r; i++) {
                    int flat_out = batch_off_out + self_grad->strides[nout - 2] * i + self_grad->strides[nout - 1] * 0;
                    float val = sg_ptr[flat_out] / c;
                    for (int j = 0; j < c; j++) {
                        int flat_a = batch_off_a + in_strides[ndim - 2] * i + in_strides[ndim - 1] * j;
                        ga_ptr[flat_a] = val;
                    }
                }
            }
        }
        return {grad_a};
    }
};

inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->ndim();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;
    
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    int nout = out->ndim();

    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    // forward
    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int64_t> batch_idx = unravel(batch, std::vector<int64_t>(a->shape.begin(), a->shape.end() - 2));

        int batch_off_a = 0;
        int batch_off_out = 0;
        for (int i = 0; i < ndim - 2; i++) {
            batch_off_a   += batch_idx[i] * a->strides[i];
            batch_off_out += batch_idx[i] * out->strides[i];
        }

        if (axis == 0) {
            for (int i = 0; i < c; i++) {
                float total = 0.0f;
                for (int j = 0; j < r; j++) {
                    int flat_a = batch_off_a + a->strides[ndim - 2] * j + a->strides[ndim - 1] * i;
                    total += a_ptr[flat_a];
                }
                int flat_out = batch_off_out + out->strides[nout - 2] * 0 + out->strides[nout - 1] * i;
                out_ptr[flat_out] = total / r;
            }
        } else {
            for (int i = 0; i < r; i++) {
                float total = 0.0f;
                for (int j = 0; j < c; j++) {
                    int flat_a = batch_off_a + a->strides[ndim - 2] * i + a->strides[ndim - 1] * j;
                    total += a_ptr[flat_a];
                }
                int flat_out = batch_off_out + out->strides[nout - 2] * i + out->strides[nout - 1] * 0;
                out_ptr[flat_out] = total / c;
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<MeanBackward>(a->shape, a->strides, axis, r, c, batch_size, ndim, nout);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ========================================
// From scalar_ops.h
// ========================================
// --- ADD SCALAR BACKWARD NODE ---
struct AddScalarBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        return {grads[0]};
    }
};

inline std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float s) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] + s;
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<AddScalarBackward>();
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SUB SCALAR BACKWARD NODE ---
struct SubScalarBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        return {grads[0]};
    }
};

inline std::shared_ptr<Tensor> sub_scalar(const std::shared_ptr<Tensor>& a, float s) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] - s;
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SubScalarBackward>();
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- MUL SCALAR BACKWARD NODE ---
struct MulScalarBackward : public Node {
    float scalar;
    explicit MulScalarBackward(float s) : scalar(s) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(grad->shape, false);
        da->fill_(0.0f);
        const float* g_ptr = grad->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = grad->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = g_ptr[i] * scalar;
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float s) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] * s;
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<MulScalarBackward>(s);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- DIV SCALAR BACKWARD NODE ---
struct DivScalarBackward : public Node {
    float scalar;
    explicit DivScalarBackward(float s) : scalar(s) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(grad->shape, false);
        da->fill_(0.0f);
        const float* g_ptr = grad->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = grad->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = g_ptr[i] / scalar;
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> div_scalar(const std::shared_ptr<Tensor>& a, float s) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] / s;
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<DivScalarBackward>(s);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

inline std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& a) {
    return mul_scalar(a, -1.0f);
}

// --- RSUB SCALAR BACKWARD NODE ---
struct RsubScalarBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(grad->shape, false);
        da->fill_(0.0f);
        const float* g_ptr = grad->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = grad->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = -g_ptr[i];
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> rsub_scalar(float s, const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = s - a_ptr[i];
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<RsubScalarBackward>();
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- RDIV SCALAR BACKWARD NODE ---
struct RdivScalarBackward : public Node {
    std::shared_ptr<Tensor> a;
    float scalar;
    RdivScalarBackward(std::shared_ptr<Tensor> a, float s) : a(a), scalar(s) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        da->fill_(0.0f);
        const float* g_ptr = grad->data_ptr<float>();
        const float* a_ptr = a->data_ptr<float>();
        float* da_ptr = da->data_ptr<float>();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            float val = a_ptr[i];
            da_ptr[i] = -scalar * g_ptr[i] / (val * val);
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> rdiv_scalar(float s, const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = s / a_ptr[i];
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<RdivScalarBackward>(a, s);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

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
        auto grad_out = grads[0];
        
        // Create full gradient tensor filled with zeros
        auto grad_in = std::make_shared<Tensor>(original_tensor->shape, original_tensor->device, original_tensor->dtype, false);
        grad_in->fill_(0.0f);

        // Map grad_out directly into the specific slice window of grad_in
        auto grad_in_slice = std::make_shared<Tensor>(grad_out->shape, grad_in->device, grad_in->dtype, false);
        grad_in_slice->fill_(0.0f);
        grad_in_slice->storage = grad_in->storage;
        grad_in_slice->strides = grad_in->strides;
        grad_in_slice->global_offset = grad_in->global_offset + start * grad_in->strides[dim];

        // In-place addition to funnel gradients into the right spot
        tensor_add_inplace(grad_in_slice, grad_out);

        return {grad_in};
    }
};

// O(1) Tensor Slicing (Memory Sharing)
inline std::shared_ptr<Tensor> slice(const std::shared_ptr<Tensor>& x, int dim, int start, int end) {
    if (dim < 0 || dim >= x->ndim()) throw std::runtime_error("slice: dim out of bounds");
    if (start < 0) start += x->shape[dim];
    if (end < 0) end += x->shape[dim];
    if (start < 0 || end > x->shape[dim] || start >= end) throw std::runtime_error("slice: invalid bounds");

    // Inherit memory pointers but shrink the shape window
    auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, x->requires_grad);
    out->storage = x->storage; 
    out->strides = x->strides;
    out->shape[dim] = end - start;
    out->global_offset = x->global_offset + start * x->strides[dim];

    if (x->requires_grad) {
        auto grad_fn = std::make_shared<SliceBackward>(x, dim, start, end);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(s->shape, s->device, s->dtype, false);
        ga->fill_(0.0f);

#ifdef USE_CUDA
        if (s->device == Device::CUDA) {
            cuda::softmax_backward(s->data_ptr<float>(), self_grad->data_ptr<float>(), ga->data_ptr<float>(), batch_size * r, c);
            return {ga};
        }
#endif

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

inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a, int axis = -1) {
    int ndim = a->shape.size();
    if (axis < 0) axis = ndim + axis;
    if (axis != ndim - 1) throw std::runtime_error("softmax currently optimized only for last dimension (axis=-1)");

    int r = (ndim >= 2) ? a->shape[ndim - 2] : 1;
    int c = a->shape[ndim - 1];
    int batch_size = a->size() / (r * c);

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, req_grad);

#ifdef USE_CUDA
    if (a->device == Device::CUDA) {
        cuda::softmax_forward(a->data_ptr<float>(), out->data_ptr<float>(), batch_size * r, c);
        if (req_grad) {
            auto grad_fn = std::make_shared<SoftmaxBackward>(out, r, c, batch_size, ndim);
            grad_fn->add_next_edge(get_grad_edge(a).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif

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
        auto grad_fn = std::make_shared<SoftmaxBackward>(out, r, c, batch_size, ndim);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(a->shape, false);
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
inline std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(in_shape, false);
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
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    if (axis < 0) axis = ndim + axis;
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int64_t> out_shape = a->shape;
    out_shape[axis] = 1;

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);

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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        return {transpose(grads[0], dim0, dim1)};
    }
};

// Functional transpose with autograd support
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a, int dim0, int dim1) {
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

    if (a->requires_grad) {
        auto grad_fn = std::make_shared<TransposeBackward>(d0, d1);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(a->shape, false);
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
inline std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    std::vector<int64_t> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
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
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        // Reshape incoming gradient back to input shape
        return {grads[0]->_reshape(in_shape)};
    }
};

// Fast zero-copy functional view (throws error if non-contiguous)
inline std::shared_ptr<Tensor> view(const std::shared_ptr<Tensor>& a, const std::vector<int64_t>& new_shape) {
    auto out = a->_view(new_shape);

    if (a->requires_grad) {
        auto grad_fn = std::make_shared<ViewBackward>(a->shape);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// General functional reshape (handles any layout)
inline std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& a, const std::vector<int64_t>& new_shape) {
    auto out = a->_reshape(new_shape);

    if (a->requires_grad) {
        auto grad_fn = std::make_shared<ViewBackward>(a->shape);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
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
        auto grad_out = grads[0];
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

inline std::shared_ptr<Tensor> im2col(const std::shared_ptr<Tensor>& x, int kH, int kW, int padH, int padW, int strideH, int strideW) {
    if (x->ndim() != 4) throw std::runtime_error("im2col requires 4D tensor (B, C, H, W)");
    
    auto xc = make_contiguous(x);
    int B = xc->shape[0];
    int C = xc->shape[1];
    int H = xc->shape[2];
    int W = xc->shape[3];

    int H_out = (H + 2 * padH - kH) / strideH + 1;
    int W_out = (W + 2 * padW - kW) / strideW + 1;

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{B, C * kH * kW, H_out, W_out}, xc->device, xc->dtype, xc->requires_grad);
    
    const float* x_data = xc->data_ptr<float>();
    float* out_data = out->data_ptr<float>();

    int x_batch_stride = C * H * W;
    int out_batch_stride = (C * kH * kW) * (H_out * W_out);

    for (int b = 0; b < B; ++b) {
        im2col_cpu(x_data + b * x_batch_stride, C, H, W, kH, kW, padH, padW, strideH, strideW, 1, 1, out_data + b * out_batch_stride);
    }

    if (xc->requires_grad) {
        auto grad_fn = std::make_shared<Im2ColBackward>(xc, kH, kW, padH, padW, strideH, strideW);
        grad_fn->add_next_edge(get_grad_edge(xc).function, 0);
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
        std::shared_ptr<Tensor> self_grad = grads[0];

        auto grad_a = std::make_shared<Tensor>(a->shape, a->device, a->dtype, false);
        grad_a->fill_(0.0f);
        auto grad_b = std::make_shared<Tensor>(b->shape, b->device, b->dtype, false);
        grad_b->fill_(0.0f);
        
        // Critical Fix: Zero out gradients because we are accumulating across batches
        grad_a->fill_(0.0f);
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

    bool req_grad = _a->requires_grad || _b->requires_grad;
    
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
        grad_fn->add_next_edge(get_grad_edge(_a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(_b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
#ifdef USE_CUDA
    if (a->device == Device::CUDA || b->device == Device::CUDA) {
        return cuda::matmul(a, b);
    }
#endif
    return manual_matmul(a, b);
}

