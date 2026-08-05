#pragma once
#include "engine.h"

// --- RELU BACKWARD NODE ---
struct ReluBackward : public Node {
    std::shared_ptr<Tensor> a;
    explicit ReluBackward(std::shared_ptr<Tensor> a) : a(a) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* da_ptr = da->data_ptr();
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
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
        auto da = std::make_shared<Tensor>(out_val->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* out_ptr = out_val->data_ptr();
        float* da_ptr = da->data_ptr();
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
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
        auto da = std::make_shared<Tensor>(a->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* da_ptr = da->data_ptr();
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
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
