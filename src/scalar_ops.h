#pragma once
#include "engine.h"

// --- ADD SCALAR BACKWARD NODE ---
struct AddScalarBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        return {grads[0]};
    }
};

inline std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float s) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
        const float* g_ptr = grad->data_ptr();
        float* da_ptr = da->data_ptr();
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
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
        const float* g_ptr = grad->data_ptr();
        float* da_ptr = da->data_ptr();
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
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
        const float* g_ptr = grad->data_ptr();
        float* da_ptr = da->data_ptr();
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
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* da_ptr = da->data_ptr();
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
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
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
