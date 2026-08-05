#pragma once
#include "engine.h"

// --- ADD OPERATION ---
struct AddBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        return {grad, grad};
    }
};

inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) {
        throw std::runtime_error("add: shape mismatch");
    }
    bool req_grad = a->requires_grad || b->requires_grad; 

    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    const float* b_ptr = b->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = out->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<AddBackward>();
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SUB OPERATION ---
struct SubBackward : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto neg_grad = std::make_shared<Tensor>(grad->shape, false);
        const float* g_ptr = grad->data_ptr();
        float* ng_ptr = neg_grad->data_ptr();
        int size = grad->size();
        for (int i = 0; i < size; i++) {
            ng_ptr[i] = -g_ptr[i];
        }
        return {grad, neg_grad};
    }
};

inline std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) {
        throw std::runtime_error("sub: shape mismatch");
    }
    bool req_grad = a->requires_grad || b->requires_grad;

    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    const float* b_ptr = b->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = out->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] - b_ptr[i];
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SubBackward>();
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- MUL OPERATION ---
struct MulBackward : public Node {
    std::shared_ptr<Tensor> a, b;
    MulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) : a(a), b(b) {}

    void release_variables() override { a = nullptr; b = nullptr; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        auto db = std::make_shared<Tensor>(b->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        const float* b_ptr = b->data_ptr();
        float* da_ptr = da->data_ptr();
        float* db_ptr = db->data_ptr();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = g_ptr[i] * b_ptr[i];
            db_ptr[i] = g_ptr[i] * a_ptr[i];
        }
        return {da, db};
    }
};

inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) {
        throw std::runtime_error("mul: shape mismatch");
    }
    bool req_grad = a->requires_grad || b->requires_grad;

    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    const float* b_ptr = b->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = out->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] * b_ptr[i];
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<MulBackward>(a, b);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- DIV OPERATION ---
struct DivBackward : public Node {
    std::shared_ptr<Tensor> a, b;
    DivBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) : a(a), b(b) {}

    void release_variables() override { a = nullptr; b = nullptr; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        auto db = std::make_shared<Tensor>(b->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        const float* b_ptr = b->data_ptr();
        float* da_ptr = da->data_ptr();
        float* db_ptr = db->data_ptr();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            float b_val = b_ptr[i];
            da_ptr[i] = g_ptr[i] / b_val;
            db_ptr[i] = -g_ptr[i] * a_ptr[i] / (b_val * b_val);
        }
        return {da, db};
    }
};

inline std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if (a->shape != b->shape) {
        throw std::runtime_error("div: shape mismatch");
    }
    bool req_grad = a->requires_grad || b->requires_grad;

    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    const float* b_ptr = b->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = out->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = a_ptr[i] / b_ptr[i];
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<DivBackward>(a, b);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- POW OPERATION ---
struct PowBackward : public Node {
    std::shared_ptr<Tensor> a;
    int n;
    PowBackward(std::shared_ptr<Tensor> a, int n) : a(a), n(n) {}

    void release_variables() override { a = nullptr; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* da_ptr = da->data_ptr();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = n * std::pow(a_ptr[i], n - 1) * g_ptr[i];
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const int n) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::pow(a_ptr[i], n);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<PowBackward>(a, n);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- SQRT OPERATION ---
struct SqrtBackward : public Node {
    std::shared_ptr<Tensor> a;
    explicit SqrtBackward(std::shared_ptr<Tensor> a) : a(a) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* da_ptr = da->data_ptr();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = (0.5f / std::sqrt(a_ptr[i])) * g_ptr[i];
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = a->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::sqrt(a_ptr[i]);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SqrtBackward>(a);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- LOG OPERATION ---
struct LogBackward : public Node {
    std::shared_ptr<Tensor> a;
    float epsilon;
    LogBackward(std::shared_ptr<Tensor> a, float eps) : a(a), epsilon(eps) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(a->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* da_ptr = da->data_ptr();
        int size = a->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = g_ptr[i] * (1.0f / (a_ptr[i] + epsilon));
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = out->size();

    float epsilon = 1e-8f;
    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::log(a_ptr[i] + epsilon);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<LogBackward>(a, epsilon);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- EXP OPERATION ---
struct ExpBackward : public Node {
    std::shared_ptr<Tensor> out_val;
    explicit ExpBackward(std::shared_ptr<Tensor> out_val) : out_val(out_val) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> grad = grads[0];
        auto da = std::make_shared<Tensor>(out_val->shape, false);
        const float* g_ptr = grad->data_ptr();
        const float* out_ptr = out_val->data_ptr();
        float* da_ptr = da->data_ptr();
        int size = out_val->size();

        for (int i = 0; i < size; i++) {
            da_ptr[i] = g_ptr[i] * out_ptr[i];
        }
        return {da};
    }
};

inline std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();
    int size = out->size();

    for (int i = 0; i < size; i++) {
        out_ptr[i] = std::exp(a_ptr[i]);
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<ExpBackward>(out);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out; 
}

// --- SIMPLE SUM OPERATION ---
struct SimpleSumBackward : public Node {
    std::vector<int> in_shape;
    int in_size;
    SimpleSumBackward(const std::vector<int>& shape, int size) : in_shape(shape), in_size(size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto out_grad = std::make_shared<Tensor>(in_shape, false);
        float g_val = grads[0]->data_ptr()[0];
        float* og_ptr = out_grad->data_ptr();
        for (int i = 0; i < in_size; i++) {
            og_ptr[i] = g_val;
        }
        return {out_grad};
    }
};

inline std::shared_ptr<Tensor> simple_sum(const std::shared_ptr<Tensor>& a) {
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}, req_grad);
    const float* a_ptr = a->data_ptr();
    int size = a->size();

    float total = 0.0f;
    for (int i = 0; i < size; i++) {
        total += a_ptr[i];
    }
    out->data_ptr()[0] = total;

    if (req_grad) {
        auto grad_fn = std::make_shared<SimpleSumBackward>(a->shape, size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
