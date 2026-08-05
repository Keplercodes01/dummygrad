#pragma once 
#include "engine.h"

// --- CROSS ENTROPY BACKWARD NODE ---
struct CrossEntropyBackward : public Node {
    std::shared_ptr<Tensor> pred, target;
    int n;
    CrossEntropyBackward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, int n)
        : pred(pred), target(target), n(n) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_pred = std::make_shared<Tensor>(pred->shape, false);
        const float* p_ptr = pred->data_ptr();
        const float* t_ptr = target->data_ptr();
        float* gp_ptr = grad_pred->data_ptr();
        float g_val = self_grad->data_ptr()[0];

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
    const float* p_ptr = pred->data_ptr();
    const float* t_ptr = target->data_ptr();

    float sum_loss = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_loss -= t_ptr[i] * std::log(p_ptr[i] + 1e-8f);
    }

    bool req_grad = pred->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}, req_grad);
    out->data_ptr()[0] = sum_loss / n;

    if (req_grad) {
        auto grad_fn = std::make_shared<CrossEntropyBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
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
        auto grad_pred = std::make_shared<Tensor>(pred->shape, false);
        const float* p_ptr = pred->data_ptr();
        const float* t_ptr = target->data_ptr();
        float* gp_ptr = grad_pred->data_ptr();
        float g_val = self_grad->data_ptr()[0];

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
    const float* p_ptr = pred->data_ptr();
    const float* t_ptr = target->data_ptr();

    float sq_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = p_ptr[i] - t_ptr[i];
        sq_sum += diff * diff;
    }

    bool req_grad = pred->requires_grad;
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}, req_grad);
    out->data_ptr()[0] = sq_sum / n;

    if (req_grad) {
        auto grad_fn = std::make_shared<MseBackward>(pred, target, n);
        grad_fn->add_next_edge(get_grad_edge(pred).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
