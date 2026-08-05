#pragma once 
#include "engine.h"

// --- VIEW / RESHAPE BACKWARD NODE ---
struct ViewBackward : public Node {
    std::vector<int> in_shape;
    explicit ViewBackward(const std::vector<int>& in_shape) : in_shape(in_shape) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        // Reshape incoming gradient back to input shape
        return {grads[0]->_reshape(in_shape)};
    }
};

// Fast zero-copy functional view (throws error if non-contiguous)
inline std::shared_ptr<Tensor> view(const std::shared_ptr<Tensor>& a, const std::vector<int>& new_shape) {
    auto out = a->_view(new_shape);

    if (a->requires_grad) {
        auto grad_fn = std::make_shared<ViewBackward>(a->shape);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// General functional reshape (handles any layout)
inline std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& a, const std::vector<int>& new_shape) {
    auto out = a->_reshape(new_shape);

    if (a->requires_grad) {
        auto grad_fn = std::make_shared<ViewBackward>(a->shape);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
