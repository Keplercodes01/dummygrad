#pragma once 
#include "engine.h" 

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
