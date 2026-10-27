// Writing your own op: squared ReLU, y = relu(x)^2, with a hand-written backward pass.
//
// An op is a forward function plus a backward Node:
//   1. Densify inputs with contiguous() if your kernel indexes raw memory.
//   2. Compute the output.
//   3. If needs_grad(input), attach a Node that holds what backward needs and add one graph
//      edge per input, in the same order apply() returns their gradients.
// Then verify the backward with dummygrad::gradcheck before using the op in a model.
//
// Build and run from the repository root:
//   cmake -S . -B build && cmake --build build -j && ./build/custom_op

#include "dummy_core.h"
#include <algorithm>
#include <iostream>

// Backward: dL/dx = dL/dy * 2 * relu(x)
struct SquaredReluBackward : public Node {
    std::shared_ptr<Tensor> x;  // saved forward input

    explicit SquaredReluBackward(std::shared_ptr<Tensor> x) : x(std::move(x)) {}

    // Called after backward unless retain_graph is set: free saved tensors early.
    void release_variables() override { x.reset(); }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !x) return {nullptr};
        auto dy = grads[0]->cpu();  // the engine always passes dense gradients
        auto xs = x->cpu();
        auto dx = std::make_shared<Tensor>(x->shape, Device::CPU, DType::Float32, false);

        const float* dy_p = dy->data_ptr<float>();
        const float* x_p = xs->data_ptr<float>();
        float* dx_p = dx->data_ptr<float>();
        for (int64_t i = 0; i < x->size(); ++i) dx_p[i] = dy_p[i] * 2.0f * std::max(0.0f, x_p[i]);

        return {dx->to(x->device)};  // one gradient per input
    }
};

std::shared_ptr<Tensor> squared_relu(const std::shared_ptr<Tensor>& x_in) {
    DG_PROFILE_OP("squared_relu");  // named entry in profile_report(ProfilePhase::Forward)
    auto x = contiguous(x_in);  // the loop below indexes raw memory
    auto xs = x->cpu();
    auto y = std::make_shared<Tensor>(x->shape, Device::CPU, DType::Float32, false);

    const float* x_p = xs->data_ptr<float>();
    float* y_p = y->data_ptr<float>();
    for (int64_t i = 0; i < x->size(); ++i) {
        float r = std::max(0.0f, x_p[i]);
        y_p[i] = r * r;
    }

    auto out = y->to(x->device);
    out->requires_grad = needs_grad(x);
    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<SquaredReluBackward>(x);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

int main() {
    manual_seed(0);

    // 1. Check the backward pass against finite differences
    std::string err = dummygrad::gradcheck(
        [](const std::vector<std::shared_ptr<Tensor>>& in) { return squared_relu(in[0]); },
        {randn({4, 5})});
    if (!err.empty()) {
        std::cerr << "gradcheck failed: " << err << "\n";
        return 1;
    }
    std::cout << "gradcheck passed for squared_relu\n";

    // 2. Use it like any built-in op, here inside a tiny network
    Linear layer(3, 2);
    auto x = randn({4, 3}, Device::CPU, DType::Float32, /*requires_grad=*/false);
    auto loss = mean(mean(squared_relu(layer.forward(x)), 1), 0);
    loss->backward();
    std::cout << "loss " << loss->data_ptr<float>()[0]
              << ", gradient reached the layer weights: " << (layer.W->grad ? "yes" : "no") << "\n";
    return layer.W->grad ? 0 : 1;
}
