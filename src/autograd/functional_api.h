#pragma once
// Derivatives of functions (as in torch.func / torch.autograd.functional), gradient checkpointing and
// custom differentiable functions.

#include "autograd.h"
#include "functional.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace autograd {

using Function = std::function<TensorPtr(const TensorPtr&)>;

namespace api_detail {
// A fresh leaf with x's value
inline TensorPtr leaf(const TensorPtr& x) {
    auto t = make_contiguous(x)->detach();
    t->requires_grad = true;
    return t;
}
inline TensorPtr basis(const std::vector<int64_t>& shape, int64_t i, Device device) {
    auto e = op_detail::filled(shape, Device::CPU, 0.0f);
    e->data_ptr<float>()[i] = 1.0f;
    return device == Device::CPU ? e : e->to(device)->detach();
}
}  // namespace api_detail

// f(x) and v^T J: the gradient of <f(x), v>. v defaults to 1 for one-element outputs.
inline std::pair<TensorPtr, TensorPtr> vjp(const Function& f, const TensorPtr& x, const TensorPtr& v = nullptr) {
    GradModeGuard record(true);
    auto l = api_detail::leaf(x);
    auto y = f(l);
    auto g = grad({y}, {l}, {v}, false, false, true)[0];
    return {y->detach(), g ? g : zeros_like(x)};
}
// f(x) and J v, by differentiating the (linear) vjp: two backward passes, no Jacobian
inline std::pair<TensorPtr, TensorPtr> jvp(const Function& f, const TensorPtr& x, const TensorPtr& v) {
    GradModeGuard record(true);
    auto l = api_detail::leaf(x);
    auto y = f(l);
    auto u = api_detail::leaf(zeros_like(y->detach()));
    auto g = grad({y}, {l}, {u}, true, true, true)[0];  // u^T J, linear in u
    if (!g || !g->requires_grad) return {y->detach(), zeros_like(y->detach())};
    auto out = grad({g}, {u}, {v}, false, false, true)[0];
    return {y->detach(), out ? out : zeros_like(y->detach())};
}
// [outputs..., inputs...] matrix of partial derivatives, one backward pass per output element
inline TensorPtr jacobian(const Function& f, const TensorPtr& x) {
    GradModeGuard record(true);
    auto l = api_detail::leaf(x);
    auto y = f(l);
    TensorList rows;
    for (int64_t i = 0; i < y->size(); ++i) {
        auto g = grad({y}, {l}, {api_detail::basis(y->shape, i, y->device)}, true, false, true)[0];
        rows.push_back(reshape(g ? g->detach() : zeros_like(x), {x->size()}));
    }
    auto shape = y->shape;
    shape.insert(shape.end(), x->shape.begin(), x->shape.end());
    return reshape(stack(rows, 0), shape);
}
// [inputs..., inputs...] second derivatives of a one-element f
inline TensorPtr hessian(const Function& f, const TensorPtr& x) {
    return jacobian([&f](const TensorPtr& t) -> TensorPtr {
        auto y = f(t);
        auto g = grad({y}, {t}, {}, true, true, true)[0];
        return g ? g : zeros_like(t);
    }, x);
}
// The Hessian of a one-element f times v, without forming it
inline TensorPtr hvp(const Function& f, const TensorPtr& x, const TensorPtr& v) {
    GradModeGuard record(true);
    auto l = api_detail::leaf(x);
    auto g = grad({f(l)}, {l}, {}, true, true, true)[0];
    if (!g || !g->requires_grad) return zeros_like(x);
    auto h = grad({g}, {l}, {v}, false, false, true)[0];
    return h ? h : zeros_like(x);
}

// Runs f without keeping its intermediate activations and runs it again during backward: less memory
// for one more forward pass. Random draws (dropout) are replayed identically; the autocast dtype is too.
// f may read parameters it captures: their gradients accumulate as usual. Not differentiable twice.
inline TensorPtr checkpoint(const std::function<TensorPtr(const TensorList&)>& f, const TensorList& inputs) {
    if (!grad_enabled()) return f(inputs);
    uint64_t seed = random_state().seed, stream = random_state().stream;
    DType autocast = autocast_dtype();
    TensorPtr out;
    {
        NoGradGuard no_grad;
        out = f(inputs)->detach();
    }
    struct CheckpointBackward : public Node {
        std::function<TensorPtr(const TensorList&)> f;
        TensorList inputs;  // detached
        uint64_t seed, stream;
        DType autocast;
        std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
            TensorList leaves;
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto l = inputs[i] ? inputs[i]->detach() : nullptr;
                if (l) l->requires_grad = next_edges[i].function != nullptr;
                leaves.push_back(l);
            }
            TensorPtr y;
            {
                GradModeGuard record(true);
                RandomStateGuard restore;
                random_state().seed = seed;
                random_state().stream = stream;
                AutocastGuard same(autocast);
                y = f(leaves);
            }
            if (y->requires_grad) autograd::backward({y}, {grads[0]});
            TensorList out;
            for (auto& l : leaves) out.push_back(l ? l->grad : nullptr);
            return out;
        }
        void release_variables() override {
            inputs.clear();
            f = nullptr;
        }
        std::string name() const override { return "CheckpointBackward"; }
    };
    auto node = std::make_shared<CheckpointBackward>();
    node->f = f;
    node->seed = seed;
    node->stream = stream;
    node->autocast = autocast;
    for (const auto& t : inputs) {
        node->inputs.push_back(t ? t->detach() : nullptr);
        node->add_next_edge(get_grad_edge(t).function);
    }
    // The output requires grad whenever grad is on: parameters f captures may need it
    out = out->detach();
    out->requires_grad = true;
    out->grad_fn = node;
    return out;
}
inline TensorPtr checkpoint(const Function& f, const TensorPtr& x) {
    return checkpoint([f](const TensorList& in) { return f(in[0]); }, TensorList{x});
}

// A function with a hand-written gradient: forward runs without recording; backward gets the output's
// gradient and the context (saved inputs by index, output()) and returns one gradient per input. Write it
// with differentiable ops and it supports create_graph too.
inline TensorPtr custom_function(const char* name, const std::function<TensorPtr(const TensorList&)>& forward, const TensorList& inputs,
                                 std::initializer_list<int> save, FnBackward::Vjp backward) {
    TensorPtr out;
    {
        NoGradGuard no_grad;
        out = forward(inputs)->detach();
    }
    return make_op(name, out, inputs, save, std::move(backward));
}

}  // namespace autograd
