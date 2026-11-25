#pragma once
// Building differentiable ops out of a forward value and a vector-Jacobian product.
//
//   auto out = /* compute, without a graph */;
//   return make_op("sin", out, {x}, {/*save input*/ 0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
//       return {mul(g, cos(c.input(0)))};
//   });
//
// The vjp gets the gradient of the output and returns one gradient per input (null for those that
// need none: c.needs(i)). It is written with differentiable ops. An ordinary backward runs it without
// recording; under create_graph it runs with recording on, and c.input() / c.output() hand out tensors
// connected to their history, so the op supports higher-order gradients with no further code.
//
// Saved tensors remember their storage's version: using one that was modified in place since
// (an optimizer step, fill_) throws instead of computing a wrong gradient.

#include "node.h"
#include "tensor.h"

#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using TensorPtr = std::shared_ptr<Tensor>;
using TensorList = std::vector<std::shared_ptr<Tensor>>;

// A saved input of node n, connected to the history its edge i leads to (create_graph)
inline TensorPtr graph_input(const Node& n, size_t i, const TensorPtr& saved) {
    if (!saved || i >= n.next_edges.size() || !n.next_edges[i].function) return saved;
    const auto& fn = n.next_edges[i].function;
    if (auto* leaf = dynamic_cast<AccumulateGrad*>(fn.get())) {
        if (auto var = leaf->variable.lock()) return var;
    }
    auto t = std::make_shared<Tensor>(saved->storage, saved->shape, saved->strides, saved->global_offset, saved->device,
                                      saved->dtype, true);
    t->grad_fn = fn;
    return t;
}

// A saved output of node n, connected to n itself (create_graph)
inline TensorPtr graph_output(Node& n, const TensorPtr& saved) {
    if (!saved) return saved;
    auto t = std::make_shared<Tensor>(saved->storage, saved->shape, saved->strides, saved->global_offset, saved->device,
                                      saved->dtype, true);
    t->grad_fn = n.shared_from_this();
    return t;
}

class FnBackward;

// What a vjp can ask for
class OpContext {
public:
    OpContext(FnBackward& node, bool graph) : node_(node), graph_(graph) {}
    bool needs(size_t i) const;       // whether input i wants a gradient
    TensorPtr input(size_t i) const;  // a saved input
    TensorPtr output() const;         // the saved output
    bool graph() const { return graph_; }  // whether the gradient is itself being recorded (create_graph)

private:
    FnBackward& node_;
    bool graph_;
};

class FnBackward : public Node {
public:
    using Vjp = std::function<TensorList(const OpContext&, const TensorPtr&)>;

    FnBackward(const char* label, Vjp vjp) : label_(label), vjp_(std::move(vjp)) {}

    std::string name() const override { return std::string(label_) + "Backward"; }

    void release_variables() override {
        saved_.clear();
        vjp_ = nullptr;  // drops tensors the closure holds
    }

    TensorList apply(const TensorList& grads) override { return run(grads, false); }
    TensorList apply_graph(const TensorList& grads) override { return run(grads, true); }

    void save(int index, const TensorPtr& t) { saved_.push_back({index, t->detach(), t->storage->version}); }

    TensorPtr saved(int index, bool graph) {
        for (auto& s : saved_) {
            if (s.index != index) continue;
            if (s.tensor->storage->version != s.version) {
                throw std::runtime_error(name() + ": a tensor needed for backward was modified in place after the forward pass");
            }
            if (!graph) return s.tensor;
            return index == kOutput ? graph_output(*this, s.tensor) : graph_input(*this, (size_t)index, s.tensor);
        }
        throw std::logic_error(name() + ": a tensor its vjp uses was not saved");
    }

    static constexpr int kOutput = -1;

private:
    struct Saved {
        int index;
        TensorPtr tensor;
        uint64_t version;
    };

    TensorList run(const TensorList& grads, bool graph) {
        if (grads.empty() || !grads[0]) return TensorList(next_edges.size());
        return vjp_(OpContext(*this, graph), grads[0]);
    }

    const char* label_;
    Vjp vjp_;
    std::vector<Saved> saved_;
};

inline bool OpContext::needs(size_t i) const { return i < node_.next_edges.size() && node_.next_edges[i].function != nullptr; }
inline TensorPtr OpContext::input(size_t i) const { return node_.saved((int)i, graph_); }
inline TensorPtr OpContext::output() const { return node_.saved(FnBackward::kOutput, graph_); }

// Saving the output: make_op(..., {0, kSaveOutput}, ...)
constexpr int kSaveOutput = FnBackward::kOutput;

// Connects out, a fresh tensor, to its inputs. save lists the inputs the vjp reads (and kSaveOutput if
// it reads the output). Without any input that requires grad it only returns out.
inline TensorPtr make_op(const char* name, TensorPtr out, const TensorList& inputs, std::initializer_list<int> save,
                         FnBackward::Vjp vjp) {
    bool any = false;
    if (grad_enabled()) {
        for (const auto& t : inputs) any = any || (t && t->requires_grad);
    }
    out->requires_grad = any;
    if (!any) return out;
    auto node = std::make_shared<FnBackward>(name, std::move(vjp));
    for (const auto& t : inputs) node->add_next_edge(get_grad_edge(t).function);
    for (int i : save) node->save(i, i == kSaveOutput ? out : inputs[(size_t)i]);
    out->grad_fn = node;
    return out;
}
