#include "node.h"
#include "tensor.h"
#include "functional.h"
#include "ops.h"

void Node::add_next_edge(const std::shared_ptr<Node>& fn, size_t input_slot) {
    next_edges.push_back({fn, input_slot});
}

std::vector<std::shared_ptr<Tensor>> AccumulateGrad::apply(const std::vector<std::shared_ptr<Tensor>>& grads) {
    if (auto var = variable.lock()) {
        if (!grads.empty() && grads[0]) {
            tensor_add_inplace(var->grad, grads[0]);
        }
    }
    return {};
}
