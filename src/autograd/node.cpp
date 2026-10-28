#include "node.h"
#include "tensor.h"
#include "functional.h"
#include "ops.h"
#include <cstdlib>
#include <typeinfo>
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

std::string Node::name() const {
    const char* mangled = typeid(*this).name();
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string result(demangled);
        std::free(demangled);
        return result;
    }
#endif
    return mangled;
}

void Node::add_next_edge(const std::shared_ptr<Node>& fn, size_t input_slot) {
    next_edges.push_back({fn, input_slot});
}

// The first gradient a parameter receives is adopted rather than copied when nothing else can
// see it (grad_is_exclusive). Adopting saves a device copy per parameter per step.
std::vector<std::shared_ptr<Tensor>> AccumulateGrad::apply(const std::vector<std::shared_ptr<Tensor>>& grads) {
    if (auto var = variable.lock()) {
        if (!grads.empty() && grads[0]) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!var->grad && grad_is_exclusive(grads[0])) {
                var->grad = grads[0];
                var->grad->requires_grad = false;
            } else {
                tensor_add_inplace(var->grad, grads[0]);
            }
        }
    }
    return {};
}
