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

std::vector<std::shared_ptr<Tensor>> AccumulateGrad::apply(const std::vector<std::shared_ptr<Tensor>>& grads) {
    if (auto var = variable.lock()) {
        if (!grads.empty() && grads[0]) {
            std::lock_guard<std::mutex> lock(mutex);
            tensor_add_inplace(var->grad, grads[0]);
        }
    }
    return {};
}
