#pragma once
// Gradient checking for custom ops and layers.
//
//   auto err = dummygrad::gradcheck(
//       [](const std::vector<TensorPtr>& x) { return my_op(x[0], x[1]); },
//       {randn({4, 8}), randn({8, 3})});
//   if (!err.empty()) std::cerr << err << "\n";   // e.g. "input 1 element 5: analytic 0.12 vs numeric 0.48"
//
// Compares the gradients autograd computes against central finite differences of the forward
// pass. Run it on CPU inputs: finite differences need full float32 precision, which TF32
// tensor-core matmuls on the GPU don't provide. Inputs near a kink (relu at 0, abs at 0, max
// ties) can fail spuriously; keep them away from such points.

#include "tensor.h"
#include "functional.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dummygrad {

using TensorPtr = std::shared_ptr<Tensor>;
using GradcheckFn = std::function<TensorPtr(const std::vector<TensorPtr>&)>;

struct GradcheckOptions {
    float eps = 1e-2f;        // finite-difference step
    float atol = 5e-3f;       // absolute tolerance
    float rtol = 2e-2f;       // tolerance relative to the larger of the two gradients
    int max_checks = 24;      // elements sampled per input (all of them for smaller inputs)
    unsigned seed = 0;        // controls the output weighting and the sampled elements
};

namespace detail {

inline std::vector<float> host_values(const TensorPtr& t) {
    auto c = make_contiguous(t);
    auto h = (c->device == Device::CPU) ? c : c->to(Device::CPU);
    const float* p = h->data_ptr<float>();
    return std::vector<float>(p, p + h->size());
}

inline TensorPtr make_input(const std::vector<float>& values, const std::vector<int64_t>& shape,
                            Device device, bool requires_grad) {
    auto t = std::make_shared<Tensor>(shape, Device::CPU, DType::Float32, false);
    t->copy_from_vector(values);
    if (device != Device::CPU) t = t->to(device);
    t->requires_grad = requires_grad;
    return t;
}

// Non-scalar outputs are reduced with fixed random weights so that every output element
// contributes a distinct amount; a plain sum would hide permutation bugs.
inline TensorPtr weighted_sum(const TensorPtr& out, const std::vector<float>& weights) {
    if (out->size() == 1) return out;
    auto w = make_input(weights, out->shape, out->device, false);
    auto prod = mul(out, w);
    return sum(reshape(prod, {1, prod->size()}), 1);
}

inline float scalar_value(const TensorPtr& t) { return host_values(t)[0]; }

}  // namespace detail

// Returns an empty string when the analytic gradient of every input matches finite differences,
// otherwise a description of the first mismatch. The inputs' values are used as the evaluation
// point; their requires_grad flags are ignored (every input is checked).
inline std::string gradcheck(const GradcheckFn& fn, const std::vector<TensorPtr>& inputs,
                             const GradcheckOptions& opts = {}) {
    if (inputs.empty()) return "gradcheck: no inputs";
    for (const auto& t : inputs) {
        if (!t) return "gradcheck: null input";
        if (t->dtype != DType::Float32) return "gradcheck: inputs must be Float32";
    }
    Device device = inputs[0]->device;
    std::vector<std::vector<float>> values;
    std::vector<std::vector<int64_t>> shapes;
    for (const auto& t : inputs) {
        values.push_back(detail::host_values(t));
        shapes.push_back(t->shape);
    }

    auto build = [&](const std::vector<std::vector<float>>& vals, bool requires_grad) {
        std::vector<TensorPtr> leaves;
        for (size_t i = 0; i < vals.size(); ++i) leaves.push_back(detail::make_input(vals[i], shapes[i], device, requires_grad));
        return leaves;
    };

    std::mt19937 rng(opts.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> weights;
    {
        auto out = fn(build(values, false));
        if (!out) return "gradcheck: function returned null";
        weights.resize(out->size());
        for (auto& w : weights) w = dist(rng);
    }
    auto loss_at = [&](const std::vector<std::vector<float>>& vals) {
        return detail::scalar_value(detail::weighted_sum(fn(build(vals, false)), weights));
    };

    // Analytic gradients
    auto leaves = build(values, true);
    detail::weighted_sum(fn(leaves), weights)->backward();

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<float> analytic = leaves[i]->grad ? detail::host_values(leaves[i]->grad)
                                                      : std::vector<float>(values[i].size(), 0.0f);
        int64_t n = (int64_t)values[i].size();
        std::vector<int64_t> idx(n);
        for (int64_t k = 0; k < n; ++k) idx[k] = k;
        if (n > opts.max_checks) {
            std::mt19937 pick(opts.seed + 77 + (unsigned)i);
            std::shuffle(idx.begin(), idx.end(), pick);
            idx.resize(opts.max_checks);
        }

        float max_numeric = 0.0f;
        for (int64_t k : idx) {
            auto plus = values, minus = values;
            plus[i][k] += opts.eps;
            minus[i][k] -= opts.eps;
            float numeric = (loss_at(plus) - loss_at(minus)) / (2.0f * opts.eps);
            float a = analytic[k];
            max_numeric = std::max(max_numeric, std::fabs(numeric));
            float tol = opts.atol + opts.rtol * std::max(std::fabs(numeric), std::fabs(a));
            if (!std::isfinite(a) || std::fabs(numeric - a) > tol) {
                std::ostringstream os;
                os << "input " << i << " element " << k << ": analytic " << a << " vs numeric " << numeric;
                if (!leaves[i]->grad) os << " (no gradient reached this input)";
                return os.str();
            }
        }
        if (!leaves[i]->grad && max_numeric > 1e-3f) return "input " + std::to_string(i) + " received no gradient";
    }
    return "";
}

}  // namespace dummygrad
