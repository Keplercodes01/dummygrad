// Tests of the op library, autograd API, layers, optimizers, linear algebra, transforms and distributions.
//
//   1. Every differentiable op: gradients against finite differences, second derivatives (create_graph)
//      against finite differences of the gradient, and, with a GPU, CUDA against the CPU.
//   2. Values: against brute-force definitions and closed forms.
//
// Exit code is 0 only if every test passes. An argument runs only tests whose name contains it.

#include "dummy_core.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Fn = std::function<TensorPtr(const TensorList&, Device)>;

static std::string g_filter;
static int g_passed = 0;
static std::vector<std::string> g_failures;
static std::mt19937 g_rng(7);

static void run(const std::string& name, const std::function<std::string()>& body) {
    if (!g_filter.empty() && name.find(g_filter) == std::string::npos) return;
    std::string err;
    try {
        err = body();
    } catch (const std::exception& e) {
        err = std::string("threw: ") + e.what();
    }
    if (err.empty()) {
        ++g_passed;
    } else {
        g_failures.push_back(name + ": " + err);
        std::cout << "[FAIL] " << name << ": " << err << std::endl;
    }
}

static bool has_cuda() {
#ifdef USE_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

// Values uniform on [lo, hi], at least min_abs away from 0 and from each other's kinks
static TensorPtr input(const std::vector<int64_t>& shape, float lo = -1.0f, float hi = 1.0f, float min_abs = 0.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> v((size_t)n);
    for (auto& x : v) do { x = dist(g_rng); } while (std::fabs(x) < min_abs);
    return tensor(v, shape);
}
// Shuffled, evenly spaced values, so maxima and orderings survive a finite-difference step
static TensorPtr distinct(const std::vector<int64_t>& shape, float gap = 0.15f) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = gap * (float)(i - n / 2) + 0.05f;
    std::shuffle(v.begin(), v.end(), g_rng);
    return tensor(v, shape);
}

static std::string near(const std::vector<float>& a, const std::vector<float>& b, float atol, float rtol = 0.0f) {
    if (a.size() != b.size()) return "sizes " + std::to_string(a.size()) + " and " + std::to_string(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) && std::isnan(b[i])) continue;
        if (a[i] == b[i]) continue;
        if (!(std::fabs(a[i] - b[i]) <= atol + rtol * std::max(std::fabs(a[i]), std::fabs(b[i])))) {
            std::ostringstream os;
            os << "element " << i << ": got " << a[i] << ", expected " << b[i];
            return os.str();
        }
    }
    return "";
}
static std::string near(const TensorPtr& a, const std::vector<float>& b, float atol = 1e-5f, float rtol = 1e-5f) { return near(a->to_vector(), b, atol, rtol); }
static std::string near(const TensorPtr& a, const TensorPtr& b, float atol = 1e-5f, float rtol = 1e-5f) {
    if (a->shape != b->shape) return "shapes " + shape_string(a->shape) + " and " + shape_string(b->shape);
    return near(a->to_vector(), b->to_vector(), atol, rtol);
}
static std::string scalar(const TensorPtr& a, float b, float atol = 1e-5f, float rtol = 1e-5f) { return near(std::vector<float>{a->item()}, {b}, atol, rtol); }
static std::string shape_is(const TensorPtr& a, const std::vector<int64_t>& s) { return a->shape == s ? "" : "shape " + shape_string(a->shape) + ", expected " + shape_string(s); }
static std::string expect_throw(const std::function<void()>& f, const std::string& what) {
    try {
        f();
    } catch (const std::exception&) {
        return "";
    }
    return what + " did not throw";
}
#define CHECK(expr)                             \
    do {                                        \
        std::string err_ = (expr);              \
        if (!err_.empty()) return #expr ": " + err_; \
    } while (0)

static TensorPtr leaf_of(const TensorPtr& t) {
    if (t->requires_grad) return t;
    auto l = t->detach();
    l->requires_grad = true;
    return l;
}

struct OpTest {
    std::string name;
    TensorList inputs;
    Fn fn;
    bool second_order = true;
    float eps = 1e-2f;
    float cuda_rtol = 2e-3f;
};
static std::vector<OpTest> g_ops;
static OpTest& op(const std::string& name, TensorList inputs, Fn fn) {
    g_ops.push_back({name, std::move(inputs), std::move(fn)});
    return g_ops.back();
}

static std::string first_order(const OpTest& t) {
    dummygrad::GradcheckOptions o;
    o.eps = t.eps;
    return dummygrad::gradcheck([&](const TensorList& x) { return t.fn(x, Device::CPU); }, t.inputs, o);
}
// Gradient-checks x -> weighted sum of f's gradients, which create_graph makes differentiable
static std::string second_order(const OpTest& t) {
    dummygrad::GradcheckOptions o;
    o.eps = t.eps;
    o.atol = 1.5e-2f;
    o.rtol = 4e-2f;
    o.max_checks = 8;
    auto grads = [&](const TensorList& x_in) -> TensorPtr {
        TensorList x;
        for (const auto& i : x_in) x.push_back(leaf_of(i));
        auto y = t.fn(x, Device::CPU);
        std::vector<float> w((size_t)y->size());
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.3f + 0.17f * (float)(i % 5);
        auto g = autograd::grad({sum(mul(y, tensor(w, y->shape)))}, x, {}, true, true, true);
        TensorPtr total;
        for (size_t k = 0; k < g.size(); ++k) {
            if (!g[k]) continue;
            std::vector<float> v((size_t)g[k]->size());
            for (size_t i = 0; i < v.size(); ++i) v[i] = 0.5f - 0.21f * (float)((i + k) % 4);
            auto part = sum(mul(g[k], tensor(v, g[k]->shape)));
            total = total ? add(total, part) : part;
        }
        // Keeps the result connected to every input even where the gradient is constant
        for (const auto& i : x) total = add(total, mul_scalar(sum(i), 0.1f));
        return total;
    };
    return dummygrad::gradcheck(grads, t.inputs, o);
}
static std::string cuda_parity(const OpTest& t) {
    auto eval = [&](Device dev, std::vector<std::vector<float>>& out) {
        TensorList x;
        for (const auto& i : t.inputs) {
            auto l = make_contiguous(i)->to(dev)->detach();
            if (dev == Device::CPU) l = i->clone()->detach();
            l->requires_grad = true;
            x.push_back(l);
        }
        auto y = t.fn(x, dev);
        out.push_back(y->to_vector());
        std::vector<float> w((size_t)y->size());
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.4f - 0.13f * (float)(i % 7);
        autograd::backward({y}, {tensor(w, y->shape, dev)});
        for (const auto& l : x) out.push_back(l->grad ? l->grad->to_vector() : std::vector<float>());
    };
    std::vector<std::vector<float>> cpu, gpu;
    eval(Device::CPU, cpu);
    eval(Device::CUDA, gpu);
    for (size_t i = 0; i < cpu.size(); ++i) {
        auto err = near(gpu[i], cpu[i], 1e-4f, t.cuda_rtol);
        if (!err.empty()) return (i == 0 ? "output " : "gradient " + std::to_string(i - 1) + " ") + err;
    }
    return "";
}

// ---------------------------------------------------------------------------------------------
// Differentiable ops
// ---------------------------------------------------------------------------------------------

static void unary_op_test(const std::string& name, TensorPtr (*f)(const TensorPtr&), float lo = -1.0f, float hi = 1.0f, float min_abs = 0.0f, bool second = true) {
    op(name, {input({2, 5}, lo, hi, min_abs)}, [f](const TensorList& x, Device) { return f(x[0]); }).second_order = second;
}

static void build_ops() {
    // Functions of one tensor
    unary_op_test("exp", exp);
    unary_op_test("expm1", expm1);
    unary_op_test("log", log, 0.5f, 2.0f);
    unary_op_test("log2", log2, 0.5f, 2.0f);
    unary_op_test("log10", log10, 0.5f, 2.0f);
    unary_op_test("log1p", log1p, -0.4f, 1.0f);
    unary_op_test("sqrt", sqrt, 0.5f, 2.0f);
    unary_op_test("rsqrt", rsqrt, 0.6f, 2.0f);
    unary_op_test("square", square);
    unary_op_test("reciprocal", reciprocal, 0.6f, 2.0f);
    unary_op_test("abs", abs, -1.0f, 1.0f, 0.1f);
    unary_op_test("sin", sin);
    unary_op_test("cos", cos);
    unary_op_test("tan", tan, -0.8f, 0.8f);
    unary_op_test("asin", asin, -0.7f, 0.7f);
    unary_op_test("acos", acos, -0.7f, 0.7f);
    unary_op_test("atan", atan);
    unary_op_test("sinh", sinh);
    unary_op_test("cosh", cosh);
    unary_op_test("asinh", asinh);
    unary_op_test("acosh", acosh, 1.5f, 2.5f);
    unary_op_test("atanh", atanh, -0.6f, 0.6f);
    unary_op_test("erf", erf);
    unary_op_test("erfc", erfc);
    unary_op_test("erfinv", erfinv, -0.6f, 0.6f);
    unary_op_test("lgamma", lgamma, 1.0f, 3.0f);
    unary_op_test("digamma", digamma, 1.2f, 3.0f, 0.0f, false);
    unary_op_test("i0", i0);
    unary_op_test("i1", i1, 0.3f, 1.5f);
    unary_op_test("sinc", sinc, 0.3f, 1.0f);
    unary_op_test("relu", relu, -1.0f, 1.0f, 0.1f);
    unary_op_test("tanh", tanh);
    unary_op_test("sigmoid", sigmoid);
    unary_op_test("gelu", gelu);
    unary_op_test("gelu_erf", gelu_erf);
    unary_op_test("silu", silu);
    unary_op_test("mish", mish);
    unary_op_test("selu", selu, -1.0f, 1.0f, 0.1f);
    unary_op_test("hardswish", hardswish, -2.0f, 2.0f);
    unary_op_test("hardsigmoid", hardsigmoid, -2.0f, 2.0f);
    unary_op_test("softsign", softsign, -1.0f, 1.0f, 0.1f);
    unary_op_test("log_sigmoid", log_sigmoid, -2.0f, 2.0f);
    unary_op_test("tanhshrink", tanhshrink);
    unary_op_test("relu6", relu6, -1.0f, 2.0f, 0.1f);
    op("leaky_relu", {input({6}, -1, 1, 0.1f)}, [](const TensorList& x, Device) { return leaky_relu(x[0], 0.2f); });
    op("elu", {input({6}, -1, 1, 0.1f)}, [](const TensorList& x, Device) { return elu(x[0], 1.3f); });
    op("celu", {input({6}, -1, 1, 0.1f)}, [](const TensorList& x, Device) { return celu(x[0], 0.7f); });
    op("softplus", {input({6}, -1, 1)}, [](const TensorList& x, Device) { return softplus(x[0], 2.0f); });
    op("softshrink", {input({6}, 0.6f, 1.0f)}, [](const TensorList& x, Device) { return softshrink(x[0]); });
    op("hardshrink", {input({6}, 0.6f, 1.0f)}, [](const TensorList& x, Device) { return hardshrink(x[0]); });
    op("clamp", {input({8}, -1, 1, 0.05f)}, [](const TensorList& x, Device) { return clamp(x[0], -0.52f, 0.03f); });
    op("threshold", {input({6}, 0.2f, 1.0f)}, [](const TensorList& x, Device) { return threshold(x[0], 0.1f, 7.0f); });
    op("pow_scalar", {input({6}, 0.5f, 1.5f)}, [](const TensorList& x, Device) { return pow(x[0], 2.5f); });
    op("pow_cube_negative", {input({6}, -1.5f, -0.5f)}, [](const TensorList& x, Device) { return pow(x[0], 3.0f); });
    op("scalar_pow", {input({6})}, [](const TensorList& x, Device) { return pow(2.0f, x[0]); });
    op("prelu", {input({2, 3, 4}, -1, 1, 0.1f), input({3}, 0.1f, 0.4f)}, [](const TensorList& x, Device) { return prelu(x[0], x[1]); });
    op("logit", {input({6}, 0.2f, 0.8f)}, [](const TensorList& x, Device) { return logit(x[0]); });
    op("scalar_chain", {input({6}, 0.5f, 1.5f)}, [](const TensorList& x, Device) {
        return add(rsub_scalar(2.0f, mul_scalar(x[0], 1.5f)), rdiv_scalar(1.0f, div_scalar(add_scalar(x[0], 0.5f), 2.0f)));
    });

    // Two tensors, with both broadcasting
    op("add_both", {input({3, 1}), input({1, 4})}, [](const TensorList& x, Device) { return add(x[0], x[1]); });
    op("sub_both", {input({3, 1}), input({2, 1, 4})}, [](const TensorList& x, Device) { return sub(x[0], x[1]); });
    op("mul_both", {input({3, 1}), input({1, 4})}, [](const TensorList& x, Device) { return mul(x[0], x[1]); });
    op("div_both", {input({4}), input({3, 1}, 0.6f, 1.5f)}, [](const TensorList& x, Device) { return div(x[0], x[1]); });
    op("pow", {input({3, 1}, 0.6f, 1.5f), input({4}, 0.5f, 2.0f)}, [](const TensorList& x, Device) { return pow(x[0], x[1]); });
    op("maximum", {distinct({2, 4}), distinct({2, 4}, 0.31f)}, [](const TensorList& x, Device) { return maximum(x[0], x[1]); });
    op("minimum", {distinct({2, 4}), distinct({4}, 0.37f)}, [](const TensorList& x, Device) { return minimum(x[0], x[1]); });
    op("atan2", {input({4}, 0.4f, 1.0f), input({4}, 0.4f, 1.0f)}, [](const TensorList& x, Device) { return atan2(x[0], x[1]); });
    op("hypot", {input({4}, 0.4f, 1.0f), input({4}, 0.4f, 1.0f)}, [](const TensorList& x, Device) { return hypot(x[0], x[1]); });
    op("copysign", {input({4}, 0.4f, 1.0f), input({4}, -1.0f, 1.0f, 0.1f)}, [](const TensorList& x, Device) { return copysign(x[0], x[1]); });
    op("where", {input({2, 3}), input({3})}, [](const TensorList& x, Device d) { return where(tensor({1, 0, 1, 0, 0, 1}, {2, 3}, d), x[0], x[1]); });
    op("lerp", {input({4}), input({4}), input({4}, 0.1f, 0.9f)}, [](const TensorList& x, Device) { return lerp(x[0], x[1], x[2]); });
    op("xlogy", {input({4}, 0.3f, 1.0f), input({4}, 0.5f, 1.5f)}, [](const TensorList& x, Device) { return xlogy(x[0], x[1]); });
    op("logaddexp", {distinct({4}), distinct({4}, 0.4f)}, [](const TensorList& x, Device) { return logaddexp(x[0], x[1]); });
    op("addcdiv", {input({4}), input({4}), input({4}, 0.6f, 1.5f)}, [](const TensorList& x, Device) { return addcdiv(x[0], x[1], x[2], 0.5f); });

    // Views and joining
    op("expand", {input({3, 1})}, [](const TensorList& x, Device) { return expand(x[0], {2, 3, 4}); });
    op("permute", {input({2, 3, 4})}, [](const TensorList& x, Device) { return permute(x[0], {2, 0, 1}); });
    op("permute_then_op", {input({2, 3, 4})}, [](const TensorList& x, Device) { return tanh(permute(x[0], {1, 2, 0})); });
    op("unsqueeze_squeeze", {input({2, 3})}, [](const TensorList& x, Device) { return squeeze(unsqueeze(transpose(x[0], 0, 1), 1), 1); });
    op("flatten", {input({2, 3, 2})}, [](const TensorList& x, Device) { return flatten(permute(x[0], {2, 0, 1}), 1); });
    op("slice_step", {input({7, 2})}, [](const TensorList& x, Device) { return slice_step(x[0], 0, 1, 7, 2); });
    op("slice", {input({5, 2})}, [](const TensorList& x, Device) { return slice(x[0], 0, 1, 4); });
    op("select", {input({3, 4})}, [](const TensorList& x, Device) { return mul(select(x[0], 1, 2), select(x[0], 1, 0)); });
    op("diagonal", {input({3, 4})}, [](const TensorList& x, Device) { return diagonal(x[0], 1); });
    op("diag_embed", {input({2, 3})}, [](const TensorList& x, Device) { return diag_embed(x[0]); });
    op("cat", {input({2, 3}), input({2, 1}), input({2, 2})}, [](const TensorList& x, Device) { return cat({x[0], x[1], x[2]}, 1); });
    op("cat_strided", {input({3, 2}), input({2, 3})}, [](const TensorList& x, Device) { return cat({transpose(x[0], 0, 1), x[1]}, 0); });
    op("concat", {input({2, 3}), input({2, 3})}, [](const TensorList& x, Device) { return concat(x[0], x[1], 1); });
    op("stack", {input({3}), input({3})}, [](const TensorList& x, Device) { return stack({x[0], x[1]}, 1); });
    op("split", {input({5, 2})}, [](const TensorList& x, Device) { auto p = split(x[0], 2); return add(p[0], mul(p[1], p[0])); });
    op("unbind", {input({3, 2})}, [](const TensorList& x, Device) { auto p = unbind(x[0], 0); return mul(p[0], add(p[1], p[2])); });
    op("repeat", {input({2, 3})}, [](const TensorList& x, Device) { return repeat(x[0], {2, 1, 2}); });
    op("repeat_interleave", {input({2, 3})}, [](const TensorList& x, Device) { return repeat_interleave(x[0], 2, 1); });
    op("flip", {input({2, 3})}, [](const TensorList& x, Device) { return flip(x[0], {0, 1}); });
    op("roll", {input({5})}, [](const TensorList& x, Device) { return roll(x[0], 2, 0); });
    op("pad_constant", {input({2, 3})}, [](const TensorList& x, Device) { return pad(x[0], {1, 2, 1, 0}, PadMode::Constant, 0.5f); });
    op("pad_negative", {input({2, 5})}, [](const TensorList& x, Device) { return pad(x[0], {-1, 2}); });
    op("pad_reflect", {input({1, 4})}, [](const TensorList& x, Device) { return pad(x[0], {2, 3}, PadMode::Reflect); });
    op("pad_replicate", {input({1, 4})}, [](const TensorList& x, Device) { return pad(x[0], {2, 1}, PadMode::Replicate); });
    op("pad_circular", {input({1, 4})}, [](const TensorList& x, Device) { return pad(x[0], {2, 1}, PadMode::Circular); });
    op("tril", {input({3, 4})}, [](const TensorList& x, Device) { return tril(x[0], 1); });
    op("triu", {input({3, 4})}, [](const TensorList& x, Device) { return triu(x[0], -1); });

    // Reductions
    op("sum_all", {input({2, 3})}, [](const TensorList& x, Device) { return sum(square(x[0])); });
    op("sum_dims", {input({2, 3, 4})}, [](const TensorList& x, Device) { return sum(x[0], {0, 2}, false); });
    op("sum_first", {input({4, 3, 2})}, [](const TensorList& x, Device) { return sum(x[0], 0, true); });
    op("sum_strided", {input({4, 3})}, [](const TensorList& x, Device) { return sum(transpose(x[0], 0, 1), 0, false); });
    op("mean_dims", {input({2, 3, 4})}, [](const TensorList& x, Device) { return mean(x[0], std::vector<int>{1}, false); });
    op("var", {input({3, 4})}, [](const TensorList& x, Device) { return var(x[0], 0, true, 1); });
    op("std", {input({3, 4})}, [](const TensorList& x, Device) { return std_dev(x[0], 1); });
    op("amax", {distinct({3, 4})}, [](const TensorList& x, Device) { return amax(x[0], 1); });
    op("amin_dims", {distinct({2, 3, 2})}, [](const TensorList& x, Device) { return amin(x[0], {0, 2}, false); });
    op("max_dim", {distinct({3, 4})}, [](const TensorList& x, Device) { return max(x[0], 0).first; });
    op("prod", {input({3, 4}, 0.5f, 1.5f)}, [](const TensorList& x, Device) { return prod(x[0], 1); });
    op("cumsum", {input({3, 4})}, [](const TensorList& x, Device) { return cumsum(x[0], 1); });
    op("cumprod", {input({3, 4}, 0.5f, 1.5f)}, [](const TensorList& x, Device) { return cumprod(x[0], 0); });
    op("cummax", {distinct({3, 4})}, [](const TensorList& x, Device) { return cummax(x[0], 1).first; });
    op("logcumsumexp", {input({2, 4})}, [](const TensorList& x, Device) { return logcumsumexp(x[0], 1); });
    op("logsumexp", {input({3, 4})}, [](const TensorList& x, Device) { return logsumexp(x[0], 1); });
    op("softmax_dim0", {input({3, 4})}, [](const TensorList& x, Device) { return softmax(x[0], 0); });
    op("softmax_last", {input({3, 4})}, [](const TensorList& x, Device) { return softmax(x[0]); });
    op("log_softmax", {input({3, 4})}, [](const TensorList& x, Device) { return log_softmax(x[0], 1); });
    op("norm2", {input({3, 4}, 0.3f, 1.0f)}, [](const TensorList& x, Device) { return norm(x[0], 2.0f, 1); });
    op("norm1", {input({3, 4}, -1, 1, 0.1f)}, [](const TensorList& x, Device) { return norm(x[0], 1.0f, 0); });
    op("norm3", {input({3, 4}, 0.3f, 1.0f)}, [](const TensorList& x, Device) { return norm(x[0], 3.0f); });
    op("norm_inf", {distinct({3, 4})}, [](const TensorList& x, Device) { return norm(x[0], INFINITY, 1); });
    op("normalize", {input({3, 4}, 0.3f, 1.0f)}, [](const TensorList& x, Device) { return normalize(x[0]); });
    op("cosine_similarity", {input({3, 4}, 0.3f, 1.0f), input({3, 4}, 0.3f, 1.0f)}, [](const TensorList& x, Device) { return cosine_similarity(x[0], x[1]); });

    // Indexing
    op("gather", {input({3, 4})}, [](const TensorList& x, Device d) { return gather(x[0], 1, tensor({0, 3, 2, 2, 1, 1}, {3, 2}, d)); });
    op("gather_smaller", {input({3, 4})}, [](const TensorList& x, Device d) { return gather(x[0], 0, tensor({2, 0, 1, 1}, {2, 2}, d)); });
    op("scatter_add", {input({3, 4}), input({3, 2})}, [](const TensorList& x, Device d) { return scatter_add(x[0], 1, tensor({0, 0, 2, 3, 1, 1}, {3, 2}, d), x[1]); });
    op("scatter", {input({3, 4}), input({3, 2})}, [](const TensorList& x, Device d) { return scatter(x[0], 1, tensor({0, 1, 2, 3, 1, 0}, {3, 2}, d), x[1]); });
    op("scatter_mean", {input({3}), input({4})}, [](const TensorList& x, Device d) { return scatter_reduce(x[0], 0, tensor({0, 0, 2, 1}, {4}, d), x[1], ScatterReduce::Mean); });
    op("scatter_amax", {distinct({3}), distinct({4}, 0.4f)}, [](const TensorList& x, Device d) { return scatter_reduce(x[0], 0, tensor({0, 0, 2, 1}, {4}, d), x[1], ScatterReduce::Amax); });
    op("index_select", {input({4, 3})}, [](const TensorList& x, Device d) { return index_select(x[0], 0, tensor({3, 1, 1}, {3}, d)); });
    op("index_select_dim1", {input({2, 4, 2})}, [](const TensorList& x, Device d) { return index_select(x[0], 1, tensor({0, 0, 3}, {3}, d)); });
    op("index_add", {input({4, 2}), input({3, 2})}, [](const TensorList& x, Device d) { return index_add(x[0], 0, tensor({1, 1, 3}, {3}, d), x[1], 0.5f); });
    op("index_copy", {input({4, 2}), input({2, 2})}, [](const TensorList& x, Device d) { return index_copy(x[0], 0, tensor({1, 3}, {2}, d), x[1]); });
    op("take", {input({2, 3})}, [](const TensorList& x, Device d) { return take(x[0], tensor({5, 0, 2, 2}, {2, 2}, d)); });
    op("masked_fill", {input({2, 3})}, [](const TensorList& x, Device d) { return masked_fill(x[0], tensor({1, 0, 0, 1, 0, 0}, {2, 3}, d), 3.0f); });
    op("masked_select", {input({2, 3})}, [](const TensorList& x, Device d) { return masked_select(x[0], tensor({1, 0, 0, 1, 1, 0}, {2, 3}, d)); });
    op("sort", {distinct({3, 4})}, [](const TensorList& x, Device) { return sort(x[0], 1, true).first; });
    op("topk", {distinct({3, 5})}, [](const TensorList& x, Device) { return topk(x[0], 2).first; });
    op("median", {distinct({3, 5})}, [](const TensorList& x, Device) { return median(x[0], 1).first; });
    op("quantile", {distinct({3, 4})}, [](const TensorList& x, Device) { return quantile(x[0], 0.4f, 1); });

    // Products
    op("matmul", {input({2, 3, 4}), input({4, 2})}, [](const TensorList& x, Device) { return matmul(x[0], x[1]); });
    op("matmul_partial_broadcast", {input({2, 1, 2, 3}), input({1, 3, 3, 2})}, [](const TensorList& x, Device) { return matmul(x[0], x[1]); });
    op("matmul_left_broadcast", {input({2, 3}), input({4, 3, 2})}, [](const TensorList& x, Device) { return matmul(x[0], x[1]); });
    op("matmul_vectors", {input({4}), input({4})}, [](const TensorList& x, Device) { return matmul(x[0], x[1]); });
    op("matmul_mv", {input({3, 4}), input({4})}, [](const TensorList& x, Device) { return matmul(x[0], x[1]); });
    op("matmul_vm", {input({3}), input({2, 3, 2})}, [](const TensorList& x, Device) { return matmul(x[0], x[1]); });
    op("matmul_nt", {input({2, 3, 4}), input({5, 4})}, [](const TensorList& x, Device) { return matmul_nt(x[0], x[1]); });
    op("linear", {input({2, 3, 4}), input({4, 5}), input({1, 5})}, [](const TensorList& x, Device) { return linear(x[0], x[1], x[2]); });
    op("einsum_bmm", {input({2, 3, 4}), input({2, 4, 2})}, [](const TensorList& x, Device) { return einsum("bij,bjk->bik", {x[0], x[1]}); });
    op("einsum_trace", {input({3, 3})}, [](const TensorList& x, Device) { return einsum("ii", {x[0]}); });
    op("einsum_three", {input({2, 3}), input({3, 4}), input({4})}, [](const TensorList& x, Device) { return einsum("ij,jk,k->i", {x[0], x[1], x[2]}); });
    op("einsum_outer", {input({2, 3}), input({2, 4})}, [](const TensorList& x, Device) { return einsum("bi,bj->bij", {x[0], x[1]}); });
    op("einsum_ellipsis", {input({2, 3, 4}), input({4, 5})}, [](const TensorList& x, Device) { return einsum("...i,ij->...j", {x[0], x[1]}); });
    op("outer", {input({3}), input({4})}, [](const TensorList& x, Device) { return outer(x[0], x[1]); });
    op("kron", {input({2, 2}), input({2, 3})}, [](const TensorList& x, Device) { return kron(x[0], x[1]); });
    op("tensordot", {input({2, 3, 4}), input({3, 4, 2})}, [](const TensorList& x, Device) { return tensordot(x[0], x[1], 2); });
    op("cross", {input({2, 3}), input({2, 3})}, [](const TensorList& x, Device) { return cross(x[0], x[1]); });
    op("cdist", {input({3, 2}, 1, 2), input({2, 2}, -1, 0)}, [](const TensorList& x, Device) { return cdist(x[0], x[1]); });
    op("trace", {input({2, 3, 3})}, [](const TensorList& x, Device) { return trace(x[0]); });
    op("trapezoid", {input({2, 5})}, [](const TensorList& x, Device) { return trapezoid(x[0], 0.5f); });

    // Convolution, pooling, resampling
    op("conv1d", {input({2, 4, 7}), input({6, 2, 3}), input({6})}, [](const TensorList& x, Device) { return conv_nd(x[0], x[1], x[2], {2}, {1}, {1}, 2); });
    op("conv2d", {input({1, 2, 5, 5}), input({3, 2, 3, 2})}, [](const TensorList& x, Device) { return conv_nd(x[0], x[1], nullptr, {1, 2}, {1, 0}, {1}, 1); });
    op("conv2d_dilated", {input({1, 1, 6, 6}), input({1, 1, 2, 2})}, [](const TensorList& x, Device) { return conv_nd(x[0], x[1], nullptr, {1}, {1}, {2}, 1); });
    op("conv2d_depthwise", {input({1, 3, 4, 4}), input({3, 1, 3, 3})}, [](const TensorList& x, Device) { return conv_nd(x[0], x[1], nullptr, {1}, {1}, {1}, 3); });
    op("conv3d", {input({1, 1, 3, 3, 3}), input({2, 1, 2, 2, 2})}, [](const TensorList& x, Device) { return conv_nd(x[0], x[1], nullptr, {1}, {0}, {1}, 1); });
    op("conv_transpose2d", {input({1, 2, 3, 3}), input({2, 3, 3, 3}), input({3})}, [](const TensorList& x, Device) { return conv_transpose_nd(x[0], x[1], x[2], {2}, {1}, {1}, {1}, 1); });
    op("conv_transpose1d_groups", {input({1, 4, 3}), input({4, 2, 2})}, [](const TensorList& x, Device) { return conv_transpose_nd(x[0], x[1], nullptr, {1}, {0}, {0}, {1}, 2); });
    op("unfold", {input({1, 2, 4, 4})}, [](const TensorList& x, Device) { return unfold(x[0], {2, 3}, {1}, {1}); });
    op("fold", {input({1, 8, 9})}, [](const TensorList& x, Device) { return fold(x[0], {4, 4}, {2, 2}); });
    op("max_pool2d", {distinct({1, 2, 4, 4})}, [](const TensorList& x, Device) { return max_pool_nd(x[0], {3}, {2}, {1}).first; });
    op("max_pool2d_legacy", {distinct({1, 2, 4, 4})}, [](const TensorList& x, Device) { return max_pool2d(x[0], 2, 2, 2, 2); });
    op("avg_pool2d", {input({1, 2, 5, 5})}, [](const TensorList& x, Device) { return avg_pool_nd(x[0], {3}, {2}, {1}, true, false); });
    op("avg_pool1d", {input({2, 2, 6})}, [](const TensorList& x, Device) { return avg_pool_nd(x[0], {2}); });
    op("lp_pool", {input({1, 1, 4, 4}, 0.4f, 1.0f)}, [](const TensorList& x, Device) { return lp_pool_nd(x[0], 2.0f, {2}); });
    op("adaptive_avg_pool", {input({1, 2, 5, 7})}, [](const TensorList& x, Device) { return adaptive_avg_pool_nd(x[0], {2, 3}); });
    op("adaptive_max_pool", {distinct({1, 1, 5, 4})}, [](const TensorList& x, Device) { return adaptive_max_pool_nd(x[0], {2, 2}); });
    op("interpolate_nearest", {input({1, 2, 3, 3})}, [](const TensorList& x, Device) { return interpolate(x[0], {5, 6}); });
    op("interpolate_bilinear", {input({1, 2, 3, 3})}, [](const TensorList& x, Device) { return interpolate(x[0], {5, 6}, InterpolateMode::Linear); });
    op("interpolate_bicubic", {input({1, 1, 4, 4})}, [](const TensorList& x, Device) { return interpolate(x[0], {6, 7}, InterpolateMode::Cubic, true); });
    op("pixel_shuffle", {input({1, 4, 2, 2})}, [](const TensorList& x, Device) { return pixel_shuffle(x[0], 2); });
    op("pixel_unshuffle", {input({1, 1, 4, 4})}, [](const TensorList& x, Device) { return pixel_unshuffle(x[0], 2); });
    op("grid_sample", {input({1, 2, 4, 4}), input({1, 2, 2, 2}, -0.6f, 0.6f)}, [](const TensorList& x, Device) { return grid_sample(x[0], x[1]); }).second_order = false;
    op("affine_grid", {input({1, 2, 3})}, [](const TensorList& x, Device) { return affine_grid(x[0], 2, 3); });

    // Masks, normalization, attention and embedding layers
    op("causal_mask", {input({2, 3, 3})}, [](const TensorList& x, Device) { return softmax(causal_mask(x[0])); });
    op("sliding_window_mask", {input({2, 4, 4})}, [](const TensorList& x, Device) { return softmax(masking::sliding_window_mask(x[0], 1)); });
    op("prefix_causal_mask", {input({2, 4, 4})}, [](const TensorList& x, Device) { return softmax(masking::prefix_causal_mask(x[0], 2)); });
    op("repeat_kv", {input({1, 2, 2, 2})}, [](const TensorList& x, Device) { return repeat_kv(x[0], 2); });
    op("scale_n_shift", {input({2, 3, 4}), input({4}), input({4})}, [](const TensorList& x, Device) { return scale_n_shift(x[0], x[1], x[2]); });
    {
        auto ln = std::make_shared<LayerNorm>(4);
        op("layer_norm", {input({3, 4}), input({4}, 0.5f, 1.5f), input({4})}, [ln](const TensorList& x, Device) {
            ln->gamma = x[1];
            ln->beta = x[2];
            return ln->forward(x[0]);
        });
        auto rms = std::make_shared<RMSNorm>(4);
        op("rms_norm", {input({3, 4}), input({4}, 0.5f, 1.5f)}, [rms](const TensorList& x, Device) {
            rms->gamma = x[1];
            return rms->forward(x[0]);
        });
        auto emb = std::make_shared<Embedding>(5, 3);
        op("embedding", {input({5, 3})}, [emb](const TensorList& x, Device) {
            emb->weight = x[0];
            return emb->forward(tensor({0, 3, 3, 1}, {2, 2}));
        });
        auto freqs = precompute_freqs_cis(4, 8);
        auto c = freqs.first->to_vector(), s = freqs.second->to_vector();
        op("rope", {input({1, 3, 2, 4})}, [c, s](const TensorList& x, Device d) {
            return apply_rotary_emb(x[0], tensor(c, {8, 2}, d), tensor(s, {8, 2}, d), 1);
        });
        op("attention_qkv", {input({1, 2, 3, 4}), input({1, 2, 3, 4}), input({1, 2, 3, 4})}, [](const TensorList& x, Device) {
            return scaled_dot_product_attention(x[0], x[1], x[2], nullptr, 0.0f, true);
        });
        op("group_norm", {input({2, 4, 3})}, [](const TensorList& x, Device) { return GroupNorm(2, 4, 1e-5f, false).forward(x[0]); });
        op("instance_norm", {input({2, 2, 5})}, [](const TensorList& x, Device) { return InstanceNorm(2).forward(x[0]); });
        op("local_response_norm", {input({1, 5, 2}, 0.2f, 1.0f)}, [](const TensorList& x, Device) { return LocalResponseNorm(3, 1.0f).forward(x[0]); });
        op("batch_norm", {input({4, 3})}, [](const TensorList& x, Device d) {
            BatchNorm bn(3, 1e-5f, 0.1f, false);
            bn.to(d);
            return bn.forward(x[0]);
        });
        op("weight_norm", {input({3, 4}), input({3, 1}, 0.5f, 1.5f)}, [](const TensorList& x, Device) { return weight_norm(x[0], x[1]); });
    }

    // Losses
    auto near_target = [](std::vector<int64_t> shape) { return input(shape, 2.0f, 3.0f)->to_vector(); };
    {
        auto t = near_target({3, 4});
        op("mse", {input({3, 4})}, [t](const TensorList& x, Device d) { return mse(x[0], tensor(t, {3, 4}, d)); });
        op("l1", {input({3, 4})}, [t](const TensorList& x, Device d) { return l1_loss(x[0], tensor(t, {3, 4}, d)); });
        op("mse_loss_sum", {input({3, 4}), input({3, 4})}, [](const TensorList& x, Device) { return mse_loss(x[0], x[1], Reduction::Sum); });
        op("huber", {input({3, 4})}, [t](const TensorList& x, Device d) { return huber_loss(x[0], tensor(t, {3, 4}, d), 0.8f); });
        op("huber_quadratic", {input({3, 4}), input({3, 4})}, [](const TensorList& x, Device) { return huber_loss(x[0], x[1], 5.0f); });
        op("smooth_l1", {input({3, 4})}, [t](const TensorList& x, Device d) { return smooth_l1_loss(x[0], tensor(t, {3, 4}, d)); });
    }
    op("cross_entropy", {input({4, 3})}, [](const TensorList& x, Device d) {
        return cross_entropy_loss(x[0], tensor({0, 2, -100, 1}, {4}, d), tensor({1.0f, 2.0f, 0.5f}, {3}, d), -100, 0.1f);
    });
    op("cross_entropy_spatial", {input({2, 3, 2})}, [](const TensorList& x, Device d) { return cross_entropy_loss(x[0], tensor({0, 2, 1, 1}, {2, 2}, d)); });
    op("fused_cross_entropy", {input({4, 3})}, [](const TensorList& x, Device d) { return fused_cross_entropy(x[0], tensor({0, 2, -100, 1}, {4}, d)); });
    op("cross_entropy_probs", {input({3, 4}, 0.2f, 1.0f)}, [](const TensorList& x, Device d) { return cross_entropy(x[0], tensor({0.1f, 0.2f, 0.3f, 0.4f, 1, 0, 0, 0, 0.25f, 0.25f, 0.25f, 0.25f}, {3, 4}, d)); });
    op("bce", {input({5}, 0.2f, 0.8f), input({5}, 0.1f, 0.9f)}, [](const TensorList& x, Device) { return binary_cross_entropy(x[0], x[1]); });
    op("bce_logits", {input({5})}, [](const TensorList& x, Device d) {
        return binary_cross_entropy_with_logits(x[0], tensor({1, 0, 1, 0.3f, 0}, {5}, d), nullptr, tensor({1.5f, 1, 2, 1, 1}, {5}, d));
    });
    op("kl_div", {input({2, 3}), input({2, 3}, 0.2f, 0.8f)}, [](const TensorList& x, Device) { return kl_div(log_softmax(x[0], 1), x[1]); });
    op("poisson_nll", {input({5})}, [](const TensorList& x, Device d) { return poisson_nll_loss(x[0], tensor({1, 2, 0, 3, 1}, {5}, d)); });
    op("gaussian_nll", {input({5}), input({5}, 0.5f, 1.5f)}, [](const TensorList& x, Device d) { return gaussian_nll_loss(x[0], tensor({1, 2, 0, 3, 1}, {5}, d), x[1]); });
    op("triplet", {input({3, 4}, 1, 2), input({3, 4}, -2, -1), input({3, 4}, -1, 0)}, [](const TensorList& x, Device) { return triplet_margin_loss(x[0], x[1], x[2], 5.0f); });
    op("margin_ranking", {input({4}), input({4})}, [](const TensorList& x, Device d) { return margin_ranking_loss(x[0], x[1], tensor({1, -1, 1, -1}, {4}, d), 3.0f); });
    op("cosine_embedding", {input({3, 4}, 0.3f, 1), input({3, 4}, 0.3f, 1)}, [](const TensorList& x, Device d) { return cosine_embedding_loss(x[0], x[1], tensor({1, 1, -1}, {3}, d)); });
    op("multi_margin", {input({3, 4}, -0.3f, 0.3f)}, [](const TensorList& x, Device d) { return multi_margin_loss(x[0], tensor({0, 3, 1}, {3}, d)); });
    op("hinge_embedding", {input({4}, -0.5f, 0.5f)}, [](const TensorList& x, Device d) { return hinge_embedding_loss(x[0], tensor({1, -1, 1, -1}, {4}, d)); });
    op("focal", {input({5})}, [](const TensorList& x, Device d) { return sigmoid_focal_loss(x[0], tensor({1, 0, 1, 0, 0}, {5}, d)); });
    op("ctc", {input({4, 2, 3})}, [](const TensorList& x, Device d) { return ctc_loss(log_softmax(x[0], 2), tensor({1, 2, 1, 1}, {2, 2}, d), {4, 4}, {2, 1}); }).second_order = false;

    // Random ops with fixed draws
    op("dropout", {input({3, 5})}, [](const TensorList& x, Device) {
        RandomStateGuard keep;
        manual_seed(11);
        return dropout(x[0], 0.4f);
    });
    op("checkpoint", {input({3, 4})}, [](const TensorList& x, Device) { return autograd::checkpoint([](const TensorPtr& t) { return tanh(mul(t, t)); }, x[0]); }).second_order = false;
    op("custom_function", {input({4})}, [](const TensorList& x, Device) {
        return autograd::custom_function("cube", [](const TensorList& in) { return pow(in[0], 3.0f); }, {x[0]}, {0},
                                         [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, mul_scalar(square(c.input(0)), 3.0f))}; });
    });

    // Linear algebra
    auto spd = [](const TensorPtr& m) { return add(matmul(m, transpose(m, -2, -1)), mul_scalar(eye(m->shape.back(), m->shape.back(), m->device), 2.0f)); };
    op("cholesky", {input({3, 3})}, [spd](const TensorList& x, Device) { return linalg::cholesky(spd(x[0])); });
    op("cholesky_batch", {input({2, 2, 2})}, [spd](const TensorList& x, Device) { return linalg::cholesky(spd(x[0])); });
    op("inv", {input({3, 3})}, [spd](const TensorList& x, Device) { return linalg::inv(spd(x[0])); });
    op("solve", {input({3, 3}), input({3, 2})}, [spd](const TensorList& x, Device) { return linalg::solve(spd(x[0]), x[1]); });
    op("solve_triangular", {input({3, 3}, 0.8f, 1.5f), input({3, 2})}, [](const TensorList& x, Device) { return linalg::solve_triangular(x[0], x[1], true); });
    op("solve_triangular_transposed", {input({3, 3}, 0.8f, 1.5f), input({3, 2})}, [](const TensorList& x, Device) { return linalg::solve_triangular(x[0], x[1], false, true); });
    op("slogdet", {input({3, 3})}, [spd](const TensorList& x, Device) { return linalg::slogdet(spd(x[0])).second; });
    op("det", {input({2, 2, 2})}, [](const TensorList& x, Device) { return linalg::det(add(x[0], mul_scalar(eye(2, 2, x[0]->device), 2.0f))); });
    auto tall = [] { return add(input({4, 3}, -0.4f, 0.4f), tensor({2, 0, 0, 0, 1.5f, 0, 0, 0, 1, 0.3f, 0.2f, 0.1f}, {4, 3})); }();
    op("qr_q", {tall}, [](const TensorList& x, Device) { return linalg::qr(x[0]).first; });
    op("qr_r", {tall}, [](const TensorList& x, Device) { return linalg::qr(x[0]).second; });
    auto sym_input = add(input({3, 3}, -0.2f, 0.2f), tensor({3, 0, 0, 0, 2, 0, 0, 0, 1}, {3, 3}));
    auto sym = [](const TensorPtr& m) { return mul_scalar(add(m, transpose(m, 0, 1)), 0.5f); };
    op("eigh_values", {sym_input}, [sym](const TensorList& x, Device) { return linalg::eigh(sym(x[0])).first; });
    // Eigenvectors and singular vectors are fixed up to sign: squares remove it
    op("eigh_vectors", {sym_input}, [sym](const TensorList& x, Device) { return square(linalg::eigh(sym(x[0])).second); });
    op("svd_values", {tall}, [](const TensorList& x, Device) { return linalg::svdvals(x[0]); });
    op("svd_u", {tall}, [](const TensorList& x, Device) { return square(std::get<0>(linalg::svd(x[0]))); });
    op("svd_v", {tall}, [](const TensorList& x, Device) { return square(std::get<2>(linalg::svd(x[0]))); });
    op("svd_v_wide", {transpose(tall, 0, 1)->clone()}, [](const TensorList& x, Device) { return square(std::get<2>(linalg::svd(x[0]))); });
    op("pinv", {tall}, [](const TensorList& x, Device) { return linalg::pinv(x[0]); });
    op("lstsq", {tall, input({4, 2})}, [](const TensorList& x, Device) { return linalg::lstsq(x[0], x[1]); });
    op("matrix_exp", {input({3, 3})}, [](const TensorList& x, Device) { return linalg::matrix_exp(x[0]); });
    op("matrix_power", {input({3, 3})}, [](const TensorList& x, Device) { return linalg::matrix_power(x[0], 3); });
    op("lu_l", {add(input({3, 3}, -0.3f, 0.3f), mul_scalar(eye(3), 2.0f))}, [](const TensorList& x, Device) { return std::get<1>(linalg::lu(x[0])); });
    op("lu_u", {add(input({3, 3}, -0.3f, 0.3f), mul_scalar(eye(3), 2.0f))}, [](const TensorList& x, Device) { return std::get<2>(linalg::lu(x[0])); });

    // Fourier transforms
    op("fft", {input({2, 6, 2})}, [](const TensorList& x, Device) { return fft::fft(x[0]); });
    op("ifft_dim0", {input({5, 2, 2})}, [](const TensorList& x, Device) { return fft::ifft(x[0], 0); });
    op("rfft", {input({2, 8})}, [](const TensorList& x, Device) { return fft::rfft(x[0]); });
    op("irfft", {input({2, 5, 2})}, [](const TensorList& x, Device) { return fft::irfft(x[0]); });
    op("irfft_odd", {input({1, 4, 2})}, [](const TensorList& x, Device) { return fft::irfft(x[0], -1, 7); });
    op("fft_convolve", {input({5}), input({3})}, [](const TensorList& x, Device) { return fft::fft_convolve(x[0], x[1]); });
    op("stft", {input({1, 16})}, [](const TensorList& x, Device d) { return fft::stft(x[0], 8, 4, fft::hann_window(8)->to(d)); });

    // Distributions
    op("normal_log_prob", {input({4}), input({4}, 0.5f, 1.5f), input({4})}, [](const TensorList& x, Device) { return distributions::Normal(x[0], x[1]).log_prob(x[2]); });
    op("gamma_log_prob", {input({4}, 1.0f, 2.0f), input({4}, 0.5f, 1.5f), input({4}, 0.5f, 2.0f)}, [](const TensorList& x, Device) { return distributions::Gamma(x[0], x[1]).log_prob(x[2]); });
    op("beta_log_prob", {input({4}, 1.0f, 2.0f), input({4}, 1.0f, 2.0f), input({4}, 0.3f, 0.7f)}, [](const TensorList& x, Device) { return distributions::Beta(x[0], x[1]).log_prob(x[2]); });
    op("categorical_log_prob", {input({3, 4})}, [](const TensorList& x, Device d) { return distributions::Categorical(x[0]).log_prob(tensor({0, 3, 2}, {3}, d)); });
    op("mvn_log_prob", {input({3}), input({3, 3}, 0.6f, 1.2f), input({3})}, [](const TensorList& x, Device) {
        return distributions::MultivariateNormal(x[0], tril(x[1])).log_prob(x[2]);
    });
    op("kl_normal", {input({3}), input({3}, 0.5f, 1.5f), input({3}), input({3}, 0.5f, 1.5f)}, [](const TensorList& x, Device) {
        return distributions::kl_divergence(distributions::Normal(x[0], x[1]), distributions::Normal(x[2], x[3]));
    });
    op("kl_gamma", {input({3}, 1, 2), input({3}, 0.5f, 1.5f), input({3}, 1, 2), input({3}, 0.5f, 1.5f)}, [](const TensorList& x, Device) {
        return distributions::kl_divergence(distributions::Gamma(x[0], x[1]), distributions::Gamma(x[2], x[3]));
    });
}

// ---------------------------------------------------------------------------------------------
// Autograd API
// ---------------------------------------------------------------------------------------------

static std::string autograd_api() {
    auto x = tensor({1, 2, 3}, {3}, Device::CPU, true);
    auto y = sum(pow(x, 3.0f));
    auto g = autograd::grad({y}, {x}, {}, true, true)[0];
    CHECK(near(g, {3, 12, 27}));
    if (x->grad) return "autograd::grad wrote to .grad";
    auto h = autograd::grad({sum(g)}, {x}, {}, true, true)[0];
    CHECK(near(h, {6, 12, 18}));
    auto third = autograd::grad({sum(h)}, {x})[0];
    CHECK(near(third, {6, 6, 6}));

    // With respect to an intermediate, and a gradient penalty accumulated into a parameter
    auto w = tensor({2, -1}, {2}, Device::CPU, true);
    auto mid = mul(w, tensor({3, 4}, {2}));
    auto out = sum(square(mid));
    CHECK(near(autograd::grad({out}, {mid}, {}, true)[0], {12, -8}));
    auto gw = autograd::grad({out}, {w}, {}, true, true)[0];  // 2 c^2 w
    sum(square(gw))->backward();                              // sum 4 c^4 w^2 -> 8 c^4 w
    CHECK(near(w->grad, {8 * 81 * 2.0f, 8 * 256 * -1.0f}, 1e-2f));

    CHECK(expect_throw([&] { autograd::grad({sum(x)}, {w}); }, "an unused input"));
    if (autograd::grad({sum(x)}, {w}, {}, false, false, true)[0]) return "allow_unused should give null";
    // Several outputs, weighted
    auto a = tensor({1, 2}, {2}, Device::CPU, true);
    auto gg = autograd::grad({square(a), mul_scalar(a, 3.0f)}, {a}, {tensor({1, 10}, {2}), tensor({1, 1}, {2})})[0];
    CHECK(near(gg, {2 + 3, 40 + 3}));
    // Non-scalar backward
    auto b = tensor({1, 2}, {2}, Device::CPU, true);
    square(b)->backward(tensor({1, 0.5f}, {2}));
    CHECK(near(b->grad, {2, 2}));
    // A graph that was freed must not silently lose gradients
    auto c = tensor({1, 2}, {2}, Device::CPU, true);
    auto shared = square(c);
    sum(shared)->backward();
    CHECK(expect_throw([&] { sum(mul_scalar(shared, 2.0f))->backward(); }, "backward through a freed graph"));
    return "";
}
static std::string autograd_hooks() {
    auto x = tensor({1, 2}, {2}, Device::CPU, true);
    auto y = mul_scalar(x, 3.0f);
    y->retain_grad();
    int calls = 0;
    y->register_hook([&](const TensorPtr& g) -> TensorPtr {
        ++calls;
        return mul_scalar(g, 10.0f);
    });
    x->register_hook([&](const TensorPtr&) -> TensorPtr {
        ++calls;
        return nullptr;
    });
    sum(square(y))->backward();
    if (calls != 2) return "hooks ran " + std::to_string(calls) + " times";
    CHECK(near(y->grad, {6, 12}));
    CHECK(near(x->grad, {180, 360}));
    return "";
}
static std::string autograd_functional() {
    auto f = [](const TensorPtr& x) { return sum(mul(square(x), tensor({1, 2, 3}, {3}))); };
    auto x = tensor({1, 2, -1}, {3});
    CHECK(near(autograd::hessian(f, x), {2, 0, 0, 0, 4, 0, 0, 0, 6}));
    CHECK(near(autograd::hvp(f, x, tensor({1, 1, 2}, {3})), {2, 4, 12}));
    auto g = [](const TensorPtr& t) { return stack({mul(select(t, 0, 0), select(t, 0, 1)), sin(select(t, 0, 2))}, 0); };
    CHECK(near(autograd::jacobian(g, x), {2, 1, 0, 0, 0, std::cos(-1.0f)}));
    CHECK(near(autograd::jvp(g, x, tensor({1, 0, 1}, {3})).second, {2, std::cos(-1.0f)}));
    CHECK(near(autograd::vjp(g, x, tensor({1, 2}, {2})).second, {2, 1, 2 * std::cos(-1.0f)}));
    return "";
}
static std::string autograd_inplace_detected() {
    auto x = tensor({1, 2}, {2}, Device::CPU, true);
    auto y = sum(exp(x));
    auto z = mul(tensor({1, 1}, {2}), square(x));
    x->fill_(5.0f);
    y->backward();  // exp saved its output, which did not change
    return expect_throw([&] { sum(z)->backward(); }, "using a saved tensor modified in place");
}
static std::string checkpoint_matches() {
    manual_seed(5);
    Linear a(6, 6), b(6, 6);
    auto block = [&](const TensorPtr& t) { return dropout(tanh(b.forward(relu(a.forward(t)))), 0.3f); };
    auto x = input({4, 6});
    auto run_once = [&](bool ckpt) {
        manual_seed(9);
        a.zero_grad();
        b.zero_grad();
        auto l = leaf_of(x->clone()->detach());
        auto y = ckpt ? autograd::checkpoint(block, l) : block(l);
        sum(square(y))->backward();
        std::vector<std::vector<float>> out{y->to_vector(), l->grad->to_vector(), a.W->grad->to_vector(), b.b->grad->to_vector()};
        return out;
    };
    auto plain = run_once(false), ck = run_once(true);
    for (size_t i = 0; i < plain.size(); ++i) CHECK(near(ck[i], plain[i], 1e-6f));
    return "";
}

// ---------------------------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------------------------

static std::string values_shapes() {
    auto x = reshape(arange(0, 24), {2, 3, 4});
    CHECK(near(slice_step(x, 2, 1, 4, 2), {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}));
    CHECK(near(select(x, 1, -1), {8, 9, 10, 11, 20, 21, 22, 23}));
    CHECK(near(flip(arange(0, 4), {0}), {3, 2, 1, 0}));
    CHECK(near(roll(arange(0, 5), 2, 0), {3, 4, 0, 1, 2}));
    CHECK(near(pad(arange(0, 4), {2, 1}, PadMode::Reflect), {2, 1, 0, 1, 2, 3, 2}));
    CHECK(near(pad(arange(0, 3), {1, 2}, PadMode::Circular), {2, 0, 1, 2, 0, 1}));
    CHECK(near(repeat(tensor({1, 2}, {2}), {2, 2}), {1, 2, 1, 2, 1, 2, 1, 2}));
    CHECK(near(repeat_interleave(tensor({1, 2}, {2}), 2, 0), {1, 1, 2, 2}));
    CHECK(shape_is(movedim(x, 0, 2), {3, 4, 2}));
    CHECK(near(diagonal(reshape(arange(0, 9), {3, 3}), -1), {3, 7}));
    CHECK(near(tril(ones({3, 3})), {1, 0, 0, 1, 1, 0, 1, 1, 1}));
    auto parts = chunk(arange(0, 7), 3, 0);
    if (parts.size() != 3 || parts[2]->size() != 1) return "chunk sizes";
    auto grids = meshgrid({arange(0, 2), arange(0, 3)});
    CHECK(near(grids[1], {0, 1, 2, 0, 1, 2}));
    CHECK(near(linspace(0, 1, 5), {0, 0.25f, 0.5f, 0.75f, 1}));
    // Two-way broadcasting
    CHECK(near(add(reshape(arange(0, 2), {2, 1}), reshape(arange(0, 3), {1, 3})), {0, 1, 2, 1, 2, 3}));
    CHECK(near(sub(arange(0, 3), reshape(arange(0, 2), {2, 1})), {0, 1, 2, -1, 0, 1}));
    return "";
}
static std::string values_reductions() {
    auto x = reshape(arange(0, 24), {2, 3, 4});
    CHECK(near(sum(x, {0, 2}, false), {60, 92, 124}));
    CHECK(near(mean(x, 0, false), {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}));
    CHECK(near(amax(x, {1, 2}, false), {11, 23}));
    CHECK(near(argmax(x, 1), {2, 2, 2, 2, 2, 2, 2, 2}));
    CHECK(near(argmin(tensor({3, 1, 2, 1}, {4}), 0), {1}));
    CHECK(near(cumsum(arange(1, 5), 0), {1, 3, 6, 10}));
    CHECK(near(cumprod(arange(1, 5), 0), {1, 2, 6, 24}));
    CHECK(near(cummax(tensor({1, 3, 2, 5, 4}, {5}), 0).second, {0, 1, 1, 3, 3}));
    CHECK(near(prod(x, 2, false), {0, 840, 7920, 32760, 73440, 143640}, 1e-2f));
    CHECK(scalar(var(arange(0, 5), 0, false, 1), 2.5f));
    CHECK(scalar(norm(tensor({3, 4}, {2})), 5.0f));
    CHECK(scalar(logsumexp(tensor({1000, 1000}, {2}), 0), 1000.6931f, 1e-3f));
    CHECK(near(softmax(tensor({1, 2, 3, 4}, {2, 2}), 0), {0.1192029f, 0.1192029f, 0.8807971f, 0.8807971f}));
    CHECK(near(count_nonzero(tensor({0, 1, 2, 0, 0, 3}, {2, 3}), {1}), {2, 1}));
    // A long sum is accurate and does not depend on the thread count
    int64_t n = 1000003;
    auto big = uniform({n}, 0.0f, 1.0f);
    auto v = big->to_vector();
    double exact = 0.0;
    for (float f : v) exact += f;
    int threads = get_num_threads();
    float many = sum(big)->item();
    set_num_threads(1);
    float one = sum(big)->item();
    set_num_threads(threads);
    if (many != one) return "a sum depends on the thread count";
    CHECK(near({many}, {(float)exact}, 0.0f, 2e-7f));
    // The gradient of prod is exact with zeros
    auto z = tensor({2, 0, 3}, {3}, Device::CPU, true);
    prod(z)->backward();
    CHECK(near(z->grad, {0, 6, 0}));
    auto c = tensor({2, 0, 3}, {3}, Device::CPU, true);
    sum(cumprod(c, 0))->backward();
    CHECK(near(c->grad, {1, 2 + 6, 0}));
    return "";
}
static std::string values_indexing() {
    auto x = reshape(arange(0, 12), {3, 4});
    CHECK(near(gather(x, 1, tensor({0, 1, 3, 2, 0, 0}, {3, 2})), {0, 1, 7, 6, 8, 8}));
    CHECK(near(scatter_add(zeros({2, 3}), 1, tensor({0, 0, 2, 1}, {2, 2}), ones({2, 2})), {2, 0, 0, 0, 1, 1}));
    CHECK(near(scatter(zeros({3}), 0, tensor({2, 0}, {2}), tensor({5, 7}, {2})), {7, 0, 5}));
    CHECK(near(index_add(zeros({3, 2}), 0, tensor({1, 1, 0}, {3}), ones({3, 2})), {1, 1, 2, 2, 0, 0}));
    auto s = sort(tensor({3, 1, 2}, {3}), 0, true);
    CHECK(near(s.first, {3, 2, 1}));
    CHECK(near(s.second, {0, 2, 1}));
    CHECK(near(topk(tensor({3, 1, 4, 1, 5}, {5}), 2).second, {4, 2}));
    CHECK(near(nonzero(tensor({0, 2, 3, 0}, {2, 2})), {0, 1, 1, 0}));
    auto u = unique(tensor({3, 1, 3, 2}, {4}));
    CHECK(near(u.values, {1, 2, 3}));
    CHECK(near(u.inverse, {2, 0, 2, 1}));
    CHECK(near(u.counts, {1, 1, 2}));
    CHECK(near(searchsorted(tensor({1, 3, 5}, {3}), tensor({3, 4}, {2}), true), {2, 2}));
    CHECK(near(bincount(tensor({0, 2, 2}, {3})), {1, 0, 2}));
    CHECK(near(histc(tensor({0.1f, 0.5f, 0.9f, 1.0f}, {4}), 2, 0, 1), {1, 3}));
    CHECK(scalar(median(tensor({4, 1, 3, 2}, {4})), 2));
    CHECK(near(quantile(tensor({1, 2, 3, 4}, {4}), 0.5f, 0), {2.5f}));
    CHECK(expect_throw([] { gather(zeros({3}), 0, tensor({3}, {1})); }, "an out-of-range index"));
    CHECK(expect_throw([] { index_select(zeros({3}), 0, tensor({-1}, {1})); }, "a negative index"));
    return "";
}
static std::string values_math() {
    CHECK(near(erfinv(erf(tensor({-1.5f, -0.2f, 0.7f, 2.0f}, {4}))), {-1.5f, -0.2f, 0.7f, 2.0f}, 2e-4f, 2e-4f));
    CHECK(near(digamma(tensor({1, 0.5f, -0.5f}, {3})), {-0.5772157f, -1.9635100f, 0.0364900f}, 1e-4f));
    CHECK(near(trigamma(tensor({1, 0.5f}, {2})), {1.6449341f, 4.9348022f}, 1e-4f));
    CHECK(near(i0(tensor({0, 1, 5}, {3})), {1, 1.2660659f, 27.239872f}, 1e-4f, 1e-4f));
    CHECK(near(round(tensor({0.5f, 1.5f, 2.5f, -0.5f}, {4})), {0, 2, 2, 0}));
    CHECK(near(remainder(tensor({-3, 3}, {2}), tensor({2, -2}, {2})), {1, -1}));
    CHECK(near(fmod(tensor({-3, 3}, {2}), tensor({2, -2}, {2})), {-1, 1}));
    CHECK(near(floor_divide(tensor({-3, 7}, {2}), tensor({2, 2}, {2})), {-2, 3}));
    CHECK(near(nan_to_num(tensor({NAN, 1, INFINITY}, {3}), 2.0f), {2, 1, 3.4028235e38f}, 0, 1e-6f));
    CHECK(near(is_nan(tensor({NAN, 1, INFINITY}, {3})), {1, 0, 0}));
    CHECK(near(logical_and(tensor({1, 0, 2}, {3}), tensor({1, 1, 0}, {3})), {1, 0, 0}));
    CHECK(near(maximum(tensor({NAN, 1}, {2}), tensor({0, 2}, {2})), {NAN, 2}));
    // The vectorized exp, log and tanh against libm over their range
    std::vector<float> v;
    for (float x = -100.0f; x < 89.0f; x += 0.37f) v.push_back(x);
    auto e = exp(tensor(v, {(int64_t)v.size()}))->to_vector();
    auto t = tanh(tensor(v, {(int64_t)v.size()}))->to_vector();
    for (size_t i = 0; i < v.size(); ++i) {
        float want = std::exp(v[i]);
        if (std::fabs(e[i] - want) > 4e-7f * want + 1e-44f) return "exp(" + std::to_string(v[i]) + ") = " + std::to_string(e[i]);
        if (std::fabs(t[i] - std::tanh(v[i])) > 3e-7f) return "tanh(" + std::to_string(v[i]) + ")";
    }
    std::vector<float> p;
    for (float x = 1e-40f; x < 1e38f; x *= 1.31f) p.push_back(x);
    auto l = log(tensor(p, {(int64_t)p.size()}))->to_vector();
    for (size_t i = 0; i < p.size(); ++i) if (std::fabs(l[i] - std::log(p[i])) > 3e-7f * std::fabs(std::log(p[i])) + 2e-7f) return "log(" + std::to_string(p[i]) + ") = " + std::to_string(l[i]);
    CHECK(near(exp(tensor({INFINITY, -INFINITY, NAN, 0}, {4})), {INFINITY, 0, NAN, 1}));
    CHECK(near(log(tensor({INFINITY, 0, -1, 1}, {4})), {INFINITY, -INFINITY, NAN, 0}));
    return "";
}
static std::string values_einsum() {
    auto a = input({3, 4}), b = input({4, 5}), c = input({2, 3, 4});
    CHECK(near(einsum("ij,jk->ik", {a, b}), matmul(a, b)));
    CHECK(near(einsum("ij->ji", {a}), contiguous(transpose(a, 0, 1))));
    CHECK(near(einsum("ij,jk", {a, b}), matmul(a, b)));
    CHECK(near(einsum("bij->b", {c}), sum(c, {1, 2}, false)));
    CHECK(scalar(einsum("ij,ij->", {a, a}), sum(square(a))->item(), 1e-4f));
    auto d = input({4, 4});
    CHECK(near(einsum("ii->i", {d}), contiguous(diagonal(d))));
    CHECK(near(einsum("...ij,...jk->...ik", {c, b}), matmul(c, b)));
    CHECK(shape_is(einsum("a,b,c->cab", {input({2}), input({3}), input({4})}), {4, 2, 3}));
    CHECK(expect_throw([&] { einsum("ij,jk->iz", {a, b}); }, "an unknown output subscript"));
    return "";
}

// Direct definition of a 2-D convolution, for comparison
static std::string values_conv() {
    auto x = input({2, 4, 7, 6}), w = input({6, 2, 3, 2});
    int64_t sh = 2, sw = 1, ph = 1, pw = 2, dh = 1, dw = 2, groups = 2;
    auto y = conv_nd(x, w, nullptr, {sh, sw}, {ph, pw}, {dh, dw}, groups);
    auto xv = x->to_vector(), wv = w->to_vector(), yv = y->to_vector();
    int64_t H = y->shape[2], W = y->shape[3];
    for (int64_t b = 0; b < 2; ++b)
        for (int64_t o = 0; o < 6; ++o)
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j) {
                    double want = 0.0;
                    for (int64_t c = 0; c < 2; ++c)
                        for (int64_t ki = 0; ki < 3; ++ki)
                            for (int64_t kj = 0; kj < 2; ++kj) {
                                int64_t r = i * sh - ph + ki * dh, s = j * sw - pw + kj * dw, in = (o / 3) * 2 + c;
                                if (r < 0 || r >= 7 || s < 0 || s >= 6) continue;
                                want += (double)xv[(size_t)(((b * 4 + in) * 7 + r) * 6 + s)] * wv[(size_t)(((o * 2 + c) * 3 + ki) * 2 + kj)];
                            }
                    float got = yv[(size_t)(((b * 6 + o) * H + i) * W + j)];
                    if (std::fabs(got - want) > 1e-5) return "conv2d differs from its definition";
                }
    // A transposed convolution is the adjoint of the convolution: <conv(x), y> = <x, conv_transpose(y)>
    auto u = input({1, 2, 5, 5}), k = input({3, 2, 3, 3});
    auto cu = conv_nd(u, k, nullptr, {2}, {1}, {1}, 1);
    auto v = input(cu->shape);
    auto back = conv_transpose_nd(v, k, nullptr, {2}, {1}, {0}, {1}, 1);
    CHECK(shape_is(back, u->shape));
    CHECK(near({sum(mul(cu, v))->item()}, {sum(mul(u, back))->item()}, 1e-4f));
    CHECK(near(avg_pool_nd(reshape(arange(0, 16), {1, 1, 4, 4}), {2}), {2.5f, 4.5f, 10.5f, 12.5f}));
    auto mp = max_pool_nd(reshape(arange(0, 16), {1, 1, 4, 4}), {2});
    CHECK(near(mp.second, {5, 7, 13, 15}));
    auto un = max_unpool_nd(mp.first, mp.second, {4, 4});
    CHECK(scalar(sum(un), 5 + 7 + 13 + 15));
    CHECK(near(interpolate(tensor({0, 1, 2, 3}, {1, 1, 2, 2}), {4, 4}, InterpolateMode::Linear, true),
               {0, 1 / 3.f, 2 / 3.f, 1, 2 / 3.f, 1, 4 / 3.f, 5 / 3.f, 4 / 3.f, 5 / 3.f, 2, 7 / 3.f, 2, 7 / 3.f, 8 / 3.f, 3}));
    CHECK(near(interpolate(tensor({1, 2}, {1, 1, 2}), {4}, InterpolateMode::Linear), {1, 1.25f, 1.75f, 2}));
    // Sampling on the identity grid returns the image
    auto img = input({1, 2, 4, 5});
    auto grid = affine_grid(tensor({1, 0, 0, 0, 1, 0}, {1, 2, 3}), 4, 5);
    CHECK(near(grid_sample(img, grid), img, 1e-4f));
    return "";
}

// CTC against a sum over every alignment
static std::string values_ctc() {
    auto lp = log_softmax(input({5, 1, 3}), 2);
    auto v = lp->to_vector();
    std::vector<int> target{1, 1};
    double total = 0.0;
    for (int code = 0; code < 243; ++code) {
        int c = code;
        double logp = 0.0;
        std::vector<int> collapsed;
        int last = -1;
        for (int t = 0; t < 5; ++t) {
            int s = c % 3;
            c /= 3;
            logp += v[(size_t)(t * 3 + s)];
            if (s != last && s != 0) collapsed.push_back(s);
            last = s;
        }
        if (collapsed == target) total += std::exp(logp);
    }
    auto loss = ctc_loss(lp, tensor({1, 1}, {1, 2}), {5}, {2}, 0, Reduction::Sum);
    return scalar(loss, (float)-std::log(total), 1e-4f, 1e-4f);
}
static std::string values_losses() {
    auto logits = input({4, 3});
    auto t = tensor({0, 2, 1, 1}, {4});
    CHECK(scalar(cross_entropy_loss(logits, t), fused_cross_entropy(logits, t)->item(), 1e-5f));
    CHECK(near(one_hot(t, 3), {1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0}));
    CHECK(scalar(soft_cross_entropy_loss(logits, one_hot(t, 3)), cross_entropy_loss(logits, t)->item(), 1e-5f));
    auto x = input({5}), y = tensor({1, 0, 1, 1, 0}, {5});
    CHECK(scalar(binary_cross_entropy_with_logits(x, y), binary_cross_entropy(sigmoid(x), y)->item(), 1e-5f));
    CHECK(scalar(binary_cross_entropy_with_logits(tensor({200, -200}, {2}), tensor({1, 0}, {2})), 0.0f));
    CHECK(scalar(huber_loss(tensor({0, 3}, {2}), tensor({0.5f, 0}, {2})), (0.125f + 2.5f) / 2));
    CHECK(scalar(kl_div(log(tensor({0.5f, 0.5f}, {1, 2})), tensor({1, 0}, {1, 2})), std::log(2.0f)));
    return "";
}
static std::string values_random() {
    manual_seed(3);
    auto a = rand({1000}), n = normal({200000});
    manual_seed(3);
    CHECK(near(rand({1000}), a, 0, 0));
    set_num_threads(1);
    manual_seed(3);
    CHECK(near(rand({1000}), a, 0, 0));
    set_num_threads(4);
    CHECK(scalar(mean(n), 0.0f, 0.01f));
    CHECK(scalar(std_dev(n), 1.0f, 0.01f));
    auto kurt = mean(pow(n, 4.0f))->item();
    if (std::fabs(kurt - 3.0f) > 0.1f) return "normal draws have fourth moment " + std::to_string(kurt);
    auto u = rand({200000});
    CHECK(scalar(mean(u), 0.5f, 0.005f));
    if (amin(u)->item() < 0.0f || amax(u)->item() >= 1.0f) return "uniform draws leave [0, 1)";
    auto r = randint(2, 5, {10000});
    if (amin(r)->item() != 2.0f || amax(r)->item() != 4.0f) return "randint range";
    auto perm = sort(randperm(50), 0).first;
    CHECK(near(perm, arange(0, 50)));
    auto d = dropout(ones({100000}), 0.25f);
    CHECK(scalar(mean(d), 1.0f, 0.01f));
    CHECK(near(dropout(ones({4}), 0.5f, false), {1, 1, 1, 1}));
    auto m = multinomial(tensor({0.1f, 0.0f, 0.9f}, {3}), 20000, true);
    CHECK(scalar(mean(eq(m, 2.0f)), 0.9f, 0.01f));
    if (any(eq(m, 1.0f))) return "multinomial drew a class of weight 0";
    CHECK(near(sort(multinomial(tensor({1, 2, 3}, {3}), 3), 0).first, {0, 1, 2}));
    CHECK(scalar(mean(bernoulli({20000}, 0.3f)), 0.3f, 0.02f));
    auto g = gumbel_softmax(zeros({1000, 4}), 1.0f, true);
    CHECK(scalar(sum(g), 1000.0f, 1e-2f));
    return "";
}

static std::string values_linalg() {
    auto a = input({2, 5, 5});
    auto spd = add(matmul(a, transpose(a, -2, -1)), eye(5));
    auto L = linalg::cholesky(spd);
    CHECK(near(matmul(L, transpose(L, -2, -1)), spd, 1e-4f, 1e-4f));
    CHECK(near(matmul(spd, linalg::inv(spd)), contiguous(expand(eye(5), {2, 5, 5})), 1e-4f));
    auto b = input({2, 5, 3});
    CHECK(near(matmul(spd, linalg::solve(spd, b)), b, 1e-4f));
    CHECK(near(linalg::cholesky_solve(b, L), linalg::solve(spd, b), 1e-4f, 1e-3f));
    auto t = input({6, 4});
    auto [q, r] = linalg::qr(t);
    CHECK(near(matmul(q, r), t, 1e-5f));
    CHECK(near(matmul(transpose(q, 0, 1), q), eye(4), 1e-5f));
    CHECK(near(r, triu(r), 1e-6f));
    for (auto shape : std::vector<std::vector<int64_t>>{{6, 4}, {4, 6}, {3, 3}}) {
        auto m = input(shape);
        auto [u, s, vt] = linalg::svd(m);
        CHECK(near(matmul(mul(u, unsqueeze(s, -2)), vt), m, 1e-5f));
        CHECK(near(matmul(transpose(u, 0, 1), u), eye(s->size()), 1e-5f));
    }
    auto sym = mul_scalar(add(a, transpose(a, -2, -1)), 0.5f);
    auto [w, v] = linalg::eigh(sym);
    CHECK(near(matmul(mul(v, unsqueeze(w, -2)), transpose(v, -2, -1)), sym, 1e-5f));
    CHECK(near(linalg::det(tensor({1, 2, 3, 4}, {2, 2})), {-2}));
    CHECK(near(linalg::matrix_exp(tensor({0, -1.5707963f, 1.5707963f, 0}, {2, 2})), {0, -1, 1, 0}, 1e-5f));
    CHECK(near(linalg::matrix_power(tensor({1, 1, 1, 0}, {2, 2}), 10), {89, 55, 55, 34}, 1e-3f));
    CHECK(near(linalg::pinv(t), matmul(linalg::inv(matmul(transpose(t, 0, 1), t)), transpose(t, 0, 1)), 1e-3f, 1e-3f));
    CHECK(scalar(linalg::matrix_rank(matmul(input({5, 2}), input({2, 5}))), 2));
    auto m = input({4, 4});
    auto [p2, l2, u2] = linalg::lu(m);
    CHECK(near(matmul(matmul(p2, l2), u2), m, 1e-5f));
    // Eigenvalues of a rotation are e^{+-i theta}; their sum is the trace
    auto ev = linalg::eigvals(tensor({0.6f, -0.8f, 0.8f, 0.6f}, {2, 2}));
    CHECK(near(abs(ev), {0.6f, 0.8f, 0.6f, 0.8f}, 1e-5f));
    auto gen = input({7, 7});
    CHECK(near({sum(select(linalg::eigvals(gen), -1, 0))->item()}, {trace(gen)->item()}, 1e-4f));
    CHECK(expect_throw([] { linalg::cholesky(tensor({1, 2, 2, 1}, {2, 2})); }, "an indefinite matrix"));
    CHECK(expect_throw([] { linalg::inv(zeros({2, 2})); }, "a singular matrix"));
    return "";
}
static std::string values_fft() {
    for (int64_t n : {1, 2, 5, 8, 12, 17, 64}) {
        auto x = input({n, 2});
        auto v = x->to_vector();
        std::vector<float> want;
        for (int64_t k = 0; k < n; ++k) {
            double re = 0, im = 0;
            for (int64_t j = 0; j < n; ++j) {
                double a = -2.0 * 3.14159265358979 * (double)(j * k) / (double)n;
                re += v[(size_t)(2 * j)] * std::cos(a) - v[(size_t)(2 * j + 1)] * std::sin(a);
                im += v[(size_t)(2 * j)] * std::sin(a) + v[(size_t)(2 * j + 1)] * std::cos(a);
            }
            want.push_back((float)re);
            want.push_back((float)im);
        }
        CHECK(near(fft::fft(x), want, 1e-4f));
        CHECK(near(fft::ifft(fft::fft(x)), x, 1e-5f));
    }
    for (int64_t n : {6, 7, 16}) {
        auto x = input({3, n});
        CHECK(near(fft::irfft(fft::rfft(x), -1, n), x, 1e-5f));
    }
    CHECK(near(fft::fft_convolve(tensor({1, 2, 3}, {3}), tensor({0, 1, 0.5f}, {3})), {0, 1, 2.5f, 4, 1.5f}, 1e-5f));
    auto x = input({2, 64});
    auto win = fft::hann_window(16);
    CHECK(near(fft::istft(fft::stft(x, 16, 4, win), 16, 4, win), x, 1e-4f));
    auto img = input({4, 6, 2});
    CHECK(near(fft::ifft2(fft::fft2(img)), img, 1e-5f));
    CHECK(near(fft::fftfreq(4), {0, 0.25f, -0.5f, -0.25f}));
    return "";
}
static std::string values_distributions() {
    using namespace distributions;
    manual_seed(1);
    auto zero = tensor({0}, {1}), one = tensor({1}, {1});
    CHECK(scalar(Normal(zero, one).log_prob(tensor({1}, {1})), -1.4189385f));
    CHECK(scalar(Beta(tensor({2}, {1}), tensor({3}, {1})).log_prob(tensor({0.4f}, {1})), std::log(12 * 0.4f * 0.36f), 1e-4f));
    CHECK(scalar(Poisson(tensor({2}, {1})).log_prob(tensor({3}, {1})), 3 * std::log(2.0f) - 2 - std::log(6.0f), 1e-4f));
    CHECK(scalar(Binomial(5, tensor({0.3f}, {1}), true).log_prob(tensor({2}, {1})), std::log(10 * 0.09f * 0.343f), 1e-4f));
    // Moments of samples against the stated ones
    struct Case {
        std::string name;
        std::shared_ptr<Distribution> d;
    };
    auto p = [](float v) { return tensor({v}, {1}); };
    std::vector<Case> cases = {{"Normal", std::make_shared<Normal>(p(1), p(2))},
                               {"LogNormal", std::make_shared<LogNormal>(p(0), p(0.4f))},
                               {"HalfNormal", std::make_shared<HalfNormal>(p(1.5f))},
                               {"Uniform", std::make_shared<Uniform>(p(-1), p(3))},
                               {"Exponential", std::make_shared<Exponential>(p(2))},
                               {"Laplace", std::make_shared<Laplace>(p(1), p(0.5f))},
                               {"Gumbel", std::make_shared<Gumbel>(p(1), p(0.7f))},
                               {"Gamma", std::make_shared<Gamma>(p(2.5f), p(1.5f))},
                               {"Beta", std::make_shared<Beta>(p(2), p(3))},
                               {"Poisson", std::make_shared<Poisson>(p(3))},
                               {"Geometric", std::make_shared<Geometric>(p(0.3f))},
                               {"Bernoulli", std::make_shared<Bernoulli>(p(0.3f), true)},
                               {"Binomial", std::make_shared<Binomial>(6, p(0.4f), true)}};
    for (auto& c : cases) {
        auto s = c.d->sample({40000});
        float m = mean(s)->item(), v = var(s)->item(), wm = c.d->mean()->item(), wv = c.d->variance()->item();
        if (std::fabs(m - wm) > 0.03f * std::max(1.0f, std::sqrt(wv)) * 1.5f) return c.name + " mean " + std::to_string(m) + ", expected " + std::to_string(wm);
        if (std::fabs(v - wv) > 0.06f * wv + 0.01f) return c.name + " variance " + std::to_string(v) + ", expected " + std::to_string(wv);
    }
    // KL against a Monte Carlo estimate
    Normal a(p(0.3f), p(0.8f)), b(p(-0.2f), p(1.1f));
    auto s = a.sample({100000});
    CHECK(scalar(kl_divergence(a, b), mean(sub(a.log_prob(s), b.log_prob(s)))->item(), 0.02f));
    Categorical cat(tensor({0, 1, 2}, {1, 3}));
    CHECK(scalar(mean(eq(cat.sample({20000}), 2.0f)), 0.66524f, 0.02f));
    CHECK(scalar(sum(Dirichlet(tensor({1, 2, 3}, {3})).sample()), 1.0f, 1e-5f));
    auto cov = tensor({2, 0.5f, 0.5f, 1}, {2, 2});
    auto mvn = MultivariateNormal::from_covariance(tensor({1, -1}, {2}), cov);
    auto draws = sub(mvn.sample({50000}), tensor({1, -1}, {2}));
    CHECK(near(div_scalar(matmul(transpose(draws, 0, 1), draws), 50000.0f), cov, 0.06f));
    return "";
}

// ---------------------------------------------------------------------------------------------
// Layers and optimizers
// ---------------------------------------------------------------------------------------------

static std::string layer_batch_norm() {
    BatchNorm bn(2);
    auto x = tensor({1, 10, 3, 20, 5, 60}, {3, 2});
    auto y = bn.forward(x);
    CHECK(near(mean(y, 0, false), {0, 0}, 1e-5f));
    CHECK(near(bn.running_mean, {0.3f, 3.0f}, 1e-5f));
    CHECK(near(bn.running_var, {0.9f + 0.1f * 4, 0.9f + 0.1f * 700}, 1e-3f));
    bn.eval();
    auto e = bn.forward(x);
    CHECK(near({e->to_vector()[0]}, {(1 - 0.3f) / std::sqrt(1.3f + 1e-5f)}, 1e-4f));
    if (bn.state_dict().size() != 4) return "running statistics are not saved";
    return "";
}
static std::string layer_sequential() {
    manual_seed(2);
    auto net = std::make_shared<Sequential>();
    net->add<Conv2d>(1, 4, 3, 1, 1).add<BatchNorm2d>(4).add<ReLU>().add<MaxPool2d>(2).add<Flatten>().add<Dropout>(0.1f).add<Linear>(4 * 4 * 4, 3);
    net->add([](const TensorPtr& t) { return mul_scalar(t, 1.0f); });
    auto y = net->forward(input({2, 1, 8, 8}));
    CHECK(shape_is(y, {2, 3}));
    if (net->parameters().size() != 6) return "parameters: " + std::to_string(net->parameters().size());
    net->eval();
    auto x = input({2, 1, 8, 8});
    CHECK(near(net->forward(x), net->forward(x), 0, 0));
    sum(net->forward(x))->backward();
    for (auto& [name, t] : net->named_parameters()) if (!t->grad) return "no gradient for " + name;
    return "";
}
static std::string layer_recurrent() {
    manual_seed(4);
    LSTM lstm(3, 4, {2, 0.0f, true});
    auto x = input({2, 5, 3});
    auto out = lstm.forward(x);
    CHECK(shape_is(out.output, {2, 5, 8}));
    CHECK(shape_is(out.h, {4, 2, 4}));
    CHECK(shape_is(out.c, {4, 2, 4}));
    // The module against stepping its cell by hand
    LSTM one(3, 4);
    auto got = one.forward(x);
    LSTMCell::State s;
    for (int64_t t = 0; t < 5; ++t) s = one.cells[0]->forward(contiguous(select(x, 1, t)), s);
    CHECK(near(contiguous(select(got.h, 0, 0)), s.first, 1e-6f));
    CHECK(near(contiguous(select(got.output, 1, 4)), s.first, 1e-6f));
    GRU gru(3, 4);
    RNN rnn(3, 4);
    sum(add(gru.forward(x).output, rnn.forward(x).output))->backward();
    for (auto& p : gru.parameters()) if (!p->grad) return "GRU parameter without gradient";
    auto err = dummygrad::gradcheck([&](const TensorList& in) { return one.forward(in[0]).output; }, {x});
    if (!err.empty()) return "LSTM " + err;
    return dummygrad::gradcheck([&](const TensorList& in) { return gru.forward(in[0]).output; }, {x});
}
static std::string layer_transformer() {
    manual_seed(6);
    TransformerOptions o;
    o.d_model = 8;
    o.n_heads = 2;
    o.d_ff = 16;
    o.encoder_layers = o.decoder_layers = 1;
    o.dropout = 0.0f;
    Transformer model(o);
    auto src = input({2, 4, 8}), tgt = input({2, 3, 8});
    auto y = model.forward(src, tgt);
    CHECK(shape_is(y, {2, 3, 8}));
    // Causal: changing the last target step leaves earlier outputs alone
    auto changed = tgt->clone();
    for (int64_t i = 0; i < 8; ++i) changed->data_ptr<float>()[2 * 8 + i] += 1.0f;
    auto y2 = model.forward(src, changed);
    CHECK(near(contiguous(narrow(y2, 1, 0, 2)), contiguous(narrow(y, 1, 0, 2)), 1e-6f));
    // A padded key has no influence
    CrossAttention attn(8, 2);
    auto padding = tensor({0, 0, 1, 0, 0, 1}, {2, 3});
    auto base = attn.forward(src, tgt, nullptr, padding);
    CHECK(near(attn.forward(src, changed, nullptr, padding), base, 1e-6f));
    sum(y)->backward();
    for (auto& [name, p] : model.named_parameters()) if (!p->grad) return "no gradient for " + name;
    return "";
}
static std::string layer_misc() {
    manual_seed(8);
    Bilinear bl(3, 4, 2);
    CHECK(shape_is(bl.forward(input({5, 3}), input({5, 4})), {5, 2}));
    EmbeddingBag bag(6, 3);
    CHECK(shape_is(bag.forward(tensor({0, 1, 2, 3}, {2, 2})), {2, 3}));
    SpectralNorm sn(5, 30);
    auto w = input({5, 4});
    CHECK(scalar(amax(linalg::svdvals(sn.forward(w))), 1.0f, 1e-3f));
    auto big = std::make_shared<Tensor>(std::vector<int64_t>{300, 200}, true);
    init::kaiming_normal_(big);
    CHECK(scalar(std_dev(big), std::sqrt(2.0f / 200), 2e-3f));
    init::trunc_normal_(big);
    if (amax(big)->item() > 2.0f || amin(big)->item() < -2.0f) return "trunc_normal_ bounds";
    CHECK(shape_is(Conv1d(2, 4, 3).forward(input({1, 2, 8})), {1, 4, 6}));
    CHECK(shape_is(Conv3d(2, 4, 3, 1, 1).forward(input({1, 2, 4, 4, 4})), {1, 4, 4, 4, 4}));
    CHECK(shape_is(ConvTranspose2d(2, 3, 4, 2, 1).forward(input({1, 2, 5, 5})), {1, 3, 10, 10}));
    CHECK(shape_is(Upsample(2.0).forward(input({1, 2, 3, 3})), {1, 2, 6, 6}));
    CHECK(shape_is(AdaptiveAvgPool({1}).forward(input({1, 2, 7, 7})), {1, 2, 1, 1}));
    return "";
}

// Every optimizer on a quadratic, and the first steps of some against hand computations
template <typename Opt>
static std::string converges(Opt& opt, int steps, float tol, const std::string& name, bool matrix = false) {
    auto target = matrix ? tensor({1, -2, 0.5f, 3}, {2, 2}) : tensor({1, -2, 0.5f, 3}, {4});
    auto x = matrix ? tensor({0, 0, 0.1f, 0}, {2, 2}, Device::CPU, true) : tensor({0, 0, 0.1f, 0}, {4}, Device::CPU, true);
    float loss = 0.0f;
    for (int i = 0; i < steps; ++i) {
        opt.zero_grad({x});
        auto l = sum(mul(square(sub(x, target)), matrix ? tensor({1, 2, 3, 0.5f}, {2, 2}) : tensor({1, 2, 3, 0.5f}, {4})));
        l->backward();
        opt.step({x});
        loss = l->item();
    }
    return loss < tol ? "" : name + " ended at loss " + std::to_string(loss);
}
static std::string optimizers() {
    {
        auto x = tensor({1}, {1}, Device::CPU, true);
        optim::SGD sgd(0.1f, 0.9f, 0.0f, true);
        for (int i = 0; i < 2; ++i) {
            sgd.zero_grad({x});
            sum(square(x))->backward();
            sgd.step({x});
        }
        // g=2, buf=2, step 0.1*(2+1.8)=0.38 -> 0.62; g=1.24, buf=1.8+1.24=3.04, step 0.1*(1.24+2.736)
        CHECK(near(x, {0.62f - 0.1f * (1.24f + 0.9f * 3.04f)}, 1e-5f));
        auto y = tensor({1}, {1}, Device::CPU, true);
        optim::Adam adam(0.1f);
        sum(square(y))->backward();
        adam.step({y});
        CHECK(near(y, {0.9f}, 1e-5f));
    }
    { optim::SGD o(0.05f, 0.9f); CHECK(converges(o, 300, 1e-4f, "SGD")); }
    { optim::Adam o(0.1f); CHECK(converges(o, 500, 1e-3f, "Adam")); }
    { optim::Adam o(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, true); CHECK(converges(o, 500, 1e-2f, "AMSGrad")); }
    { optim::AdamW o(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f); CHECK(converges(o, 500, 1e-3f, "AdamW")); }
    { optim::NAdam o(0.1f); CHECK(converges(o, 500, 1e-3f, "NAdam")); }
    { optim::RAdam o(0.1f); CHECK(converges(o, 800, 1e-2f, "RAdam")); }
    { optim::Adamax o(0.1f); CHECK(converges(o, 500, 1e-2f, "Adamax")); }
    { optim::Adagrad o(1.0f); CHECK(converges(o, 300, 1e-3f, "Adagrad")); }
    { optim::Adadelta o(30.0f); CHECK(converges(o, 1500, 1e-1f, "Adadelta")); }
    { optim::RMSprop o(0.02f); CHECK(converges(o, 800, 1e-2f, "RMSprop")); }
    { optim::RMSprop o(0.01f, 0.9f, 1e-8f, 0.0f, 0.5f, true); CHECK(converges(o, 1000, 5e-2f, "centered RMSprop")); }
    { optim::Rprop o(0.05f); CHECK(converges(o, 100, 1e-4f, "Rprop")); }
    { optim::Lion o(0.01f); CHECK(converges(o, 600, 1e-2f, "Lion")); }
    { optim::Adafactor o(0.05f); CHECK(converges(o, 800, 5e-2f, "Adafactor", true)); }
    { optim::Muon o(0.02f); CHECK(converges(o, 800, 5e-2f, "Muon", true)); }
    {
        // Rosenbrock by LBFGS
        auto x = tensor({-1.2f, 1.0f}, {2}, Device::CPU, true);
        optim::LBFGS opt({x}, 1.0f, 60);
        auto closure = [&] {
            zero_grad({x});
            auto a = select(x, 0, 0), b = select(x, 0, 1);
            auto l = add(square(rsub_scalar(1.0f, a)), mul_scalar(square(sub(b, square(a))), 100.0f));
            l->backward();
            return l->item();
        };
        for (int i = 0; i < 3; ++i) opt.step(closure);
        CHECK(near(x, {1, 1}, 1e-2f));
    }
    {
        // State survives a round trip: the resumed run equals the uninterrupted one
        auto run_steps = [](optim::NAdam& o, const TensorPtr& x, int n) {
            for (int i = 0; i < n; ++i) {
                o.zero_grad({x});
                sum(square(square(x)))->backward();
                o.step({x});
            }
        };
        auto x = tensor({1, -2}, {2}, Device::CPU, true);
        optim::NAdam o(0.05f);
        run_steps(o, x, 3);
        auto y = x->clone()->detach();
        y->requires_grad = true;
        std::unordered_map<std::string, TensorPtr> saved;
        for (auto& [k, v] : o.state_dict({{"x", x}})) saved[k] = v->clone();
        optim::NAdam o2(0.05f);
        o2.load_state_dict({{"x", y}}, saved);
        run_steps(o, x, 3);
        run_steps(o2, y, 3);
        CHECK(near(y, x, 0, 0));
    }
    {
        auto a = tensor({1}, {1}, Device::CPU, true), b = tensor({1}, {1}, Device::CPU, true);
        optim::SGD o(0.1f);
        o.set_lr_scale({b}, 0.5f);
        sum(add(a, b))->backward();
        o.step({a, b});
        CHECK(near(a, {0.9f}));
        CHECK(near(b, {0.95f}));
    }
    return "";
}
static std::string schedules() {
    optim::StepLR st(1.0f, 2, 0.5f);
    CHECK(near({st.lr_at(0), st.lr_at(2), st.lr_at(5)}, {1, 0.5f, 0.25f}, 1e-6f));
    optim::MultiStepLR ms(1.0f, {2, 4}, 0.1f);
    CHECK(near({ms.lr_at(1), ms.lr_at(2), ms.lr_at(4)}, {1, 0.1f, 0.01f}, 1e-6f));
    optim::CosineAnnealingWarmRestarts cw(1.0f, 4, 2);
    CHECK(near({cw.lr_at(0), cw.lr_at(2), cw.lr_at(4), cw.lr_at(8)}, {1, 0.5f, 1, 0.5f}, 1e-5f));
    optim::OneCycleLR oc(1.0f, 100);
    CHECK(near({oc.lr_at(0), oc.lr_at(29), oc.lr_at(99)}, {0.04f, 1.0f, 4e-6f}, 1e-5f));
    optim::SequentialLR seq({std::make_shared<optim::LinearLR>(1.0f, 10, 0.0f, 1.0f), std::make_shared<optim::ExponentialLR>(1.0f, 0.5f)}, {10});
    CHECK(near({seq.lr_at(5), seq.lr_at(10), seq.lr_at(12)}, {0.5f, 1.0f, 0.25f}, 1e-6f));
    optim::ReduceLROnPlateau pl(1.0f, 0.1f, 1);
    pl.step(1.0f);
    pl.step(1.0f);
    CHECK(near({pl.step(1.0f)}, {0.1f}, 1e-6f));
    optim::CyclicLR cy(0.0f, 1.0f, 2, 2);
    CHECK(near({cy.lr_at(0), cy.lr_at(1), cy.lr_at(2), cy.lr_at(3)}, {0, 0.5f, 1, 0.5f}, 1e-6f));
    return "";
}
static std::string averaging() {
    Linear m(2, 2);
    m.W->fill_(1.0f);
    optim::EMA ema(m, 0.5f);
    m.W->fill_(3.0f);
    ema.update();
    CHECK(near(ema.averages()[0], {2, 2, 2, 2}));
    ema.swap();
    CHECK(near(m.W, {2, 2, 2, 2}));
    ema.swap();
    CHECK(near(m.W, {3, 3, 3, 3}));
    optim::SWA swa(m);
    swa.update();
    m.W->fill_(5.0f);
    swa.update();
    CHECK(near(swa.averages()[0], {4, 4, 4, 4}));
    return "";
}
static std::string create_graph_model() {
    // Second derivatives through a whole transformer block: the gradient penalty's gradient by finite differences
    manual_seed(12);
    ModernTransformerBlock block(8, 2, 1, 16);
    auto f = precompute_freqs_cis(4, 8);
    auto x = input({1, 3, 8});
    auto penalty = [&](const TensorPtr& in) {
        auto y = sum(square(block.forward(in, nullptr, f.first, f.second)));
        auto g = autograd::grad({y}, {in}, {}, true, true)[0];
        return sum(square(g));
    };
    dummygrad::GradcheckOptions o;
    o.eps = 5e-3f;
    o.atol = 5e-2f;
    o.rtol = 5e-2f;
    o.max_checks = 10;
    return dummygrad::gradcheck([&](const TensorList& in) { return penalty(leaf_of(in[0])); }, {x}, o);
}
static std::string gemm_shapes() {
    // Sizes around the blocking boundaries of the built-in matrix multiply, against the definition
    for (auto dims : std::vector<std::vector<int64_t>>{{1, 1, 1}, {5, 7, 3}, {6, 16, 16}, {13, 33, 17}, {125, 40, 260}, {64, 513, 31}, {300, 3, 300}}) {
        int64_t m = dims[0], k = dims[1], n = dims[2];
        auto a = input({m, k}), b = input({k, n});
        auto av = a->to_vector(), bv = b->to_vector(), got = matmul(a, b)->to_vector();
        auto nt = matmul_nt(a, contiguous(transpose(b, 0, 1)))->to_vector();
        std::vector<float> want((size_t)(m * n));
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double s = 0;
                for (int64_t p = 0; p < k; ++p) s += (double)av[(size_t)(i * k + p)] * bv[(size_t)(p * n + j)];
                want[(size_t)(i * n + j)] = (float)s;
            }
        CHECK(near(got, want, 1e-4f, 1e-4f));
        CHECK(near(nt, want, 1e-4f, 1e-4f));
    }
    return "";
}
static std::string parallel_behavior() {
    int threads = get_num_threads();
    set_num_threads(4);
    std::vector<int> hits(100000, 0);
    parallel_for(0, (int64_t)hits.size(), 100, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) hits[(size_t)i]++;
    });
    for (int h : hits) if (h != 1) return "parallel_for visited an index " + std::to_string(h) + " times";
    bool threw = false;
    try {
        parallel_for(0, 1000, 1, [](int64_t b, int64_t) { if (b > 500) throw std::runtime_error("boom"); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) return "an exception in a parallel body was lost";
    std::vector<int> nested(1000, 0);
    parallel_for(0, 10, 1, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) parallel_for(0, 100, 1, [&](int64_t b2, int64_t e2) { for (int64_t j = b2; j < e2; ++j) nested[(size_t)(i * 100 + j)]++; });
    });
    for (int h : nested) if (h != 1) return "nested parallel_for";
    set_num_threads(threads);
    return "";
}

int main(int argc, char** argv) {
    if (argc > 1) g_filter = argv[1];
    manual_seed(0);
    bool cuda = has_cuda();
    std::cout << "dummygrad research test suite (CUDA " << (cuda ? "available" : "not available") << ")\n";
    build_ops();
    std::cout << g_ops.size() << " differentiable ops\n";
    for (const auto& t : g_ops) run("grad/" + t.name, [&] { return first_order(t); });
    for (const auto& t : g_ops) if (t.second_order) run("grad2/" + t.name, [&] { return second_order(t); });
    if (cuda) for (const auto& t : g_ops) run("cuda/" + t.name, [&] { return cuda_parity(t); });

    run("autograd/api", autograd_api);
    run("autograd/hooks", autograd_hooks);
    run("autograd/functional", autograd_functional);
    run("autograd/inplace_detected", autograd_inplace_detected);
    run("autograd/checkpoint_matches", checkpoint_matches);
    run("autograd/create_graph_model", create_graph_model);
    run("values/shapes", values_shapes);
    run("values/reductions", values_reductions);
    run("values/indexing", values_indexing);
    run("values/math", values_math);
    run("values/einsum", values_einsum);
    run("values/conv", values_conv);
    run("values/ctc", values_ctc);
    run("values/losses", values_losses);
    run("values/random", values_random);
    run("values/linalg", values_linalg);
    run("values/fft", values_fft);
    run("values/distributions", values_distributions);
    run("values/gemm_shapes", gemm_shapes);
    run("values/parallel", parallel_behavior);
    run("layers/batch_norm", layer_batch_norm);
    run("layers/sequential", layer_sequential);
    run("layers/recurrent", layer_recurrent);
    run("layers/transformer", layer_transformer);
    run("layers/misc", layer_misc);
    run("optim/optimizers", optimizers);
    run("optim/schedules", schedules);
    run("optim/averaging", averaging);

    std::cout << "\n" << g_passed << " passed, " << g_failures.size() << " failed\n";
    for (const auto& f : g_failures) std::cout << "  FAILED " << f << "\n";
    return g_failures.empty() ? 0 : 1;
}
