// dummygrad engine test suite
//
//   1. Gradient checks (CPU): analytic gradients vs. central finite differences.
//   2. Device parity (CUDA): forward outputs and gradients vs. the CPU implementation.
//      Finite differences are not used on the GPU because TF32 matmuls are too noisy for them.
//   3. Engine tests: exception safety, concurrency, deep graphs, retain_graph.
//   4. End-to-end: a tiny ModernTransformer must give every parameter a gradient and learn.
//
// Exit code is 0 only if every test passes.

#include "dummy_core.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using TensorPtr = std::shared_ptr<Tensor>;
using Leaves = std::vector<TensorPtr>;
// Builds the graph under test from its leaves. Device is passed so constants (targets,
// RoPE tables) can be uploaded next to the leaves.
using ForwardFn = std::function<TensorPtr(const Leaves&, Device)>;

// ---------------------------------------------------------------------------
// Host <-> tensor helpers
// ---------------------------------------------------------------------------

static std::mt19937 g_test_rng(1234);

static int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// Uniform values in [lo, hi], optionally kept at least min_abs away from zero
// (keeps relu-style kinks out of finite-difference range).
static std::vector<float> random_values(int64_t n, float lo, float hi, float min_abs = 0.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) {
        do { x = dist(g_test_rng); } while (std::fabs(x) < min_abs);
    }
    return v;
}

static TensorPtr upload(const std::vector<float>& data, const std::vector<int64_t>& shape, Device dev, bool req_grad) {
    auto t = std::make_shared<Tensor>(shape, Device::CPU, DType::Float32, false);
    t->copy_from_vector(data);
    if (dev != Device::CPU) t = t->to(dev);
    t->requires_grad = req_grad;
    return t;
}

static std::vector<float> download(const TensorPtr& t) {
    auto c = make_contiguous(t);
    auto h = (c->device == Device::CPU) ? c : c->to(Device::CPU);
    const float* p = h->data_ptr<float>();
    return std::vector<float>(p, p + h->size());
}

static const char* device_name(Device d) {
    switch (d) {
        case Device::CPU: return "cpu";
        case Device::CUDA: return "cuda";
        default: return "other";
    }
}

static bool cuda_available() {
#ifdef USE_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Gradient checking
// ---------------------------------------------------------------------------

struct LeafSpec {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

static LeafSpec leaf(std::vector<int64_t> shape, float lo = -1.0f, float hi = 1.0f, float min_abs = 0.0f) {
    LeafSpec s;
    s.shape = std::move(shape);
    s.data = random_values(numel(s.shape), lo, hi, min_abs);
    return s;
}

// Shuffled, evenly spaced values: neighbours differ by more than the finite-difference step,
// so argmax-style ops (max_pool2d) keep the same winner under perturbation.
static LeafSpec distinct_leaf(std::vector<int64_t> shape) {
    LeafSpec s;
    s.shape = std::move(shape);
    int64_t n = numel(s.shape);
    s.data.resize(n);
    for (int64_t i = 0; i < n; ++i) s.data[i] = -1.0f + 2.0f * (float)i / (float)std::max<int64_t>(1, n - 1);
    std::shuffle(s.data.begin(), s.data.end(), g_test_rng);
    return s;
}

struct GradTest {
    std::string name;
    std::vector<LeafSpec> leaves;
    ForwardFn fn;
};

// Non-scalar outputs are reduced with fixed random weights, so every output element
// contributes a distinct amount (a plain sum would hide permutation bugs).
static TensorPtr reduce_to_scalar(const TensorPtr& out, const std::vector<float>& probe, Device dev) {
    if (out->size() == 1) return out;
    auto w = upload(probe, out->shape, dev, false);
    auto prod = mul(out, w);
    return sum(reshape(prod, {1, prod->size()}), 1);
}

struct Eval {
    std::vector<float> out;
    float loss = 0.0f;
    std::vector<std::vector<float>> grads;
    std::vector<bool> has_grad;
};

static Eval evaluate(const GradTest& t, const std::vector<LeafSpec>& specs, const std::vector<float>& probe,
                     Device dev, bool with_grad) {
    Leaves leaves;
    for (const auto& s : specs) leaves.push_back(upload(s.data, s.shape, dev, with_grad));
    auto out = t.fn(leaves, dev);
    auto loss = reduce_to_scalar(out, probe, dev);

    Eval e;
    e.out = download(out);
    e.loss = download(loss)[0];
    if (with_grad) {
        loss->backward();
        for (size_t i = 0; i < leaves.size(); ++i) {
            if (leaves[i]->grad) {
                e.grads.push_back(download(leaves[i]->grad));
                e.has_grad.push_back(true);
            } else {
                e.grads.emplace_back(leaves[i]->size(), 0.0f);
                e.has_grad.push_back(false);
            }
        }
    }
    return e;
}

static std::vector<float> make_probe(const GradTest& t) {
    Leaves leaves;
    for (const auto& s : t.leaves) leaves.push_back(upload(s.data, s.shape, Device::CPU, false));
    return random_values(t.fn(leaves, Device::CPU)->size(), -1.0f, 1.0f);
}

// Returns an empty string on success, otherwise a description of the first mismatch.
static std::string gradcheck_cpu(const GradTest& t, const std::vector<float>& probe) {
    const float h = 1e-2f;
    const int max_checks = 24;  // sampled elements per leaf keeps big layers fast

    auto analytic = evaluate(t, t.leaves, probe, Device::CPU, true);

    for (size_t i = 0; i < t.leaves.size(); ++i) {
        int64_t n = (int64_t)t.leaves[i].data.size();
        std::vector<int64_t> idx(n);
        for (int64_t k = 0; k < n; ++k) idx[k] = k;
        if (n > max_checks) {
            std::mt19937 pick(77 + (unsigned)i);
            std::shuffle(idx.begin(), idx.end(), pick);
            idx.resize(max_checks);
        }

        float max_numeric = 0.0f;
        for (int64_t k : idx) {
            auto plus = t.leaves;
            auto minus = t.leaves;
            plus[i].data[k] += h;
            minus[i].data[k] -= h;
            float f_plus = evaluate(t, plus, probe, Device::CPU, false).loss;
            float f_minus = evaluate(t, minus, probe, Device::CPU, false).loss;
            float num = (f_plus - f_minus) / (2.0f * h);
            float ana = analytic.grads[i][k];
            max_numeric = std::max(max_numeric, std::fabs(num));

            float tol = 5e-3f + 2e-2f * std::max(std::fabs(num), std::fabs(ana));
            if (std::fabs(num - ana) > tol) {
                std::ostringstream os;
                os << "leaf " << i << " element " << k << ": analytic " << ana << " vs numeric " << num;
                if (!analytic.has_grad[i]) os << " (no gradient reached this leaf)";
                return os.str();
            }
        }
        if (!analytic.has_grad[i] && max_numeric > 1e-3f) {
            return "leaf " + std::to_string(i) + " received no gradient";
        }
    }
    return "";
}

static std::string compare(const std::vector<float>& ref, const std::vector<float>& got, float rtol) {
    if (ref.size() != got.size()) {
        return "size " + std::to_string(got.size()) + " vs expected " + std::to_string(ref.size());
    }
    // Absolute floor: some gradients are exactly zero in theory (e.g. an attention key bias
    // without RoPE, which softmax's shift invariance cancels) and both sides are rounding noise.
    const float atol = 1e-5f;
    float scale = 0.0f;
    for (float v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t k = 0; k < ref.size(); ++k) {
        if (!std::isfinite(got[k]) || std::fabs(ref[k] - got[k]) > atol + rtol * scale) {
            std::ostringstream os;
            os << "element " << k << ": " << got[k] << " vs expected " << ref[k] << " (scale " << scale << ")";
            return os.str();
        }
    }
    return "";
}

static std::string parity_check(const GradTest& t, const std::vector<float>& probe, Device dev) {
    const float rtol = 2e-2f;  // TF32 tensor-core matmuls
    auto ref = evaluate(t, t.leaves, probe, Device::CPU, true);
    auto got = evaluate(t, t.leaves, probe, dev, true);

    std::string err = compare(ref.out, got.out, rtol);
    if (!err.empty()) return "forward output " + err;
    for (size_t i = 0; i < t.leaves.size(); ++i) {
        if (ref.has_grad[i] && !got.has_grad[i]) return "leaf " + std::to_string(i) + " received no gradient";
        err = compare(ref.grads[i], got.grads[i], rtol);
        if (!err.empty()) return "grad of leaf " + std::to_string(i) + " " + err;
    }
    return "";
}

// ---------------------------------------------------------------------------
// Gradient test catalogue
// ---------------------------------------------------------------------------

using UnaryFn = std::function<TensorPtr(const TensorPtr&)>;

// Single-input ops exercised with strided (transposed) inputs, by both the gradient checks
// and the forward-equivalence test.
static std::vector<std::pair<std::string, UnaryFn>> strided_unary_ops() {
    return {
        {"relu", [](const TensorPtr& t) { return relu(t); }},
        {"leaky_relu", [](const TensorPtr& t) { return leaky_relu(t, 0.1f); }},
        {"tanh", [](const TensorPtr& t) { return tanh(t); }},
        {"sigmoid", [](const TensorPtr& t) { return sigmoid(t); }},
        {"gelu", [](const TensorPtr& t) { return gelu(t); }},
        {"silu", [](const TensorPtr& t) { return silu(t); }},
        {"softmax", [](const TensorPtr& t) { return softmax(t); }},
        {"sum_last", [](const TensorPtr& t) { return sum(t, -1); }},
        {"sum_rows", [](const TensorPtr& t) { return sum(t, -2); }},
        {"mean_axis0", [](const TensorPtr& t) { return mean(t, 0); }},
        {"mean_axis1", [](const TensorPtr& t) { return mean(t, 1); }},
        {"var_axis1", [](const TensorPtr& t) { return var(t, 1); }},
        {"std_dev_axis0", [](const TensorPtr& t) { return std_dev(t, 0); }},
        {"scalar_ops", [](const TensorPtr& t) { return add_scalar(mul_scalar(t, 1.5f), 0.25f); }},
    };
}

static std::vector<GradTest> build_grad_tests() {
    std::vector<GradTest> tests;
    auto add_test = [&](std::string name, std::vector<LeafSpec> leaves, ForwardFn fn) {
        tests.push_back({std::move(name), std::move(leaves), std::move(fn)});
    };

    // --- Elementwise ---
    add_test("relu", {leaf({3, 4}, -1, 1, 0.1f)}, [](const Leaves& L, Device) { return relu(L[0]); });
    add_test("leaky_relu", {leaf({3, 4}, -1, 1, 0.1f)}, [](const Leaves& L, Device) { return leaky_relu(L[0], 0.1f); });
    add_test("tanh", {leaf({3, 4})}, [](const Leaves& L, Device) { return tanh(L[0]); });
    add_test("sigmoid", {leaf({3, 4})}, [](const Leaves& L, Device) { return sigmoid(L[0]); });
    add_test("gelu", {leaf({3, 4})}, [](const Leaves& L, Device) { return gelu(L[0]); });
    add_test("silu", {leaf({3, 4})}, [](const Leaves& L, Device) { return silu(L[0]); });

    // --- Scalar ops ---
    add_test("scalar_ops", {leaf({3, 4}, 0.5f, 1.5f)}, [](const Leaves& L, Device) {
        auto a = add_scalar(mul_scalar(L[0], 1.5f), 0.25f);
        auto b = div_scalar(sub_scalar(a, 0.1f), 2.0f);
        auto c = rsub_scalar(3.0f, b);
        return add(neg(c), rdiv_scalar(1.0f, L[0]));
    });

    // --- Binary ops & broadcasting ---
    add_test("add_same_shape", {leaf({2, 3, 4}), leaf({2, 3, 4})},
             [](const Leaves& L, Device) { return add(L[0], L[1]); });
    add_test("add_bias_broadcast", {leaf({2, 3, 4}), leaf({1, 4})},
             [](const Leaves& L, Device) { return cast_n_add(L[0], L[1]); });
    add_test("sub_broadcast", {leaf({2, 3, 4}), leaf({1, 4})},
             [](const Leaves& L, Device) { return cast_n_sub(L[0], L[1]); });
    add_test("mul_same_shape", {leaf({2, 3, 4}), leaf({2, 3, 4})},
             [](const Leaves& L, Device) { return mul(L[0], L[1]); });
    add_test("div_broadcast", {leaf({2, 3, 4}), leaf({1, 4}, 0.5f, 1.5f)},
             [](const Leaves& L, Device) { return cast_n_div(L[0], L[1]); });
    // GQA-style grouped broadcast: 8 flattened x batches share 4 y batches in pairs
    add_test("mul_grouped_broadcast", {leaf({2, 4, 3, 2}), leaf({2, 2, 3, 2})},
             [](const Leaves& L, Device) { return mul(L[0], L[1]); });

    // --- Reductions & softmax ---
    add_test("sum_last", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return sum(L[0], -1); });
    add_test("sum_rows", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return sum(L[0], -2); });
    add_test("mean_axis0", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return mean(L[0], 0); });
    add_test("mean_axis1", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return mean(L[0], 1); });
    add_test("softmax", {leaf({2, 3, 5}, -2, 2)}, [](const Leaves& L, Device) { return softmax(L[0]); });
    // Wide rows exercise the multi-warp block reductions in the CUDA kernels
    add_test("softmax_wide", {leaf({2, 300}, -2, 2)}, [](const Leaves& L, Device) { return softmax(L[0]); });

    // --- Matmul, views & strided gradients ---
    add_test("matmul_2d", {leaf({3, 4}), leaf({4, 5})},
             [](const Leaves& L, Device) { return matmul(L[0], L[1]); });
    add_test("matmul_broadcast_weight", {leaf({2, 3, 4}), leaf({4, 5})},
             [](const Leaves& L, Device) { return matmul(L[0], L[1]); });
    add_test("matmul_batched", {leaf({2, 3, 4}), leaf({2, 4, 5})},
             [](const Leaves& L, Device) { return matmul(L[0], L[1]); });
    add_test("matmul_transposed_input", {leaf({2, 4, 3}), leaf({2, 4, 5})},
             [](const Leaves& L, Device) { return matmul(transpose(L[0], -2, -1), L[1]); });
    add_test("reshape_then_matmul", {leaf({2, 3, 4}), leaf({4, 2})},
             [](const Leaves& L, Device) { return matmul(reshape(L[0], {6, 4}), L[1]); });
    // Tied embedding: the weight is used directly and transposed, so it receives one
    // contiguous and one strided gradient that must be accumulated correctly.
    add_test("tied_weight", {leaf({6, 4}), leaf({5, 6})}, [](const Leaves& L, Device) {
        auto h = matmul(L[1], L[0]);                     // [5, 4]
        return matmul(h, transpose(L[0], 0, 1));         // [5, 6]
    });
    // Residual fan-out: AddBackward hands the same gradient tensor to several edges
    add_test("residual_fanout", {leaf({3, 4}, -1, 1, 0.1f)}, [](const Leaves& L, Device) {
        auto a = add(L[0], L[0]);
        auto b = add(a, relu(L[0]));
        return add(b, mul_scalar(L[0], 0.5f));
    });

    // --- Losses ---
    {
        std::vector<float> targets = {0, 3, 1, 4, 2, 2};
        add_test("fused_cross_entropy", {leaf({2, 3, 5}, -2, 2)}, [targets](const Leaves& L, Device dev) {
            return fused_cross_entropy(L[0], upload(targets, {2, 3}, dev, false));
        });
    }
    {
        std::vector<float> target = random_values(12, -1, 1);
        add_test("mse", {leaf({3, 4})}, [target](const Leaves& L, Device dev) {
            return mse(L[0], upload(target, {3, 4}, dev, false));
        });
    }

    // --- Layers (parameters are substituted with test leaves) ---
    {
        auto lin = std::make_shared<Linear>(4, 3);
        add_test("nn.Linear", {leaf({2, 5, 4}), leaf({4, 3}), leaf({1, 3})}, [lin](const Leaves& L, Device) {
            lin->W = L[1];
            lin->b = L[2];
            return lin->forward(L[0]);
        });
    }
    {
        auto ln = std::make_shared<LayerNorm>(6);
        add_test("nn.LayerNorm", {leaf({2, 3, 6}), leaf({6}, 0.5f, 1.5f), leaf({6})}, [ln](const Leaves& L, Device) {
            ln->gamma = L[1];
            ln->beta = L[2];
            return ln->forward(L[0]);
        });
    }
    {
        auto ln = std::make_shared<LayerNorm>(300);
        add_test("nn.LayerNorm_wide", {leaf({2, 300}), leaf({300}, 0.5f, 1.5f), leaf({300})}, [ln](const Leaves& L, Device) {
            ln->gamma = L[1];
            ln->beta = L[2];
            return ln->forward(L[0]);
        });
    }
    {
        auto rms = std::make_shared<RMSNorm>(6);
        add_test("nn.RMSNorm", {leaf({2, 3, 6}), leaf({6}, 0.5f, 1.5f)}, [rms](const Leaves& L, Device) {
            rms->gamma = L[1];
            return rms->forward(L[0]);
        });
    }
    {
        auto ffn = std::make_shared<SwiGLU>(4, 6);
        add_test("nn.SwiGLU",
                 {leaf({2, 3, 4}), leaf({4, 6}), leaf({1, 6}), leaf({4, 6}), leaf({1, 6}), leaf({6, 4}), leaf({1, 4})},
                 [ffn](const Leaves& L, Device) {
                     ffn->w_gate.W = L[1]; ffn->w_gate.b = L[2];
                     ffn->w_up.W = L[3];   ffn->w_up.b = L[4];
                     ffn->w_down.W = L[5]; ffn->w_down.b = L[6];
                     return ffn->forward(L[0]);
                 });
    }
    {
        auto freqs = precompute_freqs_cis(4, 8);  // head_dim 4 -> tables [8, 2]
        auto cos_h = download(freqs.first);
        auto sin_h = download(freqs.second);
        add_test("rope", {leaf({2, 3, 2, 4})}, [cos_h, sin_h](const Leaves& L, Device dev) {
            return apply_rotary_emb(L[0], upload(cos_h, {8, 2}, dev, false), upload(sin_h, {8, 2}, dev, false), 1);
        });
    }

    // Multi-head attention: d_model 8. Leaves: x, then (W, b) for q, k, v, o.
    auto attention_test = [&](const std::string& name, int n_heads, int n_kv_heads, bool use_rope) {
        auto mha = std::make_shared<MultiHeadAttention>(8, n_heads, n_kv_heads, true);
        int d_k = 8 / n_heads;
        int kv_dim = mha->n_kv_heads * d_k;
        auto freqs = precompute_freqs_cis(d_k, 8);
        auto cos_h = download(freqs.first);
        auto sin_h = download(freqs.second);
        int64_t half = d_k / 2;
        add_test(name,
                 {leaf({2, 3, 8}),
                  leaf({8, 8}, -0.5f, 0.5f), leaf({1, 8}, -0.5f, 0.5f),
                  leaf({8, kv_dim}, -0.5f, 0.5f), leaf({1, kv_dim}, -0.5f, 0.5f),
                  leaf({8, kv_dim}, -0.5f, 0.5f), leaf({1, kv_dim}, -0.5f, 0.5f),
                  leaf({8, 8}, -0.5f, 0.5f), leaf({1, 8}, -0.5f, 0.5f)},
                 [mha, use_rope, cos_h, sin_h, half](const Leaves& L, Device dev) {
                     mha->W_q.W = L[1]; mha->W_q.b = L[2];
                     mha->W_k.W = L[3]; mha->W_k.b = L[4];
                     mha->W_v.W = L[5]; mha->W_v.b = L[6];
                     mha->W_o.W = L[7]; mha->W_o.b = L[8];
                     if (!use_rope) return mha->forward(L[0]);
                     return mha->forward(L[0], nullptr,
                                         upload(cos_h, {8, half}, dev, false),
                                         upload(sin_h, {8, half}, dev, false), 0);
                 });
    };
    attention_test("nn.MultiHeadAttention", 2, 0, false);
    attention_test("nn.MultiHeadAttention_gqa_rope", 4, 2, true);

    // --- Shape ops ---
    add_test("concat_rows", {leaf({2, 3, 4}), leaf({2, 2, 4})},
             [](const Leaves& L, Device) { return concat(L[0], L[1], 0); });
    add_test("concat_cols", {leaf({2, 3, 4}), leaf({2, 3, 2})},
             [](const Leaves& L, Device) { return concat(L[0], L[1], 1); });
    add_test("broadcast_rows", {leaf({2, 1, 4})}, [](const Leaves& L, Device) { return broadcast(L[0], 0, 3); });
    add_test("broadcast_cols", {leaf({2, 3, 1})}, [](const Leaves& L, Device) { return broadcast(L[0], 1, 4); });
    // Slices are strided views; gradients must land in the right window of the input
    add_test("slice_dim0", {leaf({4, 3})}, [](const Leaves& L, Device) { return slice(L[0], 0, 1, 3); });
    add_test("slice_inner_dim", {leaf({2, 5, 3})}, [](const Leaves& L, Device) { return slice(L[0], 1, 1, 4); });
    add_test("slice_last_dim", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return slice(L[0], 2, 0, 2); });
    add_test("slice_negative_bounds", {leaf({2, 5})}, [](const Leaves& L, Device) { return slice(L[0], 1, -3, -1); });

    // --- Statistics ---
    add_test("var_axis0", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return var(L[0], 0); });
    add_test("var_axis1", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return var(L[0], 1); });
    add_test("std_dev_axis0", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return std_dev(L[0], 0); });
    add_test("std_dev_axis1", {leaf({2, 3, 4})}, [](const Leaves& L, Device) { return std_dev(L[0], 1); });

    // --- Masks (followed by softmax, as in attention; masked entries are -1e9) ---
    add_test("causal_mask", {leaf({2, 4, 4})}, [](const Leaves& L, Device) { return softmax(causal_mask(L[0])); });
    {
        std::vector<float> keep = {1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0};
        add_test("bool_mask", {leaf({2, 3, 4})}, [keep](const Leaves& L, Device dev) {
            return softmax(bool_mask(L[0], upload(keep, {2, 3, 4}, dev, false)));
        });
    }
    add_test("sliding_window_mask", {leaf({2, 5, 5})},
             [](const Leaves& L, Device) { return softmax(masking::sliding_window_mask(L[0], 2)); });
    add_test("prefix_causal_mask", {leaf({2, 5, 5})},
             [](const Leaves& L, Device) { return softmax(masking::prefix_causal_mask(L[0], 2)); });

    // --- More losses & affine ---
    {
        std::vector<float> target = random_values(12, 0.0f, 1.0f);
        add_test("cross_entropy", {leaf({3, 4}, 0.2f, 1.0f)}, [target](const Leaves& L, Device dev) {
            return cross_entropy(L[0], upload(target, {3, 4}, dev, false));
        });
    }
    {
        // Targets far from predictions keep |pred - target| away from the kink
        std::vector<float> target = random_values(12, 2.0f, 3.0f);
        add_test("l1_loss", {leaf({3, 4})}, [target](const Leaves& L, Device dev) {
            return l1_loss(L[0], upload(target, {3, 4}, dev, false));
        });
    }
    add_test("scale_n_shift", {leaf({2, 3, 4}), leaf({4}, 0.5f, 1.5f), leaf({4})},
             [](const Leaves& L, Device) { return scale_n_shift(L[0], L[1], L[2]); });

    // --- Convolution, pooling, embedding ---
    add_test("conv2d_padded", {leaf({2, 3, 5, 5}), leaf({4, 3, 3, 3}, -0.5f, 0.5f)},
             [](const Leaves& L, Device) { return conv2d(L[0], L[1], 1, 1); });
    add_test("conv2d_strided", {leaf({2, 2, 6, 5}), leaf({3, 2, 2, 3}, -0.5f, 0.5f)},
             [](const Leaves& L, Device) { return conv2d(L[0], L[1], 2, 1, 0, 1); });
    add_test("conv1d", {leaf({2, 3, 8}), leaf({4, 3, 3}, -0.5f, 0.5f)},
             [](const Leaves& L, Device) { return conv1d(L[0], L[1], 2, 1); });
    add_test("max_pool2d", {distinct_leaf({2, 2, 4, 4})},
             [](const Leaves& L, Device) { return max_pool2d(L[0], 2, 2, 2, 2); });
    {
        auto emb = std::make_shared<Embedding>(10, 4);
        std::vector<float> ids = {3, 0, 7, 3, 9, 1};
        add_test("nn.Embedding", {leaf({10, 4})}, [emb, ids](const Leaves& L, Device dev) {
            emb->weight = L[0];
            return emb->forward(upload(ids, {2, 3}, dev, false));
        });
    }

    // --- Strided inputs: every op must read transposed / sliced views correctly ---
    for (const auto& entry : strided_unary_ops()) {
        UnaryFn f = entry.second;
        add_test("strided/" + entry.first, {leaf({4, 3}, -1, 1, 0.1f)},
                 [f](const Leaves& L, Device) { return f(transpose(L[0], 0, 1)); });
    }
    add_test("strided/relu_of_slice", {leaf({3, 5}, -1, 1, 0.1f)},
             [](const Leaves& L, Device) { return relu(slice(L[0], 1, 1, 4)); });
    add_test("strided/add_same_shape", {leaf({4, 3}), leaf({3, 4})},
             [](const Leaves& L, Device) { return add(transpose(L[0], 0, 1), L[1]); });
    add_test("strided/add_bias_broadcast", {leaf({4, 3}), leaf({1, 4})},
             [](const Leaves& L, Device) { return cast_n_add(transpose(L[0], 0, 1), L[1]); });
    add_test("strided/mul_both", {leaf({4, 3}), leaf({4, 3})},
             [](const Leaves& L, Device) { return mul(transpose(L[0], 0, 1), transpose(L[1], 0, 1)); });
    add_test("strided/matmul_slice", {leaf({3, 5}), leaf({2, 4})},
             [](const Leaves& L, Device) { return matmul(slice(L[0], 1, 1, 3), L[1]); });
    add_test("strided/causal_mask", {leaf({2, 4, 4})},
             [](const Leaves& L, Device) { return softmax(causal_mask(transpose(L[0], -2, -1))); });
    {
        auto ln = std::make_shared<LayerNorm>(6);
        add_test("strided/nn.LayerNorm", {leaf({6, 2}), leaf({6}, 0.5f, 1.5f), leaf({6})}, [ln](const Leaves& L, Device) {
            ln->gamma = L[1];
            ln->beta = L[2];
            return ln->forward(transpose(L[0], 0, 1));
        });
    }
    {
        auto rms = std::make_shared<RMSNorm>(6);
        add_test("strided/nn.RMSNorm", {leaf({6, 2}), leaf({6}, 0.5f, 1.5f)}, [rms](const Leaves& L, Device) {
            rms->gamma = L[1];
            return rms->forward(transpose(L[0], 0, 1));
        });
    }
    {
        std::vector<float> targets = {0, 4, 2, 1, 3, 2};
        add_test("strided/fused_cross_entropy", {leaf({5, 6}, -2, 2)}, [targets](const Leaves& L, Device dev) {
            return fused_cross_entropy(transpose(L[0], 0, 1), upload(targets, {6}, dev, false));
        });
    }
    {
        std::vector<float> target = random_values(12, -1, 1);
        add_test("strided/mse", {leaf({4, 3})}, [target](const Leaves& L, Device dev) {
            return mse(transpose(L[0], 0, 1), upload(target, {3, 4}, dev, false));
        });
    }
    {
        auto freqs = precompute_freqs_cis(4, 8);
        auto cos_h = download(freqs.first);
        auto sin_h = download(freqs.second);
        // [B, H, S, D] memory viewed as [B, S, H, D], as attention produces before RoPE
        add_test("strided/rope", {leaf({2, 2, 3, 4})}, [cos_h, sin_h](const Leaves& L, Device dev) {
            return apply_rotary_emb(transpose(L[0], 1, 2), upload(cos_h, {8, 2}, dev, false),
                                    upload(sin_h, {8, 2}, dev, false), 0);
        });
    }

    return tests;
}

// ---------------------------------------------------------------------------
// Engine tests
// ---------------------------------------------------------------------------

static std::string expect_all(const TensorPtr& t, float expected, const std::string& what) {
    if (!t->grad) return what + ": no gradient";
    for (float v : download(t->grad)) {
        if (std::fabs(v - expected) > 1e-3f * std::max(1.0f, std::fabs(expected))) {
            return what + ": got " + std::to_string(v) + ", expected " + std::to_string(expected);
        }
    }
    return "";
}

static TensorPtr scalar_sum(const TensorPtr& t) { return sum(sum(t, -1), -2); }

// A gradient check only proves backward matches forward; this checks the forward itself:
// an op applied to a strided view must give exactly what it gives on a dense copy.
static std::string strided_forward_matches_dense() {
    auto x = upload(random_values(12, -1, 1, 0.1f), {4, 3}, Device::CPU, false);
    auto view = transpose(x, 0, 1);
    auto dense = make_contiguous(view);
    for (const auto& entry : strided_unary_ops()) {
        auto err = compare(download(entry.second(dense)), download(entry.second(view)), 1e-6f);
        if (!err.empty()) return entry.first + ": " + err;
    }
    auto sliced = slice(upload(random_values(15, -1, 1), {3, 5}, Device::CPU, false), 1, 1, 4);
    auto err = compare(download(relu(make_contiguous(sliced))), download(relu(sliced)), 1e-6f);
    return err.empty() ? "" : "relu of slice: " + err;
}

static std::string engine_retain_graph() {
    auto x = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    auto y = scalar_sum(mul_scalar(x, 3.0f));
    y->backward(true);
    y->backward(true);
    return expect_all(x, 6.0f, "two backward passes should accumulate");
}

struct ThrowingNode : public Node {
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>&) override {
        throw std::runtime_error("boom");
    }
};

static std::string engine_exception_propagates() {
    auto x = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    auto y = mul_scalar(x, 2.0f);
    auto bad = std::make_shared<ThrowingNode>();
    bad->add_next_edge(get_grad_edge(y).function);
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{1, 1}, Device::CPU, DType::Float32, true);
    out->fill_(0.0f);
    out->grad_fn = bad;

    bool threw = false;
    try {
        out->backward();
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "boom";
    }
    if (!threw) return "exception from a backward node was not propagated to backward()";

    // The engine must still be usable afterwards
    auto z = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    scalar_sum(mul_scalar(z, 5.0f))->backward();
    return expect_all(z, 5.0f, "backward after a failed backward");
}

// Many independent branches run concurrently on the worker pool and fan back into one leaf
static std::string engine_concurrent_fanin() {
    for (int iter = 0; iter < 25; ++iter) {
        auto x = upload({1.5f, -2.0f, 0.5f, 3.0f}, {2, 2}, Device::CPU, true);
        TensorPtr acc = mul_scalar(x, 1.0f);
        for (int k = 2; k <= 64; ++k) acc = add(acc, mul_scalar(x, (float)k));
        scalar_sum(acc)->backward();
        auto err = expect_all(x, 2080.0f, "iteration " + std::to_string(iter));  // 1 + 2 + ... + 64
        if (!err.empty()) return err;
    }
    return "";
}

static std::string engine_deep_chain() {
    auto x = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    TensorPtr y = x;
    for (int i = 0; i < 3000; ++i) y = add_scalar(y, 0.001f);
    scalar_sum(y)->backward();
    return expect_all(x, 1.0f, "3000-op chain");
}

// ---------------------------------------------------------------------------
// End-to-end training
// ---------------------------------------------------------------------------

static std::string train_tiny_transformer(Device dev) {
    manual_seed(42);
    const int V = 16, T = 8, B = 4;
    ModernTransformer model(V, 16, 32, 4, 2, 2);  // GQA: 4 query heads share 2 kv heads
    if (dev == Device::CUDA) model.cuda(); else model.cpu();
    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->device != dev) return "parameter " + std::to_string(i) + " is not on " + device_name(dev);
    }
    Adam opt(1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);

    // Next-token task: token i is always followed by token i + 1
    std::vector<float> in(B * T), tg(B * T);
    for (int i = 0; i < B * T; ++i) {
        in[i] = (float)(i % V);
        tg[i] = (float)((i + 1) % V);
    }

    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 60; ++step) {
        opt.zero_grad(params);
        auto inputs = upload(in, {B, T}, dev, false);
        auto targets = upload(tg, {B, T}, dev, false);
        auto logits = model.forward(inputs);
        if (logits->device != dev) return std::string("logits computed on the wrong device");
        auto loss = fused_cross_entropy(logits, targets);
        loss->backward();

        if (step == 0) {
            for (size_t i = 0; i < params.size(); ++i) {
                if (!params[i]->grad) return "parameter " + std::to_string(i) + " received no gradient";
                if (params[i]->grad->device != dev) return "gradient of parameter " + std::to_string(i) + " is not on " + device_name(dev);
                double norm = 0.0;
                for (float g : download(params[i]->grad)) norm += (double)g * g;
                if (!(norm > 0.0) || !std::isfinite(norm)) {
                    return "parameter " + std::to_string(i) + " has gradient norm " + std::to_string(norm);
                }
            }
        }
        opt.step(params);

        float l = download(loss)[0];
        if (!std::isfinite(l)) return "loss became " + std::to_string(l) + " at step " + std::to_string(step);
        if (step == 0) first = l;
        last = l;
    }
    if (!(last < 0.5f * first)) {
        return "loss did not decrease enough: " + std::to_string(first) + " -> " + std::to_string(last);
    }
    std::cout << "        loss " << first << " -> " << last << "\n";
    return "";
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

struct Results {
    int passed = 0;
    std::vector<std::string> failures;
};

static void run_case(Results& r, const std::string& name, const std::function<std::string()>& body) {
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    try {
        err = body();
    } catch (const std::exception& e) {
        err = std::string("threw: ") + e.what();
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (err.empty()) {
        r.passed++;
        std::cout << "[PASS] " << name << " (" << (int)ms << " ms)\n";
    } else {
        r.failures.push_back(name + ": " + err);
        std::cout << "[FAIL] " << name << ": " << err << "\n";
    }
}

int main() {
    manual_seed(0);
    Results r;
    bool has_cuda = cuda_available();
#ifndef DUMMYGRAD_VERSION
#define DUMMYGRAD_VERSION "dev"
#endif
    std::cout << "dummygrad " << DUMMYGRAD_VERSION << " test suite (CUDA "
              << (has_cuda ? "available" : "not available") << ")\n\n";

    auto tests = build_grad_tests();
    std::vector<std::vector<float>> probes;
    for (const auto& t : tests) probes.push_back(make_probe(t));

    std::cout << "-- gradient checks (cpu, finite differences)\n";
    for (size_t i = 0; i < tests.size(); ++i) {
        run_case(r, "grad/" + tests[i].name, [&] { return gradcheck_cpu(tests[i], probes[i]); });
    }

    if (has_cuda) {
        std::cout << "\n-- device parity (cuda vs cpu)\n";
        for (size_t i = 0; i < tests.size(); ++i) {
            run_case(r, "cuda/" + tests[i].name, [&] { return parity_check(tests[i], probes[i], Device::CUDA); });
        }
    }

    std::cout << "\n-- autograd engine\n";
    run_case(r, "engine/strided_forward_matches_dense", strided_forward_matches_dense);
    run_case(r, "engine/retain_graph", engine_retain_graph);
    run_case(r, "engine/exception_propagates", engine_exception_propagates);
    run_case(r, "engine/concurrent_fanin", engine_concurrent_fanin);
    run_case(r, "engine/deep_chain", engine_deep_chain);

    std::cout << "\n-- end-to-end training\n";
    run_case(r, "train/tiny_transformer_cpu", [] { return train_tiny_transformer(Device::CPU); });
    if (has_cuda) {
        run_case(r, std::string("train/tiny_transformer_") + device_name(Device::CUDA),
                 [] { return train_tiny_transformer(Device::CUDA); });
    }

    std::cout << "\n" << r.passed << " passed, " << r.failures.size() << " failed\n";
    for (const auto& f : r.failures) std::cout << "  FAILED " << f << "\n";
    return r.failures.empty() ? 0 : 1;
}
