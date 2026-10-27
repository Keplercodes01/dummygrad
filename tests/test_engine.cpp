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
#include <cstdio>
#include <chrono>
#include <cmath>
#include <functional>
#include <iterator>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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

// CPU gradient check through the public dummygrad::gradcheck (src/testing.h), so the suite
// also tests the tool users rely on for their own ops.
static std::string gradcheck_cpu(const GradTest& t) {
    Leaves inputs;
    for (const auto& s : t.leaves) inputs.push_back(upload(s.data, s.shape, Device::CPU, false));
    return dummygrad::gradcheck([&t](const Leaves& L) { return t.fn(L, Device::CPU); }, inputs);
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
    // Batch > 1 with per-head (ALiBi-style) and per-sequence (padding-mask-style) operands
    add_test("add_head_broadcast", {leaf({2, 3, 4, 4}), leaf({1, 3, 4, 4})},
             [](const Leaves& L, Device) { return add(L[0], L[1]); });
    add_test("add_mask_broadcast", {leaf({2, 3, 4, 4}), leaf({2, 1, 1, 4})},
             [](const Leaves& L, Device) { return add(L[0], L[1]); });
    add_test("mul_middle_broadcast", {leaf({2, 4, 3, 2}), leaf({1, 4, 1, 2})},
             [](const Leaves& L, Device) { return mul(L[0], L[1]); });
    add_test("div_scalar_tensor", {leaf({2, 3, 4}), leaf({1}, 0.5f, 1.5f)},
             [](const Leaves& L, Device) { return cast_n_div(L[0], L[1]); });
    add_test("repeat_kv", {leaf({2, 2, 3, 4})}, [](const Leaves& L, Device) { return repeat_kv(L[0], 2); });

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

    // Multi-head attention. Leaves: x, then (W, b) for q, k, v, o. Weights scale with 1/sqrt(d_model)
    // so the softmax stays smooth. On CUDA these short sequences take the GEMM path; main() runs them
    // again on the tiled kernels (head dims 16/32/64/128, with sequence lengths that cross their tile
    // and block boundaries) and the untiled ones (other head dims).
    auto attention_test = [&](const std::string& name, int d_model, int n_heads, int n_kv_heads, bool use_rope,
                              int batch, int seq) {
        auto mha = std::make_shared<MultiHeadAttention>(d_model, n_heads, n_kv_heads, true);
        int d_k = d_model / n_heads;
        int kv_dim = mha->n_kv_heads * d_k;
        auto freqs = precompute_freqs_cis(d_k, seq);
        auto cos_h = download(freqs.first);
        auto sin_h = download(freqs.second);
        int64_t half = d_k / 2;
        float w = 1.7f / std::sqrt((float)d_model);
        add_test(name,
                 {leaf({batch, seq, d_model}),
                  leaf({d_model, d_model}, -w, w), leaf({1, d_model}, -w, w),
                  leaf({d_model, kv_dim}, -w, w), leaf({1, kv_dim}, -w, w),
                  leaf({d_model, kv_dim}, -w, w), leaf({1, kv_dim}, -w, w),
                  leaf({d_model, d_model}, -w, w), leaf({1, d_model}, -w, w)},
                 [mha, use_rope, cos_h, sin_h, half, seq](const Leaves& L, Device dev) {
                     mha->W_q.W = L[1]; mha->W_q.b = L[2];
                     mha->W_k.W = L[3]; mha->W_k.b = L[4];
                     mha->W_v.W = L[5]; mha->W_v.b = L[6];
                     mha->W_o.W = L[7]; mha->W_o.b = L[8];
                     if (!use_rope) return mha->forward(L[0]);
                     return mha->forward(L[0], nullptr,
                                         upload(cos_h, {seq, half}, dev, false),
                                         upload(sin_h, {seq, half}, dev, false), 0);
                 });
    };
    attention_test("nn.MultiHeadAttention", 8, 2, 0, false, 2, 3);
    attention_test("nn.MultiHeadAttention_gqa_rope", 8, 4, 2, true, 2, 3);
    attention_test("attention_d16_two_blocks", 32, 2, 0, false, 1, 70);
    attention_test("attention_d32_gqa_rope", 128, 4, 2, true, 2, 37);
    attention_test("attention_d64", 128, 2, 0, false, 2, 20);
    attention_test("attention_d128_two_blocks", 256, 2, 0, false, 1, 40);

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

static std::string engine_no_grad() {
    auto w = upload({0.5f, -1.0f, 2.0f, 0.25f}, {2, 2}, Device::CPU, true);
    auto x = upload({1.0f, 2.0f, -3.0f, 4.0f}, {2, 2}, Device::CPU, false);
    {
        NoGradGuard no_grad;
        auto y = relu(matmul(x, w));
        auto views = {transpose(w, 0, 1), slice(w, 1, 0, 1), reshape(w, {4}), contiguous(transpose(w, 0, 1))};
        if (y->requires_grad || y->grad_fn) return std::string("ops under NoGradGuard recorded a graph");
        for (const auto& v : views) {
            if (v->requires_grad || v->grad_fn) return std::string("a view op under NoGradGuard recorded a graph");
        }
        ModernTransformer model(16, 8, 16, 2, 1, 1);
        auto logits = model.forward(upload({1, 2, 3, 4}, {1, 4}, Device::CPU, false));
        if (logits->requires_grad || logits->grad_fn) return std::string("model forward under NoGradGuard recorded a graph");
        {
            NoGradGuard nested;
        }
        if (grad_enabled()) return std::string("a nested NoGradGuard re-enabled grad mode on exit");
    }
    if (!grad_enabled()) return "NoGradGuard did not restore grad mode";
    auto z = relu(matmul(x, w));
    if (!z->requires_grad || !z->grad_fn) return "graph not recorded after the guard ended";
    scalar_sum(z)->backward();
    return w->grad ? "" : "no gradient after the guard ended";
}

static std::string engine_detect_anomaly() {
    {
        DetectAnomalyGuard detect;
        // d/dx (1/x) = -1/x^2 is infinite at x = 0
        auto x = upload({0.0f, 1.0f, 2.0f, 4.0f}, {2, 2}, Device::CPU, true);
        std::string msg;
        try {
            scalar_sum(rdiv_scalar(1.0f, x))->backward();
        } catch (const std::runtime_error& e) {
            msg = e.what();
        }
        if (msg.empty()) return "a non-finite gradient was not detected";
        if (msg.find("ScalarOpBackward(rdiv)") == std::string::npos) return "anomaly message does not name the op: " + msg;
    }
    if (detect_anomaly_enabled()) return "DetectAnomalyGuard did not restore the previous setting";
    auto y = upload({0.0f, 1.0f, 2.0f, 4.0f}, {2, 2}, Device::CPU, true);
    scalar_sum(rdiv_scalar(1.0f, y))->backward();  // must not throw with detection off
    return "";
}

// A custom op whose backward drops the factor 2 of d(x^2)/dx; gradcheck must reject it.
struct WrongSquareBackward : public Node {
    std::shared_ptr<Tensor> x;
    explicit WrongSquareBackward(std::shared_ptr<Tensor> x) : x(std::move(x)) {}
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        NoGradGuard no_grad;
        return {mul(grads[0], x)};
    }
};

static TensorPtr wrong_square(const TensorPtr& x) {
    TensorPtr out;
    {
        NoGradGuard no_grad;
        out = mul(x, x);
    }
    out->requires_grad = needs_grad(x);
    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<WrongSquareBackward>(x);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }
    return out;
}

static std::string gradcheck_rejects_wrong_backward() {
    auto err = dummygrad::gradcheck([](const Leaves& L) { return wrong_square(L[0]); }, {randn({3, 4})});
    if (err.empty()) return "gradcheck accepted a backward pass that is off by a factor of 2";
    auto ok = dummygrad::gradcheck([](const Leaves& L) { return mul(L[0], L[0]); }, {randn({3, 4})});
    return ok.empty() ? "" : "gradcheck rejected a correct op: " + ok;
}

// Gradient checks cannot catch a wrong broadcast mapping (forward and backward would agree), so
// check NumPy semantics on values directly.
static std::string broadcast_semantics() {
    auto x = upload(std::vector<float>(12, 0.0f), {2, 3, 1, 2}, Device::CPU, false);
    auto per_head = upload({1, 2, 3, 4, 5, 6}, {1, 3, 1, 2}, Device::CPU, false);
    auto out = download(add(x, per_head));
    for (int b = 0; b < 2; ++b)
        for (int k = 0; k < 6; ++k)
            if (out[b * 6 + k] != (float)(k + 1)) return "a [1, H, ...] operand was not repeated across the batch";
    auto per_seq = upload({10, 20, 30, 40}, {2, 1, 1, 2}, Device::CPU, false);
    out = download(add(x, per_seq));
    for (int b = 0; b < 2; ++b)
        for (int h = 0; h < 3; ++h)
            for (int j = 0; j < 2; ++j)
                if (out[b * 6 + h * 2 + j] != (float)(10 * (b * 2 + j + 1))) return "a [B, 1, 1, S] operand was misapplied";
    try {
        add(upload({1, 2, 3, 4, 5, 6}, {2, 3}, Device::CPU, false), upload(std::vector<float>(9, 1.0f), {3, 3}, Device::CPU, false));
    } catch (const std::invalid_argument&) {
        return "";
    }
    return "incompatible shapes were accepted";
}

// Forward ops and backward nodes land in separate phases; ops a backward node runs internally are
// part of that node, not forward entries; a nested op's time is not counted again by its caller
static std::string engine_profiler() {
    auto find = [](ProfilePhase phase, const std::string& name) -> ProfileEntry {
        for (const auto& e : profile_report(phase)) {
            if (e.name == name) return e;
        }
        return ProfileEntry{};
    };
    auto x = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    reset_profile();
    set_profiling(true);
    scalar_sum(mul_scalar(relu(x), 2.0f))->backward();
    set_profiling(false);
    if (find(ProfilePhase::Backward, "ScalarOpBackward(mul)").calls != 1) return "ScalarOpBackward(mul) was not recorded once";
    if (find(ProfilePhase::Forward, "mul_scalar").calls != 1) return "forward mul_scalar was not recorded once";
    if (find(ProfilePhase::Forward, "relu").calls != 1) return "forward relu was not recorded once";
    if (find(ProfilePhase::Forward, "sum").calls != 2) return "forward sum was not recorded twice";
    for (const auto& e : profile_report(ProfilePhase::Forward)) {
        if (e.phase != ProfilePhase::Forward) return "profile_report(Forward) returned a backward entry";
    }
    if (profile_report().size() != profile_report(ProfilePhase::Forward).size() + profile_report(ProfilePhase::Backward).size()) {
        return "profile_report() is not the union of both phases";
    }

    // An op called from inside a backward node belongs to that node
    struct OpInsideBackward : public Node {
        std::string name() const override { return "OpInsideBackward"; }
        std::vector<TensorPtr> apply(const std::vector<TensorPtr>& grads) override {
            DG_PROFILE_OP("op_inside_backward");
            return {grads[0]};
        }
    };
    reset_profile();
    set_profiling(true);
    auto y = mul_scalar(x, 1.0f);
    auto node = std::make_shared<OpInsideBackward>();
    node->add_next_edge(get_grad_edge(x).function);
    y->grad_fn = node;
    scalar_sum(y)->backward();
    set_profiling(false);
    if (find(ProfilePhase::Backward, "OpInsideBackward").calls != 1) return "the custom backward node was not recorded";
    if (find(ProfilePhase::Forward, "op_inside_backward").calls != 0) return "an op inside a backward node got a forward entry";

    // Self time: the outer op's entry excludes the inner op it called
    reset_profile();
    set_profiling(true);
    {
        DG_PROFILE_OP("outer_op");
        {
            DG_PROFILE_OP("inner_op");
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
    }
    set_profiling(false);
    auto outer = find(ProfilePhase::Forward, "outer_op"), inner = find(ProfilePhase::Forward, "inner_op");
    reset_profile();
    if (outer.calls != 1 || inner.calls != 1) return "nested ops were not recorded once each";
    if (inner.total_ms < 40.0) return "the inner op's time is shorter than its sleep";
    if (outer.total_ms >= inner.total_ms) return "the outer op's entry includes the inner op's time";
    return "";
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

// AccumulateGrad adopts a parameter's first gradient instead of copying it. AddBackward hands
// one tensor to both of its edges, so adopting must never leave two parameters sharing a buffer.
static std::string engine_adopted_grads_do_not_alias() {
    auto a = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    auto b = upload({5, 6, 7, 8}, {2, 2}, Device::CPU, true);
    scalar_sum(add(a, b))->backward();
    if (!a->grad || !b->grad) return "a parameter received no gradient";
    if (a->grad->storage == b->grad->storage) return "two parameters share one gradient buffer";
    a->grad->fill_(5.0f);
    auto err = expect_all(b, 1.0f, "writing a's gradient changed b's");
    if (!err.empty()) return err;
    // A second backward accumulates into the adopted buffers
    scalar_sum(add(a, b))->backward();
    return expect_all(b, 2.0f, "second backward");
}

static std::string engine_deep_chain() {
    auto x = upload({1, 2, 3, 4}, {2, 2}, Device::CPU, true);
    TensorPtr y = x;
    for (int i = 0; i < 3000; ++i) y = add_scalar(y, 0.001f);
    scalar_sum(y)->backward();
    return expect_all(x, 1.0f, "3000-op chain");
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

static std::string expect_names(Module& m, const std::vector<std::string>& required, size_t expected_count) {
    auto named = m.named_parameters();
    if (named.size() != expected_count) {
        return "expected " + std::to_string(expected_count) + " parameters, got " + std::to_string(named.size());
    }
    for (const auto& want : required) {
        bool found = false;
        for (const auto& [name, t] : named) found |= (name == want);
        if (!found) return "missing parameter name " + want;
    }
    int64_t total = 0;
    for (const auto& [name, t] : named) total += t->size();
    return total == m.num_parameters() ? "" : "num_parameters() disagrees with named_parameters()";
}

// Checkpoint names must stay exactly as earlier releases wrote them
static std::string module_checkpoint_names() {
    ModernTransformer modern(16, 8, 16, 2, 2, 1);
    auto err = expect_names(modern, {"token_emb.weight", "blocks.0.rms1.gamma", "blocks.0.attn.W_q.weight",
                                     "blocks.1.attn.W_k.bias", "blocks.1.ffn.w_down.bias", "rms_f.gamma"},
                            1 + 2 * 16 + 1);
    if (!err.empty()) return "ModernTransformer: " + err;
    GPT gpt(16, 8, 16, 2, 2);
    err = expect_names(gpt, {"token_emb.weight", "pos_emb.weight", "blocks.0.ln1.beta", "blocks.0.attn.W_o.weight",
                             "blocks.1.ffn.fc2.bias", "ln_f.gamma"},
                       2 + 2 * 16 + 2);
    if (!err.empty()) return "GPT: " + err;
    // Non-persistent buffers move with the model but are not saved
    for (const auto& [name, t] : modern.state_dict()) {
        if (name.find("freqs") != std::string::npos) return "RoPE table " + name + " ended up in the state dict";
    }
    return modern.named_buffers().size() == 2 ? "" : "expected the two RoPE buffers";
}

// Changing only the last token must not change the logits at earlier positions
template <typename Model>
static std::string expect_causal(Model& model, int vocab, const std::string& what) {
    NoGradGuard no_grad;
    std::vector<float> a = {1, 5, 2, 7, 3}, b = a;
    b.back() = (float)(((int)b.back() + 4) % vocab);
    auto la = download(model.forward(upload(a, {1, 5}, Device::CPU, false)));
    auto lb = download(model.forward(upload(b, {1, 5}, Device::CPU, false)));
    for (int pos = 0; pos < 4; ++pos) {
        for (int v = 0; v < vocab; ++v) {
            int k = pos * vocab + v;
            if (std::fabs(la[k] - lb[k]) > 1e-5f) {
                return what + ": logits at position " + std::to_string(pos) + " depend on a later token";
            }
        }
    }
    return "";
}

static std::string module_causality() {
    GPT gpt(12, 8, 16, 2, 2);
    auto err = expect_causal(gpt, 12, "GPT");
    if (!err.empty()) return err;
    ModernTransformer modern(12, 8, 16, 2, 2, 1);
    return expect_causal(modern, 12, "ModernTransformer");
}

static std::string module_save_load_roundtrip() {
    manual_seed(7);
    ModernTransformer a(16, 8, 16, 2, 1, 1);
    manual_seed(8);
    ModernTransformer b(16, 8, 16, 2, 1, 1);
    auto input = upload({1, 2, 3, 4}, {1, 4}, Device::CPU, false);
    const std::string path = "test_module_roundtrip.safetensors";
    a.save_safetensors(path);

    auto params_before = b.parameters();
    b.load_safetensors(path);
    std::remove(path.c_str());

    auto params_after = b.parameters();
    for (size_t i = 0; i < params_before.size(); ++i) {
        if (params_before[i].get() != params_after[i].get()) return "loading replaced parameter tensors (optimizer state would detach)";
    }
    NoGradGuard no_grad;
    auto err = compare(download(a.forward(input)), download(b.forward(input)), 0.0f);
    return err.empty() ? "" : "reloaded model gives different outputs: " + err;
}

static std::string module_strict_loading() {
    ModernTransformer modern(16, 8, 16, 2, 1, 1);
    const std::string path = "test_module_strict.safetensors";
    modern.save_safetensors(path);

    std::string msg;
    GPT gpt(16, 8, 16, 2, 1);
    try { gpt.load_safetensors(path); } catch (const std::runtime_error& e) { msg = e.what(); }
    if (msg.find("missing from checkpoint") == std::string::npos || msg.find("unexpected in checkpoint") == std::string::npos) {
        std::remove(path.c_str());
        return "strict load of a mismatched checkpoint did not report missing and unexpected names: " + msg;
    }

    msg.clear();
    ModernTransformer wider(16, 8, 32, 2, 1, 1);
    try { wider.load_safetensors(path, /*strict=*/false); } catch (const std::runtime_error& e) { msg = e.what(); }
    if (msg.find("shape mismatch") == std::string::npos) {
        std::remove(path.c_str());
        return "a shape mismatch was not reported even with strict=false";
    }

    // Non-strict loading copies the overlapping subset: the token embedding and attention
    // projections share names and (with full multi-head attention) shapes
    GPT donor(16, 8, 16, 2, 1);
    const std::string gpt_path = "test_module_partial.safetensors";
    donor.save_safetensors(gpt_path);
    ModernTransformer partial(16, 8, 16, 2, 1, /*n_kv_heads=*/0);
    partial.load_safetensors(gpt_path, /*strict=*/false);
    std::remove(path.c_str());
    std::remove(gpt_path.c_str());
    auto err = compare(download(donor.token_emb.weight), download(partial.token_emb.weight), 0.0f);
    if (err.empty()) err = compare(download(donor.blocks[0]->attn.W_k.W), download(partial.blocks[0]->attn.W_k.W), 0.0f);
    return err.empty() ? "" : "non-strict load did not copy the overlapping tensors";
}

struct TiedPair : public Module {
    std::shared_ptr<Tensor> shared = randn({2, 3});
    std::shared_ptr<Tensor> alias = shared;
    Linear frozen{3, 2};
    void visit(ModuleVisitor& v) override {
        v.parameter("shared", shared);
        v.parameter("alias", alias);
        v.module("frozen", frozen);
    }
};

static std::string module_to_tied_and_frozen() {
    TiedPair m;
    if (m.parameters().size() != 3) return "a tied tensor was listed twice";
    m.frozen.requires_grad_(false);
    m.to(Device::CPU, DType::Float16);
    if (m.shared.get() != m.alias.get()) return "to() split a tied tensor into two copies";
    if (m.shared->dtype != DType::Float16 || m.frozen.W->dtype != DType::Float16) return "to() did not convert every tensor";
    if (m.frozen.W->requires_grad || m.frozen.b->requires_grad) return "to() unfroze frozen parameters";
    if (!m.shared->requires_grad) return "to() dropped requires_grad from a trainable parameter";
    m.to(Device::CPU, DType::Float32);
    return m.shared.get() == m.alias.get() ? "" : "tied tensor split on the way back";
}

static std::string module_modes_and_zero_grad() {
    ModernTransformer model(16, 8, 16, 2, 1, 1);
    model.eval();
    if (model.is_training() || model.blocks[0]->attn.W_q.is_training()) return "eval() did not reach submodules";
    model.train();
    if (!model.blocks[0]->ffn.w_up.is_training()) return "train() did not reach submodules";
    auto loss = fused_cross_entropy(model.forward(upload({1, 2, 3, 4}, {1, 4}, Device::CPU, false)),
                                    upload({2, 3, 4, 5}, {1, 4}, Device::CPU, false));
    loss->backward();
    model.zero_grad();
    for (auto& p : model.parameters()) {
        if (p->grad) return "zero_grad() left a gradient";
    }
    return "";
}

// ---------------------------------------------------------------------------
// Workflow: determinism, checkpoints, data, logging
// ---------------------------------------------------------------------------

static std::vector<std::vector<float>> all_grads(Module& m) {
    std::vector<std::vector<float>> out;
    for (auto& p : m.parameters()) out.push_back(p->grad ? download(p->grad) : std::vector<float>{});
    return out;
}

// Gradients meeting at a node are summed in a fixed order and every kernel reduces in a fixed
// order (no atomics), so repeated runs are bitwise equal on CPU and GPU.
template <typename Model>
static std::string deterministic_backward(Model& model, Device dev, int batch, int seq) {
    if (dev == Device::CUDA) model.cuda();
    std::vector<float> in(batch * seq), tg(batch * seq);
    for (int i = 0; i < batch * seq; ++i) {
        in[i] = (float)((i * 7) % 16);
        tg[i] = (float)((i * 7 + 1) % 16);
    }
    auto run = [&]() {
        model.zero_grad();
        fused_cross_entropy(model.forward(upload(in, {batch, seq}, dev, false)), upload(tg, {batch, seq}, dev, false))->backward();
        return all_grads(model);
    };
    auto first = run();
    for (int i = 0; i < 5; ++i) {
        auto again = run();
        for (size_t k = 0; k < first.size(); ++k) {
            if (first[k] != again[k]) return "gradients of parameter " + std::to_string(k) + " differ between identical runs";
        }
    }
    return "";
}

static std::string workflow_deterministic_backward() {
    ModernTransformer model(16, 8, 16, 2, 2, 1);
    return deterministic_backward(model, Device::CPU, 2, 3);
}

// 128 rows, so LayerNorm's gamma/beta reductions take the multi-chunk path
static std::string workflow_deterministic_backward_cuda_modern() {
    manual_seed(5);
    ModernTransformer model(16, 32, 32, 2, 2, 1);
    return deterministic_backward(model, Device::CUDA, 4, 32);
}

static std::string workflow_deterministic_backward_cuda_gpt() {
    manual_seed(5);
    GPT model(16, 32, 32, 2, 2);
    return deterministic_backward(model, Device::CUDA, 4, 32);
}

template <typename Optimizer>
static void train_steps(ModernTransformer& model, Optimizer& opt, int first_step, int steps) {
    auto params = model.parameters();
    for (int s = first_step; s < first_step + steps; ++s) {
        // The batch depends on the global step, so a resumed run sees the same data
        std::vector<float> in(8), tg(8);
        for (int i = 0; i < 8; ++i) {
            in[i] = (float)((i + 3 * s) % 16);
            tg[i] = (float)((i + 3 * s + 1) % 16);
        }
        opt.zero_grad(params);
        fused_cross_entropy(model.forward(upload(in, {2, 4}, Device::CPU, false)), upload(tg, {2, 4}, Device::CPU, false))->backward();
        opt.step(params);
    }
}

// Training 3 steps, checkpointing, and resuming into a fresh model and optimizer for 3 more
// must reproduce 6 uninterrupted steps exactly.
template <typename MakeOptimizer>
static std::string resume_matches_uninterrupted(MakeOptimizer make_optimizer, const std::string& what) {
    manual_seed(3);
    ModernTransformer straight(16, 8, 16, 2, 1, 1);
    auto opt_a = make_optimizer();
    train_steps(straight, opt_a, 0, 6);

    manual_seed(3);
    ModernTransformer first(16, 8, 16, 2, 1, 1);
    auto opt_b = make_optimizer();
    train_steps(first, opt_b, 0, 3);
    const std::string path = "test_resume.safetensors";
    save_checkpoint(path, first, &opt_b, TrainingState{3, {}});

    manual_seed(99);  // a different initialization, fully overwritten by the checkpoint
    ModernTransformer resumed(16, 8, 16, 2, 1, 1);
    auto opt_c = make_optimizer();
    TrainingState state = load_checkpoint(path, resumed, &opt_c);
    std::remove(path.c_str());
    if (state.step != 3) return what + ": step was not restored";
    train_steps(resumed, opt_c, 3, 3);

    auto a = straight.parameters(), c = resumed.parameters();
    for (size_t i = 0; i < a.size(); ++i) {
        if (download(a[i]) != download(c[i])) return what + ": parameter " + std::to_string(i) + " differs from the uninterrupted run";
    }
    return "";
}

static std::string workflow_resume_adam() {
    return resume_matches_uninterrupted([] { return Adam(1e-2f); }, "Adam");
}

static std::string workflow_resume_sgd_momentum() {
    return resume_matches_uninterrupted([] { return SGD(0.1f, /*momentum=*/0.9f, /*weight_decay=*/1e-3f); }, "SGD");
}

static std::string workflow_checkpoint_scalars() {
    Linear layer(2, 2);
    TrainingState saved{123456789012345LL, {{"best_val", 0.1}, {"epoch", 7.0}}};
    const std::string path = "test_scalars.safetensors";
    save_checkpoint(path, layer, saved);
    Linear fresh(2, 2);
    TrainingState loaded = load_checkpoint(path, fresh);
    std::remove(path.c_str());
    if (loaded.step != saved.step) return "step changed: " + std::to_string(loaded.step);
    if (loaded.scalars != saved.scalars) return "scalars changed in the round trip";
    return compare(download(layer.W), download(fresh.W), 0.0f);
}

static std::string workflow_clip_grad_norm() {
    auto a = upload({0, 0, 0, 0}, {2, 2}, Device::CPU, true);
    auto b = upload({0, 0}, {1, 2}, Device::CPU, true);
    a->grad = upload({3, 0, 0, 0}, {2, 2}, Device::CPU, false);
    b->grad = upload({0, 4}, {1, 2}, Device::CPU, false);
    float norm = clip_grad_norm_({a, b}, 1.0f);
    if (std::fabs(norm - 5.0f) > 1e-5f) return "reported norm " + std::to_string(norm) + ", expected 5";
    float ga = download(a->grad)[0], gb = download(b->grad)[1];
    if (std::fabs(ga - 0.6f) > 1e-5f || std::fabs(gb - 0.8f) > 1e-5f) return "gradients not rescaled to norm 1";
    float unclipped = clip_grad_norm_({a, b}, 10.0f);
    return std::fabs(unclipped - 1.0f) < 1e-4f ? "" : "a norm under the limit was changed";
}

static std::string workflow_dataloader() {
    const int64_t n = 10;
    std::vector<float> xs(n * 3), ys(n);
    for (int64_t i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) xs[i * 3 + k] = (float)(i * 10 + k);
        ys[i] = (float)i;
    }
    auto X = upload(xs, {n, 3}, Device::CPU, false), Y = upload(ys, {n}, Device::CPU, false);

    DataLoader loader(X, Y, /*batch_size=*/4, /*shuffle=*/true, /*seed=*/5);
    if (loader.num_batches() != 3) return "expected 3 batches (4 + 4 + 2)";
    auto epoch_order = [&](int64_t epoch) {
        loader.set_epoch(epoch);
        std::vector<int> seen;
        for (auto [x, y] : loader) {
            auto xv = download(x), yv = download(y);
            if (x->shape[1] != 3 || x->shape[0] != y->shape[0]) seen.push_back(-1000);
            for (int64_t r = 0; r < y->shape[0]; ++r) {
                int id = (int)yv[r];
                if (xv[r * 3] != (float)(id * 10)) seen.push_back(-1000);  // row and target misaligned
                seen.push_back(id);
            }
        }
        return seen;
    };
    auto e0 = epoch_order(0), e0_again = epoch_order(0), e1 = epoch_order(1);
    auto sorted = e0;
    std::sort(sorted.begin(), sorted.end());
    for (int64_t i = 0; i < n; ++i) {
        if (sorted.size() != (size_t)n || sorted[i] != i) return "an epoch did not visit every sample exactly once (or rows were misaligned)";
    }
    if (e0 != e0_again) return "the same epoch produced a different order";
    if (e0 == e1) return "different epochs produced the same order";

    DataLoader ordered(X, nullptr, 4, /*shuffle=*/false, 0, /*drop_last=*/true);
    if (ordered.num_batches() != 2) return "drop_last should leave 2 full batches";
    Batch b = ordered.batch(1);
    if (b.targets || download(b.inputs)[0] != 40.0f) return "unshuffled loader returned the wrong rows";
    return "";
}

static std::string workflow_token_dataset() {
    std::vector<int32_t> ids(100);
    for (int i = 0; i < 100; ++i) ids[i] = i;
    TokenDataset data(ids);
    if (data.vocab_size() != 100) return "vocab_size should be max id + 1";
    std::mt19937 rng(1);
    Batch b = data.sample(4, 8, rng);
    auto x = download(b.inputs), y = download(b.targets);
    for (size_t k = 0; k < x.size(); ++k) {
        if (y[k] != x[k] + 1) return "targets are not the inputs shifted by one token";
    }
    if (data.num_windows(8) != 12) return "expected 12 windows of 8 tokens in 100";
    auto w = data.windows(10, 4, 8);
    if (w.inputs->shape[0] != 2 || download(w.inputs)[0] != 80.0f) return "windows() returned the wrong slice";
    auto parts = data.split(0.9);
    if (parts.first.size() != 90 || parts.second.tokens()[0] != 90) return "split() cut in the wrong place";

    const std::string path = "test_tokens.bin";
    {
        std::ofstream out(path, std::ios::binary);
        for (uint16_t v : {7, 65535, 3}) out.write(reinterpret_cast<const char*>(&v), 2);
    }
    auto from_file = TokenDataset::from_file(path);
    std::remove(path.c_str());
    if (from_file.tokens() != std::vector<int32_t>{7, 65535, 3}) return "from_file misread uint16 tokens";
    auto text = TokenDataset::from_string("hi!");
    if (text.vocab_size() != 256 || text.tokens()[0] != 'h') return "from_string should give byte tokens with vocab 256";
    return "";
}

static std::string workflow_metrics_logger() {
    const std::string path = "test_metrics.csv";
    {
        MetricsLogger log(path, /*echo=*/false);
        log.log(1, {{"loss", 2.5}});
        log.log(2, {{"loss", 2.0}, {"lr", 0.001}});
    }
    {
        MetricsLogger resumed(path, /*echo=*/false, /*append=*/true);
        resumed.log(3, {{"val_loss", 1.5}});
    }
    std::ifstream in(path);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::remove(path.c_str());
    const std::string expected = "step,metric,value\n1,loss,2.5\n2,loss,2\n2,lr,0.001\n3,val_loss,1.5\n";
    return content == expected ? "" : "unexpected CSV contents:\n" + content;
}

// ---------------------------------------------------------------------------
// Mixed precision
// ---------------------------------------------------------------------------

static std::string hex16(uint16_t v) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%04x", v);
    return buf;
}

static bool half_is_nan(uint16_t h) { return (h & 0x7c00) == 0x7c00 && (h & 0x03ff) != 0; }

static std::string dtype_half_conversion() {
    struct Case { float in; uint16_t out; const char* what; };
    const Case cases[] = {
        {0.0f, 0x0000, "zero"},
        {-0.0f, 0x8000, "negative zero"},
        {1.0f, 0x3c00, "one"},
        {-2.0f, 0xc000, "minus two"},
        {1.0f + std::ldexp(1.0f, -11), 0x3c00, "tie between 1 and 1+2^-10 rounds to even (down)"},
        {1.0f + 3 * std::ldexp(1.0f, -11), 0x3c02, "tie between 1+2^-10 and 1+2^-9 rounds to even (up)"},
        {1.0f + std::ldexp(1.0f, -11) + std::ldexp(1.0f, -20), 0x3c01, "just above a tie rounds up"},
        {65504.0f, 0x7bff, "largest finite"},
        {65519.0f, 0x7bff, "below the overflow tie"},
        {65520.0f, 0x7c00, "overflow tie rounds to inf"},
        {70000.0f, 0x7c00, "overflow is inf, not NaN"},
        {-1e30f, 0xfc00, "large negative is -inf"},
        {INFINITY, 0x7c00, "inf"},
        {-INFINITY, 0xfc00, "-inf"},
        {std::ldexp(1.0f, -14), 0x0400, "smallest normal"},
        {std::ldexp(1.0f, -24), 0x0001, "smallest subnormal"},
        {std::ldexp(1.0f, -25), 0x0000, "half the smallest subnormal ties to zero"},
        {1.5f * std::ldexp(1.0f, -24), 0x0002, "subnormal tie rounds to even"},
        {std::ldexp(1023.0f, -24), 0x03ff, "largest subnormal"},
        {std::ldexp(1023.5f, -24), 0x0400, "subnormal rounding carries into the smallest normal"},
        {-std::ldexp(3.0f, -24), 0x8003, "negative subnormal"},
    };
    for (const auto& c : cases) {
        uint16_t got = float_to_half_bits(c.in);
        if (got != c.out) return std::string(c.what) + ": got " + hex16(got) + ", expected " + hex16(c.out);
    }
    if (!half_is_nan(float_to_half_bits(NAN))) return "NaN did not stay NaN";
    // Every fp16 value survives a round trip through float exactly
    for (uint32_t h = 0; h <= 0xffff; ++h) {
        float f = half_bits_to_float((uint16_t)h);
        uint16_t back = float_to_half_bits(f);
        if (half_is_nan((uint16_t)h) ? !half_is_nan(back) : back != h) {
            return "round trip of " + hex16((uint16_t)h) + " gave " + hex16(back);
        }
    }
    return "";
}

static std::string dtype_typed_access_checked() {
    auto h = std::make_shared<Tensor>(std::vector<int64_t>{4}, Device::CPU, DType::Float16, false);
    try {
        (void)h->data_ptr<float>();
        return "an fp16 tensor was readable as float";
    } catch (const std::runtime_error&) {
    }
    if (!h->data_ptr<float16>() || !h->data_ptr<void>() || !h->data_ptr<uint16_t>()) return "matching or raw access failed";
    auto f = upload({1, 2}, {2}, Device::CPU, false);
    try {
        (void)f->data_ptr<bfloat16>();
        return "a float tensor was readable as bf16";
    } catch (const std::runtime_error&) {
    }
    // Conversion between dtypes still works, and rounds to nearest even
    auto back = f->to(DType::Float16)->to(DType::Float32);
    return download(back) == std::vector<float>{1, 2} ? "" : "fp32 -> fp16 -> fp32 changed exact values";
}

static std::string amp_autocast_guard() {
    if (autocast_dtype() != DType::Float32) return "autocast is on by default";
    {
        AutocastGuard outer(DType::Float16);
        {
            AutocastGuard inner(DType::Float32);
            if (autocast_dtype() != DType::Float32) return "nested guard did not set its dtype";
        }
        if (autocast_dtype() != DType::Float16) return "nested guard did not restore the outer dtype";
        // CPU matmuls ignore autocast: still exact fp32
        auto x = upload({1.0f + std::ldexp(1.0f, -20), 1}, {1, 2}, Device::CPU, false);
        auto w = upload({1, 0}, {2, 1}, Device::CPU, false);
        if (download(matmul(x, w))[0] != 1.0f + std::ldexp(1.0f, -20)) return "a CPU matmul was rounded under autocast";
    }
    if (autocast_dtype() != DType::Float32) return "guard did not restore fp32";
    try {
        AutocastGuard bad(DType::Int8);
        return "Int8 autocast was accepted";
    } catch (const std::runtime_error&) {
    }
    return autocast_dtype() == DType::Float32 ? "" : "a rejected guard changed the dtype";
}

static std::string amp_grad_scaler() {
    auto p = upload({1, 1}, {1, 2}, Device::CPU, true);
    SGD opt(1.0f);
    GradScaler scaler(8.0f, 2.0f, 0.5f, /*growth_interval=*/2);

    // scale() multiplies the loss, so backward produces gradients scaled by 8
    auto loss = scalar_sum(mul_scalar(p, 0.5f));
    scaler.scale(loss)->backward();
    if (download(p->grad) != std::vector<float>{4, 4}) return "scaled gradient is not 8 * 0.5";
    scaler.unscale({p});
    scaler.unscale({p});  // a second unscale in the same step does nothing
    if (download(p->grad) != std::vector<float>{0.5f, 0.5f}) return "unscale did not divide the gradient by the scale exactly once";
    if (!scaler.step(opt, {p}) || download(p) != std::vector<float>{0.5f, 0.5f}) return "a clean step did not run the optimizer";
    scaler.update();
    if (scaler.get_scale() != 8.0f) return "scale changed before growth_interval clean steps";

    // An overflowed gradient skips the step and halves the scale
    p->grad = upload({INFINITY, 1}, {1, 2}, Device::CPU, false);
    if (scaler.step(opt, {p})) return "a step with an inf gradient ran";
    if (!scaler.found_inf() || download(p) != std::vector<float>{0.5f, 0.5f}) return "a skipped step changed the parameter";
    scaler.update();
    if (scaler.get_scale() != 4.0f || scaler.skipped_steps() != 1) return "scale did not back off to 4 after overflow";

    // Two clean steps in a row double it
    for (int i = 0; i < 2; ++i) {
        p->grad = upload({0, 0}, {1, 2}, Device::CPU, false);
        scaler.step(opt, {p});
        scaler.update();
    }
    if (scaler.get_scale() != 8.0f) return "scale did not grow after growth_interval clean steps";

    p->grad = upload({NAN, 0}, {1, 2}, Device::CPU, false);
    scaler.step(opt, {p});
    try {
        scaler.update();
        scaler.update();
        return "update() twice without a step was accepted";
    } catch (const std::runtime_error&) {
    }

    // Checkpoint round trip through TrainingState
    TrainingState state;
    scaler.save_to(state);
    GradScaler restored;
    restored.load_from(state);
    if (restored.get_scale() != scaler.get_scale() || restored.skipped_steps() != 2) return "scaler state did not round-trip";
    GradScaler untouched(1024.0f);
    untouched.load_from(TrainingState{});
    if (untouched.get_scale() != 1024.0f) return "loading a state without scaler entries changed the scale";

    // Disabled: a pass-through
    GradScaler off(8.0f, 2.0f, 0.5f, 2000, /*enabled=*/false);
    auto l = scalar_sum(p);
    return off.scale(l) == l ? "" : "a disabled scaler changed the loss";
}

#ifdef USE_CUDA
static bool cuda_has_bf16_tensor_cores() {
    int dev = 0, major = 0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 8;
}

// fp16/bf16 matmuls against the fp32 result: forward and both gradients, for a plain, a batched,
// a shared-weight (folded), a broadcast-a (accumulated per batch) and a transposed-weight (tied LM
// head: b is a strided view, densified before rounding) product
static std::string amp_matmul_parity(DType compute) {
    struct Shape { std::vector<int64_t> a, b; bool transpose_b = false; };  // b's shape before any transpose
    const Shape shapes[] = {
        {{48, 64}, {64, 40}},
        {{3, 32, 64}, {3, 64, 24}},
        {{4, 16, 64}, {64, 32}},
        {{16, 64}, {4, 64, 32}},
        {{4, 16, 64}, {40, 64}, true},
    };
    for (const auto& s : shapes) {
        auto av = random_values(numel(s.a), -1, 1), bv = random_values(numel(s.b), -1, 1);
        auto run = [&](DType dt, std::vector<float>& out, std::vector<float>& ga, std::vector<float>& gb) {
            auto a = upload(av, s.a, Device::CUDA, true), b = upload(bv, s.b, Device::CUDA, true);
            TensorPtr y;
            {
                AutocastGuard autocast(dt);
                y = matmul(a, s.transpose_b ? transpose(b, 0, 1) : b);
            }
            if (y->dtype != DType::Float32) throw std::runtime_error("autocast matmul output is not Float32");
            // Backward runs outside the guard: it must use the dtype the forward ran with
            scalar_sum(reshape(mul(y, y), {1, (int64_t)y->size()}))->backward();
            out = download(y);
            ga = download(a->grad);
            gb = download(b->grad);
        };
        std::vector<float> y32, ga32, gb32, y16, ga16, gb16;
        run(DType::Float32, y32, ga32, gb32);
        run(compute, y16, ga16, gb16);
        auto compare = [&](const std::vector<float>& ref, const std::vector<float>& got, const char* what) -> std::string {
            float scale = 0.0f, worst = 0.0f;
            for (size_t i = 0; i < ref.size(); ++i) {
                scale = std::max(scale, std::fabs(ref[i]));
                worst = std::max(worst, std::fabs(ref[i] - got[i]));
            }
            // 8 mantissa bits (bf16) or 11 (fp16) per operand; 2% of the largest value covers both
            if (!(worst <= 2e-2f * scale + 1e-4f)) return std::string(what) + " differs from fp32 by " + std::to_string(worst);
            if (worst == 0.0f) return std::string(what) + " is bit-identical to fp32: autocast had no effect";
            return "";
        };
        std::string label = "a" + std::to_string(s.a.size()) + "d x b" + std::to_string(s.b.size()) + "d" +
                            (s.transpose_b ? " (transposed)" : "") + ": ";
        for (auto err : {compare(y32, y16, "output"), compare(ga32, ga16, "grad a"), compare(gb32, gb16, "grad b")}) {
            if (!err.empty()) return label + err;
        }
    }
    return "";
}

// A CUDA matmul with one frozen operand (a fixed input, or a frozen weight as in LoRA fine-tuning)
// computes only the other gradient, which must be bit-identical to the one from the same product
// with both operands trainable. Covers the plain, folded, per-batch and transposed-weight paths.
static std::string matmul_frozen_operand_cuda(DType compute) {
    struct Shape { std::vector<int64_t> a, b; bool transpose_b = false; };
    const Shape shapes[] = {
        {{48, 64}, {64, 40}},
        {{4, 16, 64}, {64, 32}},
        {{16, 64}, {4, 64, 32}},
        {{4, 16, 64}, {40, 64}, true},
    };
    for (const auto& s : shapes) {
        auto av = random_values(numel(s.a), -1, 1), bv = random_values(numel(s.b), -1, 1);
        auto run = [&](bool train_a, bool train_b) {
            auto a = upload(av, s.a, Device::CUDA, train_a), b = upload(bv, s.b, Device::CUDA, train_b);
            {
                AutocastGuard autocast(compute);
                auto y = matmul(a, s.transpose_b ? transpose(b, 0, 1) : b);
                scalar_sum(reshape(mul(y, y), {1, (int64_t)y->size()}))->backward();
            }
            return std::make_pair(a->grad, b->grad);
        };
        auto both = run(true, true);
        auto only_a = run(true, false);
        auto only_b = run(false, true);
        std::string label = "a" + std::to_string(s.a.size()) + "d x b" + std::to_string(s.b.size()) + "d" +
                            (s.transpose_b ? " (transposed)" : "") + ": ";
        if (!only_a.first || only_a.second) return label + "frozen b: expected a gradient for a only";
        if (!only_b.second || only_b.first) return label + "frozen a: expected a gradient for b only";
        if (download(only_a.first) != download(both.first)) return label + "grad a changed when b was frozen";
        if (download(only_b.second) != download(both.second)) return label + "grad b changed when a was frozen";
    }
    return "";
}

// Attention under autocast against fp32: output, lse and the three gradients on the GEMM path,
// causal and not, with a sequence length that is not a multiple of 8. The tiled kernels ignore
// autocast, so there the results must be bit-identical.
static std::string amp_attention_parity(DType compute) {
    const int B = 2, H = 3, S = 40, D = 32;
    const int64_t n = (int64_t)B * H * S * D;
    const float scale = 1.0f / std::sqrt((float)D);
    auto qv = random_values(n, -1, 1), kv = random_values(n, -1, 1), vv = random_values(n, -1, 1);
    auto dov = random_values(n, -1, 1);
    std::vector<int64_t> shape{B, H, S, D};
    struct Result { std::vector<float> out, lse, dq, dk, dv; };
    auto run = [&](DType dt, bool causal) {
        auto q = upload(qv, shape, Device::CUDA, false), k = upload(kv, shape, Device::CUDA, false);
        auto v = upload(vv, shape, Device::CUDA, false), dout = upload(dov, shape, Device::CUDA, false);
        auto out = std::make_shared<Tensor>(shape, Device::CUDA, DType::Float32, false);
        auto lse = std::make_shared<Tensor>(std::vector<int64_t>{B, H, S}, Device::CUDA, DType::Float32, false);
        auto dq = std::make_shared<Tensor>(shape, Device::CUDA, DType::Float32, false);
        auto dk = std::make_shared<Tensor>(shape, Device::CUDA, DType::Float32, false);
        auto dv = std::make_shared<Tensor>(shape, Device::CUDA, DType::Float32, false);
        auto impl = cuda::attention_forward(q->data_ptr<float>(), k->data_ptr<float>(), v->data_ptr<float>(),
                                            out->data_ptr<float>(), lse->data_ptr<float>(), B, H, S, D, causal, scale, dt);
        cuda::attention_backward(q->data_ptr<float>(), k->data_ptr<float>(), v->data_ptr<float>(), out->data_ptr<float>(),
                                 dout->data_ptr<float>(), lse->data_ptr<float>(), dq->data_ptr<float>(),
                                 dk->data_ptr<float>(), dv->data_ptr<float>(), B, H, S, D, causal, scale, impl, dt);
        return Result{download(out), download(lse), download(dq), download(dk), download(dv)};
    };
    auto compare = [](const std::vector<float>& ref, const std::vector<float>& got, const std::string& what) -> std::string {
        float scale_ref = 0.0f, worst = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            scale_ref = std::max(scale_ref, std::fabs(ref[i]));
            worst = std::max(worst, std::fabs(ref[i] - got[i]));
        }
        if (!(worst <= 2e-2f * scale_ref + 1e-4f)) return what + " differs from fp32 by " + std::to_string(worst);
        if (worst == 0.0f) return what + " is bit-identical to fp32: autocast had no effect";
        return "";
    };
    for (bool causal : {false, true}) {
        std::string label = causal ? "causal " : "full ";
        Result ref = run(DType::Float32, causal), half = run(compute, causal);
        for (auto err : {compare(ref.out, half.out, label + "output"), compare(ref.lse, half.lse, label + "lse"),
                         compare(ref.dq, half.dq, label + "dq"), compare(ref.dk, half.dk, label + "dk"),
                         compare(ref.dv, half.dv, label + "dv")}) {
            if (!err.empty()) return err;
        }
    }
    int gemm_max_seq = cuda::attention_gemm_max_seq();
    cuda::set_attention_gemm_max_seq(0);
    Result tiled32 = run(DType::Float32, true), tiled_half = run(compute, true);
    cuda::set_attention_gemm_max_seq(gemm_max_seq);
    if (tiled32.out != tiled_half.out || tiled32.dq != tiled_half.dq || tiled32.dk != tiled_half.dk ||
        tiled32.dv != tiled_half.dv) {
        return "the tiled kernels changed under autocast; they compute in fp32";
    }
    return "";
}
#endif

// ---------------------------------------------------------------------------
// End-to-end training
// ---------------------------------------------------------------------------

// compute != Float32 runs the matmuls under autocast (CUDA only); Float16 adds loss scaling
static std::string train_tiny_transformer(Device dev, DType compute = DType::Float32) {
    manual_seed(42);
    const int V = 16, T = 8, B = 4;
    ModernTransformer model(V, 16, 32, 4, 2, 2);  // GQA: 4 query heads share 2 kv heads
    if (dev == Device::CUDA) model.cuda(); else model.cpu();
    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->device != dev) return "parameter " + std::to_string(i) + " is not on " + device_name(dev);
    }
    Adam opt(1e-2f, 0.9f, 0.999f, 1e-8f, 0.0f);
    GradScaler scaler(65536.0f, 2.0f, 0.5f, 2000, compute == DType::Float16);

    // Next-token task: token i is always followed by token i + 1
    std::vector<float> in(B * T), tg(B * T);
    for (int i = 0; i < B * T; ++i) {
        in[i] = (float)(i % V);
        tg[i] = (float)((i + 1) % V);
    }

    float first = 0.0f, last = 0.0f;
    bool grads_checked = false;
    for (int step = 0; step < 60; ++step) {
        opt.zero_grad(params);
        auto inputs = upload(in, {B, T}, dev, false);
        auto targets = upload(tg, {B, T}, dev, false);
        TensorPtr loss;
        {
            AutocastGuard autocast(compute);
            auto logits = model.forward(inputs);
            if (logits->device != dev) return std::string("logits computed on the wrong device");
            loss = fused_cross_entropy(logits, targets);
        }
        scaler.scale(loss)->backward();
        scaler.unscale(params);

        // Every parameter must get a finite, nonzero gradient. With loss scaling the first steps
        // may overflow and be skipped by design, so check the first step that was not.
        if (!grads_checked && !scaler.found_inf()) {
            grads_checked = true;
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
        scaler.step(opt, params);
        scaler.update();

        float l = download(loss)[0];
        if (!std::isfinite(l)) return "loss became " + std::to_string(l) + " at step " + std::to_string(step);
        if (step == 0) first = l;
        last = l;
    }
    if (!(last < 0.5f * first)) {
        return "loss did not decrease enough: " + std::to_string(first) + " -> " + std::to_string(last);
    }
    if (!grads_checked) return "every step overflowed";
    if (scaler.skipped_steps() > 5) return std::to_string(scaler.skipped_steps()) + " of 60 steps overflowed";
    std::cout << "        loss " << first << " -> " << last;
    if (scaler.enabled()) std::cout << " (scale " << scaler.get_scale() << ", " << scaler.skipped_steps() << " skipped)";
    std::cout << "\n";
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
        run_case(r, "grad/" + tests[i].name, [&] { return gradcheck_cpu(tests[i]); });
    }

    if (has_cuda) {
        std::cout << "\n-- device parity (cuda vs cpu)\n";
        for (size_t i = 0; i < tests.size(); ++i) {
            run_case(r, "cuda/" + tests[i].name, [&] { return parity_check(tests[i], probes[i], Device::CUDA); });
        }
#ifdef USE_CUDA
        // The attention tests again with the GEMM path off, so the kernels used past
        // attention_gemm_max_seq() are checked too
        int gemm_max_seq = cuda::attention_gemm_max_seq();
        cuda::set_attention_gemm_max_seq(0);
        for (size_t i = 0; i < tests.size(); ++i) {
            if (tests[i].name.find("ttention") == std::string::npos) continue;
            run_case(r, "cuda_tiled/" + tests[i].name, [&] { return parity_check(tests[i], probes[i], Device::CUDA); });
        }
        cuda::set_attention_gemm_max_seq(gemm_max_seq);
#endif
    }

    std::cout << "\n-- autograd engine\n";
    run_case(r, "engine/strided_forward_matches_dense", strided_forward_matches_dense);
    run_case(r, "engine/no_grad", engine_no_grad);
    run_case(r, "engine/detect_anomaly", engine_detect_anomaly);
    run_case(r, "engine/gradcheck_rejects_wrong_backward", gradcheck_rejects_wrong_backward);
    run_case(r, "engine/broadcast_semantics", broadcast_semantics);
    run_case(r, "engine/profiler", engine_profiler);
    run_case(r, "engine/retain_graph", engine_retain_graph);
    run_case(r, "engine/exception_propagates", engine_exception_propagates);
    run_case(r, "engine/concurrent_fanin", engine_concurrent_fanin);
    run_case(r, "engine/adopted_grads_do_not_alias", engine_adopted_grads_do_not_alias);
    run_case(r, "engine/deep_chain", engine_deep_chain);

    std::cout << "\n-- modules\n";
    run_case(r, "module/checkpoint_names", module_checkpoint_names);
    run_case(r, "module/causality", module_causality);
    run_case(r, "module/save_load_roundtrip", module_save_load_roundtrip);
    run_case(r, "module/strict_loading", module_strict_loading);
    run_case(r, "module/to_tied_and_frozen", module_to_tied_and_frozen);
    run_case(r, "module/modes_and_zero_grad", module_modes_and_zero_grad);

    std::cout << "\n-- workflow\n";
    run_case(r, "workflow/deterministic_backward", workflow_deterministic_backward);
    if (has_cuda) {
        run_case(r, "workflow/deterministic_backward_cuda_modern", workflow_deterministic_backward_cuda_modern);
        run_case(r, "workflow/deterministic_backward_cuda_gpt", workflow_deterministic_backward_cuda_gpt);
    }
    run_case(r, "workflow/resume_adam", workflow_resume_adam);
    run_case(r, "workflow/resume_sgd_momentum", workflow_resume_sgd_momentum);
    run_case(r, "workflow/checkpoint_scalars", workflow_checkpoint_scalars);
    run_case(r, "workflow/clip_grad_norm", workflow_clip_grad_norm);
    run_case(r, "workflow/dataloader", workflow_dataloader);
    run_case(r, "workflow/token_dataset", workflow_token_dataset);
    run_case(r, "workflow/metrics_logger", workflow_metrics_logger);

    std::cout << "\n-- mixed precision\n";
    run_case(r, "amp/half_conversion", dtype_half_conversion);
    run_case(r, "amp/typed_access_checked", dtype_typed_access_checked);
    run_case(r, "amp/autocast_guard", amp_autocast_guard);
    run_case(r, "amp/grad_scaler", amp_grad_scaler);
#ifdef USE_CUDA
    if (has_cuda) {
        run_case(r, "amp/matmul_frozen_operand_fp32_cuda", [] { return matmul_frozen_operand_cuda(DType::Float32); });
        run_case(r, "amp/matmul_frozen_operand_fp16_cuda", [] { return matmul_frozen_operand_cuda(DType::Float16); });
        run_case(r, "amp/matmul_parity_fp16_cuda", [] { return amp_matmul_parity(DType::Float16); });
        run_case(r, "amp/attention_parity_fp16_cuda", [] { return amp_attention_parity(DType::Float16); });
        if (cuda_has_bf16_tensor_cores()) {
            run_case(r, "amp/matmul_parity_bf16_cuda", [] { return amp_matmul_parity(DType::BFloat16); });
            run_case(r, "amp/attention_parity_bf16_cuda", [] { return amp_attention_parity(DType::BFloat16); });
        } else {
            std::cout << "[SKIP] amp/*_parity_bf16_cuda (needs compute capability 8.0+)\n";
        }
    }
#endif

    std::cout << "\n-- end-to-end training\n";
    run_case(r, "train/tiny_transformer_cpu", [] { return train_tiny_transformer(Device::CPU); });
    if (has_cuda) {
        run_case(r, std::string("train/tiny_transformer_") + device_name(Device::CUDA),
                 [] { return train_tiny_transformer(Device::CUDA); });
        run_case(r, "train/tiny_transformer_cuda_fp16",
                 [] { return train_tiny_transformer(Device::CUDA, DType::Float16); });
    }

    std::cout << "\n" << r.passed << " passed, " << r.failures.size() << " failed\n";
    for (const auto& f : r.failures) std::cout << "  FAILED " << f << "\n";
    return r.failures.empty() ? 0 : 1;
}
