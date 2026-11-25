#pragma once
// Random tensors and dropout, from one counter-based generator (Philox). Every call takes the next
// stream of the current seed, so a program run twice with the same manual_seed draws the same numbers
// on any device and with any number of threads. Included from functional.h.
//
// rand, randint and friends are constants (no grad). randn, kaiming and xavier (init.h), which create
// parameters, draw from the thread's std::mt19937.

#include "index_ops.h"

#include <atomic>
#include <random>

struct RandomState {
    std::atomic<uint64_t> seed{0x5eed5eedULL};
    std::atomic<uint64_t> stream{0};
};
inline RandomState& random_state() {
    static RandomState state;
    return state;
}
inline uint64_t next_random_stream() { return random_state().stream.fetch_add(1, std::memory_order_relaxed); }

// Saves and restores the generator, e.g. to replay the same dropout masks (gradient checkpointing)
struct RandomStateGuard {
    uint64_t seed = random_state().seed, stream = random_state().stream;
    RandomStateGuard() = default;
    ~RandomStateGuard() {
        random_state().seed = seed;
        random_state().stream = stream;
    }
    RandomStateGuard(const RandomStateGuard&) = delete;
    RandomStateGuard& operator=(const RandomStateGuard&) = delete;
};

// Uniform on [0, 1)
inline TensorPtr rand(const std::vector<int64_t>& shape, Device device = Device::CPU) {
    auto t = op_detail::empty(shape, device);
    if (t->size() > 0) DG_KERNEL(device, random_uniform(t->data_ptr<float>(), t->size(), random_state().seed, next_random_stream()));
    return t;
}
inline TensorPtr rand_like(const TensorPtr& x) { return rand(x->shape, x->device); }
// Normal with the given mean and standard deviation
inline TensorPtr normal(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f, Device device = Device::CPU) {
    auto t = op_detail::empty(shape, device);
    if (t->size() > 0) DG_KERNEL(device, random_normal(t->data_ptr<float>(), t->size(), random_state().seed, next_random_stream()));
    NoGradGuard no_grad;
    if (std != 1.0f) t = mul_scalar(t, std);
    return mean != 0.0f ? add_scalar(t, mean) : t;
}
inline TensorPtr randn_like(const TensorPtr& x) { return normal(x->shape, 0.0f, 1.0f, x->device); }
inline TensorPtr uniform(const std::vector<int64_t>& shape, float lo, float hi, Device device = Device::CPU) {
    NoGradGuard no_grad;
    return add_scalar(mul_scalar(rand(shape, device), hi - lo), lo);
}
// Integers in [lo, hi)
inline TensorPtr randint(int64_t lo, int64_t hi, const std::vector<int64_t>& shape, Device device = Device::CPU) {
    if (hi <= lo) throw std::invalid_argument("randint: needs lo < hi");
    NoGradGuard no_grad;
    return clamp(floor(uniform(shape, (float)lo, (float)hi, device)), (float)lo, (float)(hi - 1));
}
// A random permutation of 0 .. n - 1
inline TensorPtr randperm(int64_t n, Device device = Device::CPU) {
    std::mt19937_64 rng((uint64_t)random_state().seed * 0x9e3779b97f4a7c15ULL + next_random_stream());
    std::vector<float> p((size_t)n);
    for (int64_t i = 0; i < n; ++i) p[(size_t)i] = (float)i;
    for (int64_t i = n - 1; i > 0; --i) std::swap(p[(size_t)i], p[(size_t)(rng() % (uint64_t)(i + 1))]);
    return op_detail::from_values(p, {n}, device);
}
// 1 with the probability given by each element, else 0
inline TensorPtr bernoulli(const TensorPtr& p) { return lt(rand_like(p), p); }
inline TensorPtr bernoulli(const std::vector<int64_t>& shape, float p, Device device = Device::CPU) { return lt(rand(shape, device), p); }
// Standard Gumbel noise
inline TensorPtr gumbel_like(const TensorPtr& x) {
    NoGradGuard no_grad;
    return neg(log(neg(log(clamp(rand_like(x), 1e-7f, 1.0f - 6e-8f)))));
}
// num draws per row of class weights [..., classes], as class indices [..., num]
inline TensorPtr multinomial(const TensorPtr& weights_in, int64_t num, bool replacement = false) {
    NoGradGuard no_grad;
    auto w = op_detail::dense(weights_in);
    if (w->ndim() < 1) throw std::invalid_argument("multinomial: weights need a class dimension");
    int64_t classes = w->shape.back(), rows = w->size() / std::max<int64_t>(1, classes);
    if (!replacement) {
        // The largest keys log(w) + Gumbel are a sample without replacement
        if (num > classes) throw std::invalid_argument("multinomial: more draws than classes needs replacement");
        return topk(add(log(w), gumbel_like(w)), num, -1).second;
    }
    auto wv = op_detail::host_floats(w);
    auto u = op_detail::host_floats(rand({rows, num}));
    std::vector<float> out((size_t)(rows * num));
    std::vector<double> cdf((size_t)classes);
    for (int64_t r = 0; r < rows; ++r) {
        double total = 0.0;
        for (int64_t k = 0; k < classes; ++k) {
            float v = wv[(size_t)(r * classes + k)];
            if (!(v >= 0.0f)) throw std::invalid_argument("multinomial: weights must be non-negative");
            cdf[(size_t)k] = total += v;
        }
        if (!(total > 0.0)) throw std::invalid_argument("multinomial: a row's weights sum to zero");
        for (int64_t j = 0; j < num; ++j) {
            auto it = std::upper_bound(cdf.begin(), cdf.end(), (double)u[(size_t)(r * num + j)] * total);
            int64_t k = std::min<int64_t>(it - cdf.begin(), classes - 1);
            while (k > 0 && wv[(size_t)(r * classes + k)] == 0.0f) --k;
            out[(size_t)(r * num + j)] = (float)k;
        }
    }
    auto shape = w->shape;
    shape.back() = num;
    return op_detail::from_values(out, shape, w->device);
}

namespace op_detail {
// x masked by the draws of one stream. Its gradient applies the same mask, regenerated, not stored.
inline TensorPtr dropout_stream(const TensorPtr& x_in, float keep, uint64_t seed, uint64_t stream) {
    auto x = dense(x_in);
    auto out = empty(x->shape, x->device);
    if (x->size() > 0) DG_KERNEL(x->device, dropout(x->data_ptr<float>(), out->data_ptr<float>(), x->size(), keep, seed, stream));
    return make_op("dropout", out, {x}, {}, [keep, seed, stream](const OpContext&, const TensorPtr& g) -> TensorList {
        return {dropout_stream(g, keep, seed, stream)};
    });
}
}  // namespace op_detail

// Zeroes each element with probability p and scales the rest by 1 / (1 - p)
inline TensorPtr dropout(const TensorPtr& x, float p = 0.5f, bool training = true) {
    DG_PROFILE_OP("dropout");
    if (!(p >= 0.0f && p <= 1.0f)) throw std::invalid_argument("dropout: p must be in [0, 1]");
    if (!training || p == 0.0f) return x;
    return op_detail::dropout_stream(x, 1.0f - p, random_state().seed, next_random_stream());
}
// Zeroes whole channels: dimension 1 of [batch, channels, ...]
inline TensorPtr feature_dropout(const TensorPtr& x, float p = 0.5f, bool training = true) {
    if (!training || p == 0.0f) return x;
    if (x->ndim() < 2) throw std::invalid_argument("feature_dropout: input must be [batch, channels, ...]");
    std::vector<int64_t> shape((size_t)x->ndim(), 1);
    shape[0] = x->shape[0];
    shape[1] = x->shape[1];
    auto mask = dropout(op_detail::filled(shape, x->device, 1.0f), p, true);
    return mul(x, mask);
}
// Dropout that keeps the mean and variance of standard-normal inputs (for SELU networks)
inline TensorPtr alpha_dropout(const TensorPtr& x, float p = 0.5f, bool training = true) {
    if (!training || p == 0.0f) return x;
    const float alpha = -1.7580993408473766f;
    float a = 1.0f / std::sqrt((1.0f - p) * (1.0f + p * alpha * alpha));
    float b = -a * alpha * p;
    auto keep = bernoulli(x->shape, 1.0f - p, x->device);
    return add_scalar(mul_scalar(where(keep, x, alpha), a), b);
}
// Samples from the Gumbel-softmax relaxation; hard returns one-hot values with the soft gradient
inline TensorPtr gumbel_softmax(const TensorPtr& logits, float tau = 1.0f, bool hard = false, int dim = -1) {
    auto y = softmax(div_scalar(add(logits, gumbel_like(logits)), tau), dim);
    if (!hard) return y;
    auto index = argmax(y->detach(), dim, true);
    auto one_hot = scatter(op_detail::filled(y->shape, y->device, 0.0f), dim, index, 1.0f);
    return add(sub(one_hot, y->detach()), y);
}
// Drops whole residual branches per sample (stochastic depth)
inline TensorPtr drop_path(const TensorPtr& x, float p, bool training = true) {
    if (!training || p == 0.0f) return x;
    std::vector<int64_t> shape((size_t)x->ndim(), 1);
    shape[0] = x->shape[0];
    return mul(x, div_scalar(bernoulli(shape, 1.0f - p, x->device), 1.0f - p));
}
