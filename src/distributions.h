#pragma once
// Probability distributions (torch.distributions): log_prob, sampling, entropy, moments and KL
// divergences, batched by broadcasting the parameters. sample() draws without a graph; rsample() is the
// reparameterized draw gradients flow through, and throws for distributions without one.

#include "functional.h"
#include "linalg/linalg.h"

#include <memory>
#include <random>

namespace distributions {

constexpr float kHalfLog2Pi = 0.91893853320467274f;

class Distribution {
public:
    virtual ~Distribution() = default;
    virtual TensorPtr log_prob(const TensorPtr& x) const = 0;
    // Draws of shape sample_shape + the parameters' broadcast shape
    virtual TensorPtr rsample(const std::vector<int64_t>& sample_shape = {}) const {
        (void)sample_shape;
        throw std::runtime_error("this distribution has no reparameterized sampler; use sample() (e.g. with a score-function gradient)");
    }
    virtual TensorPtr sample(const std::vector<int64_t>& sample_shape = {}) const {
        NoGradGuard no_grad;
        return rsample(sample_shape);
    }
    virtual TensorPtr entropy() const { throw std::runtime_error("entropy is not available for this distribution"); }
    virtual TensorPtr mean() const { throw std::runtime_error("mean is not available for this distribution"); }
    virtual TensorPtr variance() const { throw std::runtime_error("variance is not available for this distribution"); }
    TensorPtr stddev() const { return sqrt(variance()); }

protected:
    // The shape of a batch of draws, and parameters broadcast to a common shape
    static std::vector<int64_t> join(std::vector<int64_t> a, const std::vector<int64_t>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    }
    static TensorPtr widen(const TensorPtr& t, const std::vector<int64_t>& shape) { return contiguous(expand(t, shape)); }
};

class Normal : public Distribution {
public:
    TensorPtr loc, scale;
    Normal(const TensorPtr& loc_in, const TensorPtr& scale_in) {
        auto shape = op_detail::broadcast_shapes(loc_in->shape, scale_in->shape, "Normal");
        loc = widen(loc_in, shape);
        scale = widen(scale_in, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override {
        auto z = div(sub(x, loc), scale);
        return sub_scalar(sub(mul_scalar(square(z), -0.5f), log(scale)), kHalfLog2Pi);
    }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override { return add(mul(normal(join(s, loc->shape), 0.0f, 1.0f, loc->device), scale), loc); }
    TensorPtr entropy() const override { return add_scalar(log(scale), 0.5f + kHalfLog2Pi); }
    TensorPtr mean() const override { return loc; }
    TensorPtr variance() const override { return square(scale); }
    TensorPtr cdf(const TensorPtr& x) const { return mul_scalar(erfc(mul_scalar(div(sub(x, loc), scale), -0.70710678118654752f)), 0.5f); }
    TensorPtr icdf(const TensorPtr& p) const { return add(loc, mul(scale, mul_scalar(erfinv(sub_scalar(mul_scalar(p, 2.0f), 1.0f)), 1.41421356237309505f))); }
};
// exp of a Normal
class LogNormal : public Distribution {
public:
    Normal base;
    LogNormal(const TensorPtr& loc, const TensorPtr& scale) : base(loc, scale) {}
    TensorPtr log_prob(const TensorPtr& x) const override { return sub(base.log_prob(log(x)), log(x)); }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override { return exp(base.rsample(s)); }
    TensorPtr entropy() const override { return add(base.entropy(), base.loc); }
    TensorPtr mean() const override { return exp(add(base.loc, mul_scalar(square(base.scale), 0.5f))); }
    TensorPtr variance() const override {
        auto v = square(base.scale);
        return mul(expm1(v), exp(add(mul_scalar(base.loc, 2.0f), v)));
    }
};
class HalfNormal : public Distribution {
public:
    Normal base;
    explicit HalfNormal(const TensorPtr& scale) : base(zeros_like(scale), scale) {}
    TensorPtr log_prob(const TensorPtr& x) const override {
        return where(ge(x, 0.0f), add_scalar(base.log_prob(x), 0.69314718055994531f), -std::numeric_limits<float>::infinity());
    }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override { return abs(base.rsample(s)); }
    TensorPtr entropy() const override { return sub_scalar(base.entropy(), 0.69314718055994531f); }
    TensorPtr mean() const override { return mul_scalar(base.scale, 0.79788456080286536f); }
    TensorPtr variance() const override { return mul_scalar(square(base.scale), 1.0f - 0.63661977236758134f); }
};

class Uniform : public Distribution {
public:
    TensorPtr low, high;
    Uniform(const TensorPtr& low_in, const TensorPtr& high_in) {
        auto shape = op_detail::broadcast_shapes(low_in->shape, high_in->shape, "Uniform");
        low = widen(low_in, shape);
        high = widen(high_in, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override {
        return where(logical_and(ge(x, low), lt(x, high)), neg(log(sub(high, low))), -std::numeric_limits<float>::infinity());
    }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override { return add(mul(rand(join(s, low->shape), low->device), sub(high, low)), low); }
    TensorPtr entropy() const override { return log(sub(high, low)); }
    TensorPtr mean() const override { return mul_scalar(add(low, high), 0.5f); }
    TensorPtr variance() const override { return div_scalar(square(sub(high, low)), 12.0f); }
};

class Exponential : public Distribution {
public:
    TensorPtr rate;
    explicit Exponential(const TensorPtr& rate) : rate(rate) {}
    TensorPtr log_prob(const TensorPtr& x) const override { return sub(log(rate), mul(x, rate)); }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override { return div(neg(log1p(neg(rand(join(s, rate->shape), rate->device)))), rate); }
    TensorPtr entropy() const override { return rsub_scalar(1.0f, log(rate)); }
    TensorPtr mean() const override { return reciprocal(rate); }
    TensorPtr variance() const override { return reciprocal(square(rate)); }
};
class Laplace : public Distribution {
public:
    TensorPtr loc, scale;
    Laplace(const TensorPtr& loc_in, const TensorPtr& scale_in) {
        auto shape = op_detail::broadcast_shapes(loc_in->shape, scale_in->shape, "Laplace");
        loc = widen(loc_in, shape);
        scale = widen(scale_in, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override { return neg(add(div(abs(sub(x, loc)), scale), log(mul_scalar(scale, 2.0f)))); }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override {
        auto u = clamp(uniform(join(s, loc->shape), -1.0f, 1.0f, loc->device), -0.9999999f, 0.9999999f);
        auto noise = neg(mul(sign(u), log1p(neg(abs(u)))));
        return add(loc, mul(scale, noise));
    }
    TensorPtr entropy() const override { return add_scalar(log(mul_scalar(scale, 2.0f)), 1.0f); }
    TensorPtr mean() const override { return loc; }
    TensorPtr variance() const override { return mul_scalar(square(scale), 2.0f); }
};
class Cauchy : public Distribution {
public:
    TensorPtr loc, scale;
    Cauchy(const TensorPtr& loc_in, const TensorPtr& scale_in) {
        auto shape = op_detail::broadcast_shapes(loc_in->shape, scale_in->shape, "Cauchy");
        loc = widen(loc_in, shape);
        scale = widen(scale_in, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override {
        return neg(add_scalar(add(log(scale), log1p(square(div(sub(x, loc), scale)))), 1.14472988584940017f));
    }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override {
        auto u = sub_scalar(rand(join(s, loc->shape), loc->device), 0.5f);
        return add(loc, mul(scale, tan(mul_scalar(u, 3.14159265f))));
    }
    TensorPtr entropy() const override { return add_scalar(log(scale), 2.53102424696929079f); }
};
class Gumbel : public Distribution {
public:
    TensorPtr loc, scale;
    Gumbel(const TensorPtr& loc_in, const TensorPtr& scale_in) {
        auto shape = op_detail::broadcast_shapes(loc_in->shape, scale_in->shape, "Gumbel");
        loc = widen(loc_in, shape);
        scale = widen(scale_in, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override {
        auto z = div(sub(x, loc), scale);
        return neg(add(add(z, exp(neg(z))), log(scale)));
    }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override {
        return add(loc, mul(scale, gumbel_like(op_detail::empty(join(s, loc->shape), loc->device))));
    }
    TensorPtr entropy() const override { return add_scalar(log(scale), 1.57721566490153286f); }
    TensorPtr mean() const override { return add(loc, mul_scalar(scale, 0.57721566490153286f)); }
    TensorPtr variance() const override { return mul_scalar(square(scale), 1.64493406684822644f); }
};

namespace detail {
// Gamma(shape, 1) draws by Marsaglia and Tsang (2000), on the host
inline TensorPtr gamma_draws(const TensorPtr& concentration, const std::vector<int64_t>& shape) {
    auto a = contiguous(expand(concentration->detach(), shape))->to_vector();
    std::mt19937_64 rng((uint64_t)random_state().seed * 0x9e3779b97f4a7c15ULL + next_random_stream());
    for (auto& v : a) {
        std::gamma_distribution<double> g((double)v, 1.0);
        v = (float)std::max(g(rng), 1e-37);
    }
    return op_detail::from_values(a, shape, concentration->device);
}
}  // namespace detail

class Gamma : public Distribution {
public:
    TensorPtr concentration, rate;
    Gamma(const TensorPtr& c, const TensorPtr& r) {
        auto shape = op_detail::broadcast_shapes(c->shape, r->shape, "Gamma");
        concentration = widen(c, shape);
        rate = widen(r, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override {
        auto t = add(mul(concentration, log(rate)), mul(sub_scalar(concentration, 1.0f), log(x)));
        return sub(sub(t, mul(rate, x)), lgamma(concentration));
    }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        return div(detail::gamma_draws(concentration, join(s, concentration->shape)), rate);
    }
    TensorPtr entropy() const override {
        return add(add(sub(concentration, log(rate)), lgamma(concentration)), mul(rsub_scalar(1.0f, concentration), digamma(concentration)));
    }
    TensorPtr mean() const override { return div(concentration, rate); }
    TensorPtr variance() const override { return div(concentration, square(rate)); }
};
class Chi2 : public Gamma {
public:
    explicit Chi2(const TensorPtr& df) : Gamma(mul_scalar(df, 0.5f), full_like(df->detach(), 0.5f)) {}
};
class Beta : public Distribution {
public:
    TensorPtr a, b;
    Beta(const TensorPtr& a_in, const TensorPtr& b_in) {
        auto shape = op_detail::broadcast_shapes(a_in->shape, b_in->shape, "Beta");
        a = widen(a_in, shape);
        b = widen(b_in, shape);
    }
    static TensorPtr log_beta(const TensorPtr& a, const TensorPtr& b) { return sub(add(lgamma(a), lgamma(b)), lgamma(add(a, b))); }
    TensorPtr log_prob(const TensorPtr& x) const override {
        return sub(add(mul(sub_scalar(a, 1.0f), log(x)), mul(sub_scalar(b, 1.0f), log1p(neg(x)))), log_beta(a, b));
    }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        auto shape = join(s, a->shape);
        auto x = detail::gamma_draws(a, shape), y = detail::gamma_draws(b, shape);
        return div(x, add(x, y));
    }
    TensorPtr entropy() const override {
        auto total = add(a, b);
        auto t = sub(log_beta(a, b), add(mul(sub_scalar(a, 1.0f), digamma(a)), mul(sub_scalar(b, 1.0f), digamma(b))));
        return add(t, mul(sub_scalar(total, 2.0f), digamma(total)));
    }
    TensorPtr mean() const override { return div(a, add(a, b)); }
    TensorPtr variance() const override {
        auto total = add(a, b);
        return div(mul(a, b), mul(square(total), add_scalar(total, 1.0f)));
    }
};
// Over the last dimension
class Dirichlet : public Distribution {
public:
    TensorPtr concentration;
    explicit Dirichlet(const TensorPtr& concentration) : concentration(concentration) {}
    TensorPtr log_prob(const TensorPtr& x) const override {
        auto c = concentration;
        auto t = sum(mul(sub_scalar(c, 1.0f), log(x)), -1, false);
        return sub(add(t, lgamma(sum(c, -1, false))), sum(lgamma(c), -1, false));
    }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        auto g = detail::gamma_draws(concentration, join(s, concentration->shape));
        return div(g, sum(g, -1, true));
    }
    TensorPtr mean() const override { return div(concentration, sum(concentration, -1, true)); }
    TensorPtr entropy() const override {
        auto c = concentration;
        auto total = sum(c, -1, false);
        auto log_b = sub(sum(lgamma(c), -1, false), lgamma(total));
        auto k = (float)c->shape.back();
        return sub(add(log_b, mul(sub_scalar(total, k), digamma(total))), sum(mul(sub_scalar(c, 1.0f), digamma(c)), -1, false));
    }
};
class StudentT : public Distribution {
public:
    TensorPtr df, loc, scale;
    StudentT(const TensorPtr& df_in, const TensorPtr& loc_in, const TensorPtr& scale_in) {
        auto shape = op_detail::broadcast_shapes(op_detail::broadcast_shapes(df_in->shape, loc_in->shape, "StudentT"), scale_in->shape, "StudentT");
        df = widen(df_in, shape);
        loc = widen(loc_in, shape);
        scale = widen(scale_in, shape);
    }
    TensorPtr log_prob(const TensorPtr& x) const override {
        auto z = div(sub(x, loc), scale);
        auto half = mul_scalar(add_scalar(df, 1.0f), 0.5f);
        auto norm_const = sub(sub(lgamma(half), lgamma(mul_scalar(df, 0.5f))), add(mul_scalar(log(mul_scalar(df, 3.14159265358979f)), 0.5f), log(scale)));
        return sub(norm_const, mul(half, log1p(div(square(z), df))));
    }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        auto shape = join(s, df->shape);
        auto chi = mul_scalar(detail::gamma_draws(mul_scalar(df, 0.5f), shape), 2.0f);
        return add(loc, mul(scale, mul(normal(shape, 0.0f, 1.0f, df->device), rsqrt(div(chi, df)))));
    }
    TensorPtr mean() const override { return where(gt(df, 1.0f), loc, std::numeric_limits<float>::quiet_NaN()); }
};

class Bernoulli : public Distribution {
public:
    TensorPtr logits;
    // From logits; with from_probs, from probabilities
    explicit Bernoulli(const TensorPtr& p, bool from_probs = false) : logits(from_probs ? logit(p, 1e-7f) : p) {}
    TensorPtr probs() const { return sigmoid(logits); }
    TensorPtr log_prob(const TensorPtr& x) const override { return neg(add(mul(rsub_scalar(1.0f, x), logits), softplus(neg(logits)))); }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        return bernoulli(contiguous(expand(probs(), join(s, logits->shape))));
    }
    TensorPtr entropy() const override { return add(mul(rsub_scalar(1.0f, probs()), logits), softplus(neg(logits))); }
    TensorPtr mean() const override { return probs(); }
    TensorPtr variance() const override {
        auto p = probs();
        return mul(p, rsub_scalar(1.0f, p));
    }
};
// Class indices under logits [..., classes]
class Categorical : public Distribution {
public:
    TensorPtr logits;  // normalized: log-probabilities
    explicit Categorical(const TensorPtr& values, bool from_probs = false)
        : logits(from_probs ? sub(log(clamp_min(values, 1e-38f)), log(sum(values, -1, true))) : log_softmax(values, -1)) {}
    TensorPtr probs() const { return exp(logits); }
    TensorPtr log_prob(const TensorPtr& index) const override {
        auto shape = index->shape;
        shape.push_back(logits->shape.back());
        return squeeze(gather(expand(logits, shape), -1, unsqueeze(index, -1)), -1);
    }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        // Gumbel-max: the argmax of logits + Gumbel noise is a draw
        auto l = contiguous(expand(logits->detach(), join(s, logits->shape)));
        return argmax(add(l, gumbel_like(l)), -1, false);
    }
    TensorPtr entropy() const override { return neg(sum(mul(probs(), logits), -1, false)); }
    TensorPtr mode() const { return argmax(logits, -1, false); }
};
class OneHotCategorical : public Distribution {
public:
    Categorical base;
    explicit OneHotCategorical(const TensorPtr& values, bool from_probs = false) : base(values, from_probs) {}
    TensorPtr log_prob(const TensorPtr& x) const override { return sum(mul(x, base.logits), -1, false); }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        auto i = unsqueeze(base.sample(s), -1);
        auto shape = join(s, base.logits->shape);
        return scatter(op_detail::filled(shape, i->device, 0.0f), -1, i, 1.0f);
    }
    // Straight-through: one-hot values with the probabilities' gradient
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override {
        auto p = base.probs();
        return add(sample(s), sub(p, p->detach()));
    }
    TensorPtr entropy() const override { return base.entropy(); }
    TensorPtr mean() const override { return base.probs(); }
};
class Poisson : public Distribution {
public:
    TensorPtr rate;
    explicit Poisson(const TensorPtr& rate) : rate(rate) {}
    TensorPtr log_prob(const TensorPtr& x) const override { return sub(sub(xlogy(x, rate), rate), lgamma(add_scalar(x, 1.0f))); }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        auto shape = join(s, rate->shape);
        auto v = contiguous(expand(rate->detach(), shape))->to_vector();
        std::mt19937_64 rng((uint64_t)random_state().seed * 0x9e3779b97f4a7c15ULL + next_random_stream());
        for (auto& r : v) r = r > 0.0f ? (float)std::poisson_distribution<int64_t>((double)r)(rng) : 0.0f;
        return op_detail::from_values(v, shape, rate->device);
    }
    TensorPtr mean() const override { return rate; }
    TensorPtr variance() const override { return rate; }
};
// Failures before the first success
class Geometric : public Distribution {
public:
    TensorPtr probs;
    explicit Geometric(const TensorPtr& probs) : probs(probs) {}
    TensorPtr log_prob(const TensorPtr& k) const override { return add(mul(k, log1p(neg(probs))), log(probs)); }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        auto u = clamp_min(rand(join(s, probs->shape), probs->device), 1e-12f);
        return floor(div(log(u), log1p(neg(probs))));
    }
    TensorPtr mean() const override { return sub_scalar(reciprocal(probs), 1.0f); }
    TensorPtr variance() const override { return div(rsub_scalar(1.0f, probs), square(probs)); }
};
class Binomial : public Distribution {
public:
    int64_t trials;
    Bernoulli base;
    Binomial(int64_t trials, const TensorPtr& p, bool from_probs = false) : trials(trials), base(p, from_probs) {}
    TensorPtr log_prob(const TensorPtr& k) const override {
        float n = (float)trials;
        auto choose = sub(rsub_scalar(std::lgamma(n + 1.0f), lgamma(add_scalar(k, 1.0f))), lgamma(rsub_scalar(n + 1.0f, k)));
        return add(choose, sub(mul(k, base.logits), mul_scalar(softplus(base.logits), n)));
    }
    TensorPtr sample(const std::vector<int64_t>& s = {}) const override {
        NoGradGuard no_grad;
        auto shape = join(s, base.logits->shape);
        auto p = contiguous(expand(base.probs(), shape))->to_vector();
        std::mt19937_64 rng((uint64_t)random_state().seed * 0x9e3779b97f4a7c15ULL + next_random_stream());
        for (auto& v : p) v = (float)std::binomial_distribution<int64_t>(trials, std::min(1.0, std::max(0.0, (double)v)))(rng);
        return op_detail::from_values(p, shape, base.logits->device);
    }
    TensorPtr mean() const override { return mul_scalar(base.probs(), (float)trials); }
    TensorPtr variance() const override { return mul_scalar(base.variance(), (float)trials); }
};

// Mean [..., n] and lower-triangular scale L [..., n, n] (covariance L L^T)
class MultivariateNormal : public Distribution {
public:
    TensorPtr loc, scale_tril;
    MultivariateNormal(const TensorPtr& loc, const TensorPtr& scale_tril) : loc(loc), scale_tril(scale_tril) {}
    static MultivariateNormal from_covariance(const TensorPtr& loc, const TensorPtr& covariance) { return MultivariateNormal(loc, linalg::cholesky(covariance)); }
    TensorPtr log_prob(const TensorPtr& x) const override {
        auto d = unsqueeze(sub(x, loc), -1);
        auto l_shape = std::vector<int64_t>(d->shape.begin(), d->shape.end() - 2);
        l_shape.push_back(loc->shape.back());
        l_shape.push_back(loc->shape.back());
        auto z = squeeze(linalg::solve_triangular(contiguous(expand(scale_tril, l_shape)), d, false), -1);
        float n = (float)loc->shape.back();
        auto half_logdet = sum(log(diagonal(scale_tril, 0, -2, -1)), -1, false);
        return sub_scalar(sub(mul_scalar(sum(square(z), -1, false), -0.5f), half_logdet), n * kHalfLog2Pi);
    }
    TensorPtr rsample(const std::vector<int64_t>& s = {}) const override {
        auto z = normal(join(s, loc->shape), 0.0f, 1.0f, loc->device);
        return add(squeeze(matmul(scale_tril, unsqueeze(z, -1)), -1), loc);
    }
    TensorPtr entropy() const override {
        float n = (float)loc->shape.back();
        return add_scalar(sum(log(diagonal(scale_tril, 0, -2, -1)), -1, false), n * (0.5f + kHalfLog2Pi));
    }
    TensorPtr mean() const override { return loc; }
};

// KL(p || q)
inline TensorPtr kl_divergence(const Normal& p, const Normal& q) {
    auto ratio = square(div(p.scale, q.scale));
    auto shift = square(div(sub(p.loc, q.loc), q.scale));
    return mul_scalar(sub(sub_scalar(add(ratio, shift), 1.0f), log(ratio)), 0.5f);
}
inline TensorPtr kl_divergence(const Categorical& p, const Categorical& q) { return sum(mul(p.probs(), sub(p.logits, q.logits)), -1, false); }
inline TensorPtr kl_divergence(const Bernoulli& p, const Bernoulli& q) {
    auto a = p.probs();
    auto t1 = mul(a, sub(log_sigmoid(p.logits), log_sigmoid(q.logits)));
    auto t0 = mul(rsub_scalar(1.0f, a), sub(log_sigmoid(neg(p.logits)), log_sigmoid(neg(q.logits))));
    return add(t1, t0);
}
inline TensorPtr kl_divergence(const Exponential& p, const Exponential& q) {
    auto r = div(q.rate, p.rate);
    return sub_scalar(sub(r, log(r)), 1.0f);
}
inline TensorPtr kl_divergence(const Gamma& p, const Gamma& q) {
    auto t1 = mul(q.concentration, log(div(p.rate, q.rate)));
    auto t2 = sub(lgamma(q.concentration), lgamma(p.concentration));
    auto t3 = mul(sub(p.concentration, q.concentration), digamma(p.concentration));
    auto t4 = mul(sub(q.rate, p.rate), div(p.concentration, p.rate));
    return add(add(t1, t2), add(t3, t4));
}
inline TensorPtr kl_divergence(const Beta& p, const Beta& q) {
    auto total = add(p.a, p.b);
    auto t = sub(Beta::log_beta(q.a, q.b), Beta::log_beta(p.a, p.b));
    t = add(t, add(mul(sub(p.a, q.a), digamma(p.a)), mul(sub(p.b, q.b), digamma(p.b))));
    return add(t, mul(add(sub(q.a, p.a), sub(q.b, p.b)), digamma(total)));
}
inline TensorPtr kl_divergence(const MultivariateNormal& p, const MultivariateNormal& q) {
    auto m = linalg::solve_triangular(q.scale_tril, p.scale_tril, false);
    auto d = linalg::solve_triangular(q.scale_tril, unsqueeze(sub(q.loc, p.loc), -1), false);
    float n = (float)p.loc->shape.back();
    auto logdet = sub(sum(log(diagonal(q.scale_tril, 0, -2, -1)), -1, false), sum(log(diagonal(p.scale_tril, 0, -2, -1)), -1, false));
    auto quad = add(sum(square(m), std::vector<int>{-2, -1}, false), sum(square(d), std::vector<int>{-2, -1}, false));
    return add(logdet, mul_scalar(sub_scalar(quad, n), 0.5f));
}

}  // namespace distributions
