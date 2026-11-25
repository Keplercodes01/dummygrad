#pragma once
// The rest of torch.optim: SGD with Nesterov momentum, Adam (L2 and AMSGrad), NAdam, RAdam, Adamax,
// Adagrad, Adadelta, RMSprop, Rprop, LBFGS, and Lion, Muon and Adafactor. Plus parameter groups by name,
// learning-rate schedulers, EMA and SWA weight averaging.
//
// Adam in optimizers.h is AdamW (decoupled decay); SGD there is plain momentum SGD. All updates run as one
// kernel per parameter on its device. Every optimizer has lr, step(params), zero_grad(params),
// set_weight_decay, set_lr_scale and state_dict / load_state_dict for checkpoints.

#include "optimizers.h"
#include "layers.h"

#include <array>
#include <deque>
#include <limits>

namespace optim {

// State common to optimizers whose update is one kernel over a few buffers per parameter
class Optimizer {
public:
    float lr;
    float weight_decay;
    explicit Optimizer(float lr, float weight_decay) : lr(lr), weight_decay(weight_decay) {}
    virtual ~Optimizer() = default;

    void step(const std::vector<std::shared_ptr<Tensor>>& params) {
        for (const auto& p : params) step(p);
        state_.prune();
    }
    void step(const std::shared_ptr<Tensor>& p) {
        if (!p || !p->grad) return;
        optim_detail::check_param(p, name());
        auto g = make_contiguous(p->grad);
        auto& s = state_.get(p);
        if (s.buffers.size() < buffer_names().size()) {
            s.buffers.resize(buffer_names().size());
        }
        s.step++;
        p->storage->bump_version();  // the update changes p in place
        update(p, g, s, lr * scales_.get(p, 1.0f), decay_.get(p, weight_decay));
        if (p->device == Device::TPU) p->storage->tpu_handle.reset();
    }
    void zero_grad(const std::vector<std::shared_ptr<Tensor>>& params) { ::zero_grad(params); }
    // Weight decay for these parameters instead of weight_decay
    void set_weight_decay(const std::vector<std::shared_ptr<Tensor>>& params, float wd) { decay_.set(params, wd); }
    // Multiplies lr for these parameters (e.g. 0.1 for a pretrained backbone)
    void set_lr_scale(const std::vector<std::shared_ptr<Tensor>>& params, float scale) { scales_.set(params, scale); }

    NamedTensors state_dict(const NamedTensors& named_params) const {
        NamedTensors out;
        for (const auto& [param_name, p] : named_params) {
            const State* s = state_.find(p);
            if (!s || s->step == 0) continue;
            for (size_t i = 0; i < s->buffers.size(); ++i) {
                if (s->buffers[i]) out.emplace_back(param_name + "." + buffer_names()[i], s->buffers[i]);
            }
            out.emplace_back(param_name + ".step", io::encode_int64(s->step));
        }
        return out;
    }
    void load_state_dict(const NamedTensors& named_params, const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state) {
        for (const auto& [param_name, p] : named_params) {
            auto step = state.find(param_name + ".step");
            if (step == state.end()) continue;
            auto& s = state_.get(p);
            s.buffers.assign(buffer_names().size(), nullptr);
            for (size_t i = 0; i < buffer_names().size(); ++i) {
                auto it = state.find(param_name + "." + buffer_names()[i]);
                if (it == state.end()) continue;
                s.buffers[i] = zeros(it->second->shape, p->device);
                optim_detail::load_into(s.buffers[i], it->second, it->first);
            }
            s.step = io::decode_int64(step->second);
        }
    }

protected:
    struct State {
        std::vector<std::shared_ptr<Tensor>> buffers;
        int64_t step = 0;
        void follow(const std::shared_ptr<Tensor>& p) {
            for (auto& b : buffers) optim_detail::follow_device(b, p);
        }
        // Buffer i, created as zeros of p's shape when first used
        float* buffer(size_t i, const std::shared_ptr<Tensor>& p, float initial = 0.0f) {
            if (!buffers[i]) {
                buffers[i] = zeros(p->shape, p->device);
                if (initial != 0.0f) buffers[i]->fill_(initial);
            }
            return buffers[i]->data_ptr<float>();
        }
    };
    virtual const char* name() const = 0;
    virtual const std::vector<std::string>& buffer_names() const = 0;
    virtual void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) = 0;

    optim_detail::StateMap<State> state_;
    optim_detail::DecayOverrides decay_;   // per-parameter weight decay
    optim_detail::DecayOverrides scales_;  // per-parameter lr multiplier
};

#define DG_OPTIMIZER(label, ...)                                                             \
protected:                                                                                   \
    const char* name() const override { return label; }                                      \
    const std::vector<std::string>& buffer_names() const override {                          \
        static const std::vector<std::string> names = {__VA_ARGS__};                         \
        return names;                                                                        \
    }                                                                                        \
                                                                                             \
public:

// SGD with momentum, dampening, Nesterov momentum and L2 weight decay (torch.optim.SGD)
class SGD : public Optimizer {
    DG_OPTIMIZER("SGD", "momentum_buffer")
    float momentum, dampening;
    bool nesterov;
    explicit SGD(float lr = 1e-3f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false, float dampening = 0.0f)
        : Optimizer(lr, weight_decay), momentum(momentum), dampening(dampening), nesterov(nesterov) {
        if (nesterov && (momentum <= 0.0f || dampening != 0.0f)) throw std::invalid_argument("SGD: Nesterov momentum needs momentum > 0 and no dampening");
    }

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        bool first = !s.buffers[0];
        float* buf = momentum != 0.0f ? s.buffer(0, p) : nullptr;
        DG_KERNEL(p->device, sgd_step(p->data_ptr<float>(), g->data_ptr<float>(), buf, p->size(), lr, momentum, dampening, wd, nesterov, first));
    }
};

// Adam with L2 decay (added to the gradient); amsgrad keeps the largest second moment
class Adam : public Optimizer {
    DG_OPTIMIZER("Adam", "exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    float beta1, beta2, eps;
    bool amsgrad, decoupled;
    explicit Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f, bool amsgrad = false, bool decoupled = false)
        : Optimizer(lr, weight_decay), beta1(beta1), beta2(beta2), eps(eps), amsgrad(amsgrad), decoupled(decoupled) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        float bias1 = 1.0f - (float)std::pow((double)beta1, (double)s.step), bias2 = 1.0f - (float)std::pow((double)beta2, (double)s.step);
        float *m = s.buffer(0, p), *v = s.buffer(1, p), *vmax = amsgrad ? s.buffer(2, p) : nullptr;
        DG_KERNEL(p->device, adam_step(p->data_ptr<float>(), g->data_ptr<float>(), m, v, vmax, p->size(), lr, beta1, beta2, eps, wd, decoupled, bias1, bias2));
    }
};
// Adam with decoupled weight decay
class AdamW : public Adam {
public:
    explicit AdamW(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f, bool amsgrad = false)
        : Adam(lr, beta1, beta2, eps, weight_decay, amsgrad, true) {}
};

// Adam with Nesterov momentum (Dozat, 2016)
class NAdam : public Optimizer {
    DG_OPTIMIZER("NAdam", "exp_avg", "exp_avg_sq")
    float beta1, beta2, eps, momentum_decay;
    explicit NAdam(float lr = 2e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f, float momentum_decay = 4e-3f)
        : Optimizer(lr, weight_decay), beta1(beta1), beta2(beta2), eps(eps), momentum_decay(momentum_decay) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        auto mu = [&](int64_t t) { return beta1 * (1.0 - 0.5 * std::pow(0.96, (double)t * momentum_decay)); };
        double product = 1.0;
        for (int64_t t = 1; t <= s.step; ++t) product *= mu(t);
        double mu_t = mu(s.step), mu_next = mu(s.step + 1);
        float bias2 = 1.0f - (float)std::pow((double)beta2, (double)s.step);
        float *m = s.buffer(0, p), *v = s.buffer(1, p);
        DG_KERNEL(p->device, nadam_step(p->data_ptr<float>(), g->data_ptr<float>(), m, v, p->size(), lr, beta1, beta2, eps, wd,
                                        (float)((1.0 - mu_t) / (1.0 - product)), (float)(mu_next / (1.0 - product * mu_next)), bias2));
    }
};

// Rectified Adam (Liu et al., 2020)
class RAdam : public Optimizer {
    DG_OPTIMIZER("RAdam", "exp_avg", "exp_avg_sq")
    float beta1, beta2, eps;
    explicit RAdam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
        : Optimizer(lr, weight_decay), beta1(beta1), beta2(beta2), eps(eps) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        double t = (double)s.step, b2t = std::pow((double)beta2, t);
        double rho_inf = 2.0 / (1.0 - beta2) - 1.0, rho = rho_inf - 2.0 * t * b2t / (1.0 - b2t);
        double rect = rho > 5.0 ? std::sqrt((rho - 4.0) * (rho - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho)) : 0.0;
        float *m = s.buffer(0, p), *v = s.buffer(1, p);
        DG_KERNEL(p->device, radam_step(p->data_ptr<float>(), g->data_ptr<float>(), m, v, p->size(), lr, beta1, beta2, eps, wd,
                                        1.0f - (float)std::pow((double)beta1, t), 1.0f - (float)b2t, (float)rect));
    }
};

class Adamax : public Optimizer {
    DG_OPTIMIZER("Adamax", "exp_avg", "exp_inf")
    float beta1, beta2, eps;
    explicit Adamax(float lr = 2e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
        : Optimizer(lr, weight_decay), beta1(beta1), beta2(beta2), eps(eps) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        float *m = s.buffer(0, p), *u = s.buffer(1, p);
        DG_KERNEL(p->device, adamax_step(p->data_ptr<float>(), g->data_ptr<float>(), m, u, p->size(), lr, beta1, beta2, eps, wd,
                                         1.0f - (float)std::pow((double)beta1, (double)s.step)));
    }
};

class Adagrad : public Optimizer {
    DG_OPTIMIZER("Adagrad", "sum")
    float lr_decay, eps, initial;
    explicit Adagrad(float lr = 1e-2f, float lr_decay = 0.0f, float weight_decay = 0.0f, float initial = 0.0f, float eps = 1e-10f)
        : Optimizer(lr, weight_decay), lr_decay(lr_decay), eps(eps), initial(initial) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        float* sum = s.buffer(0, p, initial);
        DG_KERNEL(p->device, adagrad_step(p->data_ptr<float>(), g->data_ptr<float>(), sum, p->size(), lr / (1.0f + (float)(s.step - 1) * lr_decay), eps, wd));
    }
};

class Adadelta : public Optimizer {
    DG_OPTIMIZER("Adadelta", "square_avg", "acc_delta")
    float rho, eps;
    explicit Adadelta(float lr = 1.0f, float rho = 0.9f, float eps = 1e-6f, float weight_decay = 0.0f) : Optimizer(lr, weight_decay), rho(rho), eps(eps) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        float *sq = s.buffer(0, p), *acc = s.buffer(1, p);
        DG_KERNEL(p->device, adadelta_step(p->data_ptr<float>(), g->data_ptr<float>(), sq, acc, p->size(), lr, rho, eps, wd));
    }
};

class RMSprop : public Optimizer {
    DG_OPTIMIZER("RMSprop", "square_avg", "grad_avg", "momentum_buffer")
    float alpha, eps, momentum;
    bool centered;
    explicit RMSprop(float lr = 1e-2f, float alpha = 0.99f, float eps = 1e-8f, float weight_decay = 0.0f, float momentum = 0.0f, bool centered = false)
        : Optimizer(lr, weight_decay), alpha(alpha), eps(eps), momentum(momentum), centered(centered) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        float *sq = s.buffer(0, p), *avg = centered ? s.buffer(1, p) : nullptr, *buf = momentum > 0.0f ? s.buffer(2, p) : nullptr;
        DG_KERNEL(p->device, rmsprop_step(p->data_ptr<float>(), g->data_ptr<float>(), sq, avg, buf, p->size(), lr, alpha, eps, wd, momentum));
    }
};

// Sign-based steps that adapt per weight, for full-batch training
class Rprop : public Optimizer {
    DG_OPTIMIZER("Rprop", "prev", "step_size")
    float eta_minus, eta_plus, step_min, step_max;
    explicit Rprop(float lr = 1e-2f, float eta_minus = 0.5f, float eta_plus = 1.2f, float step_min = 1e-6f, float step_max = 50.0f)
        : Optimizer(lr, 0.0f), eta_minus(eta_minus), eta_plus(eta_plus), step_min(step_min), step_max(step_max) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float) override {
        float *prev = s.buffer(0, p), *size = s.buffer(1, p, lr);
        DG_KERNEL(p->device, rprop_step(p->data_ptr<float>(), g->data_ptr<float>(), prev, size, p->size(), eta_minus, eta_plus, step_min, step_max));
    }
};

// Evolved sign momentum (Chen et al., 2023)
class Lion : public Optimizer {
    DG_OPTIMIZER("Lion", "exp_avg")
    float beta1, beta2;
    explicit Lion(float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.99f, float weight_decay = 0.0f) : Optimizer(lr, weight_decay), beta1(beta1), beta2(beta2) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        float* m = s.buffer(0, p);
        DG_KERNEL(p->device, lion_step(p->data_ptr<float>(), g->data_ptr<float>(), m, p->size(), lr, beta1, beta2, wd));
    }
};

// Muon (Jordan et al., 2024): momentum orthogonalized by a Newton-Schulz iteration, for matrices. Give
// embeddings, biases and norms to AdamW; this one throws on parameters that are not at least 2-D.
class Muon : public Optimizer {
    DG_OPTIMIZER("Muon", "momentum_buffer")
    float momentum;
    bool nesterov;
    int iterations;
    explicit Muon(float lr = 2e-2f, float momentum = 0.95f, float weight_decay = 0.0f, bool nesterov = true, int iterations = 5)
        : Optimizer(lr, weight_decay), momentum(momentum), nesterov(nesterov), iterations(iterations) {}

    // Roughly the orthogonal factor of m: its singular values pushed towards 1
    static std::shared_ptr<Tensor> orthogonalize(const std::shared_ptr<Tensor>& m, int iterations) {
        NoGradGuard no_grad;
        auto x = reshape(m, {m->shape[0], m->size() / m->shape[0]});
        bool tall = x->shape[0] > x->shape[1];
        if (tall) x = contiguous(transpose(x, 0, 1));
        x = mul_scalar(x, 1.0f / (norm(x)->item() + 1e-7f));
        for (int i = 0; i < iterations; ++i) {
            auto a = matmul(x, transpose(x, 0, 1));
            auto b = add(mul_scalar(a, -4.7750f), mul_scalar(matmul(a, a), 2.0315f));
            x = add(mul_scalar(x, 3.4445f), matmul(b, x));
        }
        if (tall) x = contiguous(transpose(x, 0, 1));
        return reshape(x, m->shape);
    }

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        if (p->ndim() < 2) throw std::invalid_argument("Muon: parameters must be at least 2-D (use AdamW for biases and norms)");
        NoGradGuard no_grad;
        s.buffer(0, p);
        ops::make_contiguous(*s.buffers[0], *add(mul_scalar(s.buffers[0], momentum), g));
        auto d = nesterov ? add(g, mul_scalar(s.buffers[0], momentum)) : s.buffers[0];
        auto u = orthogonalize(d, iterations);
        double rows = (double)p->shape[0], cols = (double)p->size() / rows;
        float scale = (float)std::sqrt(std::max(1.0, rows / cols));
        if (wd != 0.0f) ops::make_contiguous(*p, *mul_scalar(p, 1.0f - lr * wd));
        ops::add_inplace(*p, *mul_scalar(u, -lr * scale));
    }
};

// Adafactor (Shazeer & Stern, 2018): row and column statistics instead of a full second moment
class Adafactor : public Optimizer {
    DG_OPTIMIZER("Adafactor", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq")
    float clip_threshold, decay_rate, eps;
    explicit Adafactor(float lr = 1e-3f, float clip_threshold = 1.0f, float decay_rate = -0.8f, float weight_decay = 0.0f, float eps = 1e-30f)
        : Optimizer(lr, weight_decay), clip_threshold(clip_threshold), decay_rate(decay_rate), eps(eps) {}

protected:
    void update(const std::shared_ptr<Tensor>& p, const std::shared_ptr<Tensor>& g, State& s, float lr, float wd) override {
        NoGradGuard no_grad;
        float beta2 = 1.0f - std::pow((float)s.step, decay_rate);
        auto sq = add_scalar(square(g), eps);
        auto blend = [&](size_t i, const std::shared_ptr<Tensor>& v) {
            s.buffers[i] = s.buffers[i] ? add(mul_scalar(s.buffers[i], beta2), mul_scalar(v, 1.0f - beta2)) : mul_scalar(v, 1.0f - beta2);
            return s.buffers[i];
        };
        std::shared_ptr<Tensor> u;
        if (p->ndim() >= 2) {
            auto row = blend(0, mean(sq, -1, true)), col = blend(1, mean(sq, -2, true));
            u = mul(g, mul(rsqrt(div(row, mean(row, -2, true))), rsqrt(col)));
        } else {
            u = mul(g, rsqrt(blend(2, sq)));
        }
        float rms = norm(u)->item() / std::sqrt((float)u->size());
        u = mul_scalar(u, 1.0f / std::max(1.0f, rms / clip_threshold));
        if (wd != 0.0f) ops::make_contiguous(*p, *mul_scalar(p, 1.0f - lr * wd));
        ops::add_inplace(*p, *mul_scalar(u, -lr));
    }
};

#undef DG_OPTIMIZER

// Limited-memory BFGS with an optional strong-Wolfe line search (torch.optim.LBFGS). The closure clears
// gradients, computes the loss, calls backward and returns the loss:
//
//   LBFGS opt(params);
//   opt.step([&] { zero_grad(params); auto loss = f(); loss->backward(); return loss->item(); });
class LBFGS {
public:
    float lr;
    int max_iter, history_size;
    float tolerance_grad, tolerance_change;
    bool strong_wolfe;

    explicit LBFGS(std::vector<std::shared_ptr<Tensor>> params, float lr = 1.0f, int max_iter = 20, int history_size = 100,
                   bool strong_wolfe = true, float tolerance_grad = 1e-7f, float tolerance_change = 1e-9f)
        : lr(lr), max_iter(max_iter), history_size(history_size), tolerance_grad(tolerance_grad), tolerance_change(tolerance_change),
          strong_wolfe(strong_wolfe), params_(std::move(params)) {}

    float step(const std::function<float()>& closure) {
        using Vec = std::vector<double>;
        double loss = closure(), first = loss;
        Vec g = gather_grad();
        if (max_abs(g) <= tolerance_grad) return (float)first;
        for (int it = 0; it < max_iter; ++it) {
            Vec d = direction(g);
            double gd = dotp(g, d);
            if (gd >= 0.0) {  // not a descent direction: the history is no use here
                s_.clear();
                y_.clear();
                d = direction(g);
                gd = dotp(g, d);
            }
            if (gd > -tolerance_change) break;
            double t = s_.empty() ? std::min(1.0, 1.0 / sum_abs(g)) * lr : lr;
            Vec x0 = gather_params(), g0 = g;
            double loss0 = loss;
            if (strong_wolfe) {
                t = wolfe(closure, x0, d, loss0, gd, t, loss, g);
            } else {
                assign(x0, d, t);
                loss = closure();
                g = gather_grad();
            }
            Vec s(d.size()), y(d.size());
            for (size_t i = 0; i < d.size(); ++i) {
                s[i] = t * d[i];
                y[i] = g[i] - g0[i];
            }
            if (dotp(y, s) > 1e-10) {
                s_.push_back(std::move(s));
                y_.push_back(std::move(y));
                if ((int)s_.size() > history_size) {
                    s_.pop_front();
                    y_.pop_front();
                }
            }
            if (max_abs(g) <= tolerance_grad || std::fabs(loss - loss0) < tolerance_change || std::fabs(t) * max_abs(d) <= tolerance_change) break;
            if (!std::isfinite(loss)) break;
        }
        return (float)first;
    }

private:
    using Vec = std::vector<double>;
    std::vector<std::shared_ptr<Tensor>> params_;
    std::deque<Vec> s_, y_;

    static double dotp(const Vec& a, const Vec& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }
    static double max_abs(const Vec& a) {
        double m = 0.0;
        for (double v : a) m = std::max(m, std::fabs(v));
        return m;
    }
    static double sum_abs(const Vec& a) {
        double m = 0.0;
        for (double v : a) m += std::fabs(v);
        return m;
    }
    Vec gather_params() const {
        Vec out;
        for (const auto& p : params_) for (float v : p->to_vector()) out.push_back(v);
        return out;
    }
    Vec gather_grad() const {
        Vec out;
        for (const auto& p : params_) {
            if (p->grad) for (float v : p->grad->to_vector()) out.push_back(v);
            else out.insert(out.end(), (size_t)p->size(), 0.0);
        }
        return out;
    }
    // x0 + t d into the parameters
    void assign(const Vec& x0, const Vec& d, double t) {
        size_t at = 0;
        for (const auto& p : params_) {
            std::vector<float> v((size_t)p->size());
            for (auto& x : v) {
                x = (float)(x0[at] + t * d[at]);
                ++at;
            }
            init::set_values(p, op_detail::from_values(v, p->shape, Device::CPU));
        }
    }
    // -H g by the two-loop recursion
    Vec direction(const Vec& g) const {
        Vec q = g;
        for (double& v : q) v = -v;
        std::vector<double> alpha(s_.size());
        for (size_t i = s_.size(); i-- > 0;) {
            alpha[i] = dotp(s_[i], q) / dotp(y_[i], s_[i]);
            for (size_t j = 0; j < q.size(); ++j) q[j] -= alpha[i] * y_[i][j];
        }
        if (!s_.empty()) {
            double h = dotp(s_.back(), y_.back()) / dotp(y_.back(), y_.back());
            for (double& v : q) v *= h;
        }
        for (size_t i = 0; i < s_.size(); ++i) {
            double beta = dotp(y_[i], q) / dotp(y_[i], s_[i]);
            for (size_t j = 0; j < q.size(); ++j) q[j] += s_[i][j] * (alpha[i] - beta);
        }
        return q;
    }
    // A step satisfying the strong Wolfe conditions (Nocedal & Wright, algorithms 3.5 and 3.6, by bisection)
    double wolfe(const std::function<float()>& closure, const Vec& x0, const Vec& d, double f0, double g0d, double t, double& f, Vec& g) {
        const double c1 = 1e-4, c2 = 0.9;
        double lo = 0.0, hi = std::numeric_limits<double>::infinity();
        double best_t = 0.0, best_f = f0;
        Vec best_g = g;
        for (int i = 0; i < 25; ++i) {
            assign(x0, d, t);
            f = closure();
            g = gather_grad();
            double gd = dotp(g, d);
            bool finite = std::isfinite(f);
            if (finite && f < best_f) {
                best_f = f;
                best_t = t;
                best_g = g;
            }
            if (!finite || f > f0 + c1 * t * g0d) hi = t;  // too far
            else if (std::fabs(gd) <= -c2 * g0d) return t;
            else if (gd < 0.0) lo = t;                      // still descending
            else hi = t;                                    // past the minimum
            t = std::isinf(hi) ? 2.0 * t : 0.5 * (lo + hi);
            if (hi - lo < 1e-12) break;
        }
        assign(x0, d, best_t);
        f = best_f;
        g = best_g;
        return best_t;
    }
};

// ---------------------------------------------------------------------------------------------
// Learning-rate schedules: get_lr() is the rate for the coming step; call step() after each one
// ---------------------------------------------------------------------------------------------

class LRScheduler {
public:
    float base_lr;
    int64_t current_step = 0;
    explicit LRScheduler(float base_lr) : base_lr(base_lr) {}
    virtual ~LRScheduler() = default;
    virtual float lr_at(int64_t step) const = 0;
    float get_lr() const { return lr_at(current_step); }
    void step() { ++current_step; }
    // Steps and writes the new rate into an optimizer
    template <typename Opt>
    void step(Opt& opt) {
        step();
        opt.lr = get_lr();
    }
};

class ConstantLR : public LRScheduler {
public:
    using LRScheduler::LRScheduler;
    float lr_at(int64_t) const override { return base_lr; }
};
class LambdaLR : public LRScheduler {
public:
    std::function<float(int64_t)> factor;
    LambdaLR(float base_lr, std::function<float(int64_t)> factor) : LRScheduler(base_lr), factor(std::move(factor)) {}
    float lr_at(int64_t s) const override { return base_lr * factor(s); }
};
class StepLR : public LRScheduler {
public:
    int64_t step_size;
    float gamma;
    StepLR(float base_lr, int64_t step_size, float gamma = 0.1f) : LRScheduler(base_lr), step_size(step_size), gamma(gamma) {}
    float lr_at(int64_t s) const override { return base_lr * std::pow(gamma, (float)(s / step_size)); }
};
class MultiStepLR : public LRScheduler {
public:
    std::vector<int64_t> milestones;
    float gamma;
    MultiStepLR(float base_lr, std::vector<int64_t> milestones, float gamma = 0.1f) : LRScheduler(base_lr), milestones(std::move(milestones)), gamma(gamma) {}
    float lr_at(int64_t s) const override {
        float lr = base_lr;
        for (int64_t m : milestones) if (s >= m) lr *= gamma;
        return lr;
    }
};
class ExponentialLR : public LRScheduler {
public:
    float gamma;
    ExponentialLR(float base_lr, float gamma) : LRScheduler(base_lr), gamma(gamma) {}
    float lr_at(int64_t s) const override { return base_lr * std::pow(gamma, (float)s); }
};
// From start_factor to end_factor over total steps
class LinearLR : public LRScheduler {
public:
    float start_factor, end_factor;
    int64_t total;
    LinearLR(float base_lr, int64_t total, float start_factor = 1.0f / 3.0f, float end_factor = 1.0f) : LRScheduler(base_lr), start_factor(start_factor), end_factor(end_factor), total(total) {}
    float lr_at(int64_t s) const override {
        float t = std::min(1.0f, (float)s / (float)std::max<int64_t>(1, total));
        return base_lr * (start_factor + (end_factor - start_factor) * t);
    }
};
class PolynomialLR : public LRScheduler {
public:
    int64_t total;
    float power;
    PolynomialLR(float base_lr, int64_t total, float power = 1.0f) : LRScheduler(base_lr), total(total), power(power) {}
    float lr_at(int64_t s) const override { return base_lr * std::pow(1.0f - (float)std::min(s, total) / (float)total, power); }
};
// Cosine decay restarted after T0 steps, each period T_mult times the last (SGDR)
class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    int64_t T0, T_mult;
    float min_lr;
    CosineAnnealingWarmRestarts(float base_lr, int64_t T0, int64_t T_mult = 1, float min_lr = 0.0f) : LRScheduler(base_lr), T0(T0), T_mult(T_mult), min_lr(min_lr) {}
    float lr_at(int64_t s) const override {
        int64_t period = T0;
        while (s >= period) {
            s -= period;
            period *= T_mult;
        }
        return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + std::cos(3.14159265358979f * (float)s / (float)period));
    }
};
// Up to max_lr and back down to almost nothing, both along cosines (Smith & Topin, 2018)
class OneCycleLR : public LRScheduler {
public:
    int64_t total;
    float pct_start, div_factor, final_div_factor;
    OneCycleLR(float max_lr, int64_t total, float pct_start = 0.3f, float div_factor = 25.0f, float final_div_factor = 1e4f)
        : LRScheduler(max_lr), total(total), pct_start(pct_start), div_factor(div_factor), final_div_factor(final_div_factor) {}
    float lr_at(int64_t s) const override {
        float start = base_lr / div_factor, end = start / final_div_factor;
        float peak = pct_start * (float)total - 1.0f, last = (float)total - 1.0f;
        auto cosine = [](float a, float b, float t) { return b + (a - b) * 0.5f * (1.0f + std::cos(3.14159265358979f * std::min(1.0f, t))); };
        if ((float)s <= peak) return cosine(start, base_lr, (float)s / std::max(peak, 1e-6f));
        return cosine(base_lr, end, ((float)s - peak) / std::max(last - peak, 1e-6f));
    }
};
// Triangular waves between base_lr and max_lr; gamma shrinks them over steps
class CyclicLR : public LRScheduler {
public:
    float max_lr, gamma;
    int64_t up, down;
    CyclicLR(float base_lr, float max_lr, int64_t up = 2000, int64_t down = 2000, float gamma = 1.0f) : LRScheduler(base_lr), max_lr(max_lr), gamma(gamma), up(up), down(down) {}
    float lr_at(int64_t s) const override {
        int64_t phase = s % (up + down);
        float t = phase < up ? (float)phase / (float)up : 1.0f - (float)(phase - up) / (float)down;
        return base_lr + (max_lr - base_lr) * t * std::pow(gamma, (float)s);
    }
};
// Schedules run one after another, switching at the given steps
class SequentialLR : public LRScheduler {
public:
    std::vector<std::shared_ptr<LRScheduler>> schedules;
    std::vector<int64_t> milestones;
    SequentialLR(std::vector<std::shared_ptr<LRScheduler>> schedules, std::vector<int64_t> milestones)
        : LRScheduler(schedules.at(0)->base_lr), schedules(std::move(schedules)), milestones(std::move(milestones)) {
        if (this->milestones.size() + 1 != this->schedules.size()) throw std::invalid_argument("SequentialLR: one milestone between each pair of schedules");
    }
    float lr_at(int64_t s) const override {
        size_t i = 0;
        while (i < milestones.size() && s >= milestones[i]) ++i;
        return schedules[i]->lr_at(i == 0 ? s : s - milestones[i - 1]);
    }
};
// Lowers the rate when a metric stops improving: call step(metric) after each evaluation
class ReduceLROnPlateau {
public:
    float lr, factor, min_lr, threshold;
    int patience;
    bool maximize;
    explicit ReduceLROnPlateau(float lr, float factor = 0.1f, int patience = 10, float threshold = 1e-4f, float min_lr = 0.0f, bool maximize = false)
        : lr(lr), factor(factor), min_lr(min_lr), threshold(threshold), patience(patience), maximize(maximize) {}
    float get_lr() const { return lr; }
    float step(float metric) {
        float m = maximize ? -metric : metric;
        if (!has_best_ || m < best_ - threshold * std::fabs(best_)) {
            best_ = m;
            has_best_ = true;
            bad_ = 0;
        } else if (++bad_ > patience) {
            lr = std::max(min_lr, lr * factor);
            bad_ = 0;
        }
        return lr;
    }

private:
    float best_ = 0.0f;
    bool has_best_ = false;
    int bad_ = 0;
};

// ---------------------------------------------------------------------------------------------
// Weight averaging
// ---------------------------------------------------------------------------------------------

// Exponential moving average of a model's parameters. swap() exchanges the averages with the live weights
// (evaluate with them, then swap back).
class EMA {
public:
    float decay;
    EMA(Module& model, float decay = 0.999f) : decay(decay), params_(model.parameters()) {
        for (const auto& p : params_) averages_.push_back(p->clone()->detach());
    }
    void update() {
        NoGradGuard no_grad;
        for (size_t i = 0; i < params_.size(); ++i) {
            DG_KERNEL(params_[i]->device, ema_step(averages_[i]->data_ptr<float>(), params_[i]->data_ptr<float>(), params_[i]->size(), decay));
        }
    }
    void swap() {
        NoGradGuard no_grad;
        for (size_t i = 0; i < params_.size(); ++i) {
            auto live = params_[i]->clone()->detach();
            ops::make_contiguous(*params_[i], *averages_[i]);
            ops::make_contiguous(*averages_[i], *live);
        }
    }
    const std::vector<std::shared_ptr<Tensor>>& averages() const { return averages_; }

private:
    std::vector<std::shared_ptr<Tensor>> params_, averages_;
};

// Equal-weight average of the weights seen at each update (stochastic weight averaging)
class SWA : public EMA {
public:
    explicit SWA(Module& model) : EMA(model, 0.0f) {}
    void update() {
        decay = (float)count_ / (float)(count_ + 1);
        ++count_;
        EMA::update();
    }

private:
    int64_t count_ = 0;
};

// Clamps every gradient element to [-limit, limit]
inline void clip_grad_value_(const std::vector<std::shared_ptr<Tensor>>& params, float limit) {
    NoGradGuard no_grad;
    for (const auto& p : params) {
        if (!p || !p->grad) continue;
        p->grad = clamp(p->grad, -limit, limit);
    }
}

}  // namespace optim
