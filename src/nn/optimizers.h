#pragma once
#include "tensor.h"
#include "functional.h"
#include "init.h"
#include "module.h"
#include "ops.h"
#include "serialization.h"

#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

inline void zero_grad(const std::vector<std::shared_ptr<Tensor>>& params) {
    for (const auto& p : params) {
        if (p) p->zero_grad();
    }
}

namespace optim_detail {

// Per-parameter optimizer state, keyed by tensor address but bound to the tensor's identity:
// when a parameter is freed and another tensor reuses its address, the stale state is dropped
// instead of being applied to the new tensor.
template <typename State>
class StateMap {
public:
    // Returns the state for p, creating it (and its moment tensors) on p's device if needed.
    State& get(const std::shared_ptr<Tensor>& p) {
        auto& slot = slots_[p.get()];
        if (slot.owner.lock() != p) slot = Slot{p, State{}};
        slot.state.follow(p);
        return slot.state;
    }

    const State* find(const std::shared_ptr<Tensor>& p) const {
        auto it = slots_.find(p.get());
        if (it == slots_.end() || it->second.owner.lock() != p) return nullptr;
        return &it->second.state;
    }

    // Forget state whose parameter no longer exists (e.g. after a model moved devices).
    void prune() {
        for (auto it = slots_.begin(); it != slots_.end();) {
            it = it->second.owner.expired() ? slots_.erase(it) : std::next(it);
        }
    }

private:
    struct Slot {
        std::weak_ptr<Tensor> owner;
        State state;
    };
    std::unordered_map<Tensor*, Slot> slots_;
};

inline std::shared_ptr<Tensor> zeros_like(const std::shared_ptr<Tensor>& p) {
    return zeros(p->shape, p->device, DType::Float32, false);
}

// Moves a state tensor to p's device when the model was moved after training started.
inline void follow_device(std::shared_ptr<Tensor>& t, const std::shared_ptr<Tensor>& p) {
    if (t && t->device != p->device) t = t->to(p->device);
}

inline void check_param(const std::shared_ptr<Tensor>& p, const char* who) {
    if (p->dtype != DType::Float32) throw std::runtime_error(std::string(who) + ": only Float32 parameters are supported");
    if (!p->is_contiguous()) throw std::runtime_error(std::string(who) + ": parameters must be contiguous");
    if (p->grad->device != p->device) throw std::runtime_error(std::string(who) + ": gradient and parameter are on different devices");
}

inline void load_into(const std::shared_ptr<Tensor>& dst, const std::shared_ptr<Tensor>& src, const std::string& name) {
    if (src->shape != dst->shape) throw std::runtime_error("optimizer state " + name + " has the wrong shape");
    ops::make_contiguous(*dst, *src->to(dst->device, dst->dtype));
}

// Per-parameter weight decay overrides, bound to each tensor's identity like StateMap
class DecayOverrides {
public:
    void set(const std::vector<std::shared_ptr<Tensor>>& params, float weight_decay) {
        for (const auto& p : params) {
            if (p) slots_[p.get()] = Slot{p, weight_decay};
        }
    }
    float get(const std::shared_ptr<Tensor>& p, float fallback) const {
        auto it = slots_.find(p.get());
        return (it != slots_.end() && it->second.owner.lock() == p) ? it->second.weight_decay : fallback;
    }

private:
    struct Slot {
        std::weak_ptr<Tensor> owner;
        float weight_decay;
    };
    std::unordered_map<const Tensor*, Slot> slots_;
};

}  // namespace optim_detail

// Parameters that conventionally get no weight decay: biases and normalization gains and shifts
// (names ending in "bias", "gamma" or "beta") and any other 1-D tensor. Decaying a norm's gamma
// pulls it toward 0, which the norm exists to avoid.
//
//   Adam opt(3e-4f);  // weight_decay 0.01 for everything else
//   opt.set_weight_decay(no_weight_decay_params(model), 0.0f);
inline std::vector<std::shared_ptr<Tensor>> no_weight_decay_params(Module& model) {
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    std::vector<std::shared_ptr<Tensor>> out;
    for (const auto& [name, p] : model.named_parameters()) {
        if (p->ndim() <= 1 || ends_with(name, "bias") || ends_with(name, "gamma") || ends_with(name, "beta")) out.push_back(p);
    }
    return out;
}

// SGD with optional momentum and (coupled, L2-style) weight decay:
//   g = grad + weight_decay * p;  buf = momentum * buf + g;  p -= lr * buf
class SGD {
public:
    float lr, momentum, weight_decay;

    explicit SGD(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f)
        : lr(lr), momentum(momentum), weight_decay(weight_decay) {}

    void step(const std::vector<std::shared_ptr<Tensor>>& params) {
        for (const auto& p : params) step(p);
        state_.prune();
    }

    void step(const std::shared_ptr<Tensor>& p) {
        if (!p || !p->grad) return;
        optim_detail::check_param(p, "SGD");
        NoGradGuard no_grad;
        auto g = make_contiguous(p->grad);
        float wd = decay_.get(p, weight_decay);
        if (wd != 0.0f) g = add(g, mul_scalar(p, wd));
        if (momentum != 0.0f) {
            auto& s = state_.get(p);
            // A fresh tensor: the buffer must not alias p->grad, which backward may accumulate into
            s.buf = s.buf ? add(mul_scalar(s.buf, momentum), g) : mul_scalar(g, 1.0f);
            g = s.buf;
        }
        ops::add_inplace(*p, *mul_scalar(g, -lr));
    }

    void zero_grad(const std::vector<std::shared_ptr<Tensor>>& params) { ::zero_grad(params); }

    // Weight decay for these parameters instead of weight_decay (e.g. 0 for no_weight_decay_params)
    void set_weight_decay(const std::vector<std::shared_ptr<Tensor>>& params, float wd) { decay_.set(params, wd); }

    NamedTensors state_dict(const NamedTensors& named_params) const {
        NamedTensors out;
        for (const auto& [name, p] : named_params) {
            const State* s = state_.find(p);
            if (s && s->buf) out.emplace_back(name + ".momentum_buffer", s->buf);
        }
        return out;
    }

    void load_state_dict(const NamedTensors& named_params,
                         const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state) {
        for (const auto& [name, p] : named_params) {
            auto it = state.find(name + ".momentum_buffer");
            if (it == state.end()) continue;
            auto& s = state_.get(p);
            s.buf = optim_detail::zeros_like(p);
            optim_detail::load_into(s.buf, it->second, it->first);
        }
    }

private:
    struct State {
        std::shared_ptr<Tensor> buf;
        void follow(const std::shared_ptr<Tensor>& p) { optim_detail::follow_device(buf, p); }
    };
    optim_detail::StateMap<State> state_;
    optim_detail::DecayOverrides decay_;
};

// AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019). The moment estimates live
// in tensors on each parameter's device and are updated by a fused kernel on CUDA and Metal.
// weight_decay applies to every parameter unless set_weight_decay overrides it; exclude biases
// and norm parameters with opt.set_weight_decay(no_weight_decay_params(model), 0.0f).
class Adam {
public:
    float lr, beta1, beta2, eps, weight_decay;

    explicit Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {}

    void step(const std::vector<std::shared_ptr<Tensor>>& params) {
        for (const auto& p : params) step(p);
        state_.prune();
    }

    void step(const std::shared_ptr<Tensor>& p) {
        if (!p || !p->grad) return;
        optim_detail::check_param(p, "Adam");
        auto g = make_contiguous(p->grad);
        auto& s = state_.get(p);
        s.step++;
        int64_t n = p->size();
        float wd = decay_.get(p, weight_decay);

#ifdef USE_CUDA
        if (p->device == Device::CUDA) {
            cuda::adamw_step(p->data_ptr<float>(), g->data_ptr<float>(), s.m->data_ptr<float>(), s.v->data_ptr<float>(),
                             lr, beta1, beta2, eps, wd, (int)s.step, n);
            return;
        }
#endif
        if (p->device == Device::MPS && MetalBackend::get().is_available()) {
            MetalBackend::get().adamw(p->data_ptr<float>(), g->data_ptr<float>(), s.m->data_ptr<float>(), s.v->data_ptr<float>(),
                                      (size_t)n, lr, beta1, beta2, eps, wd, (int)s.step);
            return;
        }
        if (p->device != Device::CPU && p->device != Device::MPS) throw std::runtime_error("Adam: unsupported device");

        float* w = p->data_ptr<float>();
        const float* grad = g->data_ptr<float>();
        float* m = s.m->data_ptr<float>();
        float* v = s.v->data_ptr<float>();
        const float bias1 = 1.0f - (float)std::pow((double)beta1, (double)s.step);
        const float bias2 = 1.0f - (float)std::pow((double)beta2, (double)s.step);
        for (int64_t i = 0; i < n; ++i) {
            w[i] -= lr * wd * w[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
            w[i] -= lr * (m[i] / bias1) / (std::sqrt(v[i] / bias2) + eps);
        }
    }

    void zero_grad(const std::vector<std::shared_ptr<Tensor>>& params) { ::zero_grad(params); }

    // Weight decay for these parameters instead of weight_decay (e.g. 0 for no_weight_decay_params)
    void set_weight_decay(const std::vector<std::shared_ptr<Tensor>>& params, float wd) { decay_.set(params, wd); }

    // State of the given parameters, keyed "<name>.exp_avg", "<name>.exp_avg_sq", "<name>.step".
    // Parameters that never received a gradient have no state and are omitted.
    NamedTensors state_dict(const NamedTensors& named_params) const {
        NamedTensors out;
        for (const auto& [name, p] : named_params) {
            const State* s = state_.find(p);
            if (!s || !s->m) continue;
            out.emplace_back(name + ".exp_avg", s->m);
            out.emplace_back(name + ".exp_avg_sq", s->v);
            out.emplace_back(name + ".step", io::encode_int64(s->step));
        }
        return out;
    }

    // Restores state saved by state_dict(). Parameters without saved state start fresh;
    // wrong-shaped state throws.
    void load_state_dict(const NamedTensors& named_params,
                         const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state) {
        for (const auto& [name, p] : named_params) {
            auto m = state.find(name + ".exp_avg");
            auto v = state.find(name + ".exp_avg_sq");
            auto step = state.find(name + ".step");
            if (m == state.end() || v == state.end() || step == state.end()) continue;
            auto& s = state_.get(p);
            optim_detail::load_into(s.m, m->second, m->first);
            optim_detail::load_into(s.v, v->second, v->first);
            s.step = io::decode_int64(step->second);
        }
    }

private:
    struct State {
        std::shared_ptr<Tensor> m, v;
        int64_t step = 0;
        void follow(const std::shared_ptr<Tensor>& p) {
            if (!m) {
                m = optim_detail::zeros_like(p);
                v = optim_detail::zeros_like(p);
            }
            optim_detail::follow_device(m, p);
            optim_detail::follow_device(v, p);
        }
    };
    optim_detail::StateMap<State> state_;
    optim_detail::DecayOverrides decay_;
};

using AdamW = Adam;
