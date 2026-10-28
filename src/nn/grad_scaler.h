#pragma once
#include "tensor.h"
#include "functional.h"
#include "training.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// Dynamic loss scaling for fp16 autocast, with PyTorch's GradScaler semantics. fp16 cannot hold
// magnitudes below ~6e-8, so small gradients round to zero when backward rounds them for a matmul.
// Scaling the loss by S scales every gradient by S; the scaler divides S back out before the
// optimizer step. If any scaled gradient overflowed to inf/NaN, the step is skipped and S is
// multiplied by backoff_factor; after growth_interval clean steps in a row, S grows by growth_factor.
//
//   GradScaler scaler;
//   for (...) {
//       opt.zero_grad(params);
//       std::shared_ptr<Tensor> loss;
//       { AutocastGuard autocast(DType::Float16); loss = fused_cross_entropy(model.forward(x), y); }
//       scaler.scale(loss)->backward();
//       scaler.unscale(params);        // only needed to read true gradients first, e.g. to clip
//       clip_grad_norm_(params, 1.0f);
//       scaler.step(opt, params);      // skipped if any gradient was inf/NaN
//       scaler.update();
//   }
//
// Log the unscaled loss, not scaler.scale(loss). bf16 autocast does not need a scaler.
class GradScaler {
public:
    explicit GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f, float backoff_factor = 0.5f,
                        int64_t growth_interval = 2000, bool enabled = true)
        : scale_(init_scale), growth_factor_(growth_factor), backoff_factor_(backoff_factor),
          growth_interval_(growth_interval), enabled_(enabled) {
        if (!(init_scale > 0.0f) || !std::isfinite(init_scale)) throw std::runtime_error("GradScaler: init_scale must be positive and finite");
        if (!(growth_factor > 1.0f)) throw std::runtime_error("GradScaler: growth_factor must be > 1");
        if (!(backoff_factor > 0.0f && backoff_factor < 1.0f)) throw std::runtime_error("GradScaler: backoff_factor must be in (0, 1)");
        if (growth_interval < 1) throw std::runtime_error("GradScaler: growth_interval must be >= 1");
    }

    bool enabled() const { return enabled_; }
    float get_scale() const { return scale_; }
    // Whether the gradients of the current step (after unscale) contained inf or NaN
    bool found_inf() const { return found_inf_; }
    // Optimizer steps skipped so far. A count that keeps rising means the model itself produces
    // inf/NaN, not just fp16 overflow.
    int64_t skipped_steps() const { return skipped_steps_; }

    // loss * scale, recorded in the graph so backward produces scaled gradients
    std::shared_ptr<Tensor> scale(const std::shared_ptr<Tensor>& loss) const {
        return enabled_ ? mul_scalar(loss, scale_) : loss;
    }

    // Divides every gradient by the scale in place and records whether any was inf or NaN.
    // Runs once per step: later calls before update() do nothing.
    void unscale(const std::vector<std::shared_ptr<Tensor>>& params) {
        if (!enabled_ || unscaled_) return;
        found_inf_ = unscale_grads(params, 1.0f / scale_);
        unscaled_ = true;
    }

    // Unscales if needed, then runs opt.step(params) unless a gradient was inf or NaN.
    // Returns whether the optimizer stepped.
    template <typename Optimizer>
    bool step(Optimizer& opt, const std::vector<std::shared_ptr<Tensor>>& params) {
        if (!enabled_) {
            opt.step(params);
            return true;
        }
        unscale(params);
        if (found_inf_) {
            skipped_steps_++;
            return false;
        }
        opt.step(params);
        return true;
    }

    // Adjusts the scale for the next step. Call once per step, after step().
    void update() {
        if (!enabled_) return;
        if (!unscaled_) throw std::runtime_error("GradScaler::update: call step() or unscale() first");
        if (found_inf_) {
            scale_ = std::max(scale_ * backoff_factor_, std::numeric_limits<float>::min());
            growth_tracker_ = 0;
        } else if (++growth_tracker_ >= growth_interval_) {
            float grown = scale_ * growth_factor_;
            if (std::isfinite(grown)) scale_ = grown;
            growth_tracker_ = 0;
        }
        unscaled_ = false;
        found_inf_ = false;
    }

    // Checkpointing through TrainingState::scalars, which save_checkpoint/load_checkpoint carry:
    //   scaler.save_to(state); save_checkpoint(path, model, &opt, state);
    //   auto state = load_checkpoint(path, model, &opt); scaler.load_from(state);
    void save_to(TrainingState& state) const {
        state.scalars["grad_scaler.scale"] = scale_;
        state.scalars["grad_scaler.growth_tracker"] = (double)growth_tracker_;
        state.scalars["grad_scaler.skipped_steps"] = (double)skipped_steps_;
    }

    // Restores what save_to stored; a state without scaler entries leaves the scaler unchanged
    void load_from(const TrainingState& state) {
        auto get = [&](const char* key, double fallback) {
            auto it = state.scalars.find(key);
            return it == state.scalars.end() ? fallback : it->second;
        };
        float s = (float)get("grad_scaler.scale", scale_);
        if (!(s > 0.0f) || !std::isfinite(s)) throw std::runtime_error("GradScaler::load_from: saved scale is not positive and finite");
        scale_ = s;
        growth_tracker_ = (int64_t)get("grad_scaler.growth_tracker", (double)growth_tracker_);
        skipped_steps_ = (int64_t)get("grad_scaler.skipped_steps", (double)skipped_steps_);
        unscaled_ = false;
        found_inf_ = false;
    }

private:
    float scale_, growth_factor_, backoff_factor_;
    int64_t growth_interval_;
    bool enabled_;
    int64_t growth_tracker_ = 0;
    int64_t skipped_steps_ = 0;
    bool unscaled_ = false;
    bool found_inf_ = false;

    // Gradients are owned by their parameters (adopted or accumulated into dense buffers), so
    // they are rescaled in place; a strided gradient is first replaced by a dense copy. A
    // parameter listed twice (e.g. a tied weight in a hand-built list) is unscaled once.
    static bool unscale_grads(const std::vector<std::shared_ptr<Tensor>>& params, float inv_scale) {
        bool found = false;
        std::unordered_set<const Tensor*> seen;
#ifdef USE_CUDA
        std::vector<float*> device_grads;
        std::vector<int64_t> device_sizes;
#endif
        for (const auto& p : params) {
            if (!p || !p->grad || !seen.insert(p.get()).second) continue;
            if (p->grad->dtype != DType::Float32) throw std::runtime_error("GradScaler: gradients must be Float32");
            if (!p->grad->is_contiguous()) p->grad = make_contiguous(p->grad);
            auto& g = p->grad;
#ifdef USE_CUDA
            if (g->device == Device::CUDA) {
                device_grads.push_back(g->data_ptr<float>());
                device_sizes.push_back(g->size());
                continue;
            }
#endif
            // MPS memory is shared with the host, and host memory is a TPU tensor's authoritative copy
            if (g->device != Device::CPU && g->device != Device::MPS && g->device != Device::TPU) {
                throw std::runtime_error("GradScaler: unsupported device");
            }
            float* d = g->data_ptr<float>();
            for (int64_t i = 0; i < g->size(); ++i) {
                if (!std::isfinite(d[i])) found = true;
                d[i] *= inv_scale;
            }
            if (g->device == Device::TPU) g->storage->tpu_handle.reset();
        }
#ifdef USE_CUDA
        if (cuda::unscale_and_check(device_grads, device_sizes, inv_scale)) found = true;
#endif
        return found;
    }
};
