#pragma once
#include "tensor.h"
#include "functional.h"
#include "module.h"
#include <algorithm>
#include <cmath>

// LayerNorm over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta. Backward
// recomputes xhat from x and the saved per-row mean and inverse std, so nothing the size of x is
// saved besides x itself.

namespace layernorm_detail {

inline void forward_host(const float* x, const float* gamma, const float* beta, float* out, float* mean, float* rstd,
                         int64_t rows, int64_t cols, float eps) {
    for (int64_t row = 0; row < rows; ++row) {
        const float* xr = x + row * cols;
        float sum = 0.0f;
        for (int64_t j = 0; j < cols; ++j) sum += xr[j];
        float mu = sum / (float)cols;
        float sq = 0.0f;
        for (int64_t j = 0; j < cols; ++j) sq += (xr[j] - mu) * (xr[j] - mu);
        float rs = 1.0f / std::sqrt(sq / (float)cols + eps);
        mean[row] = mu;
        rstd[row] = rs;
        for (int64_t j = 0; j < cols; ++j) out[row * cols + j] = (xr[j] - mu) * rs * gamma[j] + beta[j];
    }
}

// dx = rstd * (u - mean(u) - xhat * mean(u * xhat)) with u = grad * gamma;
// dgamma = sum_rows(grad * xhat), dbeta = sum_rows(grad)
inline void backward_host(const float* grad, const float* x, const float* gamma, const float* mean, const float* rstd,
                          float* gx, float* ggamma, float* gbeta, int64_t rows, int64_t cols) {
    if (ggamma) std::fill(ggamma, ggamma + cols, 0.0f);
    if (gbeta) std::fill(gbeta, gbeta + cols, 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        const float* xr = x + row * cols;
        const float* gr = grad + row * cols;
        float mu = mean[row], rs = rstd[row];
        float sum_u = 0.0f, sum_u_xhat = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float xhat = (xr[j] - mu) * rs;
            float u = gr[j] * gamma[j];
            sum_u += u;
            sum_u_xhat += u * xhat;
            if (ggamma) ggamma[j] += gr[j] * xhat;
            if (gbeta) gbeta[j] += gr[j];
        }
        if (gx) {
            float mean_u = sum_u / (float)cols, mean_u_xhat = sum_u_xhat / (float)cols;
            for (int64_t j = 0; j < cols; ++j) {
                float xhat = (xr[j] - mu) * rs;
                gx[row * cols + j] = rs * (gr[j] * gamma[j] - mean_u - xhat * mean_u_xhat);
            }
        }
    }
}

}  // namespace layernorm_detail

struct LayerNormBackward : public Node {
    std::shared_ptr<Tensor> x, gamma, mean, rstd;
    int64_t rows, cols;

    LayerNormBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> gamma, std::shared_ptr<Tensor> mean,
                      std::shared_ptr<Tensor> rstd, int64_t rows, int64_t cols)
        : x(std::move(x)), gamma(std::move(gamma)), mean(std::move(mean)), rstd(std::move(rstd)), rows(rows), cols(cols) {}

    void release_variables() override {
        x.reset();
        gamma.reset();
        mean.reset();
        rstd.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !x || !gamma || !mean || !rstd) return {nullptr, nullptr, nullptr};
        bool need_x = next_edges.size() > 0 && next_edges[0].function;
        bool need_g = next_edges.size() > 1 && next_edges[1].function;
        bool need_b = next_edges.size() > 2 && next_edges[2].function;
        auto g = grads[0];
        Device dev = x->device;
        std::shared_ptr<Tensor> gx, gg, gb;
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            if (need_x) gx = std::make_shared<Tensor>(x->shape, dev, x->dtype, false);
            if (need_g) gg = std::make_shared<Tensor>(gamma->shape, dev, gamma->dtype, false);
            if (need_b) gb = std::make_shared<Tensor>(gamma->shape, dev, gamma->dtype, false);
            cuda::layernorm_backward(g->data_ptr<float>(), x->data_ptr<float>(), gamma->data_ptr<float>(),
                                     mean->data_ptr<float>(), rstd->data_ptr<float>(), gx ? gx->data_ptr<float>() : nullptr,
                                     gg ? gg->data_ptr<float>() : nullptr, gb ? gb->data_ptr<float>() : nullptr, rows, cols);
            return {gx, gg, gb};
        }
#endif
        auto host = [](const std::shared_ptr<Tensor>& t) { return t->device == Device::CPU ? t : t->cpu(); };
        auto gh = host(g), xh = host(x), gmh = host(gamma), mh = host(mean), rh = host(rstd);
        if (need_x) gx = std::make_shared<Tensor>(x->shape, Device::CPU, x->dtype, false);
        if (need_g) gg = std::make_shared<Tensor>(gamma->shape, Device::CPU, gamma->dtype, false);
        if (need_b) gb = std::make_shared<Tensor>(gamma->shape, Device::CPU, gamma->dtype, false);
        layernorm_detail::backward_host(gh->data_ptr<float>(), xh->data_ptr<float>(), gmh->data_ptr<float>(),
                                        mh->data_ptr<float>(), rh->data_ptr<float>(), gx ? gx->data_ptr<float>() : nullptr,
                                        gg ? gg->data_ptr<float>() : nullptr, gb ? gb->data_ptr<float>() : nullptr, rows, cols);
        if (dev != Device::CPU) {
            if (gx) gx = gx->to(dev);
            if (gg) gg = gg->to(dev);
            if (gb) gb = gb->to(dev);
        }
        return {gx, gg, gb};
    }
};

class LayerNorm : public Module {
public:
    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    float eps;

    LayerNorm(int features, float eps = 1e-5f) : eps(eps) {
        gamma = std::make_shared<Tensor>(std::vector<int64_t>{features});
        beta  = std::make_shared<Tensor>(std::vector<int64_t>{features});
        gamma->fill_(1.0f);
        beta->fill_(0.0f);
        gamma->requires_grad = true;
        beta->requires_grad = true;
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in) {
        DG_PROFILE_OP("layer_norm");
        if (!x_in) throw std::runtime_error("LayerNorm: input tensor is null");
        auto x = contiguous(x_in);
        if (x->ndim() < 1) throw std::runtime_error("LayerNorm: input tensor must be at least 1D");
        int64_t cols = x->shape.back();
        if (cols != gamma->shape[0]) {
            throw std::runtime_error("LayerNorm: feature dimension mismatch, expected " + std::to_string(gamma->shape[0]) +
                                     " but got " + std::to_string(cols));
        }
        if (gamma->device != x->device || beta->device != x->device) {
            throw std::runtime_error("LayerNorm: input and parameters are on different devices");
        }
        int64_t rows = x->size() / cols;
        Device dev = x->device;
        auto out = std::make_shared<Tensor>(x->shape, dev, x->dtype, false);
        auto mean = std::make_shared<Tensor>(std::vector<int64_t>{rows}, dev, x->dtype, false);
        auto rstd = std::make_shared<Tensor>(std::vector<int64_t>{rows}, dev, x->dtype, false);

#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            cuda::layernorm_forward(x->data_ptr<float>(), gamma->data_ptr<float>(), beta->data_ptr<float>(),
                                    out->data_ptr<float>(), mean->data_ptr<float>(), rstd->data_ptr<float>(), rows, cols, eps);
        } else
#endif
        if (dev == Device::CPU) {
            layernorm_detail::forward_host(x->data_ptr<float>(), gamma->data_ptr<float>(), beta->data_ptr<float>(),
                                           out->data_ptr<float>(), mean->data_ptr<float>(), rstd->data_ptr<float>(),
                                           rows, cols, eps);
        } else {
            auto xh = x->cpu(), gh = gamma->cpu(), bh = beta->cpu();
            auto oh = std::make_shared<Tensor>(x->shape, Device::CPU, x->dtype, false);
            auto mh = std::make_shared<Tensor>(std::vector<int64_t>{rows}, Device::CPU, x->dtype, false);
            auto rh = std::make_shared<Tensor>(std::vector<int64_t>{rows}, Device::CPU, x->dtype, false);
            layernorm_detail::forward_host(xh->data_ptr<float>(), gh->data_ptr<float>(), bh->data_ptr<float>(),
                                           oh->data_ptr<float>(), mh->data_ptr<float>(), rh->data_ptr<float>(),
                                           rows, cols, eps);
            out = oh->to(dev);
            mean = mh->to(dev);
            rstd = rh->to(dev);
        }

        bool req_grad = needs_grad(x) || needs_grad(gamma) || needs_grad(beta);
        out->requires_grad = req_grad;
        if (req_grad) {
            auto grad_fn = std::make_shared<LayerNormBackward>(x, gamma, mean, rstd, rows, cols);
            grad_fn->add_next_edge(get_grad_edge(x).function);
            grad_fn->add_next_edge(get_grad_edge(gamma).function);
            grad_fn->add_next_edge(get_grad_edge(beta).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    void visit(ModuleVisitor& v) override {
        v.parameter("gamma", gamma);
        v.parameter("beta", beta);
    }
};
