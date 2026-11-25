#pragma once
#include "tensor.h"
#include "functional.h"
#include "module.h"
#include <algorithm>
#include <cmath>

// LayerNorm over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta. Backward
// recomputes xhat from x and the saved per-row mean and inverse std, so nothing the size of x is
// saved besides x itself.

// Sum of term(j) for j < n in 16 fixed lanes, so it vectorizes and does not depend on threads
template <typename F>
inline float lane_sum(int64_t n, const F& term) {
    float lane[16] = {};
    int64_t j = 0;
    for (; j + 16 <= n; j += 16)
        for (int k = 0; k < 16; ++k) lane[k] += term(j + k);
    float total = 0.0f;
    for (int k = 0; k < 16; ++k) total += lane[k];
    for (; j < n; ++j) total += term(j);
    return total;
}
// Rows of cols elements worth handing to a thread
inline int64_t row_grain(int64_t cols) { return std::max<int64_t>(1, 16384 / std::max<int64_t>(1, cols)); }

namespace layernorm_detail {

inline void forward_host(const float* x, const float* gamma, const float* beta, float* out, float* mean, float* rstd,
                         int64_t rows, int64_t cols, float eps) {
    parallel_for(0, rows, row_grain(cols), [=](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const float* xr = x + row * cols;
            float* o = out + row * cols;
            float mu = lane_sum(cols, [=](int64_t j) { return xr[j]; }) / (float)cols;
            float sq = lane_sum(cols, [=](int64_t j) { return (xr[j] - mu) * (xr[j] - mu); });
            float rs = 1.0f / std::sqrt(sq / (float)cols + eps);
            mean[row] = mu;
            rstd[row] = rs;
            for (int64_t j = 0; j < cols; ++j) o[j] = (xr[j] - mu) * rs * gamma[j] + beta[j];
        }
    });
}

// dx = rstd * (u - mean(u) - xhat * mean(u * xhat)) with u = grad * gamma;
// dgamma = sum_rows(grad * xhat), dbeta = sum_rows(grad)
inline void backward_host(const float* grad, const float* x, const float* gamma, const float* mean, const float* rstd,
                          float* gx, float* ggamma, float* gbeta, int64_t rows, int64_t cols) {
    if (gx) {
        parallel_for(0, rows, row_grain(cols), [=](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                const float* xr = x + row * cols;
                const float* gr = grad + row * cols;
                float* o = gx + row * cols;
                float mu = mean[row], rs = rstd[row];
                float mean_u = lane_sum(cols, [=](int64_t j) { return gr[j] * gamma[j]; }) / (float)cols;
                float mean_u_xhat = lane_sum(cols, [=](int64_t j) { return gr[j] * gamma[j] * ((xr[j] - mu) * rs); }) / (float)cols;
                for (int64_t j = 0; j < cols; ++j) o[j] = rs * (gr[j] * gamma[j] - mean_u - (xr[j] - mu) * rs * mean_u_xhat);
            }
        });
    }
    if (!ggamma && !gbeta) return;
    // Columns split over threads, rows in order: the sums do not depend on the thread count
    parallel_for(0, cols, std::max<int64_t>(16, 32768 / std::max<int64_t>(1, rows)), [=](int64_t b, int64_t e) {
        if (ggamma) std::fill(ggamma + b, ggamma + e, 0.0f);
        if (gbeta) std::fill(gbeta + b, gbeta + e, 0.0f);
        for (int64_t row = 0; row < rows; ++row) {
            const float* xr = x + row * cols;
            const float* gr = grad + row * cols;
            float mu = mean[row], rs = rstd[row];
            if (ggamma) for (int64_t j = b; j < e; ++j) ggamma[j] += gr[j] * ((xr[j] - mu) * rs);
            if (gbeta) for (int64_t j = b; j < e; ++j) gbeta[j] += gr[j];
        }
    });
}

}  // namespace layernorm_detail

struct LayerNormBackward : public Node {
    inline std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override;
    std::shared_ptr<Tensor> x, gamma, row_mean, rstd;
    int64_t rows, cols;
    float eps;

    LayerNormBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> gamma, std::shared_ptr<Tensor> row_mean,
                      std::shared_ptr<Tensor> rstd, int64_t rows, int64_t cols, float eps)
        : x(std::move(x)), gamma(std::move(gamma)), row_mean(std::move(row_mean)), rstd(std::move(rstd)), rows(rows), cols(cols), eps(eps) {}

    void release_variables() override {
        x.reset();
        gamma.reset();
        row_mean.reset();
        rstd.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !x || !gamma || !row_mean || !rstd) return {nullptr, nullptr, nullptr};
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
                                     row_mean->data_ptr<float>(), rstd->data_ptr<float>(), gx ? gx->data_ptr<float>() : nullptr,
                                     gg ? gg->data_ptr<float>() : nullptr, gb ? gb->data_ptr<float>() : nullptr, rows, cols);
            return {gx, gg, gb};
        }
#endif
        auto host = [](const std::shared_ptr<Tensor>& t) { return t->device == Device::CPU ? t : t->cpu(); };
        auto gh = host(g), xh = host(x), gmh = host(gamma), mh = host(row_mean), rh = host(rstd);
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
        auto x = contiguous(float32_input(x_in));
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
            auto grad_fn = std::make_shared<LayerNormBackward>(x, gamma, mean, rstd, rows, cols, eps);
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

// Through the definition, with xhat = (x - mean) rstd connected to x
inline TensorList LayerNormBackward::apply_graph(const TensorList& g) {
    auto xv = reshape(graph_input(*this, 0, x), {rows, cols});
    auto gm = graph_input(*this, 1, gamma);
    auto u = reshape(g[0], {rows, cols});
    auto centered = sub(xv, sum_to(div_scalar(xv, (float)cols), {rows, 1}));
    auto rs = rsqrt(add_scalar(mean(square(centered), -1, true), eps));
    auto xhat = mul(centered, rs);
    auto ug = mul(u, gm);
    auto dx = mul(rs, sub(sub(ug, mean(ug, -1, true)), mul(xhat, mean(mul(ug, xhat), -1, true))));
    return {reshape(dx, x->shape), reshape(sum(mul(u, xhat), 0, false), gamma->shape), reshape(sum(u, 0, false), gamma->shape)};
}
