#pragma once
#include "tensor.h"
#include "functional.h"
#include <limits>
#include <stdexcept>

// 2-D max pooling over [B, C, H, W] (no padding). Each output remembers which element of its window
// won as the window offset kh * kW + kw (exact in float at any input size; -1 when every value was
// NaN), so backward routes each output gradient to that element.

namespace maxpool_detail {

struct Geometry {
    int64_t planes;  // B * C
    int H, W, H_out, W_out, kH, kW, sH, sW;
};

inline void forward_host(const float* x, float* out, float* argmax, const Geometry& g) {
    for (int64_t p = 0; p < g.planes; ++p) {
        const float* xp = x + p * g.H * g.W;
        for (int h = 0; h < g.H_out; ++h) {
            for (int w = 0; w < g.W_out; ++w) {
                float best = -std::numeric_limits<float>::infinity();
                int offset = -1;
                for (int kh = 0; kh < g.kH; ++kh) {
                    for (int kw = 0; kw < g.kW; ++kw) {
                        float v = xp[(int64_t)(h * g.sH + kh) * g.W + w * g.sW + kw];
                        if (v > best) {
                            best = v;
                            offset = kh * g.kW + kw;
                        }
                    }
                }
                int64_t o = (p * g.H_out + h) * g.W_out + w;
                out[o] = best;
                argmax[o] = (float)offset;
            }
        }
    }
}

// gx must start zeroed. Outputs are visited in order, so each input sums its gradients in the same
// order as the CUDA kernel, which gathers them by window.
inline void backward_host(const float* grad, const float* argmax, float* gx, const Geometry& g) {
    for (int64_t p = 0; p < g.planes; ++p) {
        for (int h = 0; h < g.H_out; ++h) {
            for (int w = 0; w < g.W_out; ++w) {
                int64_t o = (p * g.H_out + h) * g.W_out + w;
                int offset = (int)argmax[o];
                if (offset < 0) continue;
                int kh = offset / g.kW, kw = offset % g.kW;
                gx[(p * g.H + h * g.sH + kh) * g.W + w * g.sW + kw] += grad[o];
            }
        }
    }
}

}  // namespace maxpool_detail

struct MaxPool2dBackward : public Node {
    std::vector<int64_t> x_shape;
    std::shared_ptr<Tensor> argmax;  // [B, C, H_out, W_out] window offsets, on the input's device
    maxpool_detail::Geometry geom;

    MaxPool2dBackward(std::vector<int64_t> x_shape, std::shared_ptr<Tensor> argmax, maxpool_detail::Geometry geom)
        : x_shape(std::move(x_shape)), argmax(std::move(argmax)), geom(geom) {}

    void release_variables() override { argmax = nullptr; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !argmax) return {nullptr};
        auto grad_out = grads[0];
        Device dev = grad_out->device;
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            auto gx = std::make_shared<Tensor>(x_shape, dev, DType::Float32, false);
            cuda::max_pool2d_backward(grad_out->data_ptr<float>(), argmax->data_ptr<float>(), gx->data_ptr<float>(),
                                      geom.planes, geom.H, geom.W, geom.H_out, geom.W_out, geom.kH, geom.kW, geom.sH, geom.sW);
            return {gx};
        }
#endif
        auto gh = (dev == Device::CPU) ? grad_out : grad_out->cpu();
        auto ah = (argmax->device == Device::CPU) ? argmax : argmax->cpu();
        auto gx = std::make_shared<Tensor>(x_shape, Device::CPU, DType::Float32, false);
        gx->fill_(0.0f);
        maxpool_detail::backward_host(gh->data_ptr<float>(), ah->data_ptr<float>(), gx->data_ptr<float>(), geom);
        return {(dev == Device::CPU) ? gx : gx->to(dev)};
    }
};

inline std::shared_ptr<Tensor> max_pool2d(const std::shared_ptr<Tensor>& x_in, int kH, int kW, int strideH, int strideW) {
    DG_PROFILE_OP("max_pool2d");
    if (!x_in) throw std::runtime_error("max_pool2d: input tensor is null");
    auto x = contiguous(float32_input(x_in));
    if (x->ndim() != 4) throw std::runtime_error("max_pool2d: x must be a 4D tensor (B, C, H, W)");
    if (kH <= 0 || kW <= 0 || strideH <= 0 || strideW <= 0) throw std::runtime_error("max_pool2d: kernel and strides must be > 0");

    maxpool_detail::Geometry g;
    g.planes = x->shape[0] * x->shape[1];
    g.H = (int)x->shape[2];
    g.W = (int)x->shape[3];
    if (g.H < kH || g.W < kW) throw std::runtime_error("max_pool2d: kernel size exceeds input spatial dimensions");
    g.kH = kH;
    g.kW = kW;
    g.sH = strideH;
    g.sW = strideW;
    g.H_out = (g.H - kH) / strideH + 1;
    g.W_out = (g.W - kW) / strideW + 1;

    Device dev = x->device;
    std::vector<int64_t> out_shape{x->shape[0], x->shape[1], g.H_out, g.W_out};
    auto out = std::make_shared<Tensor>(out_shape, dev, DType::Float32, false);
    auto argmax = std::make_shared<Tensor>(out_shape, dev, DType::Float32, false);
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::max_pool2d_forward(x->data_ptr<float>(), out->data_ptr<float>(), argmax->data_ptr<float>(), g.planes, g.H,
                                 g.W, g.H_out, g.W_out, g.kH, g.kW, g.sH, g.sW);
    } else
#endif
    {
        auto xh = (dev == Device::CPU) ? x : x->cpu();
        auto oh = (dev == Device::CPU) ? out : std::make_shared<Tensor>(out_shape, Device::CPU, DType::Float32, false);
        auto ah = (dev == Device::CPU) ? argmax : std::make_shared<Tensor>(out_shape, Device::CPU, DType::Float32, false);
        maxpool_detail::forward_host(xh->data_ptr<float>(), oh->data_ptr<float>(), ah->data_ptr<float>(), g);
        if (dev != Device::CPU) {
            out = oh->to(dev);
            argmax = ah->to(dev);
        }
    }

    out->requires_grad = needs_grad(x);
    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<MaxPool2dBackward>(x->shape, argmax, g);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }
    return out;
}
