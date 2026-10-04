#pragma once
#include "tensor.h"
#include "functional.h"
#include <limits>
#include <stdexcept>

struct MaxPool2dBackward : public Node {
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> indices;
    
    MaxPool2dBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> indices)
        : x(x), indices(indices) {}

    void release_variables() override { x = nullptr; indices = nullptr; }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !x || !indices) return {nullptr};
        auto grad_out = grads[0];

        if (x->device != Device::CPU) {
            auto x_cpu = x->to(Device::CPU);
            auto idx_cpu = indices->to(Device::CPU);
            auto go_cpu = grad_out->to(Device::CPU);
            MaxPool2dBackward cpu_node(x_cpu, idx_cpu);
            auto cpu_grads = cpu_node.apply({go_cpu});
            if (cpu_grads.empty() || !cpu_grads[0]) return {nullptr};
            return {cpu_grads[0]->to(x->device)};
        }

        auto grad_x = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);
        grad_x->fill_(0.0f);

        const float* go_data = grad_out->data_ptr<float>();
        const float* idx_data = indices->data_ptr<float>();
        float* gx_data = grad_x->data_ptr<float>();

        int size = grad_out->size();
        int x_size = grad_x->size();
        for (int i = 0; i < size; ++i) {
            int max_idx = static_cast<int>(idx_data[i]);
            if (max_idx >= 0 && max_idx < x_size) {
                gx_data[max_idx] += go_data[i];
            }
        }
        
        return {grad_x};
    }
};

inline std::shared_ptr<Tensor> max_pool2d(const std::shared_ptr<Tensor>& x, int kH, int kW, int strideH, int strideW) {
    if (!x) throw std::runtime_error("max_pool2d: input tensor is null");
    if (x->ndim() != 4) throw std::runtime_error("max_pool2d: x must be a 4D tensor (B, C, H, W)");
    if (strideH <= 0 || strideW <= 0) throw std::runtime_error("max_pool2d: strides must be > 0");

    auto xc = make_contiguous(x);
    int B = xc->shape[0];
    int C = xc->shape[1];
    int H = xc->shape[2];
    int W = xc->shape[3];

    if (H < kH || W < kW) throw std::runtime_error("max_pool2d: kernel size exceeds input spatial dimensions");

    int H_out = (H - kH) / strideH + 1;
    int W_out = (W - kW) / strideW + 1;

    if (xc->device != Device::CPU) {
        auto xc_cpu = xc->to(Device::CPU);
        auto out_cpu = max_pool2d(xc_cpu, kH, kW, strideH, strideW);
        auto out_res = out_cpu->to(xc->device);
        if (xc->requires_grad) {
            auto grad_fn_cpu = std::static_pointer_cast<MaxPool2dBackward>(out_cpu->grad_fn);
            auto grad_fn = std::make_shared<MaxPool2dBackward>(xc, grad_fn_cpu->indices->to(xc->device));
            grad_fn->add_next_edge(get_grad_edge(xc).function, 0);
            out_res->grad_fn = grad_fn;
        }
        return out_res;
    }

    auto out = std::make_shared<Tensor>(std::vector<int64_t>{B, C, H_out, W_out}, xc->device, xc->dtype, xc->requires_grad);
    auto indices = std::make_shared<Tensor>(std::vector<int64_t>{B, C, H_out, W_out}, xc->device, xc->dtype, false);
    indices->fill_(0.0f);

    const float* x_data = xc->data_ptr<float>();
    float* out_data = out->data_ptr<float>();
    float* idx_data = indices->data_ptr<float>();

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;

                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = h * strideH + kh;
                            int iw = w * strideW + kw;
                            int flat_idx = ((b * C + c) * H + ih) * W + iw;
                            float val = x_data[flat_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = flat_idx;
                            }
                        }
                    }

                    int out_idx = ((b * C + c) * H_out + h) * W_out + w;
                    out_data[out_idx] = max_val;
                    idx_data[out_idx] = static_cast<float>(max_idx);
                }
            }
        }
    }

    if (xc->requires_grad) {
        auto grad_fn = std::make_shared<MaxPool2dBackward>(xc, indices);
        grad_fn->add_next_edge(get_grad_edge(xc).function, 0);
        out->grad_fn = grad_fn;
    }
    
    return out;
}
