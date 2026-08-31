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
        auto grad_out = grads[0];
        auto grad_x = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);
        grad_x->fill_(0.0f);

        const float* go_data = grad_out->data_ptr<float>();
        const float* idx_data = indices->data_ptr<float>();
        float* gx_data = grad_x->data_ptr<float>();

        int size = grad_out->size();
        for (int i = 0; i < size; ++i) {
            int max_idx = static_cast<int>(idx_data[i]);
            gx_data[max_idx] += go_data[i];
        }
        
        return {grad_x};
    }
};

inline std::shared_ptr<Tensor> max_pool2d(const std::shared_ptr<Tensor>& x, int kH, int kW, int strideH, int strideW) {
    auto xc = make_contiguous(x);
    int B = xc->shape[0];
    int C = xc->shape[1];
    int H = xc->shape[2];
    int W = xc->shape[3];

    int H_out = (H - kH) / strideH + 1;
    int W_out = (W - kW) / strideW + 1;

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
