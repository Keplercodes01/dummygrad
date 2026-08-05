#pragma once
#include "engine.h"

// --- STD DEV BACKWARD NODE ---
struct StdDevBackward : public Node {
    std::shared_ptr<Tensor> a;
    int axis, r, c, batch_size;

    StdDevBackward(std::shared_ptr<Tensor> a, int axis, int r, int c, int batch_size)
        : a(a), axis(axis), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(a->shape, false);
        const float* sg_ptr = self_grad->data_ptr();
        const float* a_ptr = a->data_ptr();
        float* ga_ptr = grad_a->data_ptr();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < c; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < r; j++) {
                        sum += a_ptr[j * c + i + batch * r * c];
                    }
                    float mean_val = sum / r;
                    float x = 0.0f;
                    for (int j = 0; j < r; j++) {
                        float diff = a_ptr[j * c + i + batch * r * c] - mean_val;
                        x += diff * diff;
                    }
                    float std_val = std::sqrt(x / r) + 1e-8f;
                    float dout = sg_ptr[i + batch * c];
                    for (int j = 0; j < r; j++) {
                        ga_ptr[j * c + i + batch * r * c] = ((a_ptr[j * c + i + batch * r * c] - mean_val) / (r * std_val)) * dout;
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < c; j++) {
                        sum += a_ptr[i * c + j + batch * r * c];
                    }
                    float mean_val = sum / c;
                    float x = 0.0f;
                    for (int j = 0; j < c; j++) {
                        float diff = a_ptr[i * c + j + batch * r * c] - mean_val;
                        x += diff * diff;
                    }
                    float std_val = std::sqrt(x / c) + 1e-8f;
                    float dout = sg_ptr[i + batch * r];
                    for (int j = 0; j < c; j++) {
                        ga_ptr[i * c + j + batch * r * c] = ((a_ptr[i * c + j + batch * r * c] - mean_val) / (c * std_val)) * dout;
                    }
                }
            }
        }
        return {grad_a};
    }
};

// standard deviation
inline std::shared_ptr<Tensor> std_dev(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < c; i++) {
                float sum = 0.0f;
                for (int j = 0; j < r; j++) {
                    sum += a_ptr[j * c + i + batch * r * c];
                }
                float mean_val = sum / r;
                float x = 0.0f;
                for (int j = 0; j < r; j++) {
                    float diff = a_ptr[j * c + i + batch * r * c] - mean_val;
                    x += diff * diff;
                }
                out_ptr[i + batch * c] = std::sqrt(x / r);
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                float sum = 0.0f;
                for (int j = 0; j < c; j++) {
                    sum += a_ptr[i * c + j + batch * r * c];
                }
                float mean_val = sum / c;
                float x = 0.0f;
                for (int j = 0; j < c; j++) {
                    float diff = a_ptr[i * c + j + batch * r * c] - mean_val;
                    x += diff * diff;
                }
                out_ptr[i + batch * r] = std::sqrt(x / c);
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<StdDevBackward>(a, axis, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
