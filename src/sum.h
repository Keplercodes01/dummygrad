#pragma once 
#include "engine.h"

// --- SUM BACKWARD NODE ---
struct SumBackward : public Node {
    std::vector<int> in_shape;
    int axis;
    int r, c, batch_size, ndim;

    SumBackward(const std::vector<int>& in_shape, int axis, int r, int c, int batch_size, int ndim)
        : in_shape(in_shape), axis(axis), r(r), c(c), batch_size(batch_size), ndim(ndim) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(in_shape, false);
        const float* sg_ptr = self_grad->data_ptr();
        float* ga_ptr = grad_a->data_ptr();

        if (axis == ndim - 2) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < c; i++) {
                    float g_val = sg_ptr[i + batch * c];
                    for (int j = 0; j < r; j++) {
                        ga_ptr[j * c + i + batch * r * c] = g_val;
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    float g_val = sg_ptr[i + batch * r];
                    for (int j = 0; j < c; j++) {
                        ga_ptr[i * c + j + batch * r * c] = g_val;
                    }
                }
            }
        }
        return {grad_a};
    }
};

// sum
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    if (axis < 0) axis = ndim + axis;
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int> out_shape = a->shape;
    out_shape[axis] = 1;

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);

    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();

    if (axis == ndim - 2) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < c; i++) {
                float total = 0.0f;
                for (int j = 0; j < r; j++) {
                    total += a_ptr[j * c + i + batch * r * c];
                }
                out_ptr[i + batch * c] = total;
            }
        }
    } else if (axis == ndim - 1) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                float total = 0.0f;
                for (int j = 0; j < c; j++) {
                    total += a_ptr[i * c + j + batch * r * c];
                }
                out_ptr[i + batch * r] = total;
            }
        }
    } else {
        throw std::runtime_error("only supported for summing along the last two dimensions");
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SumBackward>(a->shape, axis, r, c, batch_size, ndim);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
