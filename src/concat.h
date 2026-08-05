#pragma once 
#include "engine.h"

// --- CONCAT BACKWARD NODE ---
struct ConcatBackward : public Node {
    std::vector<int> a_shape, b_shape;
    int axis, r_a, c_a, r_b, c_b, r_out, c_out, batch_size;

    ConcatBackward(const std::vector<int>& a_shape, const std::vector<int>& b_shape,
                   int axis, int r_a, int c_a, int r_b, int c_b, int r_out, int c_out, int batch_size)
        : a_shape(a_shape), b_shape(b_shape), axis(axis), r_a(r_a), c_a(c_a), r_b(r_b), c_b(c_b),
          r_out(r_out), c_out(c_out), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(a_shape, false);
        auto gb = std::make_shared<Tensor>(b_shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        float* ga_ptr = ga->data_ptr();
        float* gb_ptr = gb->data_ptr();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r_a; i++) {
                    for (int j = 0; j < c_a; j++) {
                        ga_ptr[batch * r_a * c_a + i * c_a + j] = sg_ptr[batch * r_out * c_out + i * c_out + j];
                    }
                }
                for (int i = 0; i < r_b; i++) {
                    for (int j = 0; j < c_b; j++) {
                        gb_ptr[batch * r_b * c_b + i * c_b + j] = sg_ptr[batch * r_out * c_out + (r_a + i) * c_out + j];
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r_a; i++) {
                    for (int j = 0; j < c_a; j++) {
                        ga_ptr[batch * r_a * c_a + i * c_a + j] = sg_ptr[batch * r_out * c_out + i * c_out + j];
                    }
                }
                for (int i = 0; i < r_b; i++) {
                    for (int j = 0; j < c_b; j++) {
                        gb_ptr[batch * r_b * c_b + i * c_b + j] = sg_ptr[batch * r_out * c_out + i * c_out + (c_a + j)];
                    }
                }
            }
        }
        return {ga, gb};
    }
};

// concatenation
inline std::shared_ptr<Tensor> concat(const std::shared_ptr<Tensor>& a,
                                       const std::shared_ptr<Tensor>& b, int axis) {
    int ndim = a->shape.size();
    int r_a = a->shape[ndim - 2];
    int c_a = a->shape[ndim - 1];
    int r_b = b->shape[ndim - 2];
    int c_b = b->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = r_a + r_b : out_shape[ndim - 1] = c_a + c_b;

    bool req_grad = a->requires_grad || b->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    int r_out = out_shape[ndim - 2];
    int c_out = out_shape[ndim - 1];

    const float* a_ptr = a->data_ptr();
    const float* b_ptr = b->data_ptr();
    float* out_ptr = out->data_ptr();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r_a; i++) {
                for (int j = 0; j < c_a; j++) {
                    out_ptr[batch * r_out * c_out + i * c_out + j] = a_ptr[batch * r_a * c_a + i * c_a + j];
                }
            }
            for (int i = 0; i < r_b; i++) {
                for (int j = 0; j < c_b; j++) {
                    out_ptr[batch * r_out * c_out + (r_a + i) * c_out + j] = b_ptr[batch * r_b * c_b + i * c_b + j];
                }
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r_a; i++) {
                for (int j = 0; j < c_a; j++) {
                    out_ptr[batch * r_out * c_out + i * c_out + j] = a_ptr[batch * r_a * c_a + i * c_a + j];
                }
            }
            for (int i = 0; i < r_b; i++) {
                for (int j = 0; j < c_b; j++) {
                    out_ptr[batch * r_out * c_out + i * c_out + (c_a + j)] = b_ptr[batch * r_b * c_b + i * c_b + j];
                }
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<ConcatBackward>(a->shape, b->shape, axis, r_a, c_a, r_b, c_b, r_out, c_out, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}
