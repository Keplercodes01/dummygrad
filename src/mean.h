#pragma once 
#include "engine.h"

// --- MEAN BACKWARD NODE ---
struct MeanBackward : public Node {
    std::vector<int> in_shape;
    std::vector<int> in_strides;
    int axis;
    int r, c, batch_size, ndim, nout;

    MeanBackward(const std::vector<int>& in_shape, const std::vector<int>& in_strides,
                 int axis, int r, int c, int batch_size, int ndim, int nout)
        : in_shape(in_shape), in_strides(in_strides), axis(axis), r(r), c(c),
          batch_size(batch_size), ndim(ndim), nout(nout) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(in_shape, false);
        const float* sg_ptr = self_grad->data_ptr();
        float* ga_ptr = grad_a->data_ptr();

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int> batch_idx = unravel(batch, std::vector<int>(in_shape.begin(), in_shape.end() - 2));

            int batch_off_a = 0;
            int batch_off_out = 0;
            for (int i = 0; i < ndim - 2; i++) {
                batch_off_a   += batch_idx[i] * in_strides[i];
                batch_off_out += batch_idx[i] * self_grad->strides[i];
            }

            if (axis == 0) {
                for (int i = 0; i < c; i++) {
                    int flat_out = batch_off_out + self_grad->strides[nout - 2] * 0 + self_grad->strides[nout - 1] * i;
                    float val = sg_ptr[flat_out] / r;
                    for (int j = 0; j < r; j++) {
                        int flat_a = batch_off_a + in_strides[ndim - 2] * j + in_strides[ndim - 1] * i;
                        ga_ptr[flat_a] = val;
                    }
                }
            } else {
                for (int i = 0; i < r; i++) {
                    int flat_out = batch_off_out + self_grad->strides[nout - 2] * i + self_grad->strides[nout - 1] * 0;
                    float val = sg_ptr[flat_out] / c;
                    for (int j = 0; j < c; j++) {
                        int flat_a = batch_off_a + in_strides[ndim - 2] * i + in_strides[ndim - 1] * j;
                        ga_ptr[flat_a] = val;
                    }
                }
            }
        }
        return {grad_a};
    }
};

inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->ndim();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = 1 : out_shape[ndim - 1] = 1;
    
    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    int nout = out->ndim();

    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();

    // forward
    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int> batch_idx = unravel(batch, std::vector<int>(a->shape.begin(), a->shape.end() - 2));

        int batch_off_a = 0;
        int batch_off_out = 0;
        for (int i = 0; i < ndim - 2; i++) {
            batch_off_a   += batch_idx[i] * a->strides[i];
            batch_off_out += batch_idx[i] * out->strides[i];
        }

        if (axis == 0) {
            for (int i = 0; i < c; i++) {
                float total = 0.0f;
                for (int j = 0; j < r; j++) {
                    int flat_a = batch_off_a + a->strides[ndim - 2] * j + a->strides[ndim - 1] * i;
                    total += a_ptr[flat_a];
                }
                int flat_out = batch_off_out + out->strides[nout - 2] * 0 + out->strides[nout - 1] * i;
                out_ptr[flat_out] = total / r;
            }
        } else {
            for (int i = 0; i < r; i++) {
                float total = 0.0f;
                for (int j = 0; j < c; j++) {
                    int flat_a = batch_off_a + a->strides[ndim - 2] * i + a->strides[ndim - 1] * j;
                    total += a_ptr[flat_a];
                }
                int flat_out = batch_off_out + out->strides[nout - 2] * i + out->strides[nout - 1] * 0;
                out_ptr[flat_out] = total / c;
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<MeanBackward>(a->shape, a->strides, axis, r, c, batch_size, ndim, nout);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
