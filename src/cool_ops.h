#pragma once
#include "engine.h"

// --- SCALE AND SHIFT BACKWARD NODE ---
struct ScaleAndShiftBackward : public Node {
    std::shared_ptr<Tensor> x, g, be;
    int r, c, batch_size, ndim, nout;

    ScaleAndShiftBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> g, std::shared_ptr<Tensor> be,
                          int r, int c, int batch_size, int ndim, int nout)
        : x(x), g(g), be(be), r(r), c(c), batch_size(batch_size), ndim(ndim), nout(nout) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x->shape, false);
        auto gg = std::make_shared<Tensor>(g->shape, false);
        auto gbe = std::make_shared<Tensor>(be->shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        const float* x_ptr = x->data_ptr();
        const float* g_ptr = g->data_ptr();

        float* gx_ptr = gx->data_ptr();
        float* gg_ptr = gg->data_ptr();
        float* gbe_ptr = gbe->data_ptr();

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int> batch_idx = unravel(batch, std::vector<int>(x->shape.begin(), x->shape.end() - 2));

            int batch_off_x = 0;
            int batch_off_out = 0;
            for (int i = 0; i < ndim - 2; i++) {
                batch_off_x   += batch_idx[i] * x->strides[i];
                batch_off_out += batch_idx[i] * self_grad->strides[i];
            }

            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int flat_x   = batch_off_x   + x->strides[ndim - 2] * i + x->strides[ndim - 1] * j;
                    int flat_out = batch_off_out  + self_grad->strides[nout - 2] * i + self_grad->strides[nout - 1] * j;
                    int gb_idx   = j;

                    float dout = sg_ptr[flat_out];
                    gx_ptr[flat_x]  += g_ptr[gb_idx] * dout;
                    gg_ptr[gb_idx]  += x_ptr[flat_x] * dout;
                    gbe_ptr[gb_idx] += dout;
                }
            }
        }
        return {gx, gg, gbe};
    }
};

// scale_n_shift
inline std::shared_ptr<Tensor> scale_n_shift(const std::shared_ptr<Tensor>& x,
                                             const std::shared_ptr<Tensor>& gamma,
                                             const std::shared_ptr<Tensor>& beta) {
    int ndim = x->ndim();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    auto g = make_contiguous(gamma);
    auto be = make_contiguous(beta);

    bool req_grad = x->requires_grad || gamma->requires_grad || beta->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    int nout = out->ndim();

    const float* x_ptr = x->data_ptr();
    const float* g_ptr = g->data_ptr();
    const float* be_ptr = be->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int> batch_idx = unravel(batch, std::vector<int>(x->shape.begin(), x->shape.end() - 2));

        int batch_off_x = 0;
        int batch_off_out = 0;
        for (int i = 0; i < ndim - 2; i++) {
            batch_off_x   += batch_idx[i] * x->strides[i];
            batch_off_out += batch_idx[i] * out->strides[i];
        }

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int flat_x   = batch_off_x   + x->strides[ndim - 2] * i   + x->strides[ndim - 1] * j;
                int flat_out = batch_off_out  + out->strides[nout - 2] * i + out->strides[nout - 1] * j;
                int gb_idx   = i * c + j;

                out_ptr[flat_out] = x_ptr[flat_x] * g_ptr[gb_idx] + be_ptr[gb_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<ScaleAndShiftBackward>(x, g, be, r, c, batch_size, ndim, nout);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
        grad_fn->add_next_edge(get_grad_edge(beta).function, 2);
        out->grad_fn = grad_fn;
    }

    return out;
}
