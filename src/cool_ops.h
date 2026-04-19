#pragma once
#include"engine.h"

//scale and shift
inline std::shared_ptr<Tensor> scale_n_shift(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& gamma, const std::shared_ptr<Tensor>& beta) {
    int ndim = x->shape.size();
    int r = x->shape[ndim-2];
    int c = x->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= x->shape[i]; }

    auto out = std::make_shared<Tensor>(x->shape);

    // forward
    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int x_idx = batch*r*c + i*c + j;
                int gb_idx = i*c + j;
                out->data_at(x_idx) = x->data_at(x_idx) * gamma->data_at(gb_idx) 
                                    + beta->data_at(gb_idx);
            }
        }
    }

    out->prev = {x, gamma, beta};
    std::weak_ptr<Tensor> weak_out = out;

    // backward
    out->backward_fn = [x, gamma, beta, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int x_idx = batch*r*c + i*c + j;
                        int gb_idx = i*c + j;
                        // grad w.r.t x
                        x->grad_at(x_idx) += gamma->data_at(gb_idx) * self->grad_at(x_idx);
                        // grad w.r.t gamma — accumulates across batches
                        gamma->grad_at(gb_idx) += x->data_at(x_idx) * self->grad_at(x_idx);
                        // grad w.r.t beta — accumulates across batches
                        beta->grad_at(gb_idx) += self->grad_at(x_idx);
                    }
                }
            }
        }
    };

    return out;
}
