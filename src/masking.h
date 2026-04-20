#pragma once
#include "engine.h"

//boolean mask
inline std::shared_ptr<Tensor> bool_mask(const std::shared_ptr<Tensor>& a,const std::shared_ptr<Tensor>& m) {
    int ndim = a->shape.size();
    int r = a->shape[ndim-2];
    int c = a->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= a->shape[i]; }

    auto out = std::make_shared<Tensor>(a->shape);

    //forward
    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int idx = batch*r*c + i*c + j;
                out->data_at(idx) = m->data_at(idx) == 0.0f
                                  ? -std::numeric_limits<float>::infinity()
                                  : a->data_at(idx);
            }
        }
    }
    out->prev.push_back(a); //m has no gradient, it's not a learned parameter

    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, m, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int idx = batch*r*c + i*c + j;
                        //gradient only flows where mask is 1
                        if(m->data_at(idx) != 0.0f) {
                            a->grad_at(idx) += self->grad_at(idx);
                        }
                    }
                }
            }
        }
    };

    return out;
}

//casual mask
inline std::shared_ptr<Tensor> causal_mask(const std::shared_ptr<Tensor>& a) {
    int ndim = a->shape.size();
    int r = a->shape[ndim-2];
    int c = a->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= a->shape[i]; }

    auto out = std::make_shared<Tensor>(a->shape);

    //forward — keep lower triangle, set upper to -inf
    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int idx = batch*r*c + i*c + j;
                out->data_at(idx) = j > i
                                  ? -std::numeric_limits<float>::infinity()
                                  : a->data_at(idx);
            }
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    //backward — gradient only flows through lower triangle
    out->backward_fn = [a, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int idx = batch*r*c + i*c + j;
                        if(j <= i) {
                            a->grad_at(idx) += self->grad_at(idx);
                        }
                    }
                }
            }
        }
    };

    return out;
}
