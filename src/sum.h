#pragma once 
#include "engine.h"

//sum
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    if(axis < 0) axis = ndim + axis;
    int r = a->shape[ndim-2];
    int c = a->shape[ndim-1];

    int batch_size = 1;
    for(int i = 0; i<ndim-2; i++) { batch_size *= a->shape[i]; }

    //calculate out_shape
    std::vector<int> out_shape = a->shape;
    out_shape[axis] = 1;

    auto out = std::make_shared<Tensor>(out_shape);

    //forward
    if(axis == ndim-2) {
        for(int batch = 0; batch<batch_size; batch++) {
            for(int i = 0; i<c; i++) {
                float total = 0.0f;
                for(int j = 0; j<r; j++) {
                    total += a->data_at(j*c + i + batch*r*c);
                }
                out->data_at(i + batch*c) = total;
            }
        }
    }
    else if(axis == ndim-1) {
        for(int batch = 0; batch<batch_size; batch++) {
            for(int i = 0; i<r; i++) {
                float total = 0.0f;
                for(int j = 0; j<c; j++) {
                    total += a->data_at(i*c + j + batch*r*c);
                }
                out->data_at(i + batch*r) = total;
            }
        }
    }
    else {
        throw std::runtime_error("only supported for summing along the last two dimensions");
    }

    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, weak_out, r, c, axis, ndim, batch_size]() {
        if(auto self = weak_out.lock()) {
            if(axis == ndim-2) {
                for(int batch = 0; batch<batch_size; batch++) {
                    for(int i = 0; i<c; i++) {
                        for(int j = 0; j<r; j++) {
                            a->grad_at(j*c + i + batch*r*c) += self->grad_at(i + batch*c);
                        }
                    }
                }
            }
            else {
                for(int batch = 0; batch<batch_size; batch++) {
                    for(int i = 0; i<r; i++) {
                        for(int j = 0; j<c; j++) {
                            a->grad_at(i*c + j + batch*r*c) += self->grad_at(i + batch*r);
                        }
                    }
                }
            }
        }
    };

    return out;
}

