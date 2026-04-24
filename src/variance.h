#pragma once 
#include "engine.h"

//variance
inline std::shared_ptr<Tensor> var(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    int r = a->shape[ndim-2];
    int c = a->shape[ndim-1];

    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim-2] = 1 : out_shape[ndim-1] = 1;

    int batch_size = 1;
    for(int i = 0; i<ndim-2; i++) { batch_size *= a->shape[i]; }

    auto out = std::make_shared<Tensor>(out_shape);

    //forward
    if(axis == 0) {
        for(int batch = 0; batch<batch_size; batch++) {
            for(int i = 0; i<c; i++) {
                float sum = 0.0f;
                for(int j = 0; j<r; j++) {
                    sum += a->data_at(j*c + i + batch*r*c);
                }
                float mean = sum/r;
                float x = 0.0f;
                for(int j = 0; j<r; j++) {
                    x += std::pow(a->data_at(j*c + i + batch*r*c) - mean, 2);
                }
                out->data_at(i + batch*c) = x/r;
            }
        }
    }
    else {
        for(int batch = 0; batch<batch_size; batch++) {
            for(int i = 0; i<r; i++) {
                float sum = 0.0f;
                for(int j = 0; j<c; j++) {
                    sum += a->data_at(i*c + j + batch*r*c);
                }
                float mean = sum/c;
                float x = 0.0f;
                for(int j = 0; j<c; j++) {
                    x += std::pow(a->data_at(i*c + j + batch*r*c) - mean, 2);
                }
                out->data_at(i + batch*r) = x/c;
            }
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, weak_out, ndim, r, c, batch_size, axis]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int batch = 0; batch<batch_size; batch++) {
                    for(int i = 0; i<c; i++) {
                        float sum = 0.0f;
                        for(int j = 0; j<r; j++) {
                            sum += a->data_at(j*c + i + batch*r*c);
                        }
                        float mean = sum/r;
                        float x = 0.0f;
                        for(int j = 0; j<r; j++) {
                            x += std::pow(a->data_at(j*c + i + batch*r*c) - mean, 2);
                        }
                        for(int j = 0; j<r; j++) {
                            a->grad_at(j*c + i + batch*r*c) += (2 * (a->data_at(j*c + i + batch*r*c) - mean) / r) * self->grad_at(i + batch*c);
                        }
                    }
                }
            }
            else {
                for(int batch = 0; batch<batch_size; batch++) {
                    for(int i = 0; i<r; i++) {
                        float sum = 0.0f;
                        for(int j = 0; j<c; j++) {
                            sum += a->data_at(i*c + j + batch*r*c);
                        }
                        float mean = sum/c;
                        float x = 0.0f;
                        for(int j = 0; j<c; j++) {
                            x += std::pow(a->data_at(i*c + j + batch*r*c) - mean, 2);
                        }
                        for(int j = 0; j<c; j++) {
                            a->grad_at(i*c + j + batch*r*c) += (2 * (a->data_at(i*c + j + batch*r*c) - mean) / c) * self->grad_at(i + batch*r);
                        }
                    }
                }
            }
        }
    };

    return out;

}

