#pragma once 
#include "engine.h"

//concatenation
inline std::shared_ptr<Tensor> concat(const std::shared_ptr<Tensor>& a,
                                       const std::shared_ptr<Tensor>& b, int axis) {
    int ndim = a->shape.size();
    int r_a = a->shape[ndim-2];
    int c_a = a->shape[ndim-1];
    int r_b = b->shape[ndim-2];
    int c_b = b->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= a->shape[i]; }

    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim-2] = r_a + r_b : out_shape[ndim-1] = c_a + c_b;

    auto out = std::make_shared<Tensor>(out_shape);
    int r_out = out_shape[ndim-2];
    int c_out = out_shape[ndim-1];

    //forward
    if(axis == 0) {
        for(int batch = 0; batch < batch_size; batch++) {
            //copy a
            for(int i = 0; i < r_a; i++) {
                for(int j = 0; j < c_a; j++) {
                    out->data_at(batch*r_out*c_out + i*c_out + j) =
                        a->data_at(batch*r_a*c_a + i*c_a + j);
                }
            }
            //copy b after a
            for(int i = 0; i < r_b; i++) {
                for(int j = 0; j < c_b; j++) {
                    out->data_at(batch*r_out*c_out + (r_a+i)*c_out + j) =
                        b->data_at(batch*r_b*c_b + i*c_b + j);
                }
            }
        }
    }
    else {
        for(int batch = 0; batch < batch_size; batch++) {
            //copy a
            for(int i = 0; i < r_a; i++) {
                for(int j = 0; j < c_a; j++) {
                    out->data_at(batch*r_out*c_out + i*c_out + j) =
                        a->data_at(batch*r_a*c_a + i*c_a + j);
                }
            }
            //copy b after a along cols
            for(int i = 0; i < r_b; i++) {
                for(int j = 0; j < c_b; j++) {
                    out->data_at(batch*r_out*c_out + i*c_out + (c_a+j)) =
                        b->data_at(batch*r_b*c_b + i*c_b + j);
                }
            }
        }
    }

    out->prev.push_back(a);
    out->prev.push_back(b);
    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, b, weak_out, r_a, c_a, r_b, c_b, r_out, c_out, batch_size, axis, ndim]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int batch = 0; batch < batch_size; batch++) {
                    //grad to a
                    for(int i = 0; i < r_a; i++) {
                        for(int j = 0; j < c_a; j++) {
                            a->grad_at(batch*r_a*c_a + i*c_a + j) +=
                                self->grad_at(batch*r_out*c_out + i*c_out + j);
                        }
                    }
                    //grad to b
                    for(int i = 0; i < r_b; i++) {
                        for(int j = 0; j < c_b; j++) {
                            b->grad_at(batch*r_b*c_b + i*c_b + j) +=
                                self->grad_at(batch*r_out*c_out + (r_a+i)*c_out + j);
                        }
                    }
                }
            }
            else {
                for(int batch = 0; batch < batch_size; batch++) {
                    //grad to a
                    for(int i = 0; i < r_a; i++) {
                        for(int j = 0; j < c_a; j++) {
                            a->grad_at(batch*r_a*c_a + i*c_a + j) +=
                                self->grad_at(batch*r_out*c_out + i*c_out + j);
                        }
                    }
                    //grad to b
                    for(int i = 0; i < r_b; i++) {
                        for(int j = 0; j < c_b; j++) {
                            b->grad_at(batch*r_b*c_b + i*c_b + j) +=
                                self->grad_at(batch*r_out*c_out + i*c_out + (c_a+j));
                        }
                    }
                }
            }
        }
    };

    return out;
}

