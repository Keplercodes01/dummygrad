#pragma once
#include"engine.h"

//broadcast
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a, int axis, int n) {
    int ndim = a->shape.size();
    int r = a->shape[ndim-2];
    int c = a->shape[ndim-1];

    if(axis == 0 && r != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }
    if(axis == 1 && c != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }

    int batch_size = 1;
    for(int i = 0; i<ndim-2; i++) { batch_size *= a->shape[i]; } 

    //calculate out_shape
    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim-2] = n : out_shape[ndim-1] = n;

    auto out = std::make_shared<Tensor>(out_shape);

    //forward
    if(axis == 0) {
        for(int batch = 0; batch<batch_size; batch++) {
            for(int i = 0; i<n; i++) {
                for(int j = 0; j<c; j++) {
                    out->data_at(i*c + j + batch*n*c) = a->data_at(j + batch*c);
                }
            }
        }
    }
    else {
        for(int batch = 0; batch<batch_size; batch++) {
            for(int i = 0; i<r; i++) {
                for(int j = 0; j<n; j++) {
                    out->data_at(i*n + j + batch*r*n) = a->data_at(i + batch*r);
                }
            }
        }
    }
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, weak_out, r, c, n, batch_size, axis, ndim]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int batch = 0; batch < batch_size; batch++) {
                    for(int i = 0; i < n; i++) {
                        for(int j = 0; j < c; j++) {
                            a->grad_at(j + batch*c) += self->grad_at(i*c + j + batch*n*c);
                        }
                    }
                }
            }
            else {
                for(int batch = 0; batch < batch_size; batch++) {
                    for(int i = 0; i < r; i++) {
                        for(int j = 0; j < n; j++) {
                            a->grad_at(i + batch*r) += self->grad_at(i*n + j + batch*r*n);
                        }
                    }
                }
            }
        }
    };

    return out;
}

//collapse
inline std::shared_ptr<Tensor> collapse(const std::shared_ptr<Tensor>& a, int axis) {
    int ndim = a->shape.size();
    int r = a->shape[ndim-2];
    int c = a->shape[ndim-1];

    int batch_size = 1;
    for(int i = 0; i<ndim-2; i++) { batch_size *= a->shape[i]; } 

    //calculate out_shape
    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim-2] = 1 : out_shape[ndim-1] = 1;

    auto out = std::make_shared<Tensor>(out_shape);

    //forward
    if(axis == 0) {
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
    else {
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
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, weak_out, r, c, axis, ndim, batch_size]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
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

//broadcast and add
inline std::shared_ptr<Tensor> cast_add(const std::shared_ptr<Tensor>& x,
                                         const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim-2];
    int c = x->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= x->shape[i]; }

    auto out = std::make_shared<Tensor>(x->shape);

    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int x_idx = batch*r*c + i*c + j;
                int y_idx = i*c + j;
                out->data_at(x_idx) = x->data_at(x_idx) + y->data_at(y_idx);
            }
        }
    }

    out->prev = {x, y};
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [x, y, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int x_idx = batch*r*c + i*c + j;
                        int y_idx = i*c + j;
                        x->grad_at(x_idx) += self->grad_at(x_idx);
                        y->grad_at(y_idx) += self->grad_at(x_idx); 
                    }
                }
            }
        }
    };
    return out;
}

//broadcast and subtract
inline std::shared_ptr<Tensor> cast_sub(const std::shared_ptr<Tensor>& x,
                                         const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim-2];
    int c = x->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= x->shape[i]; }

    auto out = std::make_shared<Tensor>(x->shape);

    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int x_idx = batch*r*c + i*c + j;
                int y_idx = i*c + j;
                out->data_at(x_idx) = x->data_at(x_idx) - y->data_at(y_idx);
            }
        }
    }

    out->prev = {x, y};
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [x, y, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int x_idx = batch*r*c + i*c + j;
                        int y_idx = i*c + j;
                        x->grad_at(x_idx) += self->grad_at(x_idx);
                        y->grad_at(y_idx) -= self->grad_at(x_idx); 
                    }
                }
            }
        }
    };
    return out;
}

//broadcast and multiply
inline std::shared_ptr<Tensor> cast_mul(const std::shared_ptr<Tensor>& x,
                                         const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim-2];
    int c = x->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= x->shape[i]; }

    auto out = std::make_shared<Tensor>(x->shape);

    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int x_idx = batch*r*c + i*c + j;
                int y_idx = i*c + j;
                out->data_at(x_idx) = x->data_at(x_idx) * y->data_at(y_idx);
            }
        }
    }

    out->prev = {x, y};
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [x, y, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int x_idx = batch*r*c + i*c + j;
                        int y_idx = i*c + j;
                        x->grad_at(x_idx) += y->data_at(y_idx) * self->grad_at(x_idx);
                        y->grad_at(y_idx) += x->data_at(x_idx) * self->grad_at(x_idx);
                    }
                }
            }
        }
    };
    return out;
}

//broadcast and divide
inline std::shared_ptr<Tensor> cast_div(const std::shared_ptr<Tensor>& x,
                                         const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim-2];
    int c = x->shape[ndim-1];
    int batch_size = 1;
    for(int i = 0; i < ndim-2; i++) { batch_size *= x->shape[i]; }

    auto out = std::make_shared<Tensor>(x->shape);

    for(int batch = 0; batch < batch_size; batch++) {
        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                int x_idx = batch*r*c + i*c + j;
                int y_idx = i*c + j;
                out->data_at(x_idx) = x->data_at(x_idx) / y->data_at(y_idx);
            }
        }
    }

    out->prev = {x, y};
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [x, y, weak_out, r, c, batch_size]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch < batch_size; batch++) {
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < c; j++) {
                        int x_idx = batch*r*c + i*c + j;
                        int y_idx = i*c + j;
                        x->grad_at(x_idx) += self->grad_at(x_idx) / y->data_at(y_idx);
                        y->grad_at(y_idx) -= (x->data_at(x_idx) * self->grad_at(x_idx))
                                           / (y->data_at(y_idx) * y->data_at(y_idx)); 
                    }
                }
            }
        }
    };
    return out;
}
