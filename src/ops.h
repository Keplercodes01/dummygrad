#pragma once
#include"engine.h"

//add
inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in addition. Cmon man..");
    }
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) + b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i);
                b->grad_at(i) += self->grad_at(i);
            }
        }    
    };

    return out;
}
                         
//sub
inline std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in subtraction. Cmon man..");
    }
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) - b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i);
                b->grad_at(i) -= self->grad_at(i);
            }
        }
    };

    return out;
}

//mul
inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in multiplication. Cmon man..");
    }
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) * b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += b->data_at(i) * self->grad_at(i);
                b->grad_at(i) += a->data_at(i) * self->grad_at(i);
            }
        }
    };

    return out;
}

//divide
inline std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in division. Cmon man..");
    }
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) / b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out; 

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i) / b->data_at(i);
                b->grad_at(i) -= a->data_at(i) * self->grad_at(i) / (b->data_at(i) * b->data_at(i));
            }
        }
    };

    return out;
}
//transpose 
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a) {
    int n = a->shape.size();

    auto out = std::make_shared<Tensor>(a->shape);
    out->storage = a->storage;
    out->strides = a->strides;

    //swap the last two dims and strides
    std::swap(out->shape[n-2], out->shape[n-1]);
    std::swap(out->strides[n-2], out->strides[n-1]);

    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            int r = a->shape[n-2];
            int c = a->shape[n-1];
            int batch_size = 1;
            for(int i = 0; i<n-2; i++) { batch_size *= a->shape[i]; }

            for(int batch = 0; batch<batch_size; batch++) {
                for(int i=0; i<r; i++) {
                    for(int j=0; j<c; j++) {
                        a->grad_at(a->strides[n-2]*i + a->strides[n-1]*j + batch*r*c) 
                            += self->grad_at(self->strides[n-2]*j + self->strides[n-1]*i + batch*r*c);
                    }
                }
            }
        }
    };

    return out;
}

//the mighty matmul
inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    int n1 = a->shape.size();
    int n2 = b->shape.size();

    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    //check the batch dimensions match 
    for(int i = 0; i < n1-2; i++) {
        if(a->shape[i] != b->shape[i]) {
            throw std::runtime_error("Batch dimensions mismatch for matmul.. CMON MAN.");
        }
    }
    //check inner dimensions match
    if(c1 != r2) {
        throw std::runtime_error("Matmul dimensions mismatch.. CMON MAN.");  
    }

    //copy the batch dimensions from a 
    std::vector<int> out_shape;
    for(int i = 0; i<n1-2; i++) {
        out_shape.push_back(a->shape[i]);
    }
    out_shape.push_back(r1);
    out_shape.push_back(c2);

    auto out = std::make_shared<Tensor>(out_shape);
    int nout = out->shape.size();

    int batch_size = 1;
    for(int i = 0; i<n1-2; i++) { batch_size *= a->shape[i]; }

    for(int batch = 0; batch<batch_size; batch++) {
        for(int i = 0; i<r1; i++) {
            for(int j = 0; j<c2; j++) {
                float sum = 0.0f;
                for(int k = 0; k<c1; k++) {
                    sum += a->data_at(a->strides[n1-2]*i + a->strides[n1-1]*k + batch*r1*c1) 
                         * b->data_at(b->strides[n2-2]*k + b->strides[n2-1]*j + batch*r2*c2);                       
                }
                out->data_at(out->strides[nout-2]*i + out->strides[nout-1]*j + batch*r1*c2) = sum;
            }
        }
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out, r1, c1, r2, c2, batch_size, n1, n2, nout]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch<batch_size; batch++) {
                //a.grad
                for(int i=0; i<r1; i++) {
                    for(int j=0; j<c1; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<c2; k++) {
                            sum += self->grad_at(self->strides[nout-2]*i + self->strides[nout-1]*k + batch*r1*c2) 
                                 * b->data_at(b->strides[n2-2]*j + b->strides[n2-1]*k + batch*r2*c2);
                        }
                        a->grad_at(a->strides[n1-2]*i + a->strides[n1-1]*j + batch*r1*c1) += sum;
                    }
                }
                //b.grad
                for(int i=0; i<r2; i++) {
                    for(int j=0; j<c2; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<r1; k++) {
                            sum += a->data_at(a->strides[n1-2]*k + a->strides[n1-1]*i + batch*r1*c1) 
                                 * self->grad_at(self->strides[nout-2]*k + self->strides[nout-1]*j + batch*r1*c2);
                        }
                        b->grad_at(b->strides[n2-2]*i + b->strides[n2-1]*j + batch*r2*c2) += sum;
                    }    
                }
            }
        }
    };

    return out;
}

//pow
inline std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const int n) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::pow(a->data_at(i), n);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += n * std::pow(a->data_at(i), n-1) * self->grad_at(i);
            }
        }
    };

    return out;
}

//sqrt
inline std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::sqrt(a->data_at(i));
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += (0.5f / std::sqrt(a->data_at(i))) * self->grad_at(i);
            }
        }
    };

    return out;
}

//log
inline std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape); 

    float epsilon = 1e-8f;
    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = std::log(a->data_at(i) + epsilon);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, epsilon]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i) * (1.0f / (a->data_at(i) + epsilon));
            }
        }
    };

    return out;
}

//exp
inline std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = std::exp(a->data_at(i));
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i) * self->data_at(i);
            }
        }
    };

    return out; 
}

//sum
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}); 

    float total = 0.0f;
    for(int i = 0; i<a->size(); i++) {
        total += a->data_at(i);
    }
    out->data_at(0) = total;
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(0);
            }
        }
    };

    return out;
}

//mean
inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}); 

    float sum = 0.0f;
    for(int i=0; i<a->size(); i++) {
        sum += a->data_at(i);
    }
    out->data_at(0) = sum/(a->size());
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            float gradient = self->grad_at(0) / a->size();
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += gradient;
            }
        }
    };

    return out;
}

//view
inline std::shared_ptr<Tensor> view(const std::shared_ptr<Tensor>& a, std::vector<int> new_shape) {
    int total = 1;
    for(int d : new_shape) total *= d;
    if(total != a->size()) throw std::runtime_error("view: total elements must match. cmon man.");

    auto out = std::make_shared<Tensor>(new_shape);
    out->storage = a->storage;
    out->offset = a->offset;

    out->prev.push_back(a);  

    return out;
}

//standard deviation
inline std::shared_ptr<Tensor> std(const std::shared_ptr<Tensor>& a, int dim) {
    int n = a->shape.size();
    int r = a->shape[n-2];
    int c = a->shape[n-1];
    std::vector<int> out_shape = a->shape.pop_back().push_back(1);

    int batch_size = 1;
    for(int i = 0; i<n-2; i++) { batch_size *= a->shape[i]; }

    auto out = std::make_shared<Tensor>(out_shape);
    for(int batch = 0; batch<batch_size; batch++) {
        for(int i = 0; i<r; i++) {
            float sum = 0.0f;
            for(int j = 0; j<c; j++) {
                sum += a->data_at(i*r + j + batch*r*c);
            }
            out->data_at(i + batch*r) = std::sqrt(sum/c);
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out; 

    out->backward_fn = [a, weak_out, n, r, c, batch_size]() {
    }

}
