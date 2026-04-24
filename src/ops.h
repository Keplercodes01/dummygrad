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

//simple sum
inline std::shared_ptr<Tensor> simple_sum(const std::shared_ptr<Tensor>& a) {
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

