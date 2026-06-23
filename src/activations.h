#pragma once
#include"engine.h"

//relu
inline std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::max(0.0f, a->data_at(i)); 
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                if(a->data_at(i)>0.0f) {
                    a->grad_at(i) += self->grad_at(i);
                }
            }
        }
    };

    return out;
}

//tanh
inline std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::tanh(a->data_at(i));
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += (1.0f - (self->data_at(i) * self->data_at(i))) * self->grad_at(i);
            }
        }
    };

    return out;
}

//gelu
inline std::shared_ptr<Tensor> gelu(const std::shared_ptr<Tensor>& a) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) {
        float x = a->data_at(i);
        float inner = sqrt_2_over_pi * (x + coeff * x*x*x);
        float tanh_val = std::tanh(inner);
        out->data_at(i) = 0.5f * x * (1.0f + tanh_val);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, sqrt_2_over_pi, coeff]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) {
                float x = a->data_at(i);
                float inner = sqrt_2_over_pi * (x + coeff * x*x*x);
                float tanh_val = std::tanh(inner);
                float sech2 = 1.0f - tanh_val * tanh_val;
                float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x*x);
                float grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
                a->grad_at(i) += grad * self->grad_at(i);
            }
        }
    };
    return out;
}
