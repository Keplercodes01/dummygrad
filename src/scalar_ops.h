#pragma once
#include"engine.h"

//add_scalar
inline std::shared_ptr<Tensor> add_scalar(const std::shared_ptr<Tensor>& a, float s) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) out->data_at(i) = a->data_at(i) + s;
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) a->grad_at(i) += self->grad_at(i);
        }
    };
    return out;
}

//sub_scalar
inline std::shared_ptr<Tensor> sub_scalar(const std::shared_ptr<Tensor>& a, float s) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) out->data_at(i) = a->data_at(i) - s;
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) a->grad_at(i) += self->grad_at(i);
        }
    };
    return out;
}

//mul_scalar
inline std::shared_ptr<Tensor> mul_scalar(const std::shared_ptr<Tensor>& a, float s) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) out->data_at(i) = a->data_at(i) * s;
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_fn = [a, weak_out, s]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) a->grad_at(i) += s * self->grad_at(i);
        }
    };
    return out;
}

//div_scalar
inline std::shared_ptr<Tensor> div_scalar(const std::shared_ptr<Tensor>& a, float s) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) out->data_at(i) = a->data_at(i) / s;
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_fn = [a, weak_out, s]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) a->grad_at(i) += self->grad_at(i) / s;
        }
    };
    return out;
}

//neg
inline std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& a) {
    return mul_scalar(a, -1.0f);
}

//rsub_scalar
inline std::shared_ptr<Tensor> rsub_scalar(float s, const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) out->data_at(i) = s - a->data_at(i);
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) a->grad_at(i) -= self->grad_at(i);
        }
    };
    return out;
}

//rdiv_scalar
inline std::shared_ptr<Tensor> rdiv_scalar(float s, const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->size(); i++) out->data_at(i) = s / a->data_at(i);
    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_fn = [a, weak_out, s]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < a->size(); i++) 
                a->grad_at(i) -= s * self->grad_at(i) / (a->data_at(i) * a->data_at(i));
        }
    };
    return out;
}
