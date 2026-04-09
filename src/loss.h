#pragma once 
#include "engine.h"

//cross_entropy
inline std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if(pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    float sum_loss = 0.0f;
    int n = static_cast<int>(pred->size());
    for(int i=0; i<n; i++) {
        sum_loss -= target->data_at(i) * std::log(pred->data_at(i));
    }
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});

    out->data_at(0) = sum_loss / n; 
    out->prev.push_back(pred);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [pred, target, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<n; i++) {
                pred->grad_at(i) += -(target->data_at(i) / (pred->data_at(i) * n)) * self->grad_at(0); 
            }
        }
    };

    return out;
}

//mse
inline std::shared_ptr<Tensor> mse(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if(pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});

    float sq_sum = 0.0f; 
    int n = static_cast<int>(pred->size());
    for(int i = 0; i<n; i++) {
        sq_sum += (target->data_at(i) - pred->data_at(i))**2; 
    }
    out->data_at(0) = sq_sum/n;
    out->prev.push_back(pred);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [pred, target, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<n; i++) {
                pred->grad_at(i) += 2*(target->data_at(i) - pred->data_at(i)) * self->grad_at(0); 
            }
        }
    };

    return out;
}
