#include<engine.h>

//CrossEntropyLoss
inline std::shared_ptr<Tensor> CrossEntropyLoss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if(pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    float sum_loss = 0.0f;
    for(int i=0; i<pred->size(); i++) {
        sum_loss -= target->data_at(i) * std::log(pred->data_at(i));
    }

    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});

    float n = static_cast<float>(pred->size());

    out->data_at(0) = sum_loss / n; 
    out->prev.push_back(pred);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [pred, target, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<pred->size(); i++) {
                pred->grad_at(i) += -(target->data_at(i) / (pred->data_at(i) * n)) * self->grad_at(0); 
            }
        }
    };

    return out;
}
