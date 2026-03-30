#include"engine.h"

//softmax
inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a) {
    int n = a->shape.size();
    int last = a->shape[n-1];  // size of last dimension
    int outer = a->size() / last;  // product of all other dimensions

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i < outer; i++) {
        int offset = i * last;

        // find max for numerical stability
        float max_val = a->data_at(offset);
        for(int j = 1; j < last; j++) {
            max_val = std::max(max_val, a->data_at(offset + j));
        }

        // compute softmax
        float sum = 0.0f;
        for(int j = 0; j < last; j++) {
            sum += std::exp(a->data_at(offset + j) - max_val);
        }
        for(int j = 0; j < last; j++) {
            out->data_at(offset + j) = std::exp(a->data_at(offset + j) - max_val) / sum;
        }
    }

    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, last, outer]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < outer; i++) {
                int offset = i * last;

                float max_val = a->data_at(offset);
                for(int j = 1; j < last; j++) {
                    max_val = std::max(max_val, a->data_at(offset + j));
                }
                float sum = 0.0f;
                for(int j = 0; j < last; j++) {
                    sum += std::exp(a->data_at(offset + j) - max_val);
                }
                for(int k = 0; k < last; k++) {
                    float y_k = std::exp(a->data_at(offset + k) - max_val);
                    a->grad_at(offset + k) += (y_k * (sum - y_k) / (sum * sum)) * self->grad_at(offset + k);
                }
            }
        }
    };

    return out;
}

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
                a->grad_at(i) += (1.0 - (self->data_at(i) * self->data_at(i))) * self->grad_at(i);
            }
        }
    };

    return out;
}
