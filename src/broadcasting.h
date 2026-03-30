#include"engine.h"

//broadcast
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a, int axis, int n) {
    int r = a->shape[0];
    int c = a->shape[1];

    auto out = std::make_shared<Tensor>(
            axis == 0 ? std::vector<int>{n, c} : std::vector<int>{r, n}
    );

    if(axis == 0) {
        for(int i = 0; i<n; i++) {
            for(int j = 0; j<c; j++) {
                out->data_at(i*c + j) = a->data_at(j);
            }
        }
    }
    else {
        for(int i = 0; i<r; i++) {
            for(int j = 0; j<n; j++) {
                out->data_at(i*n + j) = a->data_at(i);
            }
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, r, c, n, axis]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int i = 0; i<n; i++) {
                    for(int j = 0; j<c; j++) {
                        a->grad_at(j) += self->grad_at(i*c + j);
                    }
                }
            }
            else {
                for(int i = 0; i<r; i++) {
                    for(int j = 0; j<n; j++) {
                        a->grad_at(i) += self->grad_at(i*n + j);
                    }
                }
            }
        }
    };

    return out;
}

//collapse
inline std::shared_ptr<Tensor> collapse(const std::shared_ptr<Tensor>& a, int axis) {
    int r = a->shape[0];
    int c = a->shape[1];

    auto out = std::make_shared<Tensor>(
            axis == 0 ? std::vector<int>{1, c} : std::vector<int>{r, 1}
    );

    if(axis == 0) {
        for(int i = 0; i<c; i++) {
            float total = 0.0f; 
            for(int j = 0; j<r; j++) {
                total += a->data_at(j*c + i);
            }
            out->data_at(i) = total;
        }
    }
    else {
        for(int i = 0; i<r; i++) {
            float total = 0.0f;
            for(int j = 0; j<c; j++) {
                total += a->data_at(i*c + j);
            }
            out->data_at(i) = total;
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, r, c, axis]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int i = 0; i<c; i++) {
                    for(int j = 0; j<r; j++) {
                        a->grad_at(j*c + i) += self->grad_at(i);
                    }
                }
            }
            else {
                for(int i = 0; i<r; i++) {
                    for(int j = 0; j<c; j++) {
                        a->grad_at(i*c + j) += self->grad_at(i);
                    }
                }
            }
        }
    };

    return out;
}
