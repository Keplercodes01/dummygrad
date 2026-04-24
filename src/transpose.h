#pragma once 
#include "engine.h" 

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

