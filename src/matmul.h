#pragma once
#include "engine.h"

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

