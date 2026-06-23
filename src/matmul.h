#pragma once
#include "engine.h"

//manual matmul
inline std::shared_ptr<Tensor> manual_matmul(const std::shared_ptr<Tensor>& _a, const std::shared_ptr<Tensor>& _b) {
    auto a = make_contiguous(_a);
    auto b = make_contiguous(_b);

    int n1 = a->ndim();
    int n2 = b->ndim();

    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    //check the batch dimensions match 
    for(int i = 0; i < n1-2; i++) {
        if(a->shape[i] != b->shape[i]) {
            throw std::runtime_error("matmul: batch dimensions mismatch...");
        }
    }
    //check inner dimensions match
    if(c1 != r2) {
        throw std::runtime_error("matmul: inner dimensions mismatch...");  
    }

    //copy the batch dimensions from a 
    std::vector<int> out_shape;
    for(int i = 0; i<n1-2; i++) {
        out_shape.push_back(a->shape[i]);
    }
    out_shape.push_back(r1);
    out_shape.push_back(c2);

    auto out = std::make_shared<Tensor>(out_shape);
    int nout = out->ndim();

    int batch_size = 1;
    for(int i = 0; i<n1-2; i++) { batch_size *= a->shape[i]; }

    //forward
    for(int batch = 0; batch < batch_size; batch++) {
        //get the flat batch offset for a and b 
        std::vector<int> batch_idx = unravel(batch, std::vector<int>(a->shape.begin(), a->shape.end()-2)); 

        int batch_off_a = 0;
        int batch_off_b = 0;
        int batch_off_out = 0;

        for(int i = 0; i<n1-2; i++) {
            batch_off_a += batch_idx[i] * a->strides[i];
            batch_off_b += batch_idx[i] * b->strides[i];
            batch_off_out += batch_idx[i] * out->strides[i];
        }
        for(int i = 0; i<r1; i++) {
            for(int j = 0; j<c2; j++) {
                float sum = 0.0f;
                for(int k = 0; k<c1; k++) {
                    int flat_a = batch_off_a + a->strides[n1-2]*i + a->strides[n1-1]*k;
                    int flat_b = batch_off_b + b->strides[n2-2]*k + b->strides[n2-1]*j;
                    sum += a->data_at(flat_a) * b->data_at(flat_b);
                }
                int flat_out = batch_off_out + out->strides[nout-2]*i + out->strides[nout-1]*j;
                out->data_at(flat_out) = sum; 
            }
        }
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    //backward
    out->backward_fn = [a, b, weak_out, r1, c1, r2, c2, batch_size, n1, n2, nout]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch<batch_size; batch++) {
                //get the flat batch offset for a and b 
                std::vector<int> batch_idx = unravel(batch, std::vector<int>(a->shape.begin(), a->shape.end()-2)); 

                int batch_off_a = 0;
                int batch_off_b = 0;
                int batch_off_out = 0;

                for(int i = 0; i<n1-2; i++) {
                    batch_off_a += batch_idx[i] * a->strides[i];
                    batch_off_b += batch_idx[i] * b->strides[i];
                    batch_off_out += batch_idx[i] * out->strides[i];
                }

                //a.grad
                for(int i=0; i<r1; i++) {
                    for(int j=0; j<c1; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<c2; k++) {
                            //sum += self->grad_at(self->strides[nout-2]*i + self->strides[nout-1]*k + batch*r1*c2) 
                            //     * b->data_at(b->strides[n2-2]*j + b->strides[n2-1]*k + batch*r2*c2);
                            int flat_b = batch_off_b + b->strides[n2-2]*j + b->strides[n2-1]*k;
                            int flat_out = batch_off_out + out->strides[nout-2]*i + out->strides[nout-1]*k;

                            sum += self->grad_at(flat_out) * b->data_at(flat_b);
                        }
                        //a->grad_at(a->strides[n1-2]*i + a->strides[n1-1]*j + batch*r1*c1) += sum;
                        int flat_a = batch_off_a + a->strides[n1-2]*i + a->strides[n1-1]*j;
                        a->grad_at(flat_a) += sum;
                    }
                }
                //b.grad
                for(int i=0; i<r2; i++) {
                    for(int j=0; j<c2; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<r1; k++) {
                            //sum += a->data_at(a->strides[n1-2]*k + a->strides[n1-1]*i + batch*r1*c1) 
                            //     * self->grad_at(self->strides[nout-2]*k + self->strides[nout-1]*j + batch*r1*c2);
                            int flat_a = batch_off_a + a->strides[n1-2]*k + a->strides[n1-1]*i;
                            int flat_out = batch_off_out + out->strides[nout-2]*k + out->strides[nout-1]*j;

                            sum += a->data_at(flat_a) * self->grad_at(flat_out);
                        }
                        //b->grad_at(b->strides[n2-2]*i + b->strides[n2-1]*j + batch*r2*c2) += sum;
                        int flat_b = batch_off_b + b->strides[n2-2]*i + b->strides[n2-1]*j;
                        b->grad_at(flat_b) += sum;
                    }    
                }
            }
        }
    };

    return out;
}

