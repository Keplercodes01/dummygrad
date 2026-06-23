#pragma once 
#include "engine.h" 

//transpose
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a, int dim0, int dim1) {
    int n = a->shape.size();

    //normalize negative dim
    if(dim0<0) dim0 += n;
    if(dim1<0) dim1 += n;

    if(dim0<0 || dim0>=n || dim1<0 || dim1>=n) throw std::runtime_error("transpose: out of range");
    
    auto out = std::make_shared<Tensor>(a->shape);
    out->storage = a->storage;
    out->offset = a->offset;
    out->strides = a->strides;

    //swap the strides and dimensions 
    std::swap(out->shape[dim0], out->shape[dim1]);
    std::swap(out->strides[dim0], out->strides[dim1]);

    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, dim0, dim1]() {
        if (auto self = weak_out.lock()) {
            int total = a->size();
            int ndim  = a->shape.size();

            // iterate over every element of a using a's own shape
            std::vector<int> idx(ndim, 0);
            for (int flat = 0; flat < total; flat++) {
                // compute a's flat index from idx using a's strides
                int a_idx = a->offset;
                for (int d = 0; d < ndim; d++)
                    a_idx += idx[d] * a->strides[d];

                // swap dims to get the corresponding index in out
                std::swap(idx[dim0], idx[dim1]);
                int out_idx = self->offset;
                for (int d = 0; d < ndim; d++)
                    out_idx += idx[d] * self->strides[d];
                std::swap(idx[dim0], idx[dim1]); // swap back

                a->grad_at(a_idx) += self->grad_at(out_idx);

                // increment idx odometer
                for (int d = ndim - 1; d >= 0; d--) {
                    idx[d]++;
                    if (idx[d] < a->shape[d]) break;
                    idx[d] = 0;
                }
            }
        }
    };
    return out;
}














