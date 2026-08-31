#pragma once
#include "conv2d.h"

inline std::shared_ptr<Tensor> conv1d(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& weight, int stride = 1, int padding = 0) {
    // x: [B, C_in, L] -> [B, C_in, 1, L]
    auto x_2d = x->reshape({x->shape[0], x->shape[1], 1, x->shape[2]});
    
    // weight: [C_out, C_in, k] -> [C_out, C_in, 1, k]
    auto w_2d = weight->reshape({weight->shape[0], weight->shape[1], 1, weight->shape[2]});
    
    // out_2d: [B, C_out, 1, L_out]
    auto out_2d = conv2d(x_2d, w_2d, stride, padding); 
    
    return out_2d->reshape({out_2d->shape[0], out_2d->shape[1], out_2d->shape[3]});
}
