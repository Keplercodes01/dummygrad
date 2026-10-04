#pragma once
#include "conv2d.h"

inline std::shared_ptr<Tensor> conv1d(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& weight, int stride = 1, int padding = 0) {
    if (!x || !weight) throw std::runtime_error("conv1d: input or weight is null");
    if (x->ndim() != 3) throw std::runtime_error("conv1d: x must be a 3D tensor (B, C_in, L)");
    if (weight->ndim() != 3) throw std::runtime_error("conv1d: weight must be a 3D tensor (C_out, C_in, k)");
    if (x->shape[1] != weight->shape[1]) throw std::runtime_error("conv1d: channel mismatch between x and weight");
    if (stride <= 0) throw std::runtime_error("conv1d: stride must be > 0");
    if (padding < 0) throw std::runtime_error("conv1d: padding must be >= 0");
    if (x->device != weight->device) throw std::runtime_error("conv1d: devices mismatch between x and weight");

    // x: [B, C_in, L] -> [B, C_in, 1, L]
    auto x_2d = x->reshape({x->shape[0], x->shape[1], 1, x->shape[2]});
    
    // weight: [C_out, C_in, k] -> [C_out, C_in, 1, k]
    auto w_2d = weight->reshape({weight->shape[0], weight->shape[1], 1, weight->shape[2]});
    
    // out_2d: [B, C_out, 1, L_out]
    auto out_2d = conv2d(x_2d, w_2d, stride, padding); 
    
    return out_2d->reshape({out_2d->shape[0], out_2d->shape[1], out_2d->shape[3]});
}
