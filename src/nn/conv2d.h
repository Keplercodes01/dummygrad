#pragma once
#include "tensor.h"
#include "functional.h"

inline std::shared_ptr<Tensor> conv2d(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& weight, 
                                      int stride_h, int stride_w, int pad_h, int pad_w) {
    DG_PROFILE_OP("conv2d");
    if (!x || !weight) throw std::runtime_error("conv2d: input or weight is null");
    if (x->ndim() != 4) throw std::runtime_error("conv2d: x must be a 4D tensor (B, C_in, H, W)");
    if (weight->ndim() != 4) throw std::runtime_error("conv2d: weight must be a 4D tensor (C_out, C_in, kH, kW)");
    if (x->shape[1] != weight->shape[1]) throw std::runtime_error("conv2d: input channels mismatch between x and weight");
    if (stride_h <= 0 || stride_w <= 0) throw std::runtime_error("conv2d: stride must be > 0");
    if (pad_h < 0 || pad_w < 0) throw std::runtime_error("conv2d: padding must be >= 0");
    if (x->device != weight->device) throw std::runtime_error("conv2d: devices mismatch between x and weight");

    return conv_nd(x, weight, nullptr, {stride_h, stride_w}, {pad_h, pad_w}, {1}, 1);
}

inline std::shared_ptr<Tensor> conv2d(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& weight, 
                                      int stride = 1, int padding = 0) {
    return conv2d(x, weight, stride, stride, padding, padding);
}
