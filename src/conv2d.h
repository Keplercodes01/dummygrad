#pragma once
#include "tensor.h"
#include "functional.h"

inline std::shared_ptr<Tensor> conv2d(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& weight, 
                                      int stride = 1, int padding = 0) {
    // x: [B, C_in, H, W]
    // weight: [C_out, C_in, kH, kW]
    
    int B = x->shape[0];
    int C_in = x->shape[1];
    int H = x->shape[2];
    int W = x->shape[3];

    int C_out = weight->shape[0];
    int kH = weight->shape[2];
    int kW = weight->shape[3];

    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    // 1. Unfold image patches into columns
    auto x_col = im2col(x, kH, kW, padding, padding, stride, stride); // [B, C_in * kH * kW, H_out, W_out]
    
    // 2. Reshape x_col to [B, C_in*kH*kW, H_out*W_out]
    x_col = x_col->reshape({B, C_in * kH * kW, H_out * W_out});

    // 3. Reshape weight to [C_out, C_in*kH*kW]
    auto w_flat = weight->reshape({C_out, C_in * kH * kW});

    // 4. Batched MatMul: w_flat @ x_col -> [B, C_out, H_out*W_out]
    // Wait, matmul broadcasts. w_flat is [C_out, K], x_col is [B, K, L]
    auto out = matmul(w_flat, x_col);

    // 5. Reshape to final output
    return out->reshape({B, C_out, H_out, W_out});
}
