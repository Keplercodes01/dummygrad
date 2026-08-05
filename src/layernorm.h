#pragma once 
#include "engine.h"
#include "mean.h"
#include "var.h"
#include "scalar_ops.h"
#include "broadcasting.h"
#include "cool_ops.h"
#include "ops.h"

// Layer Normalization
class LayerNorm {
public:
    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    float eps;

    LayerNorm(int features, float eps = 1e-5f) : eps(eps) {
        gamma = std::make_shared<Tensor>(std::vector<int>{1, features});
        beta  = std::make_shared<Tensor>(std::vector<int>{1, features});
        float* g_ptr = gamma->data_ptr();
        float* b_ptr = beta->data_ptr();
        int size = gamma->size();
        for (int i = 0; i < size; i++) {
            g_ptr[i] = 1.0f;
            b_ptr[i] = 0.0f;
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto mu       = mean(x, 1);
        auto xmu      = cast_n_sub(x, mu);

        auto variance = var(x, 1);
        auto var_eps  = add_scalar(variance, eps);
        auto std_dev  = sqrt(var_eps);

        auto x_norm   = cast_n_div(xmu, std_dev);
        return scale_n_shift(x_norm, gamma, beta);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {gamma, beta};
    }
};
