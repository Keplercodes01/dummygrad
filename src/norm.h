#pragma once
#include "engine.h"
#include "ops.h"
#include "broadcasting.h"
#include "scalar_ops.h"

inline std::shared_ptr<Tensor> batch_norm(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& gamma,
    const std::shared_ptr<Tensor>& beta,
    float eps = 1e-5f) {

    int batch = x->shape[0];
    int features = x->shape[1];

    // mean
    auto sum_x = collapse(x, 0);
    auto mean_x = div_scalar(sum_x, (float)batch);
    auto mean_x_b = broadcast(mean_x, 0, batch);

    // variance
    auto diff = sub(x, mean_x_b);
    auto var = div_scalar(collapse(pow(diff, 2), 0), (float)batch);
    auto var_b = broadcast(var, 0, batch);

    // normalize
    auto std_x = sqrt(add_scalar(var_b, eps));
    auto norm = div(diff, std_x);

    // scale and shift
    auto gamma_b = broadcast(gamma, 0, batch);
    auto beta_b = broadcast(beta, 0, batch);

    return add(mul(gamma_b, norm), beta_b);
}















