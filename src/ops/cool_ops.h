#pragma once
#include "tensor.h"
#include "functional.h"

// x * gamma + beta over the last two dimensions of x [..., r, c]: gamma and beta are either one
// value per column ([c], or any shape with c elements, applied to every row) or one per element of
// the [r, c] block. Built from broadcast mul and add, so it runs on every device those do, with
// their deterministic gradients.
inline std::shared_ptr<Tensor> scale_n_shift(const std::shared_ptr<Tensor>& x_in,
                                             const std::shared_ptr<Tensor>& gamma_in,
                                             const std::shared_ptr<Tensor>& beta_in) {
    DG_PROFILE_OP("scale_n_shift");
    if (!x_in || !gamma_in || !beta_in) throw std::invalid_argument("scale_n_shift: null tensor");
    int ndim = x_in->ndim();
    if (ndim < 2) {
        throw std::invalid_argument("scale_n_shift requires tensor with ndim >= 2");
    }
    int64_t r = x_in->shape[ndim - 2];
    int64_t c = x_in->shape[ndim - 1];
    // One value per column broadcasts as [c]; one per element as [r, c]
    auto as_affine = [&](const std::shared_ptr<Tensor>& t, const char* name) {
        if (t->size() == c) return reshape(t, {c});
        if (t->size() == r * c) return reshape(t, {r, c});
        throw std::invalid_argument(std::string("scale_n_shift: ") + name + " must have " + std::to_string(c) + " or " +
                                    std::to_string(r * c) + " elements");
    };
    return add(mul(x_in, as_affine(gamma_in, "gamma")), as_affine(beta_in, "beta"));
}
