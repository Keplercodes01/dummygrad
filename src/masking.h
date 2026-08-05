#pragma once
#include "engine.h"

// --- BOOL MASK BACKWARD NODE ---
struct BoolMaskBackward : public Node {
    std::shared_ptr<Tensor> m;
    int r, c, batch_size;
    BoolMaskBackward(std::shared_ptr<Tensor> m, int r, int c, int batch_size)
        : m(m), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(self_grad->shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        const float* m_ptr = m->data_ptr();
        float* ga_ptr = ga->data_ptr();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    ga_ptr[idx] = (m_ptr[idx] != 0.0f) ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {ga};
    }
};

// boolean mask
inline std::shared_ptr<Tensor> bool_mask(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& m) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    const float* m_ptr = m->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int idx = batch * r * c + i * c + j;
                out_ptr[idx] = m_ptr[idx] == 0.0f
                              ? -1e9f
                              : a_ptr[idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<BoolMaskBackward>(m, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAUSAL MASK BACKWARD NODE ---
struct CausalMaskBackward : public Node {
    int r, c, batch_size;
    CausalMaskBackward(int r, int c, int batch_size) : r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(self_grad->shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        float* ga_ptr = ga->data_ptr();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    ga_ptr[idx] = (j <= i) ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {ga};
    }
};

// causal mask
inline std::shared_ptr<Tensor> causal_mask(const std::shared_ptr<Tensor>& a) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; }

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int idx = batch * r * c + i * c + j;
                out_ptr[idx] = j > i
                              ? -1e9f
                              : a_ptr[idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CausalMaskBackward>(r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
