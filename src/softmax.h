#pragma once
#include "engine.h"

// --- SOFTMAX BACKWARD NODE ---
struct SoftmaxBackward : public Node {
    std::shared_ptr<Tensor> out_val;
    int outer;
    int last;

    SoftmaxBackward(std::shared_ptr<Tensor> out_val, int outer, int last)
        : out_val(out_val), outer(outer), last(last) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(out_val->shape, false);
        const float* sg_ptr = self_grad->data_ptr();
        const float* s_ptr = out_val->data_ptr();
        float* ga_ptr = grad_a->data_ptr();

        for (int i = 0; i < outer; i++) {
            int offset = i * last;

            // sum_dot = sum_j (grad_j * s_j)
            float sum_dot = 0.0f;
            for (int j = 0; j < last; j++) {
                sum_dot += sg_ptr[offset + j] * s_ptr[offset + j];
            }

            // da_k = s_k * (grad_k - sum_dot)
            for (int k = 0; k < last; k++) {
                float s_k = s_ptr[offset + k];
                ga_ptr[offset + k] = s_k * (sg_ptr[offset + k] - sum_dot);
            }
        }
        return {grad_a};
    }
};

// softmax
inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a) {
    int n = a->shape.size();
    int last = a->shape[n - 1];      // size of last dimension
    int outer = a->size() / last;    // product of all other dimensions

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(a->shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int i = 0; i < outer; i++) {
        int offset = i * last;

        // find max for numerical stability
        float max_val = a_ptr[offset];
        for (int j = 1; j < last; j++) {
            max_val = std::max(max_val, a_ptr[offset + j]);
        }

        // compute softmax
        float sum = 0.0f;
        for (int j = 0; j < last; j++) {
            float exp_val = std::exp(a_ptr[offset + j] - max_val);
            out_ptr[offset + j] = exp_val;
            sum += exp_val;
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < last; j++) {
            out_ptr[offset + j] *= inv_sum;
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SoftmaxBackward>(out, outer, last);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}
