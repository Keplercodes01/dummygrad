#pragma once 
#include "engine.h"

// SGD
inline void SGD(const std::shared_ptr<Tensor>& param, const float& lr) {
    if (!param || !param->grad) return;
    float* data = param->data_ptr();
    const float* grad = param->grad_ptr();
    int size = param->size();

    for (int i = 0; i < size; i++) {
        data[i] -= lr * grad[i];
    }
}

// Per-parameter state for Adam
struct ParamState {
    std::vector<float> m;
    std::vector<float> v;
    int t = 0;
};

// Adam optimizer (supports arbitrary number of parameters with distinct shapes)
class Adam {
public:
    std::unordered_map<Tensor*, ParamState> state;
    float lr, b1, b2, E;

    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float E = 1e-8f)
        : lr(lr), b1(b1), b2(b2), E(E) {}

    void step(const std::shared_ptr<Tensor>& param) {
        if (!param || !param->grad) return;
        int size = param->size();

        auto& pstate = state[param.get()];
        if (pstate.m.empty()) {
            pstate.m.resize(size, 0.0f);
            pstate.v.resize(size, 0.0f);
        }
        pstate.t++;

        float* data = param->data_ptr();
        const float* grad = param->grad_ptr();

        float b1_corr = 1.0f - std::pow(b1, pstate.t);
        float b2_corr = 1.0f - std::pow(b2, pstate.t);

        for (int i = 0; i < size; i++) {
            float g = grad[i];
            pstate.m[i] = b1 * pstate.m[i] + (1.0f - b1) * g;
            pstate.v[i] = b2 * pstate.v[i] + (1.0f - b2) * g * g;

            float m_hat = pstate.m[i] / b1_corr; 
            float v_hat = pstate.v[i] / b2_corr; 

            data[i] -= lr * m_hat / (std::sqrt(v_hat) + E);
        }
    }
};
