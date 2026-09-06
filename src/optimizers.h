#pragma once 
#include "tensor.h"
#include "functional.h"

// SGD
inline void SGD(const std::shared_ptr<Tensor>& param, const float& lr) {
    if (!param || !param->grad) return;
    float* data = param->data_ptr<float>();
    const float* grad = param->grad->data_ptr<float>();
    int size = param->size();

    for (int i = 0; i < size; i++) {
        data[i] -= lr * grad[i];
    }
}

// Per-parameter state for Adam
struct ParamState {
    std::vector<float> m;
    std::vector<float> v;
    float* gpu_m = nullptr;
    float* gpu_v = nullptr;
    int t = 0;
};

// Adam optimizer (supports arbitrary number of parameters with distinct shapes)
class Adam {
public:
    std::unordered_map<Tensor*, ParamState> state;
    float lr, b1, b2, E, weight_decay;

    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float E = 1e-8f, float weight_decay = 0.01f)
        : lr(lr), b1(b1), b2(b2), E(E), weight_decay(weight_decay) {}

    void step(const std::shared_ptr<Tensor>& param) {
        if (!param || !param->grad) return;
        int size = param->size();

#ifdef USE_CUDA
        if (param->device == Device::CUDA) {
            auto& pstate = state[param.get()];
            if (!pstate.gpu_m) {
                pstate.gpu_m = static_cast<float*>(get_memory(Device::CUDA, size * sizeof(float)));
                pstate.gpu_v = static_cast<float*>(get_memory(Device::CUDA, size * sizeof(float)));
                cudaMemset(pstate.gpu_m, 0, size * sizeof(float));
                cudaMemset(pstate.gpu_v, 0, size * sizeof(float));
            }
            pstate.t++;
            cuda::adamw_step(param->data_ptr<float>(), param->grad->data_ptr<float>(),
                             pstate.gpu_m, pstate.gpu_v,
                             lr, b1, b2, E, weight_decay, pstate.t, size);
            return;
        }
#endif

        auto& pstate = state[param.get()];
        if (pstate.m.empty()) {
            pstate.m.resize(size, 0.0f);
            pstate.v.resize(size, 0.0f);
        }
        pstate.t++;

        float* data = param->data_ptr<float>();
        const float* grad = param->grad->data_ptr<float>();

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
