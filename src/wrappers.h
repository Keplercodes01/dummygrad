#pragma once 
#include "engine.h"
#include "activations.h"
#include "broadcasting.h"
#include "loss.h"
#include "scalar_ops.h"
#include "ops.h"
#include "cool_ops.h"
#include "optimizers.h"
#include "init.h"

//linear layer
class linear {
    public:    
        std::shared_ptr<Tensor> W;
        std::shared_ptr<Tensor> b;

        Linear(int fan_in, int fan_out, float bias=0.0f) {
            W = kaiming({fan_in, fan_out});
            b = std::make_shared<Tensor>(std::vector<int>{1, fan_out});
            if(bias!=0.0f) {
                for(int i=0; i<b->size(); i++) {
                    b->data_at(i) = bias;
                }
            }
        }

        std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x, int batch_size) {
            return cast_add(matmul(x, W), b);
        }

        std::vector<std::shared_ptr<Tensor>> parameters() {
            return {W, b};
        }
};

//layernorm
class layernorm {
public:
    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    float eps;

    LayerNorm(int features, float eps = 1e-5f) : eps(eps) {
        gamma = std::make_shared<Tensor>(std::vector<int>{1, features});
        beta  = std::make_shared<Tensor>(std::vector<int>{1, features});
        for(int i = 0; i < gamma->size(); i++) {
            gamma->data_at(i) = 1.0f;
            beta->data_at(i)  = 0.0f;
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto mu       = mean(x, 1);
        auto xmu      = cast_sub(x, mu);

        auto variance = var(x, 1);
        auto var_eps  = scalar_add(variance, eps);
        auto std_dev  = sqrt(var_eps);

        auto x_norm   = cast_div(xmu, std_dev);
        return scale_n_shift(x_norm, gamma, beta);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {gamma, beta};
    }
};
