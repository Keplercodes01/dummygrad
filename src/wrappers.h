#pragma once 
#include "engine.h"
#include "activations.h"
#include "broadcasting.h"
#include "norm.h"
#include "loss.h"
#include "scalar_ops.h"
#include "ops.h"
#include "optimizers.h"
#include "init.h"

//linear layer
class Linear {
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
            return add(matmul(x, W), broadcast(b, 0, batch_size));
        }

        std::vector<std::shared_ptr<Tensor>> parameters() {
            return {W, b};
        }
};

//layernorm
class Layer_norm {
    public:
        float epsilon = 1e-8f;
        std::shared_ptr<Tensor> gamma;
        std::shared_ptr<Tensor> beta;

        Layer_norm(int features, float epsilon = 1e-5f) : epsilon(epsilon) {
            gamma = std::make_shared<Tensor>(
        }
}
