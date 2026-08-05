#pragma once 
#include "engine.h"
#include "matmul.h"
#include "broadcasting.h"
#include "init.h"

// Linear / Fully-Connected Layer
class Linear {
public:    
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;

    Linear(int fan_in, int fan_out, float bias = 0.0f) {
        W = kaiming({fan_in, fan_out});
        b = std::make_shared<Tensor>(std::vector<int>{1, fan_out});
        if (bias != 0.0f) {
            float* b_ptr = b->data_ptr();
            int size = b->size();
            for (int i = 0; i < size; i++) {
                b_ptr[i] = bias;
            }
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        return cast_n_add(matmul(x, W), b);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {W, b};
    }
};
