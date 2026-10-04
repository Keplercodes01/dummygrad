#pragma once 
#include "tensor.h"
#include "functional.h"
#include "init.h"

// Linear / Fully-Connected Layer
class Linear {
public:    
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;

    Linear(int fan_in, int fan_out, float bias = 0.0f) {
        W = kaiming({fan_in, fan_out});
        b = std::make_shared<Tensor>(std::vector<int64_t>{1, fan_out});
        b->fill_(bias);
        b->requires_grad = true;
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        return cast_n_add(matmul(x, W), b);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {W, b};
    }

    void to(Device device, DType dtype = DType::Float32) {
        W = W->to(device, dtype);
        b = b->to(device, dtype);
        W->requires_grad = true;
        b->requires_grad = true;
    }
};
