#pragma once 
#include "tensor.h"
#include "functional.h"
#include "module.h"
#include "init.h"

// Linear / Fully-Connected Layer
class Linear : public Module {
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
        return linear(x, W, b);
    }

    void visit(ModuleVisitor& v) override {
        v.parameter("weight", W);
        v.parameter("bias", b);
    }
};
