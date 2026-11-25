#pragma once 
#include "tensor.h"
#include "functional.h"
#include "module.h"
#include "init.h"

// Linear / Fully-Connected Layer: x W + b, with W stored [in, out]. use_bias = false leaves b null.
//
// A low-rank adapter (LoRA, Hu et al. 2021) adds scale * x A B; enable_lora creates it, and add_lora
// (lora.h) does so across a model and freezes everything else.
class Linear : public Module {
public:    
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;  // null without a bias
    std::shared_ptr<Tensor> lora_A, lora_B;  // [in, rank], [rank, out]; null without an adapter
    float lora_scale = 0.0f, lora_dropout = 0.0f;

    Linear(int fan_in, int fan_out, float bias = 0.0f, bool use_bias = true) {
        W = kaiming({fan_in, fan_out});
        if (!use_bias) return;
        b = std::make_shared<Tensor>(std::vector<int64_t>{1, fan_out});
        b->fill_(bias);
        b->requires_grad = true;
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto y = linear(x, W, b);
        return lora_A ? add(y, lora_delta(x)) : y;
    }

    // Under autocast on a GPU, the output stays in the autocast dtype; otherwise the same as forward
    std::shared_ptr<Tensor> forward_autocast(const std::shared_ptr<Tensor>& x) {
        return lora_A ? forward(x) : linear(x, W, b, autocast_dtype());
    }

    // Adds an adapter of the given rank; B starts at zero, so the output does not change until it trains
    void enable_lora(int rank, float alpha, float dropout_p = 0.0f) {
        if (rank < 1) throw std::invalid_argument("Linear::enable_lora: the rank must be positive");
        lora_A = std::make_shared<Tensor>(std::vector<int64_t>{W->shape[0], rank}, Device::CPU, DType::Float32, true);
        lora_B = std::make_shared<Tensor>(std::vector<int64_t>{rank, W->shape[1]}, Device::CPU, DType::Float32, true);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float)W->shape[0]));
        for (int64_t i = 0; i < lora_A->size(); ++i) lora_A->data_ptr<float>()[i] = dist(g_gen);
        lora_B->fill_(0.0f);
        if (W->device != Device::CPU) {
            lora_A = lora_A->to(W->device);
            lora_B = lora_B->to(W->device);
            lora_A->requires_grad = lora_B->requires_grad = true;
        }
        lora_scale = alpha / (float)rank;
        lora_dropout = dropout_p;
    }

    // Adds the adapter into W and removes it
    void merge_lora() {
        if (!lora_A) return;
        NoGradGuard no_grad;
        ops::make_contiguous(*W, *add(W, mul_scalar(matmul(lora_A, lora_B), lora_scale)));
        lora_A = lora_B = nullptr;
    }

    void visit(ModuleVisitor& v) override {
        v.parameter("weight", W);
        if (b) v.parameter("bias", b);
        if (lora_A) {
            v.parameter("lora_A", lora_A);
            v.parameter("lora_B", lora_B);
        }
    }

private:
    std::shared_ptr<Tensor> lora_delta(const std::shared_ptr<Tensor>& x) {
        auto in = dropout(x, lora_dropout, training_);
        return mul_scalar(matmul(matmul(in, lora_A), lora_B), lora_scale);
    }
};
