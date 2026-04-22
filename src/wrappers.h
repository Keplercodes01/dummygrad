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
#include "masking.h"

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

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        return cast_n_add(matmul(x, W), b);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() {
        return {W, b};
    }
};

//layernorm
class LayerNorm {
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
        auto xmu      = cast_n_sub(x, mu);

        auto variance = var(x, 1);
        auto var_eps  = add_scalar(variance, eps);
        auto std_dev  = sqrt(var_eps);

        auto x_norm   = cast_n_div(xmu, std_dev);
        return scale_n_shift(x_norm, gamma, beta);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {gamma, beta};
    }
};

//simple self-attention block
class SelfAttention {
public:    
    Linear W_q;
    Linear W_k;
    Linear W_v;
    int d_k;
    bool casual;

    SelfAttention(int d_model, bool casual = false)
        : W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model), d_k(d_model), casual(casual) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor>& x) {
        auto Q = W_q.forward(x);
        auto K = W_k.forward(x);
        auto V = W_v.forward(x); 

        //scaled dot product
        auto scores = mul_scalar(matmul(Q, transpose(K)), 1.0f/float(sqrt(d_k)));

        //apply casual_mask
        if(casual) scores = casual_mask(scores); 

        auto weights = softmax(scores); 
        return matmul(scores, V);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = W_q.parameters();
        auto pk = W_k.parameters();
        auto pv = W_v.parameters();
        p.insert(p.end(), pk.begin(), pk.end());
        p.insert(p.end(), pv.begin(), pv.end());
        return p;
    }
};
    
//multihead attention
class MultiHeadAttention {
public:
    Linear W_q;
    Linear W_k;
    Linear W_v;
    int d_k;
    bool casual;

    MultiHeadAttention(int d_model, int h, bool casual=false) 
        : W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model), d_k(d_model/h), casual(casual) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor>& x) {
        auto Q = W_q.forward(x)  
    }
}

    




