#pragma once
#include "engine.h"
#include "linear.h"
#include "matmul.h"
#include "transpose.h"
#include "view.h"
#include "scalar_ops.h"
#include "softmax.h"
#include "masking.h"

// Single-Head Self-Attention
class SelfAttention {
public:    
    Linear W_q;
    Linear W_k;
    Linear W_v;
    int d_k;
    bool causal;

    SelfAttention(int d_model, bool causal = false)
        : W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model), d_k(d_model), causal(causal) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto Q = W_q.forward(x);
        auto K = W_k.forward(x);
        auto V = W_v.forward(x); 

        // Scaled dot-product attention
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        auto scores = mul_scalar(matmul(Q, transpose(K, -2, -1)), scale);

        if (causal) scores = causal_mask(scores); 

        auto weights = softmax(scores); 
        return matmul(weights, V);
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

// Multi-Head Self-Attention
class MultiHeadAttention {
public:
    Linear W_q;
    Linear W_k;
    Linear W_v;
    Linear W_o;
    int d_model;
    int n_heads;
    int d_k;
    bool causal;

    MultiHeadAttention(int d_model, int n_heads, bool causal = false)
        : W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model), W_o(d_model, d_model),
          d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads), causal(causal) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in) {
        auto x = x_in;
        bool is_2d = (x->ndim() == 2);
        if (is_2d) {
            x = reshape(x, {1, x->shape[0], x->shape[1]});
        }

        int B = x->shape[0];
        int S = x->shape[1];

        auto Q = W_q.forward(x); // [B, S, d_model]
        auto K = W_k.forward(x);
        auto V = W_v.forward(x);

        // Reshape Q, K, V to [B, S, n_heads, d_k] -> Transpose to [B, n_heads, S, d_k]
        auto Q_split = transpose(reshape(Q, {B, S, n_heads, d_k}), 1, 2);
        auto K_split = transpose(reshape(K, {B, S, n_heads, d_k}), 1, 2);
        auto V_split = transpose(reshape(V, {B, S, n_heads, d_k}), 1, 2);

        // Scaled dot product
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        auto scores = mul_scalar(matmul(Q_split, transpose(K_split, -2, -1)), scale);

        if (causal) scores = causal_mask(scores);

        auto weights = softmax(scores);
        auto attn_out = matmul(weights, V_split); // [B, n_heads, S, d_k]

        // Transpose back to [B, S, n_heads, d_k] -> Reshape to [B, S, d_model]
        auto merged = reshape(transpose(attn_out, 1, 2), {B, S, d_model});

        // Final output projection
        auto out = W_o.forward(merged);

        if (is_2d) {
            out = reshape(out, {S, d_model});
        }
        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = W_q.parameters();
        auto pk = W_k.parameters();
        auto pv = W_v.parameters();
        auto po = W_o.parameters();
        p.insert(p.end(), pk.begin(), pk.end());
        p.insert(p.end(), pv.begin(), pv.end());
        p.insert(p.end(), po.begin(), po.end());
        return p;
    }
};
