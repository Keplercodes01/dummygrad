#pragma once
#include "tensor.h"
#include "functional.h"
#include "linear.h"
#include "modern_layers.h"

#ifdef USE_CUDA
struct FlashAttentionBackward : public Node {
    std::shared_ptr<Tensor> q, k, v, out;
    int batch_size, num_heads, seq_len, d_k;
    bool causal;
    float scale;

    FlashAttentionBackward(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k,
                           std::shared_ptr<Tensor> v, std::shared_ptr<Tensor> out,
                           int batch_size, int num_heads, int seq_len, int d_k,
                           bool causal, float scale)
        : q(q), k(k), v(v), out(out),
          batch_size(batch_size), num_heads(num_heads), seq_len(seq_len), d_k(d_k),
          causal(causal), scale(scale) {}

    void release_variables() override {
        q = nullptr; k = nullptr; v = nullptr; out = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto grad_out = grads[0];
        auto gq = std::make_shared<Tensor>(q->shape, q->device, q->dtype, false);
        auto gk = std::make_shared<Tensor>(k->shape, k->device, k->dtype, false);
        auto gv = std::make_shared<Tensor>(v->shape, v->device, v->dtype, false);

        cuda::flash_attention_backward(q->data_ptr<float>(), k->data_ptr<float>(), v->data_ptr<float>(),
                                       out->data_ptr<float>(), grad_out->data_ptr<float>(),
                                       gq->data_ptr<float>(), gk->data_ptr<float>(), gv->data_ptr<float>(),
                                       batch_size, num_heads, seq_len, d_k, causal, scale);

        return {gq, gk, gv};
    }
};
#endif

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

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in,
                                    const std::shared_ptr<Tensor>& mask = nullptr) {
        auto x = x_in;
        bool is_2d = (x->ndim() == 2);
        if (is_2d) {
            x = reshape(x, {1, x->shape[0], x->shape[1]});
        }
        int B = x->shape[0];
        int S = x->shape[1];

        auto Q = W_q.forward(x);
        auto K = W_k.forward(x);
        auto V = W_v.forward(x); 

        // Scaled dot-product attention
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

#ifdef USE_CUDA
        std::shared_ptr<Tensor> out;
        if (x->device == Device::CUDA && !mask) {
            auto Q_4d = make_contiguous(reshape(Q, {B, 1, S, d_k}));
            auto K_4d = make_contiguous(reshape(K, {B, 1, S, d_k}));
            auto V_4d = make_contiguous(reshape(V, {B, 1, S, d_k}));
            auto out_4d = std::make_shared<Tensor>(Q_4d->shape, Device::CUDA, x->dtype, x->requires_grad);
            cuda::flash_attention_forward(Q_4d->data_ptr<float>(), K_4d->data_ptr<float>(), V_4d->data_ptr<float>(),
                                          out_4d->data_ptr<float>(),
                                          B, 1, S, d_k, causal, scale);
            if (x->requires_grad) {
                auto grad_fn = std::make_shared<FlashAttentionBackward>(Q_4d, K_4d, V_4d, out_4d,
                                                                        B, 1, S, d_k, causal, scale);
                grad_fn->add_next_edge(get_grad_edge(Q).function, 0);
                grad_fn->add_next_edge(get_grad_edge(K).function, 1);
                grad_fn->add_next_edge(get_grad_edge(V).function, 2);
                out_4d->grad_fn = grad_fn;
            }
            out = reshape(out_4d, {B, S, d_k});
        } else {
            auto scores = mul_scalar(matmul(Q, transpose(K, -2, -1)), scale);
            if (mask) scores = add(scores, mask);
            else if (causal) scores = causal_mask(scores); 
            auto weights = softmax(scores); 
            out = matmul(weights, V);
        }
#else
        auto scores = mul_scalar(matmul(Q, transpose(K, -2, -1)), scale);
        if (mask) scores = add(scores, mask);
        else if (causal) scores = causal_mask(scores); 
        auto weights = softmax(scores); 
        auto out = matmul(weights, V);
#endif

        if (is_2d) {
            out = reshape(out, {S, d_k});
        }
        return out;
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

// Backward Node for GQA head repetition (aggregates gradients from repeated heads to original KV heads)
struct GQARepeatBackward : public Node {
    std::shared_ptr<Tensor> orig;
    int B, n_heads, n_kv_heads, S, d_k;

    GQARepeatBackward(std::shared_ptr<Tensor> orig, int B, int n_heads, int n_kv_heads, int S, int d_k)
        : orig(orig), B(B), n_heads(n_heads), n_kv_heads(n_kv_heads), S(S), d_k(d_k) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto dout = grads[0];
        auto din = std::make_shared<Tensor>(orig->shape, orig->device, orig->dtype, false);
        din->fill_(0.0f);
        int group_size = n_heads / n_kv_heads;
        const float* dout_ptr = dout->data_ptr<float>();
        float* din_ptr = din->data_ptr<float>();

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_heads; ++h) {
                int kv_h = h / group_size;
                int src_offset = (b * n_heads + h) * S * d_k;
                int dst_offset = (b * n_kv_heads + kv_h) * S * d_k;
                for (int i = 0; i < S * d_k; ++i) {
                    din_ptr[dst_offset + i] += dout_ptr[src_offset + i];
                }
            }
        }
        return {din};
    }
};

// Multi-Head Self-Attention (Supports MHA, GQA, and MQA)
class MultiHeadAttention {
public:
    Linear W_q;
    Linear W_k;
    Linear W_v;
    Linear W_o;
    int d_model;
    int n_heads;
    int n_kv_heads;
    int d_k;
    bool causal;

    MultiHeadAttention(int d_model, int n_heads, int n_kv_heads = 0, bool causal = false)
        : W_q(d_model, d_model),
          W_k(d_model, (n_kv_heads > 0 ? n_kv_heads : n_heads) * (d_model / n_heads)),
          W_v(d_model, (n_kv_heads > 0 ? n_kv_heads : n_heads) * (d_model / n_heads)),
          W_o(d_model, d_model),
          d_model(d_model),
          n_heads(n_heads),
          n_kv_heads(n_kv_heads > 0 ? n_kv_heads : n_heads),
          d_k(d_model / n_heads),
          causal(causal) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in,
                                    const std::shared_ptr<Tensor>& mask = nullptr,
                                    const std::shared_ptr<Tensor>& cos_freqs = nullptr,
                                    const std::shared_ptr<Tensor>& sin_freqs = nullptr,
                                    int start_pos = 0) {
        auto x = x_in;
        bool is_2d = (x->ndim() == 2);
        if (is_2d) {
            x = reshape(x, {1, x->shape[0], x->shape[1]});
        }

        int B = x->shape[0];
        int S = x->shape[1];

        auto Q = W_q.forward(x); // [B, S, n_heads * d_k]
        auto K = W_k.forward(x); // [B, S, n_kv_heads * d_k]
        auto V = W_v.forward(x); // [B, S, n_kv_heads * d_k]

        // Reshape Q, K, V to 4D tensors
        auto Q_4d = reshape(Q, {B, S, n_heads, d_k});
        auto K_4d = reshape(K, {B, S, n_kv_heads, d_k});

        // Apply Rotary Position Embeddings (RoPE) if frequency tables provided
        if (cos_freqs && sin_freqs) {
            Q_4d = apply_rotary_emb(Q_4d, cos_freqs, sin_freqs, start_pos);
            K_4d = apply_rotary_emb(K_4d, cos_freqs, sin_freqs, start_pos);
        }

        // Transpose to [B, heads, S, d_k]
        auto Q_split = transpose(Q_4d, 1, 2);
        auto K_split = transpose(K_4d, 1, 2);
        auto V_split = transpose(reshape(V, {B, S, n_kv_heads, d_k}), 1, 2);

        // Grouped Query Attention (GQA): Repeat KV heads across query head groups
        if (n_kv_heads < n_heads) {
            int group_size = n_heads / n_kv_heads;
            auto K_rep = std::make_shared<Tensor>(
                std::vector<int64_t>{B, n_heads, S, d_k}, K_split->device, K_split->dtype, K_split->requires_grad
            );
            auto V_rep = std::make_shared<Tensor>(
                std::vector<int64_t>{B, n_heads, S, d_k}, V_split->device, V_split->dtype, V_split->requires_grad
            );

            const float* k_src = K_split->data_ptr<float>();
            const float* v_src = V_split->data_ptr<float>();
            float* k_dst = K_rep->data_ptr<float>();
            float* v_dst = V_rep->data_ptr<float>();

            int head_elements = S * d_k;
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < n_heads; ++h) {
                    int kv_h = h / group_size;
                    int src_offset = (b * n_kv_heads + kv_h) * head_elements;
                    int dst_offset = (b * n_heads + h) * head_elements;
                    std::memcpy(k_dst + dst_offset, k_src + src_offset, head_elements * sizeof(float));
                    std::memcpy(v_dst + dst_offset, v_src + src_offset, head_elements * sizeof(float));
                }
            }

            if (K_split->requires_grad) {
                auto grad_fn = std::make_shared<GQARepeatBackward>(K_split, B, n_heads, n_kv_heads, S, d_k);
                grad_fn->add_next_edge(get_grad_edge(K_split).function, 0);
                K_rep->grad_fn = grad_fn;
            }
            if (V_split->requires_grad) {
                auto grad_fn = std::make_shared<GQARepeatBackward>(V_split, B, n_heads, n_kv_heads, S, d_k);
                grad_fn->add_next_edge(get_grad_edge(V_split).function, 0);
                V_rep->grad_fn = grad_fn;
            }

            K_split = K_rep;
            V_split = V_rep;
        }

        // Scaled dot product
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

#ifdef USE_CUDA
        std::shared_ptr<Tensor> attn_out;
        if (x->device == Device::CUDA && !mask) {
            auto Q_c = make_contiguous(Q_split);
            auto K_c = make_contiguous(K_split);
            auto V_c = make_contiguous(V_split);
            attn_out = std::make_shared<Tensor>(Q_split->shape, Device::CUDA, x->dtype, x->requires_grad);
            cuda::flash_attention_forward(Q_c->data_ptr<float>(), K_c->data_ptr<float>(), V_c->data_ptr<float>(),
                                          attn_out->data_ptr<float>(),
                                          B, n_heads, S, d_k, causal, scale);
            if (x->requires_grad) {
                auto grad_fn = std::make_shared<FlashAttentionBackward>(Q_c, K_c, V_c, attn_out,
                                                                        B, n_heads, S, d_k, causal, scale);
                grad_fn->add_next_edge(get_grad_edge(Q_split).function, 0);
                grad_fn->add_next_edge(get_grad_edge(K_split).function, 1);
                grad_fn->add_next_edge(get_grad_edge(V_split).function, 2);
                attn_out->grad_fn = grad_fn;
            }
        } else {
            auto scores = mul_scalar(matmul(Q_split, transpose(K_split, -2, -1)), scale);
            if (mask) scores = add(scores, mask);
            else if (causal) scores = causal_mask(scores);
            auto weights = softmax(scores);
            attn_out = matmul(weights, V_split); // [B, n_heads, S, d_k]
        }
#else
        auto scores = mul_scalar(matmul(Q_split, transpose(K_split, -2, -1)), scale);
        if (mask) scores = add(scores, mask);
        else if (causal) scores = causal_mask(scores);
        auto weights = softmax(scores);
        attn_out = matmul(weights, V_split); // [B, n_heads, S, d_k]
#endif

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
