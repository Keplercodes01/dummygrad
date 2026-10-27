#pragma once
#include "tensor.h"
#include "functional.h"
#include "linear.h"
#include "modern_layers.h"

#ifdef USE_CUDA
// Backward of the CUDA attention kernel; q, k, v, out are contiguous [batch, heads, seq, head_dim]
struct FlashAttentionBackward : public Node {
    std::shared_ptr<Tensor> q, k, v, out, lse;
    int batch_size, num_heads, seq_len, d_k;
    bool causal;
    float scale;
    cuda::AttentionImpl impl;  // what the forward ran, so backward reads matching statistics
    DType compute;             // GEMM precision the forward ran with (autocast)

    FlashAttentionBackward(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k, std::shared_ptr<Tensor> v,
                           std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> lse,
                           int batch_size, int num_heads, int seq_len, int d_k, bool causal, float scale,
                           cuda::AttentionImpl impl, DType compute)
        : q(std::move(q)), k(std::move(k)), v(std::move(v)), out(std::move(out)), lse(std::move(lse)),
          batch_size(batch_size), num_heads(num_heads), seq_len(seq_len), d_k(d_k), causal(causal), scale(scale),
          impl(impl), compute(compute) {}

    void release_variables() override {
        q = nullptr; k = nullptr; v = nullptr; out = nullptr; lse = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !q || !k || !v || !out) return {nullptr, nullptr, nullptr};
        auto grad_out = grads[0];
        auto gq = std::make_shared<Tensor>(q->shape, q->device, q->dtype, false);
        auto gk = std::make_shared<Tensor>(k->shape, k->device, k->dtype, false);
        auto gv = std::make_shared<Tensor>(v->shape, v->device, v->dtype, false);
        cuda::attention_backward(q->data_ptr<float>(), k->data_ptr<float>(), v->data_ptr<float>(),
                                 out->data_ptr<float>(), grad_out->data_ptr<float>(), lse->data_ptr<float>(),
                                 gq->data_ptr<float>(), gk->data_ptr<float>(), gv->data_ptr<float>(),
                                 batch_size, num_heads, seq_len, d_k, causal, scale, impl, compute);
        return {gq, gk, gv};
    }
};
#endif

// Single-Head Self-Attention
class SelfAttention : public Module {
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
        if (!x_in) throw std::runtime_error("SelfAttention: input tensor is null");
        if (x_in->ndim() < 2) throw std::runtime_error("SelfAttention: input must be at least 2D (S, D) or (B, S, D)");
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

        std::shared_ptr<Tensor> out;
#ifdef USE_CUDA
        if (x->device == Device::CUDA && !mask) {
            auto Q_4d = make_contiguous(reshape(Q, {B, 1, S, d_k}));
            auto K_4d = make_contiguous(reshape(K, {B, 1, S, d_k}));
            auto V_4d = make_contiguous(reshape(V, {B, 1, S, d_k}));
            // Q/K/V carry the weights' requires_grad even when the input x does not
            bool qkv_grad = grad_enabled() && (Q->requires_grad || K->requires_grad || V->requires_grad);
            auto out_4d = std::make_shared<Tensor>(Q_4d->shape, Device::CUDA, x->dtype, qkv_grad);
            auto lse = std::make_shared<Tensor>(std::vector<int64_t>{B, 1, S}, Device::CUDA, x->dtype, false);
            DType compute = autocast_dtype();
            auto impl = cuda::attention_forward(Q_4d->data_ptr<float>(), K_4d->data_ptr<float>(), V_4d->data_ptr<float>(),
                                                out_4d->data_ptr<float>(), lse->data_ptr<float>(), B, 1, S, d_k, causal, scale,
                                                compute);
            if (qkv_grad) {
                auto grad_fn = std::make_shared<FlashAttentionBackward>(Q_4d, K_4d, V_4d, out_4d, lse,
                                                                        B, 1, S, d_k, causal, scale, impl, compute);
                grad_fn->add_next_edge(get_grad_edge(Q).function);
                grad_fn->add_next_edge(get_grad_edge(K).function);
                grad_fn->add_next_edge(get_grad_edge(V).function);
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
        out = matmul(weights, V);
#endif

        if (is_2d) {
            out = reshape(out, {S, d_k});
        }
        return out;
    }

    void visit(ModuleVisitor& v) override {
        v.module("W_q", W_q);
        v.module("W_k", W_k);
        v.module("W_v", W_v);
    }
};

// GQA: each of the kv heads serves `group` consecutive query heads (query head h reads kv head
// h / group). [B, kv, S, D] -> [B, kv * group, S, D]; the gradient sums each group back.
struct RepeatKVBackward : public Node {
    int64_t batch, kv_heads, group, head_elems;
    std::vector<int64_t> in_shape;

    RepeatKVBackward(int64_t batch, int64_t kv_heads, int64_t group, int64_t head_elems, std::vector<int64_t> in_shape)
        : batch(batch), kv_heads(kv_heads), group(group), head_elems(head_elems), in_shape(std::move(in_shape)) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr};
        auto g = grads[0];
        Device dev = g->device;
        auto gin = std::make_shared<Tensor>(in_shape, dev, g->dtype, false);
#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            cuda::repeat_kv_backward(g->data_ptr<float>(), gin->data_ptr<float>(), batch, kv_heads, group, head_elems);
            return {gin};
        }
#endif
        auto gh = (dev == Device::CPU) ? g : g->cpu();
        auto oh = (dev == Device::CPU) ? gin : std::make_shared<Tensor>(in_shape, Device::CPU, g->dtype, false);
        const float* gp = gh->data_ptr<float>();
        float* op = oh->data_ptr<float>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t k = 0; k < kv_heads; ++k) {
                float* dst = op + (b * kv_heads + k) * head_elems;
                std::fill(dst, dst + head_elems, 0.0f);
                for (int64_t r = 0; r < group; ++r) {
                    const float* src = gp + ((b * kv_heads + k) * group + r) * head_elems;
                    for (int64_t e = 0; e < head_elems; ++e) dst[e] += src[e];
                }
            }
        }
        return {(dev == Device::CPU) ? oh : oh->to(dev)};
    }
};

inline std::shared_ptr<Tensor> repeat_kv(const std::shared_ptr<Tensor>& x_in, int64_t group) {
    if (group == 1) return x_in;
    auto x = contiguous(x_in);
    if (x->ndim() != 4) throw std::invalid_argument("repeat_kv: expected [batch, kv_heads, seq, head_dim]");
    int64_t batch = x->shape[0], kv_heads = x->shape[1], head_elems = x->shape[2] * x->shape[3];
    std::vector<int64_t> out_shape{batch, kv_heads * group, x->shape[2], x->shape[3]};
    Device dev = x->device;
    auto out = std::make_shared<Tensor>(out_shape, dev, x->dtype, false);
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::repeat_kv_forward(x->data_ptr<float>(), out->data_ptr<float>(), batch, kv_heads, group, head_elems);
    } else
#endif
    {
        auto xh = (dev == Device::CPU) ? x : x->cpu();
        auto oh = (dev == Device::CPU) ? out : std::make_shared<Tensor>(out_shape, Device::CPU, x->dtype, false);
        const float* xp = xh->data_ptr<float>();
        float* op = oh->data_ptr<float>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < kv_heads * group; ++h) {
                const float* src = xp + (b * kv_heads + h / group) * head_elems;
                std::copy(src, src + head_elems, op + (b * kv_heads * group + h) * head_elems);
            }
        }
        if (dev != Device::CPU) out = oh->to(dev);
    }
    out->requires_grad = needs_grad(x);
    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<RepeatKVBackward>(batch, kv_heads, group, head_elems, x->shape);
        grad_fn->add_next_edge(get_grad_edge(x).function);
        out->grad_fn = grad_fn;
    }
    return out;
}


// Multi-Head Self-Attention (Supports MHA, GQA, and MQA)
class MultiHeadAttention : public Module {
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
        if (!x_in) throw std::runtime_error("MultiHeadAttention: input tensor is null");
        if (x_in->ndim() < 2) throw std::runtime_error("MultiHeadAttention: input must be at least 2D (S, D) or (B, S, D)");
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

        // Grouped Query Attention (GQA): each kv head serves a group of query heads
        if (n_kv_heads < n_heads) {
            int64_t group = n_heads / n_kv_heads;
            K_split = repeat_kv(K_split, group);
            V_split = repeat_kv(V_split, group);
        }

        // Scaled dot product
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

        std::shared_ptr<Tensor> attn_out;
#ifdef USE_CUDA
        if (x->device == Device::CUDA && !mask) {
            auto Q_c = make_contiguous(Q_split);
            auto K_c = make_contiguous(K_split);
            auto V_c = make_contiguous(V_split);
            // Q/K/V carry the weights' requires_grad even when the input x does not
            bool qkv_grad = grad_enabled() && (Q_split->requires_grad || K_split->requires_grad || V_split->requires_grad);
            attn_out = std::make_shared<Tensor>(Q_split->shape, Device::CUDA, x->dtype, qkv_grad);
            auto lse = std::make_shared<Tensor>(std::vector<int64_t>{B, n_heads, S}, Device::CUDA, x->dtype, false);
            DType compute = autocast_dtype();
            auto impl = cuda::attention_forward(Q_c->data_ptr<float>(), K_c->data_ptr<float>(), V_c->data_ptr<float>(),
                                                attn_out->data_ptr<float>(), lse->data_ptr<float>(), B, n_heads, S, d_k,
                                                causal, scale, compute);
            if (qkv_grad) {
                auto grad_fn = std::make_shared<FlashAttentionBackward>(Q_c, K_c, V_c, attn_out, lse,
                                                                        B, n_heads, S, d_k, causal, scale, impl, compute);
                grad_fn->add_next_edge(get_grad_edge(Q_split).function);
                grad_fn->add_next_edge(get_grad_edge(K_split).function);
                grad_fn->add_next_edge(get_grad_edge(V_split).function);
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

    void visit(ModuleVisitor& v) override {
        v.module("W_q", W_q);
        v.module("W_k", W_k);
        v.module("W_v", W_v);
        v.module("W_o", W_o);
    }
};
