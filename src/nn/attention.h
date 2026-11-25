#pragma once
#include "tensor.h"
#include "functional.h"
#include "linear.h"
#include "modern_layers.h"
#include "fused_attention.h"
#include "kv_cache.h"

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

        auto out = fused_attention(Q, K, V, mask, causal && !mask, scale);

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
    std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        return {reshape(sum(reshape(grads[0], {batch, kv_heads, group, head_elems}), 2, false), in_shape)};
    }

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
            cuda::repeat_kv_backward(g->data_ptr<void>(), gin->data_ptr<void>(), batch, kv_heads, group, head_elems, g->dtype);
            return {gin};
        }
#endif
        auto gh = (dev == Device::CPU) ? g : g->cpu();
        auto oh = (dev == Device::CPU) ? gin : std::make_shared<Tensor>(in_shape, Device::CPU, g->dtype, false);
        const float* gp = gh->data_ptr<float>();
        float* op = oh->data_ptr<float>();
        const int64_t copies = group, elems = head_elems;
        parallel_for(0, batch * kv_heads, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, copies * elems)), [=](int64_t begin, int64_t end) {
            for (int64_t h = begin; h < end; ++h) {
                float* dst = op + h * elems;
                std::fill(dst, dst + elems, 0.0f);
                for (int64_t r = 0; r < copies; ++r) {
                    const float* src = gp + (h * copies + r) * elems;
                    for (int64_t e = 0; e < elems; ++e) dst[e] += src[e];
                }
            }
        });
        return {(dev == Device::CPU) ? oh : oh->to(dev)};
    }
};

inline std::shared_ptr<Tensor> repeat_kv(const std::shared_ptr<Tensor>& x_in, int64_t group) {
    DG_PROFILE_OP("repeat_kv");
    if (group == 1) return x_in;
    auto x = contiguous(native_input(x_in));
    if (x->ndim() != 4) throw std::invalid_argument("repeat_kv: expected [batch, kv_heads, seq, head_dim]");
    int64_t batch = x->shape[0], kv_heads = x->shape[1], head_elems = x->shape[2] * x->shape[3];
    std::vector<int64_t> out_shape{batch, kv_heads * group, x->shape[2], x->shape[3]};
    Device dev = x->device;
    auto out = std::make_shared<Tensor>(out_shape, dev, x->dtype, false);
#ifdef USE_CUDA
    if (dev == Device::CUDA) {
        cuda::repeat_kv_forward(x->data_ptr<void>(), out->data_ptr<void>(), batch, kv_heads, group, head_elems, x->dtype);
    } else
#endif
    {
        auto xh = (dev == Device::CPU) ? x : x->cpu();
        auto oh = (dev == Device::CPU) ? out : std::make_shared<Tensor>(out_shape, Device::CPU, x->dtype, false);
        const float* xp = xh->data_ptr<float>();
        float* op = oh->data_ptr<float>();
        parallel_for(0, batch * kv_heads * group, std::max<int64_t>(1, 65536 / std::max<int64_t>(1, head_elems)), [=](int64_t begin, int64_t end) {
            for (int64_t h = begin; h < end; ++h) {
                const float* src = xp + (h / group) * head_elems;
                std::copy(src, src + head_elems, op + h * head_elems);
            }
        });
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

    struct Options {
        int n_kv_heads = 0;  // key and value heads (0: n_heads)
        bool causal = false;
        bool qkv_bias = true, out_bias = true;
        int head_dim = 0;    // width of one head (0: d_model / n_heads)
    };

    // Width of one head, after checking the layout: the kv heads must divide the heads, or the attention
    // kernels would index past K and V
    static int checked_head_dim(int d_model, int n_heads, const Options& o) {
        if (n_heads <= 0 || d_model <= 0 || o.head_dim < 0 || (o.head_dim == 0 && d_model % n_heads != 0)) {
            throw std::invalid_argument("MultiHeadAttention: d_model (" + std::to_string(d_model) +
                                        ") must be a positive multiple of n_heads (" + std::to_string(n_heads) + ")");
        }
        int kv = o.n_kv_heads > 0 ? o.n_kv_heads : n_heads;
        if (o.n_kv_heads < 0 || kv > n_heads || n_heads % kv != 0) {
            throw std::invalid_argument("MultiHeadAttention: n_kv_heads (" + std::to_string(o.n_kv_heads) +
                                        ") must divide n_heads (" + std::to_string(n_heads) + ")");
        }
        return o.head_dim > 0 ? o.head_dim : d_model / n_heads;
    }

    MultiHeadAttention(int d_model, int n_heads, int n_kv_heads = 0, bool causal = false)
        : MultiHeadAttention(d_model, n_heads, make_options(n_kv_heads, causal)) {}

    // W_q comes first, so a bad layout throws before anything is allocated
    MultiHeadAttention(int d_model, int n_heads, const Options& o)
        : W_q(d_model, n_heads * checked_head_dim(d_model, n_heads, o), 0.0f, o.qkv_bias),
          W_k(d_model, (o.n_kv_heads > 0 ? o.n_kv_heads : n_heads) * checked_head_dim(d_model, n_heads, o), 0.0f, o.qkv_bias),
          W_v(d_model, (o.n_kv_heads > 0 ? o.n_kv_heads : n_heads) * checked_head_dim(d_model, n_heads, o), 0.0f, o.qkv_bias),
          W_o(n_heads * checked_head_dim(d_model, n_heads, o), d_model, 0.0f, o.out_bias),
          d_model(d_model),
          n_heads(n_heads),
          n_kv_heads(o.n_kv_heads > 0 ? o.n_kv_heads : n_heads),
          d_k(checked_head_dim(d_model, n_heads, o)),
          causal(o.causal) {}

    static Options make_options(int n_kv_heads, bool causal) {
        Options o;
        o.n_kv_heads = n_kv_heads;
        o.causal = causal;
        return o;
    }

    // Queries attend to this call's tokens, or with a cache to those of earlier calls as well (inference; the
    // cache decides start_pos, and a mask must then cover every key: [.., S, cached + S])
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in,
                                    const std::shared_ptr<Tensor>& mask = nullptr,
                                    const std::shared_ptr<Tensor>& cos_freqs = nullptr,
                                    const std::shared_ptr<Tensor>& sin_freqs = nullptr,
                                    int start_pos = 0, KVCache* cache = nullptr) {
        if (!x_in) throw std::runtime_error("MultiHeadAttention: input tensor is null");
        if (cache) start_pos = (int)cache->length;
        if (x_in->ndim() < 2) throw std::runtime_error("MultiHeadAttention: input must be at least 2D (S, D) or (B, S, D)");
        auto x = x_in;
        bool is_2d = (x->ndim() == 2);
        if (is_2d) {
            x = reshape(x, {1, x->shape[0], x->shape[1]});
        }

        int B = x->shape[0];
        int S = x->shape[1];

        // Under autocast these stay in half precision through RoPE, GQA repetition and the attention kernel
        auto Q = W_q.forward_autocast(x); // [B, S, n_heads * d_k]
        auto K = W_k.forward_autocast(x); // [B, S, n_kv_heads * d_k]
        auto V = W_v.forward_autocast(x); // [B, S, n_kv_heads * d_k]

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

        // With a cache, the queries see the tokens of earlier calls too
        int64_t past = 0;
        if (cache) {
            past = cache->length;
            auto stored = cache->append(K_split, V_split);
            if (past > 0) {
                K_split = stored.first;
                V_split = stored.second;
            }
        }

        // Grouped Query Attention (GQA): each kv head serves a group of query heads
        if (n_kv_heads < n_heads) {
            int64_t group = n_heads / n_kv_heads;
            K_split = repeat_kv(K_split, group);
            V_split = repeat_kv(V_split, group);
        }

        // Scaled dot product
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

        auto attn_out = fused_attention(Q_split, K_split, V_split, mask, causal && !mask, scale, past);  // [B, n_heads, S, d_k]

        // Transpose back to [B, S, n_heads, d_k] -> Reshape to [B, S, n_heads * d_k]
        auto merged = reshape(transpose(attn_out, 1, 2), {B, S, n_heads * d_k});

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
