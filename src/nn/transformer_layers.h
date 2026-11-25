#pragma once
// General attention and the encoder-decoder Transformer of "Attention Is All You Need", with separate
// query, key and value inputs. The language-model blocks (fused attention, RoPE, GQA) are in attention.h
// and models/transformer.h.

#include "layers.h"
#include "layernorm.h"

// softmax(q k^T / sqrt(d)) v over the last two dimensions. attn_mask is added to the scores (use a large
// negative value to block); is_causal blocks later keys.
inline std::shared_ptr<Tensor> scaled_dot_product_attention(const std::shared_ptr<Tensor>& q, const std::shared_ptr<Tensor>& k,
                                                            const std::shared_ptr<Tensor>& v, const std::shared_ptr<Tensor>& attn_mask = nullptr,
                                                            float dropout_p = 0.0f, bool is_causal = false, bool training = true,
                                                            std::shared_ptr<Tensor>* weights_out = nullptr) {
    DG_PROFILE_OP("scaled_dot_product_attention");
    auto scores = mul_scalar(matmul(q, transpose(k, -2, -1)), 1.0f / std::sqrt((float)q->shape.back()));
    if (attn_mask) scores = add(scores, attn_mask);
    if (is_causal) scores = causal_mask(scores);
    auto p = softmax(scores);
    if (weights_out) *weights_out = p;
    return matmul(dropout(p, dropout_p, training), v);
}

// Multi-head attention between any query and key-value sequences [batch, time, d_model]
class CrossAttention : public Module {
public:
    Linear W_q, W_k, W_v, W_o;
    int64_t d_model, n_heads;
    float dropout_p;

    // d_context: the width of keys and values (0: d_model)
    CrossAttention(int64_t d_model, int64_t n_heads, float dropout_p = 0.0f, int64_t d_context = 0)
        : W_q((int)d_model, (int)d_model), W_k((int)(d_context ? d_context : d_model), (int)d_model),
          W_v((int)(d_context ? d_context : d_model), (int)d_model), W_o((int)d_model, (int)d_model),
          d_model(d_model), n_heads(n_heads), dropout_p(dropout_p) {
        if (n_heads < 1 || d_model % n_heads != 0) throw std::invalid_argument("CrossAttention: n_heads must divide d_model");
    }
    // key_padding_mask [batch, key time]: nonzero keys are ignored
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& query, const std::shared_ptr<Tensor>& context,
                                    const std::shared_ptr<Tensor>& attn_mask = nullptr, const std::shared_ptr<Tensor>& key_padding_mask = nullptr,
                                    bool is_causal = false, std::shared_ptr<Tensor>* weights_out = nullptr) {
        auto heads = [&](const std::shared_ptr<Tensor>& t) {
            return transpose(reshape(t, {t->shape[0], t->shape[1], n_heads, d_model / n_heads}), 1, 2);
        };
        auto mask = attn_mask;
        if (key_padding_mask) {
            auto blocked = mul_scalar(ne(key_padding_mask, 0.0f), -1e9f);
            blocked = reshape(blocked, {blocked->shape[0], 1, 1, blocked->shape[1]});
            mask = mask ? add(mask, blocked) : blocked;
        }
        auto out = scaled_dot_product_attention(heads(W_q.forward(query)), heads(W_k.forward(context)), heads(W_v.forward(context)), mask,
                                                dropout_p, is_causal, training_, weights_out);
        return W_o.forward(reshape(transpose(out, 1, 2), {query->shape[0], query->shape[1], d_model}));
    }
    void visit(ModuleVisitor& v) override {
        v.module("W_q", W_q);
        v.module("W_k", W_k);
        v.module("W_v", W_v);
        v.module("W_o", W_o);
    }
};

struct TransformerOptions {
    int64_t d_model = 512, n_heads = 8, d_ff = 2048;
    int64_t encoder_layers = 6, decoder_layers = 6;
    float dropout = 0.1f;
    bool norm_first = false;  // normalize before each sublayer instead of after the residual sum
    bool use_gelu = false;
};

class TransformerEncoderLayer : public Module {
public:
    CrossAttention self_attn;
    Linear fc1, fc2;
    LayerNorm norm1, norm2;
    TransformerOptions opt;

    explicit TransformerEncoderLayer(const TransformerOptions& o)
        : self_attn(o.d_model, o.n_heads, o.dropout), fc1((int)o.d_model, (int)o.d_ff), fc2((int)o.d_ff, (int)o.d_model),
          norm1((int)o.d_model), norm2((int)o.d_model), opt(o) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in, const std::shared_ptr<Tensor>& mask = nullptr,
                                    const std::shared_ptr<Tensor>& padding = nullptr, bool is_causal = false) {
        auto attend = [&](const std::shared_ptr<Tensor>& t) { return dropout(self_attn.forward(t, t, mask, padding, is_causal), opt.dropout, training_); };
        auto feed = [&](const std::shared_ptr<Tensor>& t) {
            auto h = fc1.forward(t);
            h = dropout(opt.use_gelu ? gelu_erf(h) : relu(h), opt.dropout, training_);
            return dropout(fc2.forward(h), opt.dropout, training_);
        };
        auto x = x_in;
        if (opt.norm_first) {
            x = add(x, attend(norm1.forward(x)));
            return add(x, feed(norm2.forward(x)));
        }
        x = norm1.forward(add(x, attend(x)));
        return norm2.forward(add(x, feed(x)));
    }
    void visit(ModuleVisitor& v) override {
        v.module("self_attn", self_attn);
        v.module("fc1", fc1);
        v.module("fc2", fc2);
        v.module("norm1", norm1);
        v.module("norm2", norm2);
    }
};

class TransformerDecoderLayer : public Module {
public:
    CrossAttention self_attn, cross_attn;
    Linear fc1, fc2;
    LayerNorm norm1, norm2, norm3;
    TransformerOptions opt;

    explicit TransformerDecoderLayer(const TransformerOptions& o)
        : self_attn(o.d_model, o.n_heads, o.dropout), cross_attn(o.d_model, o.n_heads, o.dropout),
          fc1((int)o.d_model, (int)o.d_ff), fc2((int)o.d_ff, (int)o.d_model),
          norm1((int)o.d_model), norm2((int)o.d_model), norm3((int)o.d_model), opt(o) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in, const std::shared_ptr<Tensor>& memory,
                                    const std::shared_ptr<Tensor>& self_padding = nullptr, const std::shared_ptr<Tensor>& memory_padding = nullptr) {
        auto attend = [&](const std::shared_ptr<Tensor>& t) { return dropout(self_attn.forward(t, t, nullptr, self_padding, true), opt.dropout, training_); };
        auto cross = [&](const std::shared_ptr<Tensor>& t) { return dropout(cross_attn.forward(t, memory, nullptr, memory_padding), opt.dropout, training_); };
        auto feed = [&](const std::shared_ptr<Tensor>& t) {
            auto h = fc1.forward(t);
            h = dropout(opt.use_gelu ? gelu_erf(h) : relu(h), opt.dropout, training_);
            return dropout(fc2.forward(h), opt.dropout, training_);
        };
        auto x = x_in;
        if (opt.norm_first) {
            x = add(x, attend(norm1.forward(x)));
            x = add(x, cross(norm2.forward(x)));
            return add(x, feed(norm3.forward(x)));
        }
        x = norm1.forward(add(x, attend(x)));
        x = norm2.forward(add(x, cross(x)));
        return norm3.forward(add(x, feed(x)));
    }
    void visit(ModuleVisitor& v) override {
        v.module("self_attn", self_attn);
        v.module("cross_attn", cross_attn);
        v.module("fc1", fc1);
        v.module("fc2", fc2);
        v.module("norm1", norm1);
        v.module("norm2", norm2);
        v.module("norm3", norm3);
    }
};

// Encoder-decoder: source and target [batch, time, d_model] (already embedded), target causal
class Transformer : public Module {
public:
    ModuleList<TransformerEncoderLayer> encoder;
    ModuleList<TransformerDecoderLayer> decoder;
    LayerNorm encoder_norm, decoder_norm;
    TransformerOptions opt;

    explicit Transformer(const TransformerOptions& o) : encoder_norm((int)o.d_model), decoder_norm((int)o.d_model), opt(o) {
        for (int64_t i = 0; i < o.encoder_layers; ++i) encoder.emplace_back(o);
        for (int64_t i = 0; i < o.decoder_layers; ++i) decoder.emplace_back(o);
    }
    std::shared_ptr<Tensor> encode(const std::shared_ptr<Tensor>& src, const std::shared_ptr<Tensor>& padding = nullptr) {
        auto x = src;
        for (auto& layer : encoder) x = layer->forward(x, nullptr, padding);
        return encoder_norm.forward(x);
    }
    std::shared_ptr<Tensor> decode(const std::shared_ptr<Tensor>& tgt, const std::shared_ptr<Tensor>& memory,
                                   const std::shared_ptr<Tensor>& tgt_padding = nullptr, const std::shared_ptr<Tensor>& memory_padding = nullptr) {
        auto x = tgt;
        for (auto& layer : decoder) x = layer->forward(x, memory, tgt_padding, memory_padding);
        return decoder_norm.forward(x);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& src, const std::shared_ptr<Tensor>& tgt) { return decode(tgt, encode(src)); }
    void visit(ModuleVisitor& v) override {
        v.module("encoder", encoder);
        v.module("decoder", decoder);
        v.module("encoder_norm", encoder_norm);
        v.module("decoder_norm", decoder_norm);
    }
};

// Sinusoidal positions [1, time, d_model] to add to embeddings
inline std::shared_ptr<Tensor> sinusoidal_positions(int64_t time, int64_t d_model, Device device = Device::CPU) {
    std::vector<float> p((size_t)(time * d_model));
    for (int64_t t = 0; t < time; ++t)
        for (int64_t i = 0; i < d_model; ++i) {
            double angle = (double)t / std::pow(10000.0, (double)(i - i % 2) / (double)d_model);
            p[(size_t)(t * d_model + i)] = (float)(i % 2 == 0 ? std::sin(angle) : std::cos(angle));
        }
    return op_detail::from_values(p, {1, time, d_model}, device);
}
