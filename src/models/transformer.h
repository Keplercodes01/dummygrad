#pragma once
#include "tensor.h"
#include "functional.h"
#include "linear.h"
#include "layernorm.h"
#include "attention.h"
#include "ops.h"
#include "embedding.h"
#include "modern_layers.h"
#include "serialization.h"

// Transformer Feed-Forward Network (MLP Block)
class FeedForward : public Module {
public:
    Linear fc1;
    Linear fc2;

    FeedForward(int d_model, int d_ff = 0)
        : fc1(d_model, (d_ff > 0 ? d_ff : 4 * d_model)),
          fc2((d_ff > 0 ? d_ff : 4 * d_model), d_model) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto h = gelu(fc1.forward(x));
        return fc2.forward(h);
    }

    void visit(ModuleVisitor& v) override {
        v.module("fc1", fc1);
        v.module("fc2", fc2);
    }
};

// Single Transformer Decoder Block (Pre-LayerNorm)
class TransformerBlock : public Module {
public:
    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    FeedForward ffn;

    TransformerBlock(int d_model, int n_heads, bool causal = true, int d_ff = 0)
        : ln1(d_model), attn(d_model, n_heads, /*n_kv_heads=*/0, causal), ln2(d_model), ffn(d_model, d_ff) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x,
                                    const std::shared_ptr<Tensor>& mask = nullptr) {
        // Pre-LN Residual Connection 1: x = x + Attention(LN1(x))
        auto norm1 = ln1.forward(x);
        auto attn_out = attn.forward(norm1, mask);
        auto x1 = add(x, attn_out);

        // Pre-LN Residual Connection 2: x = x + FFN(LN2(x))
        auto norm2 = ln2.forward(x1);
        auto ffn_out = ffn.forward(norm2);
        return add(x1, ffn_out);
    }

    void visit(ModuleVisitor& v) override {
        v.module("ln1", ln1);
        v.module("attn", attn);
        v.module("ln2", ln2);
        v.module("ffn", ffn);
    }
};

// GPT Language Model (Decoder-only Transformer)
class GPT : public Module {
public:
    Embedding token_emb;
    Embedding pos_emb;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    LayerNorm ln_f;

    int vocab_size;
    int max_seq_len;
    int d_model;

    GPT(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers)
        : token_emb(vocab_size, d_model),
          pos_emb(max_seq_len, d_model),
          ln_f(d_model),
          vocab_size(vocab_size),
          max_seq_len(max_seq_len),
          d_model(d_model) {
        if (n_layers <= 0) throw std::invalid_argument("GPT: n_layers must be > 0");
        if (n_heads <= 0) throw std::invalid_argument("GPT: n_heads must be > 0");
        if (d_model <= 0 || d_model % n_heads != 0) throw std::invalid_argument("GPT: d_model must be positive and divisible by n_heads");
        
        for (int i = 0; i < n_layers; i++) {
            blocks.push_back(std::make_shared<TransformerBlock>(d_model, n_heads, true));

            // GPT-2 Training Improvement: Scale residual projections at initialization
            // This prevents variance explosion deep in the network
            float scale = 1.0f / std::sqrt(2.0f * n_layers);
            
            auto attn_proj = blocks.back()->attn.W_o.W;
            int size_a = attn_proj->size();
            for (int j = 0; j < size_a; j++) {
                attn_proj->data_ptr<float>()[j] *= scale;
            }

            auto ffn_proj = blocks.back()->ffn.fc2.W;
            int size_f = ffn_proj->size();
            for (int j = 0; j < size_f; j++) {
                ffn_proj->data_ptr<float>()[j] *= scale;
            }
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_ids,
                                    const std::shared_ptr<Tensor>& pos_ids = nullptr,
                                    const std::shared_ptr<Tensor>& mask = nullptr) {
        if (!input_ids) throw std::invalid_argument("GPT::forward: input_ids is null");

        // Token + Positional Embeddings
        auto tok_x = token_emb.forward(input_ids);

        std::shared_ptr<Tensor> pos_x = nullptr;
        if (pos_ids) {
            pos_x = pos_emb.forward(pos_ids);
        } else {
            // Positions 0..T-1 as one row on the host: the [T, d_model] embedding broadcasts over the
            // batch, and the embedding keeps the host ids for its backward (no copy back from the GPU)
            int64_t T = (input_ids->ndim() >= 2) ? input_ids->shape[1] : input_ids->shape[0];
            auto positions = std::make_shared<Tensor>(std::vector<int64_t>{T}, Device::CPU, DType::Float32, false);
            float* p_ptr = positions->data_ptr<float>();
            for (int64_t t = 0; t < T; ++t) p_ptr[t] = static_cast<float>(t);
            pos_x = pos_emb.forward(positions);
        }

        auto x = add(tok_x, pos_x);

        // Pass through Transformer blocks
        for (auto& block : blocks) {
            x = block->forward(x, mask);
        }

        // Final LayerNorm 
        auto x_norm = ln_f.forward(x);
        
        // GPT-2 Training Improvement: Weight Tying
        // The output projection shares weights with the token embedding: x_norm @ weight^T, read in place
        return matmul_nt(x_norm, token_emb.weight);
    }

    void visit(ModuleVisitor& v) override {
        v.module("token_emb", token_emb);
        v.module("pos_emb", pos_emb);
        for (size_t i = 0; i < blocks.size(); ++i) v.module("blocks." + std::to_string(i), *blocks[i]);
        v.module("ln_f", ln_f);
    }
};

// ============================================================================
// Modern Transformer Block (RMSNorm + Attention/RoPE + SwiGLU)
// Architecture standard across LLaMA 3, Mistral, Gemma 2, Qwen 2, and DeepSeek
// ============================================================================
class ModernTransformerBlock : public Module {
public:
    RMSNorm rms1;
    MultiHeadAttention attn;
    RMSNorm rms2;
    SwiGLU ffn;

    ModernTransformerBlock(int d_model, int n_heads, int n_kv_heads = 0, int hidden_dim = 0, bool causal = true)
        : rms1(d_model),
          attn(d_model, n_heads, n_kv_heads, causal),
          rms2(d_model),
          ffn(d_model, hidden_dim) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x,
                                    const std::shared_ptr<Tensor>& mask = nullptr,
                                    const std::shared_ptr<Tensor>& cos_t = nullptr,
                                    const std::shared_ptr<Tensor>& sin_t = nullptr,
                                    int start_pos = 0) {
        // Pre-RMSNorm Residual Connection 1: x = x + Attention(RMSNorm1(x), RoPE)
        auto h1 = rms1.forward(x);
        auto attn_out = attn.forward(h1, mask, cos_t, sin_t, start_pos);
        auto x1 = add(x, attn_out);

        // Pre-RMSNorm Residual Connection 2: x = x + SwiGLU(RMSNorm2(x1))
        auto h2 = rms2.forward(x1);
        auto ffn_out = ffn.forward(h2);
        return add(x1, ffn_out);
    }

    void visit(ModuleVisitor& v) override {
        v.module("rms1", rms1);
        v.module("attn", attn);
        v.module("rms2", rms2);
        v.module("ffn", ffn);
    }
};

// ============================================================================
// Modern Transformer Language Model (LLaMA / Mistral / DeepSeek Stack)
// Features:
// - Zero learned absolute positional embeddings (uses RoPE)
// - RMSNorm (no mean reduction overhead)
// - SwiGLU gated activations
// - Tied input/output embeddings
// ============================================================================
class ModernTransformer : public Module {
public:
    Embedding token_emb;
    std::vector<std::shared_ptr<ModernTransformerBlock>> blocks;
    RMSNorm rms_f;

    int vocab_size;
    int max_seq_len;
    int d_model;
    int n_heads;
    int n_kv_heads;
    int head_dim;

    std::shared_ptr<Tensor> cos_freqs;
    std::shared_ptr<Tensor> sin_freqs;

    ModernTransformer(int vocab_size, int max_seq_len, int d_model, int n_heads, int n_layers, int n_kv_heads = 0, int hidden_dim = 0)
        : token_emb(vocab_size, d_model),
          rms_f(d_model),
          vocab_size(vocab_size),
          max_seq_len(max_seq_len),
          d_model(d_model),
          n_heads(n_heads),
          n_kv_heads(n_kv_heads > 0 ? n_kv_heads : n_heads),
          head_dim(d_model / n_heads) {
        if (n_layers <= 0) throw std::invalid_argument("ModernTransformer: n_layers must be > 0");
        if (n_heads <= 0) throw std::invalid_argument("ModernTransformer: n_heads must be > 0");
        if (d_model <= 0 || d_model % n_heads != 0) throw std::invalid_argument("ModernTransformer: d_model must be positive and divisible by n_heads");

        // Precompute RoPE rotary frequency tables once at model construction
        auto [c_t, s_t] = precompute_freqs_cis(head_dim, max_seq_len);
        cos_freqs = c_t;
        sin_freqs = s_t;

        for (int i = 0; i < n_layers; i++) {
            blocks.push_back(std::make_shared<ModernTransformerBlock>(d_model, n_heads, n_kv_heads, hidden_dim, true));

            // Scale residual projections at initialization to prevent variance explosion
            float scale = 1.0f / std::sqrt(2.0f * n_layers);
            auto attn_proj = blocks.back()->attn.W_o.W;
            int size_a = attn_proj->size();
            for (int j = 0; j < size_a; j++) {
                attn_proj->data_ptr<float>()[j] *= scale;
            }

            auto ffn_proj = blocks.back()->ffn.w_down.W;
            int size_f = ffn_proj->size();
            for (int j = 0; j < size_f; j++) {
                ffn_proj->data_ptr<float>()[j] *= scale;
            }
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_ids,
                                    const std::shared_ptr<Tensor>& mask = nullptr,
                                    int start_pos = 0) {
        if (!input_ids) throw std::invalid_argument("ModernTransformer::forward: input_ids is null");

        // Token Embeddings (No learned positional embeddings needed with RoPE)
        auto x = token_emb.forward(input_ids);

        // Pass through ModernTransformer blocks with Rotary Embeddings
        for (auto& block : blocks) {
            x = block->forward(x, mask, cos_freqs, sin_freqs, start_pos);
        }

        // Final RMSNorm
        auto x_norm = rms_f.forward(x);

        // Tied LM head: x_norm @ embedding^T, without materializing the transpose
        return matmul_nt(x_norm, token_emb.weight);
    }

    void visit(ModuleVisitor& v) override {
        v.module("token_emb", token_emb);
        for (size_t i = 0; i < blocks.size(); ++i) v.module("blocks." + std::to_string(i), *blocks[i]);
        v.module("rms_f", rms_f);
        // RoPE tables are recomputed at construction, so they move with the model but aren't saved
        v.buffer("cos_freqs", cos_freqs, /*persistent=*/false);
        v.buffer("sin_freqs", sin_freqs, /*persistent=*/false);
    }
};
