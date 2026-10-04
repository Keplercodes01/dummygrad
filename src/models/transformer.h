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
class FeedForward {
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

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = fc1.parameters();
        auto p2 = fc2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        return p;
    }

    void to(Device device, DType dtype = DType::Float32) {
        fc1.to(device, dtype);
        fc2.to(device, dtype);
    }
};

// Single Transformer Decoder Block (Pre-LayerNorm)
class TransformerBlock {
public:
    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    FeedForward ffn;

    TransformerBlock(int d_model, int n_heads, bool causal = true, int d_ff = 0)
        : ln1(d_model), attn(d_model, n_heads, causal), ln2(d_model), ffn(d_model, d_ff) {}

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

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = ln1.parameters();
        auto pa = attn.parameters();
        auto pl = ln2.parameters();
        auto pf = ffn.parameters();
        p.insert(p.end(), pa.begin(), pa.end());
        p.insert(p.end(), pl.begin(), pl.end());
        p.insert(p.end(), pf.begin(), pf.end());
        return p;
    }

    void to(Device device, DType dtype = DType::Float32) {
        ln1.to(device, dtype);
        attn.to(device, dtype);
        ln2.to(device, dtype);
        ffn.to(device, dtype);
    }
};

// GPT Language Model (Decoder-only Transformer)
class GPT {
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
            // Automatically generate frame position indices [0, 1, ..., T-1] safely on CPU
            int64_t T = (input_ids->ndim() >= 2) ? input_ids->shape[1] : input_ids->shape[0];
            int64_t B = (input_ids->ndim() >= 2) ? input_ids->shape[0] : 1;
            auto auto_pos = std::make_shared<Tensor>(std::vector<int64_t>{B, T}, Device::CPU, DType::Float32, false);
            float* p_ptr = auto_pos->data_ptr<float>();
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t t = 0; t < T; ++t) {
                    p_ptr[b * T + t] = static_cast<float>(t);
                }
            }
            auto p_dev = (input_ids->device != Device::CPU) ? auto_pos->to(input_ids->device) : auto_pos;
            pos_x = pos_emb.forward(p_dev);
        }

        auto x = add(tok_x, pos_x);

        // Pass through Transformer blocks
        for (auto& block : blocks) {
            x = block->forward(x, mask);
        }

        // Final LayerNorm 
        auto x_norm = ln_f.forward(x);
        
        // GPT-2 Training Improvement: Weight Tying
        // The output projection shares weights with the token embedding
        auto tied_weights = transpose(token_emb.weight, 0, 1);
        return matmul(x_norm, tied_weights);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = token_emb.parameters();
        auto pp = pos_emb.parameters();
        p.insert(p.end(), pp.begin(), pp.end());

        for (auto& block : blocks) {
            auto pb = block->parameters();
            p.insert(p.end(), pb.begin(), pb.end());
        }

        auto pln = ln_f.parameters();
        p.insert(p.end(), pln.begin(), pln.end());
        
        return p;
    }

    std::unordered_map<std::string, std::shared_ptr<Tensor>> named_parameters() const {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> named;
        named["token_emb.weight"] = token_emb.weight;
        named["pos_emb.weight"] = pos_emb.weight;
        for (size_t i = 0; i < blocks.size(); i++) {
            std::string prefix = "blocks." + std::to_string(i) + ".";
            named[prefix + "ln1.gamma"] = blocks[i]->ln1.gamma;
            named[prefix + "ln1.beta"] = blocks[i]->ln1.beta;
            named[prefix + "attn.W_q.weight"] = blocks[i]->attn.W_q.W;
            named[prefix + "attn.W_q.bias"] = blocks[i]->attn.W_q.b;
            named[prefix + "attn.W_k.weight"] = blocks[i]->attn.W_k.W;
            named[prefix + "attn.W_k.bias"] = blocks[i]->attn.W_k.b;
            named[prefix + "attn.W_v.weight"] = blocks[i]->attn.W_v.W;
            named[prefix + "attn.W_v.bias"] = blocks[i]->attn.W_v.b;
            named[prefix + "attn.W_o.weight"] = blocks[i]->attn.W_o.W;
            named[prefix + "attn.W_o.bias"] = blocks[i]->attn.W_o.b;
            named[prefix + "ln2.gamma"] = blocks[i]->ln2.gamma;
            named[prefix + "ln2.beta"] = blocks[i]->ln2.beta;
            named[prefix + "ffn.fc1.weight"] = blocks[i]->ffn.fc1.W;
            named[prefix + "ffn.fc1.bias"] = blocks[i]->ffn.fc1.b;
            named[prefix + "ffn.fc2.weight"] = blocks[i]->ffn.fc2.W;
            named[prefix + "ffn.fc2.bias"] = blocks[i]->ffn.fc2.b;
        }
        named["ln_f.gamma"] = ln_f.gamma;
        named["ln_f.beta"] = ln_f.beta;
        return named;
    }

    void save_safetensors(const std::string& filepath) const {
        io::save_safetensors(filepath, named_parameters());
    }

    void load_safetensors(const std::string& filepath) {
        auto loaded = io::load_safetensors(filepath, token_emb.weight->device);
        auto named = named_parameters();
        for (const auto& [name, src_tensor] : loaded) {
            auto it = named.find(name);
            if (it != named.end() && it->second && src_tensor) {
                auto dst = it->second;
                if (dst->size() == src_tensor->size()) {
                    auto converted = (src_tensor->device != dst->device || src_tensor->dtype != dst->dtype)
                                     ? src_tensor->to(dst->device, dst->dtype) : src_tensor;
                    dst->storage = converted->storage;
                    dst->strides = converted->strides;
                    dst->global_offset = converted->global_offset;
                }
            }
        }
    }

    void to(Device device, DType dtype = DType::Float32) {
        token_emb.to(device, dtype);
        pos_emb.to(device, dtype);
        for (auto& block : blocks) {
            block->to(device, dtype);
        }
        ln_f.to(device, dtype);
    }

    void cuda() { to(Device::CUDA); }
    void mps()  { to(Device::MPS); }
    void cpu()  { to(Device::CPU); }
};

// ============================================================================
// Modern Transformer Block (RMSNorm + Attention/RoPE + SwiGLU)
// Architecture standard across LLaMA 3, Mistral, Gemma 2, Qwen 2, and DeepSeek
// ============================================================================
class ModernTransformerBlock {
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

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = rms1.parameters();
        auto pa = attn.parameters();
        auto p2 = rms2.parameters();
        auto pf = ffn.parameters();
        p.insert(p.end(), pa.begin(), pa.end());
        p.insert(p.end(), p2.begin(), p2.end());
        p.insert(p.end(), pf.begin(), pf.end());
        return p;
    }

    void to(Device device, DType dtype = DType::Float32) {
        rms1.to(device, dtype);
        attn.to(device, dtype);
        rms2.to(device, dtype);
        ffn.to(device, dtype);
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
class ModernTransformer {
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

        // Tied LM head weights
        auto tied_weights = transpose(token_emb.weight, 0, 1);
        return matmul(x_norm, tied_weights);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = token_emb.parameters();
        for (auto& block : blocks) {
            auto pb = block->parameters();
            p.insert(p.end(), pb.begin(), pb.end());
        }
        auto pr = rms_f.parameters();
        p.insert(p.end(), pr.begin(), pr.end());
        return p;
    }

    std::unordered_map<std::string, std::shared_ptr<Tensor>> named_parameters() const {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> named;
        named["token_emb.weight"] = token_emb.weight;
        for (size_t i = 0; i < blocks.size(); i++) {
            std::string prefix = "blocks." + std::to_string(i) + ".";
            named[prefix + "rms1.gamma"] = blocks[i]->rms1.gamma;
            named[prefix + "attn.W_q.weight"] = blocks[i]->attn.W_q.W;
            named[prefix + "attn.W_q.bias"] = blocks[i]->attn.W_q.b;
            named[prefix + "attn.W_k.weight"] = blocks[i]->attn.W_k.W;
            named[prefix + "attn.W_k.bias"] = blocks[i]->attn.W_k.b;
            named[prefix + "attn.W_v.weight"] = blocks[i]->attn.W_v.W;
            named[prefix + "attn.W_v.bias"] = blocks[i]->attn.W_v.b;
            named[prefix + "attn.W_o.weight"] = blocks[i]->attn.W_o.W;
            named[prefix + "attn.W_o.bias"] = blocks[i]->attn.W_o.b;
            named[prefix + "rms2.gamma"] = blocks[i]->rms2.gamma;
            named[prefix + "ffn.w_gate.weight"] = blocks[i]->ffn.w_gate.W;
            named[prefix + "ffn.w_gate.bias"] = blocks[i]->ffn.w_gate.b;
            named[prefix + "ffn.w_up.weight"] = blocks[i]->ffn.w_up.W;
            named[prefix + "ffn.w_up.bias"] = blocks[i]->ffn.w_up.b;
            named[prefix + "ffn.w_down.weight"] = blocks[i]->ffn.w_down.W;
            named[prefix + "ffn.w_down.bias"] = blocks[i]->ffn.w_down.b;
        }
        named["rms_f.gamma"] = rms_f.gamma;
        return named;
    }

    void save_safetensors(const std::string& filepath) const {
        io::save_safetensors(filepath, named_parameters());
    }

    void load_safetensors(const std::string& filepath) {
        auto loaded = io::load_safetensors(filepath, token_emb.weight->device);
        auto named = named_parameters();
        for (const auto& [name, src_tensor] : loaded) {
            auto it = named.find(name);
            if (it != named.end() && it->second && src_tensor) {
                auto dst = it->second;
                if (dst->size() == src_tensor->size()) {
                    auto converted = (src_tensor->device != dst->device || src_tensor->dtype != dst->dtype)
                                     ? src_tensor->to(dst->device, dst->dtype) : src_tensor;
                    dst->storage = converted->storage;
                    dst->strides = converted->strides;
                    dst->global_offset = converted->global_offset;
                }
            }
        }
    }

    void to(Device device, DType dtype = DType::Float32) {
        token_emb.to(device, dtype);
        for (auto& block : blocks) {
            block->to(device, dtype);
        }
        rms_f.to(device, dtype);
        if (cos_freqs && cos_freqs->device != device) {
            cos_freqs = cos_freqs->to(device, dtype);
        }
        if (sin_freqs && sin_freqs->device != device) {
            sin_freqs = sin_freqs->to(device, dtype);
        }
    }

    void cuda() { to(Device::CUDA); }
    void mps()  { to(Device::MPS); }
    void cpu()  { to(Device::CPU); }
};
