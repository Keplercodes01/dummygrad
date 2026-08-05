#pragma once
#include "engine.h"
#include "linear.h"
#include "layernorm.h"
#include "attention.h"
#include "activations.h"
#include "ops.h"
#include "embedding.h"

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
};

// Single Transformer Decoder Block (Pre-LayerNorm)
class TransformerBlock {
public:
    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    FeedForward ffn;

    TransformerBlock(int d_model, int n_heads, bool causal = true)
        : ln1(d_model), attn(d_model, n_heads, causal), ln2(d_model), ffn(d_model) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        // Pre-LN Residual Connection 1: x = x + Attention(LN1(x))
        auto norm1 = ln1.forward(x);
        auto attn_out = attn.forward(norm1);
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
        
        for (int i = 0; i < n_layers; i++) {
            blocks.push_back(std::make_shared<TransformerBlock>(d_model, n_heads, true));

            // GPT-2 Training Improvement: Scale residual projections at initialization
            // This prevents variance explosion deep in the network
            float scale = 1.0f / std::sqrt(2.0f * n_layers);
            
            auto attn_proj = blocks.back()->attn.W_o.W;
            int size_a = attn_proj->size();
            for (int j = 0; j < size_a; j++) {
                attn_proj->data_ptr()[j] *= scale;
            }

            auto ffn_proj = blocks.back()->ffn.fc2.W;
            int size_f = ffn_proj->size();
            for (int j = 0; j < size_f; j++) {
                ffn_proj->data_ptr()[j] *= scale;
            }
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_ids, const std::shared_ptr<Tensor>& pos_ids) {
        // Token + Positional Embeddings
        auto tok_x = token_emb.forward(input_ids);
        auto pos_x = pos_emb.forward(pos_ids);
        auto x = add(tok_x, pos_x);

        // Pass through Transformer blocks
        for (auto& block : blocks) {
            x = block->forward(x);
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
};
