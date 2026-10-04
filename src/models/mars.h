#pragma once
#include "tensor.h"
#include "functional.h"
#include "linear.h"
#include "layernorm.h"
#include "attention.h"
#include "transformer.h"
#include "embedding.h"
#include "serialization.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <iostream>

// ============================================================================
// MARS: Multimodal Action Reasoning System
// Continuous Action Decoder-only Transformer for Kinematic Trajectory Modeling
//
// Core Hypothesis:
// Treats continuous joint angles (e.g. 156-dim CMU Motion Capture states) as
// primary tokens in an autoregressive decoder-only transformer. Next-frame
// action prediction follows identical scaling behavior to LLM next-token prediction.
// ============================================================================

struct MARSConfig {
    int action_dim = 156;    // Number of skeletal joint degrees of freedom per frame
    int d_model = 256;       // Transformer hidden / latent dimension
    int n_heads = 8;         // Number of causal attention heads
    int n_layers = 4;        // Depth of transformer decoder (1 to 6 layers in scaling study)
    int max_seq_len = 128;   // Maximum temporal context window (number of frames)
    int d_ff = 0;            // Feed-Forward hidden dim (0 defaults to 4 * d_model)
    bool causal = true;      // Causal masking for strictly autoregressive prediction
};

class MARS {
public:
    MARSConfig config;

    // 1. Continuous Action Encoder: projects raw joint angles [action_dim -> d_model]
    Linear action_encoder;

    // 2. Learned Temporal Positional Embeddings: [max_seq_len, d_model]
    Embedding pos_emb;

    // 3. Stack of Pre-LN Causal Transformer Decoder Blocks
    std::vector<std::shared_ptr<TransformerBlock>> blocks;

    // 4. Pre-Head Final LayerNorm
    LayerNorm ln_f;

    // 5. Continuous Action Head: projects [d_model -> action_dim] to predict next-frame joint angles
    Linear action_head;

    explicit MARS(const MARSConfig& cfg = MARSConfig())
        : config(cfg),
          action_encoder(cfg.action_dim, cfg.d_model),
          pos_emb(cfg.max_seq_len, cfg.d_model),
          ln_f(cfg.d_model),
          action_head(cfg.d_model, cfg.action_dim) {
        
        if (cfg.n_layers <= 0) throw std::invalid_argument("MARS: n_layers must be > 0");
        if (cfg.n_heads <= 0) throw std::invalid_argument("MARS: n_heads must be > 0");
        if (cfg.d_model <= 0 || cfg.d_model % cfg.n_heads != 0) throw std::invalid_argument("MARS: d_model must be positive and divisible by n_heads");
        if (cfg.action_dim <= 0) throw std::invalid_argument("MARS: action_dim must be > 0");

        int d_ff_dim = (cfg.d_ff > 0) ? cfg.d_ff : (4 * cfg.d_model);

        for (int i = 0; i < cfg.n_layers; i++) {
            auto block = std::make_shared<TransformerBlock>(cfg.d_model, cfg.n_heads, cfg.causal, d_ff_dim);
            // Residual projection scaling at initialization (stabilizes deep transformer scaling)
            float scale = 1.0f / std::sqrt(2.0f * cfg.n_layers);

            auto attn_proj = block->attn.W_o.W;
            int size_a = attn_proj->size();
            for (int j = 0; j < size_a; j++) {
                attn_proj->data_ptr<float>()[j] *= scale;
            }

            auto ffn_proj = block->ffn.fc2.W;
            int size_f = ffn_proj->size();
            for (int j = 0; j < size_f; j++) {
                ffn_proj->data_ptr<float>()[j] *= scale;
            }

            blocks.push_back(block);
        }
    }

    MARS(int action_dim, int d_model, int n_heads, int n_layers, int max_seq_len = 128)
        : MARS(MARSConfig{action_dim, d_model, n_heads, n_layers, max_seq_len}) {}

    // Forward pass:
    // actions: Tensor of shape [B, T, action_dim] or [T, action_dim]
    // returns: Predicted next-frame actions of shape [B, T, action_dim] (or [T, action_dim])
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& actions,
                                    const std::shared_ptr<Tensor>& mask = nullptr) {
        if (!actions) {
            throw std::runtime_error("MARS::forward received null actions tensor");
        }
        if (actions->ndim() != 2 && actions->ndim() != 3) {
            throw std::invalid_argument("MARS::forward: actions must be 2D [T, action_dim] or 3D [B, T, action_dim]");
        }

        bool is_2d = (actions->ndim() == 2);
        auto x_in = actions;
        if (is_2d) {
            x_in = reshape(actions, {1, actions->shape[0], actions->shape[1]});
        }

        int B = x_in->shape[0];
        int T = x_in->shape[1];
        int act_dim = x_in->shape[2];

        if (act_dim != config.action_dim) {
            throw std::runtime_error("MARS::forward: expected action_dim " +
                                     std::to_string(config.action_dim) + ", got " + std::to_string(act_dim));
        }
        if (T > config.max_seq_len) {
            throw std::runtime_error("MARS::forward: sequence length " + std::to_string(T) +
                                     " exceeds max_seq_len " + std::to_string(config.max_seq_len));
        }

        // 1. Encode continuous action tokens into latent space: [B, T, d_model]
        auto h = action_encoder.forward(x_in);

        // 2. Generate frame position indices [0, 1, ..., T-1] safely on CPU
        auto pos_ids_cpu = std::make_shared<Tensor>(std::vector<int64_t>{B, T}, Device::CPU, DType::Float32, false);
        float* p_ptr = pos_ids_cpu->data_ptr<float>();
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                p_ptr[b * T + t] = static_cast<float>(t);
            }
        }
        auto pos_ids = (actions->device != Device::CPU) ? pos_ids_cpu->to(actions->device) : pos_ids_cpu;

        // 3. Positional Embedding lookup [B, T, d_model] and residual addition
        auto pos_h = pos_emb.forward(pos_ids);
        h = add(h, pos_h);

        // 4. Pass through Causal Transformer Blocks
        for (auto& block : blocks) {
            h = block->forward(h, mask);
        }

        // 5. Final LayerNorm
        auto h_norm = ln_f.forward(h);

        // 6. Action Head projection to next-frame kinematics: [B, T, action_dim]
        auto out = action_head.forward(h_norm);

        if (is_2d) {
            out = reshape(out, {T, config.action_dim});
        }

        return out;
    }

    // Compute next-frame autoregressive loss from a motion trajectory sequence [B, T, action_dim].
    // Slices inputs actions[:, 0:T-1, :] to predict target actions[:, 1:T, :].
    std::shared_ptr<Tensor> compute_loss(const std::shared_ptr<Tensor>& trajectory,
                                         const std::string& loss_type = "mse") {
        if (!trajectory) throw std::runtime_error("compute_loss: null trajectory");

        bool is_2d = (trajectory->ndim() == 2);
        auto seq = is_2d ? reshape(trajectory, {1, trajectory->shape[0], trajectory->shape[1]}) : trajectory;
        int T = seq->shape[1];

        if (T < 2) {
            throw std::runtime_error("compute_loss: trajectory must contain at least 2 frames for next-frame prediction");
        }

        auto inputs = slice(seq, 1, 0, T - 1);
        auto targets = slice(seq, 1, 1, T);

        auto preds = forward(inputs);

        if (loss_type == "l1" || loss_type == "mae") {
            return l1_loss(preds, targets);
        }
        return mse(preds, targets);
    }

    // Compute loss given explicit paired inputs and targets
    std::shared_ptr<Tensor> compute_loss(const std::shared_ptr<Tensor>& inputs,
                                         const std::shared_ptr<Tensor>& targets,
                                         const std::string& loss_type = "mse") {
        if (!inputs || !targets) throw std::invalid_argument("compute_loss: inputs and targets cannot be null");
        if (inputs->shape != targets->shape) throw std::invalid_argument("compute_loss: inputs and targets shape mismatch");
        auto preds = forward(inputs);
        if (loss_type == "l1" || loss_type == "mae") {
            return l1_loss(preds, targets);
        }
        return mse(preds, targets);
    }

    // Autoregressive Rollout:
    // Given an initial sequence of seed frames [B, T_seed, action_dim] (or [T_seed, action_dim]),
    // iteratively predicts and appends n_steps future frames into the trajectory.
    std::shared_ptr<Tensor> rollout(const std::shared_ptr<Tensor>& seed_frames, int n_steps) {
        if (!seed_frames) throw std::runtime_error("rollout: null seed_frames");
        if (n_steps <= 0) return seed_frames;

        bool is_2d = (seed_frames->ndim() == 2);
        auto current_traj = is_2d ? reshape(seed_frames, {1, seed_frames->shape[0], seed_frames->shape[1]}) : seed_frames;

        for (int step = 0; step < n_steps; step++) {
            int T_curr = current_traj->shape[1];
            // If trajectory exceeds context length, slide window to last max_seq_len frames
            std::shared_ptr<Tensor> model_input = current_traj;
            if (T_curr > config.max_seq_len) {
                model_input = slice(current_traj, 1, T_curr - config.max_seq_len, T_curr);
            }

            // Run forward pass
            auto preds = forward(model_input);
            int T_in = model_input->shape[1];

            // Slice out the final predicted frame: [B, 1, action_dim]
            auto next_frame = slice(preds, 1, T_in - 1, T_in);

            // Detach to prevent autograd graph accumulation and memory explosion during rollout
            next_frame->requires_grad = false;
            next_frame->grad_fn = nullptr;
            current_traj->requires_grad = false;
            current_traj->grad_fn = nullptr;

            // Concatenate along the temporal sequence dimension (axis 0 in concat for 3D tensors)
            current_traj = concat(current_traj, next_frame, 0);
        }

        if (is_2d) {
            current_traj = reshape(current_traj, {current_traj->shape[1], config.action_dim});
        }

        return current_traj;
    }

    // Parameter collection for optimizers
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        std::vector<std::shared_ptr<Tensor>> params;

        auto enc_p = action_encoder.parameters();
        params.insert(params.end(), enc_p.begin(), enc_p.end());

        auto pos_p = pos_emb.parameters();
        params.insert(params.end(), pos_p.begin(), pos_p.end());

        for (const auto& block : blocks) {
            auto bp = block->parameters();
            params.insert(params.end(), bp.begin(), bp.end());
        }

        auto lnp = ln_f.parameters();
        params.insert(params.end(), lnp.begin(), lnp.end());

        auto head_p = action_head.parameters();
        params.insert(params.end(), head_p.begin(), head_p.end());

        return params;
    }

    // Named parameters for zero-copy SafeTensors export & PyTorch state_dict interoperability
    std::unordered_map<std::string, std::shared_ptr<Tensor>> named_parameters() const {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> named;

        named["action_encoder.weight"] = action_encoder.W;
        named["action_encoder.bias"] = action_encoder.b;

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

        named["action_head.weight"] = action_head.W;
        named["action_head.bias"] = action_head.b;

        return named;
    }

    // Save model weights directly to Hugging Face SafeTensors
    void save_safetensors(const std::string& filepath) const {
        io::save_safetensors(filepath, named_parameters());
    }

    // Load model weights directly from SafeTensors checkpoint
    void load_safetensors(const std::string& filepath) {
        auto loaded = io::load_safetensors(filepath, action_encoder.W->device);
        auto named = named_parameters();
        for (const auto& [name, src_tensor] : loaded) {
            auto it = named.find(name);
            if (it != named.end() && it->second && src_tensor) {
                auto dst_tensor = it->second;
                if (dst_tensor->size() == src_tensor->size()) {
                    auto converted = (src_tensor->device != dst_tensor->device || src_tensor->dtype != dst_tensor->dtype)
                                     ? src_tensor->to(dst_tensor->device, dst_tensor->dtype) : src_tensor;
                    dst_tensor->storage = converted->storage;
                    dst_tensor->strides = converted->strides;
                    dst_tensor->global_offset = converted->global_offset;
                }
            }
        }
    }

    // Transfer all model parameters to target device (CPU, CUDA, MPS, TPU)
    void to(Device dev, DType dtype = DType::Float32) {
        action_encoder.to(dev, dtype);
        pos_emb.to(dev, dtype);
        for (auto& block : blocks) {
            block->to(dev, dtype);
        }
        ln_f.to(dev, dtype);
        action_head.to(dev, dtype);
    }

    void cuda() { to(Device::CUDA); }
    void mps()  { to(Device::MPS); }
    void cpu()  { to(Device::CPU); }
};
