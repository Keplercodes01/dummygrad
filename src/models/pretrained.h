#pragma once
// Models saved in the Hugging Face format: a folder with config.json and model.safetensors, or several
// .safetensors files and model.safetensors.index.json.
//
//   auto model = load_llama("path/to/Llama-3.2-1B");   // Llama, Mistral and Qwen2 models
//   model.to(Device::CUDA);
//   auto gpt2 = load_gpt2("path/to/gpt2");
//   save_llama(model, "my-model");                     // readable by transformers

#include "generate.h"
#include "init.h"
#include "json.h"
#include "serialization.h"
#include "transformer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pretrained {

// The tensors of a model folder, read one at a time
class Checkpoint {
public:
    explicit Checkpoint(const std::string& dir) {
        const std::string index = dir + "/model.safetensors.index.json";
        if (std::ifstream(index).good()) {
            auto parsed = json::parse_file(index);
            std::unordered_map<std::string, size_t> opened;
            for (const auto& [tensor, shard] : parsed.at("weight_map").object) {
                const std::string& name = shard.as_string();
                if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos) throw std::runtime_error("checkpoint index names a file outside the folder: " + name);
                auto it = opened.find(name);
                if (it == opened.end()) {
                    it = opened.emplace(name, files_.size()).first;
                    files_.push_back(std::make_shared<io::SafeTensorsFile>(dir + "/" + name));
                }
                where_[tensor] = it->second;
            }
        } else {
            files_.push_back(std::make_shared<io::SafeTensorsFile>(dir + "/model.safetensors"));
            for (const auto& e : files_[0]->entries()) where_[e.first] = 0;
        }
    }

    bool has(const std::string& name) const { return where_.count(name) > 0; }

    // As Float32, checked against the shape the model expects
    std::shared_ptr<Tensor> read(const std::string& name, const std::vector<int64_t>& shape) const {
        auto it = where_.find(name);
        if (it == where_.end()) throw std::runtime_error("the checkpoint has no tensor " + name);
        auto t = files_[it->second]->read(name);
        if (t->shape != shape) throw std::runtime_error(name + " is " + shape_string(t->shape) + " in the checkpoint, the model expects " + shape_string(shape));
        return t->to(Device::CPU, DType::Float32);
    }

private:
    std::vector<std::shared_ptr<io::SafeTensorsFile>> files_;
    std::unordered_map<std::string, size_t> where_;
};

inline void copy_into(const std::shared_ptr<Tensor>& dst, const std::shared_ptr<Tensor>& src) {
    if (dst->size() != src->size()) throw std::logic_error("copy_into: sizes differ");
    std::memcpy(dst->data_ptr<float>(), src->data_ptr<float>(), (size_t)dst->size() * sizeof(float));
    dst->storage->bump_version();
}

// Where each output of a rotary projection lives in a Hugging Face weight. There the two halves of a rotated pair
// are head_dim / 2 apart, here they are neighbours.
inline std::vector<int64_t> rotary_order(int64_t heads, int64_t head_dim) {
    std::vector<int64_t> order;
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t j = 0; j < head_dim / 2; ++j)
            for (int64_t half = 0; half < 2; ++half) order.push_back(h * head_dim + half * (head_dim / 2) + j);
    return order;
}

// A torch Linear ([out, in], rows reordered) into a Linear here ([in, out])
inline void load_linear(Linear& l, const Checkpoint& c, const std::string& name, const std::vector<int64_t>& order = {}) {
    int64_t in = l.W->shape[0], out = l.W->shape[1];
    auto w = c.read(name + ".weight", {out, in});
    const float* src = w->data_ptr<float>();
    float* dst = l.W->data_ptr<float>();
    parallel_for(0, in, std::max<int64_t>(1, 32768 / std::max<int64_t>(1, out)), [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i)
            for (int64_t o = 0; o < out; ++o) dst[i * out + o] = src[(order.empty() ? o : order[(size_t)o]) * in + i];
    });
    l.W->storage->bump_version();
    if (!l.b) {
        if (c.has(name + ".bias")) throw std::runtime_error(name + " has a bias in the checkpoint but not in the model");
        return;
    }
    auto bias = c.read(name + ".bias", {out});
    for (int64_t o = 0; o < out; ++o) l.b->data_ptr<float>()[o] = bias->data_ptr<float>()[order.empty() ? o : order[(size_t)o]];
    l.b->storage->bump_version();
}

}  // namespace pretrained

// The model a Llama-family config.json describes. max_seq_len limits the positions (0: the config's limit).
inline ModernTransformerConfig llama_config(const json::Value& c, int64_t max_seq_len = 0) {
    const std::string type = c.get_string("model_type", "llama");
    if (type != "llama" && type != "mistral" && type != "qwen2") throw std::runtime_error("load_llama: model type \"" + type + "\" is not supported (llama, mistral and qwen2 are)");
    const std::string act = c.get_string("hidden_act", "silu");
    if (act != "silu" && act != "swish") throw std::runtime_error("load_llama: activation \"" + act + "\" is not supported");
    ModernTransformerConfig m;
    m.vocab_size = (int)c.at("vocab_size").as_int();
    m.d_model = (int)c.at("hidden_size").as_int();
    m.hidden_dim = (int)c.at("intermediate_size").as_int();
    m.n_layers = (int)c.at("num_hidden_layers").as_int();
    m.n_heads = (int)c.at("num_attention_heads").as_int();
    m.n_kv_heads = (int)c.get_int("num_key_value_heads", m.n_heads);
    m.head_dim = (int)c.get_int("head_dim", 0);
    m.norm_eps = (float)c.get("rms_norm_eps", 1e-6);
    m.tie_embeddings = c.get_bool("tie_word_embeddings", false);
    m.qkv_bias = c.get_bool("attention_bias", type == "qwen2");
    m.out_bias = type == "qwen2" ? false : c.get_bool("attention_bias", false);
    m.ffn_bias = c.get_bool("mlp_bias", false);

    // Newer files keep all of RoPE in rope_parameters, older ones in rope_theta and rope_scaling
    const json::Value* rope = c.find("rope_parameters");
    if (!rope || !rope->is_object()) rope = c.find("rope_scaling");
    m.rope_theta = (float)c.get("rope_theta", rope && rope->is_object() ? rope->get("rope_theta", 10000.0) : 10000.0);
    if (rope && rope->is_object()) {
        const std::string kind = rope->get_string("rope_type", rope->get_string("type", "default"));
        m.rope_scaling.factor = (float)rope->get("factor", 1.0);
        if (kind == "linear") {
            m.rope_scaling.type = RopeScaling::Type::Linear;
        } else if (kind == "llama3") {
            m.rope_scaling.type = RopeScaling::Type::Llama3;
            m.rope_scaling.low_freq_factor = (float)rope->get("low_freq_factor", 1.0);
            m.rope_scaling.high_freq_factor = (float)rope->get("high_freq_factor", 4.0);
            m.rope_scaling.original_max_positions = (float)rope->get("original_max_position_embeddings", 8192.0);
        } else if (kind != "default") {
            throw std::runtime_error("load_llama: RoPE scaling \"" + kind + "\" is not supported (linear and llama3 are)");
        }
    }
    int64_t positions = c.get_int("max_position_embeddings", 2048);
    // Within its window, sliding-window attention is ordinary attention
    if (c.get_bool("use_sliding_window", type == "mistral")) positions = std::min(positions, c.get_int("sliding_window", positions));
    m.max_seq_len = (int)(max_seq_len > 0 ? std::min(max_seq_len, positions) : positions);
    return m;
}

// A Llama, Mistral or Qwen2 model in float32 on the CPU
inline ModernTransformer load_llama(const std::string& dir, int64_t max_seq_len = 0) {
    auto config = llama_config(json::parse_file(dir + "/config.json"), max_seq_len);
    pretrained::Checkpoint c(dir);
    SkipInitGuard skip_init;
    ModernTransformer m(config);
    const int64_t D = config.d_model;
    pretrained::copy_into(m.token_emb.weight, c.read("model.embed_tokens.weight", {config.vocab_size, D}));
    if (m.lm_head) pretrained::copy_into(m.lm_head, c.read("lm_head.weight", {config.vocab_size, D}));
    pretrained::copy_into(m.rms_f.gamma, c.read("model.norm.weight", {D}));
    for (int i = 0; i < config.n_layers; ++i) {
        auto& b = *m.blocks[(size_t)i];
        const std::string p = "model.layers." + std::to_string(i) + ".";
        pretrained::copy_into(b.rms1.gamma, c.read(p + "input_layernorm.weight", {D}));
        pretrained::copy_into(b.rms2.gamma, c.read(p + "post_attention_layernorm.weight", {D}));
        pretrained::load_linear(b.attn.W_q, c, p + "self_attn.q_proj", pretrained::rotary_order(m.n_heads, m.head_dim));
        pretrained::load_linear(b.attn.W_k, c, p + "self_attn.k_proj", pretrained::rotary_order(m.n_kv_heads, m.head_dim));
        pretrained::load_linear(b.attn.W_v, c, p + "self_attn.v_proj");
        pretrained::load_linear(b.attn.W_o, c, p + "self_attn.o_proj");
        pretrained::load_linear(b.ffn.w_gate, c, p + "mlp.gate_proj");
        pretrained::load_linear(b.ffn.w_up, c, p + "mlp.up_proj");
        pretrained::load_linear(b.ffn.w_down, c, p + "mlp.down_proj");
    }
    return m;
}

// Writes config.json and model.safetensors, in float32, into an existing folder
inline void save_llama(ModernTransformer& m, const std::string& dir) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> out;
    auto host = [](const std::shared_ptr<Tensor>& t) { return make_contiguous(t)->to(Device::CPU, DType::Float32); };
    auto save_linear = [&](Linear& l, const std::string& name, const std::vector<int64_t>& order = {}) {
        if (l.lora_A) throw std::runtime_error("save_llama: merge the LoRA adapters first");
        int64_t in = l.W->shape[0], n_out = l.W->shape[1];
        auto w = host(l.W);
        auto t = std::make_shared<Tensor>(std::vector<int64_t>{n_out, in}, Device::CPU, DType::Float32, false);
        for (int64_t o = 0; o < n_out; ++o) {
            float* row = t->data_ptr<float>() + (order.empty() ? o : order[(size_t)o]) * in;
            for (int64_t i = 0; i < in; ++i) row[i] = w->data_ptr<float>()[i * n_out + o];
        }
        out[name + ".weight"] = t;
        if (!l.b) return;
        auto b = host(l.b);
        auto tb = std::make_shared<Tensor>(std::vector<int64_t>{n_out}, Device::CPU, DType::Float32, false);
        for (int64_t o = 0; o < n_out; ++o) tb->data_ptr<float>()[order.empty() ? o : order[(size_t)o]] = b->data_ptr<float>()[o];
        out[name + ".bias"] = tb;
    };
    const auto& cfg = m.config;
    // The only bias layouts config.json can express
    bool qwen = cfg.qkv_bias && !cfg.out_bias;
    if (!cfg.qkv_bias && cfg.out_bias) throw std::runtime_error("save_llama: a bias on the attention output alone has no Hugging Face equivalent");
    if (qwen && cfg.ffn_bias) throw std::runtime_error("save_llama: these biases have no Hugging Face equivalent");
    out["model.embed_tokens.weight"] = host(m.token_emb.weight);
    if (m.lm_head) out["lm_head.weight"] = host(m.lm_head);
    out["model.norm.weight"] = host(m.rms_f.gamma);
    for (size_t i = 0; i < m.blocks.size(); ++i) {
        auto& b = *m.blocks[i];
        const std::string p = "model.layers." + std::to_string(i) + ".";
        out[p + "input_layernorm.weight"] = host(b.rms1.gamma);
        out[p + "post_attention_layernorm.weight"] = host(b.rms2.gamma);
        save_linear(b.attn.W_q, p + "self_attn.q_proj", pretrained::rotary_order(m.n_heads, m.head_dim));
        save_linear(b.attn.W_k, p + "self_attn.k_proj", pretrained::rotary_order(m.n_kv_heads, m.head_dim));
        save_linear(b.attn.W_v, p + "self_attn.v_proj");
        save_linear(b.attn.W_o, p + "self_attn.o_proj");
        save_linear(b.ffn.w_gate, p + "mlp.gate_proj");
        save_linear(b.ffn.w_up, p + "mlp.up_proj");
        save_linear(b.ffn.w_down, p + "mlp.down_proj");
    }
    io::save_safetensors(dir + "/model.safetensors", out);

    std::ostringstream os;
    os.precision(9);
    auto yes = [](bool v) { return v ? "true" : "false"; };
    os << "{\n  \"architectures\": [\"" << (qwen ? "Qwen2ForCausalLM" : "LlamaForCausalLM") << "\"],\n"
       << "  \"model_type\": \"" << (qwen ? "qwen2" : "llama") << "\",\n"
       << "  \"hidden_act\": \"silu\",\n"
       << "  \"vocab_size\": " << m.vocab_size << ",\n"
       << "  \"hidden_size\": " << m.d_model << ",\n"
       << "  \"intermediate_size\": " << m.blocks[0]->ffn.w_gate.W->shape[1] << ",\n"
       << "  \"num_hidden_layers\": " << m.blocks.size() << ",\n"
       << "  \"num_attention_heads\": " << m.n_heads << ",\n"
       << "  \"num_key_value_heads\": " << m.n_kv_heads << ",\n"
       << "  \"head_dim\": " << m.head_dim << ",\n"
       << "  \"max_position_embeddings\": " << m.max_seq_len << ",\n"
       << "  \"rms_norm_eps\": " << cfg.norm_eps << ",\n"
       << "  \"rope_theta\": " << cfg.rope_theta << ",\n";
    if (cfg.rope_scaling.type == RopeScaling::Type::Linear) {
        os << "  \"rope_scaling\": {\"rope_type\": \"linear\", \"factor\": " << cfg.rope_scaling.factor << "},\n";
    } else if (cfg.rope_scaling.type == RopeScaling::Type::Llama3) {
        os << "  \"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": " << cfg.rope_scaling.factor
           << ", \"low_freq_factor\": " << cfg.rope_scaling.low_freq_factor << ", \"high_freq_factor\": " << cfg.rope_scaling.high_freq_factor
           << ", \"original_max_position_embeddings\": " << (int64_t)cfg.rope_scaling.original_max_positions << "},\n";
    }
    os << "  \"attention_bias\": " << yes(cfg.qkv_bias) << ",\n"
       << "  \"mlp_bias\": " << yes(cfg.ffn_bias) << ",\n"
       << "  \"tie_word_embeddings\": " << yes(!m.lm_head) << ",\n"
       << "  \"torch_dtype\": \"float32\"\n}\n";
    std::ofstream config(dir + "/config.json");
    if (!config) throw std::runtime_error("save_llama: cannot write into " + dir);
    config << os.str();
}

// GPT-2 (any size) in float32 on the CPU
inline GPT load_gpt2(const std::string& dir) {
    auto cfg = json::parse_file(dir + "/config.json");
    const std::string act = cfg.get_string("activation_function", "gelu_new");
    if (act != "gelu_new" && act != "gelu_pytorch_tanh") throw std::runtime_error("load_gpt2: activation \"" + act + "\" is not supported");
    if (std::abs(cfg.get("layer_norm_epsilon", 1e-5) - 1e-5) > 1e-9) throw std::runtime_error("load_gpt2: layer_norm_epsilon must be 1e-5");
    const int64_t V = cfg.at("vocab_size").as_int(), P = cfg.at("n_positions").as_int(), D = cfg.at("n_embd").as_int();
    const int n_layers = (int)cfg.at("n_layer").as_int();
    if (cfg.get_int("n_inner", 4 * D) != 4 * D) throw std::runtime_error("load_gpt2: n_inner must be 4 * n_embd");
    pretrained::Checkpoint c(dir);
    const std::string root = c.has("wte.weight") ? "" : "transformer.";
    SkipInitGuard skip_init;
    GPT m((int)V, (int)P, (int)D, (int)cfg.at("n_head").as_int(), n_layers);
    pretrained::copy_into(m.token_emb.weight, c.read(root + "wte.weight", {V, D}));
    pretrained::copy_into(m.pos_emb.weight, c.read(root + "wpe.weight", {P, D}));
    auto load_norm = [&](LayerNorm& n, const std::string& name) {
        pretrained::copy_into(n.gamma, c.read(name + ".weight", {D}));
        pretrained::copy_into(n.beta, c.read(name + ".bias", {D}));
    };
    // GPT-2 keeps its weights [in, out], as here
    auto load_conv1d = [&](Linear& l, const std::string& name) {
        pretrained::copy_into(l.W, c.read(name + ".weight", l.W->shape));
        pretrained::copy_into(l.b, c.read(name + ".bias", {l.W->shape[1]}));
    };
    load_norm(m.ln_f, root + "ln_f");
    for (int i = 0; i < n_layers; ++i) {
        auto& b = *m.blocks[(size_t)i];
        const std::string p = root + "h." + std::to_string(i) + ".";
        load_norm(b.ln1, p + "ln_1");
        load_norm(b.ln2, p + "ln_2");
        load_conv1d(b.attn.W_o, p + "attn.c_proj");
        load_conv1d(b.ffn.fc1, p + "mlp.c_fc");
        load_conv1d(b.ffn.fc2, p + "mlp.c_proj");
        // One matrix holds the query, key and value projections side by side
        auto w = c.read(p + "attn.c_attn.weight", {D, 3 * D}), bias = c.read(p + "attn.c_attn.bias", {3 * D});
        Linear* parts[3] = {&b.attn.W_q, &b.attn.W_k, &b.attn.W_v};
        for (int k = 0; k < 3; ++k) {
            for (int64_t r = 0; r < D; ++r) std::memcpy(parts[k]->W->data_ptr<float>() + r * D, w->data_ptr<float>() + r * 3 * D + k * D, (size_t)D * sizeof(float));
            std::memcpy(parts[k]->b->data_ptr<float>(), bias->data_ptr<float>() + k * D, (size_t)D * sizeof(float));
            parts[k]->W->storage->bump_version();
            parts[k]->b->storage->bump_version();
        }
    }
    return m;
}
