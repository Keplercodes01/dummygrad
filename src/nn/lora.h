#pragma once
// Low-rank adapters (LoRA, Hu et al. 2021): finetuning a model by training a small x A B beside some of its
// Linear layers, everything else frozen.
//
//   int n = add_lora(model, 8, 16.0f, {"W_q", "W_v"});   // rank, alpha, layers
//   Adam opt(1e-4f);
//   auto params = trainable_parameters(model);           // just the adapters
//   ...
//   save_lora(model, "adapters.safetensors");
//   merge_lora(model);                                   // into the weights

#include "linear.h"
#include "module.h"
#include "serialization.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Calls fn with the name of every Linear inside a module
inline void for_each_linear(Module& model, const std::function<void(const std::string&, Linear&)>& fn) {
    model.for_each_module([&](const std::string& name, Module& m) {
        if (auto* l = dynamic_cast<Linear*>(&m)) fn(name, *l);
    });
}

// Freezes the model and gives every Linear whose name contains one of targets (all of them when empty) an adapter.
// Returns how many got one. Move the model to its device before or after, either works.
inline int add_lora(Module& model, int rank, float alpha, const std::vector<std::string>& targets = {}, float dropout_p = 0.0f) {
    model.requires_grad_(false);
    int count = 0;
    for_each_linear(model, [&](const std::string& name, Linear& l) {
        bool wanted = targets.empty();
        for (const auto& t : targets) wanted = wanted || name.find(t) != std::string::npos;
        if (!wanted) return;
        l.enable_lora(rank, alpha, dropout_p);
        ++count;
    });
    if (count == 0) throw std::invalid_argument("add_lora: no Linear layer matches");
    return count;
}

// Adds every adapter into its weight and removes it. The weights stay frozen; requires_grad_(true) frees them.
inline void merge_lora(Module& model) {
    for_each_linear(model, [](const std::string&, Linear& l) { l.merge_lora(); });
}

// The parameters that want gradients: what to give an optimizer
inline std::vector<std::shared_ptr<Tensor>> trainable_parameters(Module& model) {
    std::vector<std::shared_ptr<Tensor>> out;
    for (auto& p : model.parameters()) if (p->requires_grad) out.push_back(p);
    return out;
}

namespace lora_detail {
inline bool is_adapter(const std::string& name) {
    auto ends_with = [&](const std::string& s) { return name.size() >= s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0; };
    return ends_with("lora_A") || ends_with("lora_B");
}
}  // namespace lora_detail

// The adapters alone: a few MB for a model of GBs. The scale (alpha / rank) is saved with each.
inline void save_lora(Module& model, const std::string& path) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    for_each_linear(model, [&](const std::string& name, Linear& l) {
        if (!l.lora_A) return;
        tensors[name + ".lora_A"] = l.lora_A;
        tensors[name + ".lora_B"] = l.lora_B;
        auto scale = std::make_shared<Tensor>(std::vector<int64_t>{1}, Device::CPU, DType::Float32, false);
        scale->fill_(l.lora_scale);
        tensors[name + ".lora_scale"] = scale;
    });
    if (tensors.empty()) throw std::invalid_argument("save_lora: the model has no adapters");
    io::save_safetensors(path, tensors);
}

// Adapters written by save_lora, into a model with or without adapters of its own. Freezes everything else.
inline void load_lora(Module& model, const std::string& path) {
    auto tensors = io::load_safetensors(path);
    model.requires_grad_(false);
    size_t used = 0;
    for_each_linear(model, [&](const std::string& name, Linear& l) {
        auto a = tensors.find(name + ".lora_A"), b = tensors.find(name + ".lora_B"), s = tensors.find(name + ".lora_scale");
        if (a == tensors.end()) return;
        if (b == tensors.end() || s == tensors.end() || a->second->ndim() != 2 || a->second->shape[0] != l.W->shape[0] ||
            b->second->shape != std::vector<int64_t>{a->second->shape[1], l.W->shape[1]}) {
            throw std::runtime_error("load_lora: the adapter of " + name + " does not fit the layer");
        }
        int rank = (int)a->second->shape[1];
        l.enable_lora(rank, 1.0f, l.lora_dropout);
        ops::make_contiguous(*l.lora_A, *a->second->to(l.lora_A->device));
        ops::make_contiguous(*l.lora_B, *b->second->to(l.lora_B->device));
        l.lora_scale = s->second->item();
        used += 3;
    });
    if (used != tensors.size()) throw std::runtime_error("load_lora: some adapters belong to layers the model does not have");
}
