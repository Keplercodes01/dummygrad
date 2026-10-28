#pragma once
// Module: base class for layers and models.
//
// A module lists its parameters, buffers and child modules in visit(); everything else —
// parameters(), named_parameters(), to(), train()/eval(), saving and loading — is derived
// from that one declaration.
//
//   struct MyBlock : Module {
//       Linear proj{64, 64};
//       RMSNorm norm{64};
//       void visit(ModuleVisitor& v) override {
//           v.module("proj", proj);
//           v.module("norm", norm);
//       }
//       TensorPtr forward(const TensorPtr& x) { return proj.forward(norm.forward(x)); }
//   };
//
// visit() is called on demand rather than recording pointers at construction, so modules stay
// safe to move (models are returned by value, e.g. from MultiGPURunner factories). Copying is
// disabled: tensors and submodules are held by shared_ptr, so a copy would share its parameters
// with the original, and moving one to another device would move parts of the other.

#include "tensor.h"
#include "ops.h"
#include "serialization.h"

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class Module;

struct ModuleVisitor {
    virtual ~ModuleVisitor() = default;
    // A trainable tensor. Passed by reference so to() can replace it with a moved copy.
    virtual void parameter(const std::string& name, std::shared_ptr<Tensor>& t) = 0;
    // A non-trainable tensor that moves with the module. Persistent buffers are saved in
    // checkpoints; non-persistent ones (e.g. recomputable RoPE tables) are not.
    virtual void buffer(const std::string& name, std::shared_ptr<Tensor>& t, bool persistent = true) = 0;
    virtual void module(const std::string& name, Module& m) = 0;
};

using NamedTensors = std::vector<std::pair<std::string, std::shared_ptr<Tensor>>>;

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;

    // Declare parameters, buffers and child modules. Names compose into dotted paths
    // ("blocks.0.attn.W_q.weight").
    virtual void visit(ModuleVisitor& v) = 0;

    // All parameters, recursively, in declaration order. A tensor reachable under several
    // names (tied weights) is listed once.
    std::vector<std::shared_ptr<Tensor>> parameters() {
        std::vector<std::shared_ptr<Tensor>> out;
        for (auto& [name, t] : named_parameters()) out.push_back(t);
        return out;
    }

    NamedTensors named_parameters() {
        NamedTensors out;
        std::unordered_set<Tensor*> seen;
        walk([&](const std::string& name, std::shared_ptr<Tensor>& t, Kind kind) {
            if (kind == Kind::Parameter && t && seen.insert(t.get()).second) out.emplace_back(name, t);
        });
        return out;
    }

    NamedTensors named_buffers() {
        NamedTensors out;
        walk([&](const std::string& name, std::shared_ptr<Tensor>& t, Kind kind) {
            if (kind != Kind::Parameter && t) out.emplace_back(name, t);
        });
        return out;
    }

    // Everything a checkpoint holds: parameters and persistent buffers.
    NamedTensors state_dict() {
        NamedTensors out;
        std::unordered_set<Tensor*> seen;
        walk([&](const std::string& name, std::shared_ptr<Tensor>& t, Kind kind) {
            if (kind == Kind::Buffer) {
                if (t) out.emplace_back(name, t);
            } else if (kind == Kind::Parameter && t && seen.insert(t.get()).second) {
                out.emplace_back(name, t);
            }
        });
        return out;
    }

    int64_t num_parameters() {
        int64_t n = 0;
        for (auto& t : parameters()) n += t->size();
        return n;
    }

    // Move parameters and buffers to device, each keeping its dtype. requires_grad is preserved, so
    // frozen parameters stay frozen, and tied references keep pointing at one shared tensor.
    void to(Device device) { move_to(device, nullptr); }
    // Move and convert: floating-point tensors become dtype (Float32, Float16 or BFloat16); others
    // (Int8) keep theirs.
    void to(Device device, DType dtype) { move_to(device, &dtype); }
    void cuda() { to(Device::CUDA); }
    void cpu() { to(Device::CPU); }
    void mps() { to(Device::MPS); }

private:
    void move_to(Device device, const DType* dtype) {
        std::unordered_map<Tensor*, std::shared_ptr<Tensor>> moved;
        walk([&](const std::string&, std::shared_ptr<Tensor>& t, Kind) {
            if (!t) return;
            bool is_float = t->dtype == DType::Float32 || t->dtype == DType::Float16 || t->dtype == DType::BFloat16;
            DType target = (dtype && is_float) ? *dtype : t->dtype;
            if (t->device == device && t->dtype == target) return;
            auto it = moved.find(t.get());
            if (it == moved.end()) {
                auto copy = t->to(device, target);
                copy->requires_grad = t->requires_grad;
                it = moved.emplace(t.get(), copy).first;
            }
            t = it->second;
        });
    }

public:

    // Train/eval mode for this module and all children (for layers like dropout that differ).
    void train(bool mode = true) {
        training_ = mode;
        walk(nullptr, [mode](Module& m) { m.training_ = mode; });
    }
    void eval() { train(false); }
    bool is_training() const { return training_; }

    // Freeze (false) or unfreeze (true) every parameter.
    void requires_grad_(bool requires_grad) {
        for (auto& t : parameters()) t->requires_grad = requires_grad;
    }

    void zero_grad() {
        for (auto& t : parameters()) t->grad = nullptr;
    }

    // Copy values from a checkpoint into the existing tensors, keeping their identity (optimizer
    // state stays attached). Shape mismatches always throw. With strict, missing or unexpected
    // names throw too; otherwise they are skipped and the rest is loaded.
    void load_state_dict(const std::unordered_map<std::string, std::shared_ptr<Tensor>>& state, bool strict = true) {
        auto own = state_dict();
        std::vector<std::string> missing, unexpected, mismatched;
        std::unordered_set<std::string> own_names;
        for (auto& [name, dst] : own) {
            own_names.insert(name);
            auto it = state.find(name);
            if (it == state.end() || !it->second) {
                missing.push_back(name);
            } else if (it->second->shape != dst->shape) {
                mismatched.push_back(name + " (checkpoint " + shape_str(it->second->shape) + ", model " + shape_str(dst->shape) + ")");
            }
        }
        for (auto& [name, t] : state) {
            if (!own_names.count(name)) unexpected.push_back(name);
        }

        std::string err;
        if (!mismatched.empty()) err += "shape mismatch: " + join(mismatched) + "\n";
        if (strict && !missing.empty()) err += "missing from checkpoint: " + join(missing) + "\n";
        if (strict && !unexpected.empty()) err += "unexpected in checkpoint: " + join(unexpected) + "\n";
        if (!err.empty()) throw std::runtime_error("load_state_dict failed (pass strict=false to load a subset):\n" + err);

        for (auto& [name, dst] : own) {
            auto it = state.find(name);
            if (it == state.end() || !it->second) continue;
            auto src = it->second->to(dst->device, dst->dtype);
            ops::make_contiguous(*dst, *src);  // copies into dst's existing storage
        }
    }

    void save_safetensors(const std::string& path) {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
        for (auto& [name, t] : state_dict()) tensors[name] = t;
        io::save_safetensors(path, tensors);
    }

    void load_safetensors(const std::string& path, bool strict = true) {
        load_state_dict(io::load_safetensors(path, Device::CPU), strict);
    }

protected:
    bool training_ = true;

private:
    enum class Kind { Parameter, Buffer, NonPersistentBuffer };
    using TensorFn = std::function<void(const std::string&, std::shared_ptr<Tensor>&, Kind)>;
    using ModuleFn = std::function<void(Module&)>;

    struct Walker : ModuleVisitor {
        std::string prefix;
        const TensorFn& on_tensor;
        const ModuleFn& on_module;
        Walker(const TensorFn& t, const ModuleFn& m) : on_tensor(t), on_module(m) {}

        void parameter(const std::string& name, std::shared_ptr<Tensor>& t) override {
            if (on_tensor) on_tensor(prefix + name, t, Kind::Parameter);
        }
        void buffer(const std::string& name, std::shared_ptr<Tensor>& t, bool persistent) override {
            if (on_tensor) on_tensor(prefix + name, t, persistent ? Kind::Buffer : Kind::NonPersistentBuffer);
        }
        void module(const std::string& name, Module& m) override {
            if (on_module) on_module(m);
            std::string saved = prefix;
            prefix += name + ".";
            m.visit(*this);
            prefix = saved;
        }
    };

    void walk(const TensorFn& on_tensor, const ModuleFn& on_module = nullptr) {
        Walker w(on_tensor, on_module);
        visit(w);
    }

    static std::string shape_str(const std::vector<int64_t>& shape) {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
        os << "]";
        return os.str();
    }

    static std::string join(const std::vector<std::string>& names, size_t limit = 10) {
        std::string out;
        for (size_t i = 0; i < names.size() && i < limit; ++i) out += (i ? ", " : "") + names[i];
        if (names.size() > limit) out += ", ... (" + std::to_string(names.size() - limit) + " more)";
        return out;
    }
};
