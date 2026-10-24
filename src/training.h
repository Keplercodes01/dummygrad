#pragma once
// Training workflow utilities: resumable checkpoints and metric logging.

#include "tensor.h"
#include "module.h"
#include "optimizers.h"
#include "serialization.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Everything besides weights and optimizer state that a run needs to resume.
struct TrainingState {
    int64_t step = 0;
    // Anything else to restore exactly: epoch, scheduler step, best validation loss, ...
    std::map<std::string, double> scalars;
};

namespace training_detail {

inline void write_checkpoint(const std::string& path, std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors) {
    // Write to a temporary file and rename over the target, so a crash mid-write never leaves
    // a truncated checkpoint where the previous good one was.
    const std::string tmp = path + ".tmp";
    io::save_safetensors(tmp, tensors);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("save_checkpoint: could not move " + tmp + " to " + path);
    }
}

template <typename Optimizer>
inline void add_optimizer_state(std::unordered_map<std::string, std::shared_ptr<Tensor>>& tensors,
                                const Optimizer* optimizer, Module& model) {
    if (!optimizer) return;
    for (auto& [name, t] : optimizer->state_dict(model.named_parameters())) tensors["optim." + name] = t;
}

inline std::unordered_map<std::string, std::shared_ptr<Tensor>> with_prefix(
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& all, const std::string& prefix) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> out;
    for (const auto& [name, t] : all) {
        if (name.compare(0, prefix.size(), prefix) == 0) out[name.substr(prefix.size())] = t;
    }
    return out;
}

}  // namespace training_detail

// Saves everything needed to resume training into one SafeTensors file: model parameters and
// buffers ("model.*"), optimizer state ("optim.*"), the step and scalars ("meta.*"). The file is
// written atomically. Works with Adam and SGD (pass nullptr to save the model alone).
template <typename Optimizer>
inline void save_checkpoint(const std::string& path, Module& model, const Optimizer* optimizer,
                            const TrainingState& state = {}) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    for (auto& [name, t] : model.state_dict()) tensors["model." + name] = t;
    training_detail::add_optimizer_state(tensors, optimizer, model);
    tensors["meta.step"] = io::encode_int64(state.step);
    for (const auto& [key, value] : state.scalars) tensors["meta.scalar." + key] = io::encode_double(value);
    training_detail::write_checkpoint(path, tensors);
}

inline void save_checkpoint(const std::string& path, Module& model, const TrainingState& state = {}) {
    save_checkpoint<Adam>(path, model, nullptr, state);
}

// Restores a checkpoint written by save_checkpoint into an existing model (and optimizer) and
// returns the saved step and scalars. strict applies to the model's tensors, as in
// Module::load_state_dict.
template <typename Optimizer>
inline TrainingState load_checkpoint(const std::string& path, Module& model, Optimizer* optimizer, bool strict = true) {
    auto all = io::load_safetensors(path, Device::CPU);
    model.load_state_dict(training_detail::with_prefix(all, "model."), strict);
    if (optimizer) optimizer->load_state_dict(model.named_parameters(), training_detail::with_prefix(all, "optim."));

    TrainingState state;
    auto meta = training_detail::with_prefix(all, "meta.");
    auto step = meta.find("step");
    if (step != meta.end()) state.step = io::decode_int64(step->second);
    for (const auto& [name, t] : training_detail::with_prefix(meta, "scalar.")) state.scalars[name] = io::decode_double(t);
    return state;
}

inline TrainingState load_checkpoint(const std::string& path, Module& model, bool strict = true) {
    return load_checkpoint<Adam>(path, model, nullptr, strict);
}

// Logs metrics to a CSV file in long format (step,metric,value), so any metric can be logged at
// any step: training loss every step, validation loss every few hundred. Load it with
// pandas.read_csv(...).pivot(index="step", columns="metric", values="value").
//
//   MetricsLogger logger("runs/exp1/metrics.csv");
//   logger.log(step, {{"train_loss", loss}, {"lr", lr}});
class MetricsLogger {
public:
    // append=true continues an existing log (when resuming); otherwise the file is replaced.
    // echo prints each call as one line: "step 100 | train_loss 2.3104 | lr 0.001".
    explicit MetricsLogger(const std::string& path, bool echo = true, bool append = false) : echo_(echo) {
        bool exists = std::ifstream(path).good();
        file_.open(path, append ? std::ios::app : std::ios::trunc);
        if (!file_) throw std::runtime_error("MetricsLogger: cannot open " + path);
        if (!append || !exists) file_ << "step,metric,value\n";
        file_ << std::setprecision(9);
    }

    void log(int64_t step, const std::vector<std::pair<std::string, double>>& metrics) {
        for (const auto& [name, value] : metrics) file_ << step << ',' << name << ',' << value << '\n';
        file_.flush();
        if (echo_) {
            std::cout << "step " << step;
            for (const auto& [name, value] : metrics) std::cout << " | " << name << ' ' << value;
            std::cout << '\n';
        }
    }

private:
    std::ofstream file_;
    bool echo_;
};
