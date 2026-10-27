// Training-step benchmark: ModernTransformer (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) at a
// few sizes, timed as forward / backward / optimizer, with host<->device traffic per step.
// bench/bench_torch.py runs the same architecture in PyTorch, timed the same way.
//
// Each configuration runs 10 warmup steps, then timed steps until it has at least 50 and at least
// 2 seconds' worth, and reports the median step and its 10th-90th percentile spread. Small models
// are paced by the host CPU, which on shared cloud machines varies from run to run, so a short
// mean is not a reliable number there.
//
//   ./dg_bench                  all configurations
//   ./dg_bench small            one configuration (tiny, small, medium, large)
//   ./dg_bench --profile        also print the most expensive forward and backward ops per configuration
//   ./dg_bench --fp16           matmuls and attention on fp16 tensor cores (AutocastGuard + GradScaler)
//   ./dg_bench --bf16           matmuls and attention on bf16 tensor cores (Ampere or newer)
//   ./dg_bench --worker-threads run backward on the engine's worker pool (the GPU default is the calling thread)
//   ./dg_bench --seconds 5      time each configuration for at least 5 seconds (default 2)
//   ./dg_bench --json out.json  also write the results as JSON (bench/compare.py reads it)

#include "dummy_core.h"
#include "allocator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct BenchConfig {
    std::string name;
    int vocab, d_model, n_layers, n_heads, n_kv_heads, batch, seq_len;
};

static const std::vector<BenchConfig> kConfigs = {
    {"tiny", 512, 128, 2, 4, 2, 8, 64},
    {"small", 4096, 256, 4, 8, 4, 16, 128},
    {"medium", 8192, 512, 6, 8, 8, 8, 256},
    {"large", 32000, 768, 12, 12, 4, 8, 512},
};

static void sync(Device device) {
#ifdef USE_CUDA
    if (device == Device::CUDA) cudaDeviceSynchronize();
#else
    (void)device;
#endif
}

// q-quantile of v (0 <= q <= 1), interpolating between neighbouring order statistics
static double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double pos = q * (double)(v.size() - 1);
    size_t lo = (size_t)pos;
    size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - (double)lo);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string only;
    bool profile = false;
    double min_seconds = 2.0;
    std::string json_path;
    DType compute = DType::Float32;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--profile") profile = true;
        else if (arg == "--seconds" && i + 1 < argc) min_seconds = std::stod(argv[++i]);
        else if (arg == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (arg == "--fp16") compute = DType::Float16;
        else if (arg == "--bf16") compute = DType::BFloat16;
        else if (arg == "--worker-threads") set_backward_execution(BackwardExecution::ThreadPool);
        else only = arg;
    }
    const char* compute_name = compute == DType::Float16 ? "fp16" : compute == DType::BFloat16 ? "bf16" : "fp32";
    std::vector<std::string> profiles;
    Device device = Device::CPU;
#ifdef USE_CUDA
    int gpus = 0;
    if (cudaGetDeviceCount(&gpus) == cudaSuccess && gpus > 0) device = Device::CUDA;
#endif
    const int warmup = 10, min_steps = 50, max_steps = 5000;
    BackwardExecution mode = backward_execution();
    if (mode == BackwardExecution::Auto) {
        mode = device == Device::CPU ? BackwardExecution::ThreadPool : BackwardExecution::CallingThread;
    }
    std::string device_name = "CPU";
#ifdef USE_CUDA
    if (device == Device::CUDA) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) device_name = prop.name;
    }
#endif
    std::printf("dummygrad %s benchmark on %s, %s autocast, backward on %s\n", DUMMYGRAD_VERSION,
                device_name.c_str(), compute_name,
                mode == BackwardExecution::CallingThread ? "the calling thread" : "worker threads");
    std::printf("median of >= %d steps and >= %.0f s per config after %d warmup; p10-p90 is the spread of step times\n\n",
                min_steps, min_seconds, warmup);
    std::printf("| config | params | ms/step | p10-p90 | tokens/s | forward | backward | optimizer | steps | peak MB | copies/step | MB/step |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    std::ostringstream json;
    json << "{\"framework\": \"dummygrad\", \"version\": \"" << DUMMYGRAD_VERSION << "\", \"device\": \"" << device_name
         << "\", \"precision\": \"" << compute_name << "\", \"configs\": [";
    bool first_json = true;

    for (const auto& c : kConfigs) {
        if (!only.empty() && only != c.name) continue;
        // Each configuration starts from an empty cache, so earlier ones don't shape its memory
        CUDACachingAllocator::get().empty_cache();
        manual_seed(0);
        ModernTransformer model(c.vocab, c.seq_len, c.d_model, c.n_heads, c.n_layers, c.n_kv_heads);
        model.to(device);
        auto params = model.parameters();
        Adam optimizer(3e-4f);
        // Loss scaling is part of the fp16 step cost; its unscale pass is timed with the optimizer
        GradScaler scaler(65536.0f, 2.0f, 0.5f, 2000, compute == DType::Float16);

        std::vector<float> in(c.batch * c.seq_len), tg(c.batch * c.seq_len);
        for (size_t i = 0; i < in.size(); ++i) {
            in[i] = (float)((i * 7) % c.vocab);
            tg[i] = (float)((i * 7 + 1) % c.vocab);
        }
        auto make = [&](const std::vector<float>& v) {
            auto t = std::make_shared<Tensor>(std::vector<int64_t>{c.batch, c.seq_len}, Device::CPU, DType::Float32, false);
            t->copy_from_vector(v);
            auto d = t->to(device);
            d->requires_grad = false;
            return d;
        };
        auto inputs = make(in), targets = make(tg);
        auto forward_loss = [&] {
            AutocastGuard autocast(compute);
            return fused_cross_entropy(model.forward(inputs), targets);
        };

        std::vector<double> t_fwd, t_bwd, t_opt, t_step;
        double timed_ms = 0.0;
        for (int s = 0; s < warmup + max_steps; ++s) {
            bool timed = s >= warmup;
            if (timed && (int)t_step.size() >= min_steps && timed_ms >= min_seconds * 1000.0) break;
            if (s == warmup) {
                reset_transfer_stats();
#ifdef DUMMYGRAD_HAS_MEMORY_STATS
                reset_peak_memory_stats();
#endif
            }
            optimizer.zero_grad(params);
            sync(device);
            double t0 = now_ms();
            auto loss = forward_loss();
            sync(device);
            double t1 = now_ms();
            scaler.scale(loss)->backward();
            sync(device);
            double t2 = now_ms();
            scaler.step(optimizer, params);
            scaler.update();
            sync(device);
            double t3 = now_ms();
            if (timed) {
                t_fwd.push_back(t1 - t0);
                t_bwd.push_back(t2 - t1);
                t_opt.push_back(t3 - t2);
                t_step.push_back(t3 - t0);
                timed_ms += t3 - t0;
            }
        }
        TransferStats traffic = transfer_stats();
        double steps = (double)t_step.size();
        double per_step = quantile(t_step, 0.5);
        double copies = (double)(traffic.to_device_count + traffic.to_host_count) / steps;
        double mb = (double)(traffic.to_device_bytes + traffic.to_host_bytes) / steps / 1e6;
        // Peak GPU memory held by tensors over the timed steps (parameters, gradients, optimizer
        // state and activations), like torch.cuda.max_memory_allocated
        // (0 when built against a version without memory statistics, as a same-session baseline may be)
        double peak_mb = 0.0;
#ifdef DUMMYGRAD_HAS_MEMORY_STATS
        if (device == Device::CUDA) peak_mb = (double)cuda_memory_stats().peak_in_use / 1e6;
#endif
        double tokens_per_s = c.batch * c.seq_len * 1000.0 / per_step;
        std::printf("| %s | %.1fM | %.2f | %.2f-%.2f | %.0f | %.2f | %.2f | %.2f | %d | %.0f | %.0f | %.1f |\n", c.name.c_str(),
                    model.num_parameters() / 1e6, per_step, quantile(t_step, 0.1), quantile(t_step, 0.9), tokens_per_s,
                    quantile(t_fwd, 0.5), quantile(t_bwd, 0.5), quantile(t_opt, 0.5), (int)t_step.size(), peak_mb,
                    copies, mb);
        std::fflush(stdout);
        json << (first_json ? "" : ", ") << "{\"name\": \"" << c.name << "\", \"params_m\": " << model.num_parameters() / 1e6
             << ", \"ms\": " << per_step << ", \"p10\": " << quantile(t_step, 0.1) << ", \"p90\": " << quantile(t_step, 0.9)
             << ", \"forward\": " << quantile(t_fwd, 0.5) << ", \"backward\": " << quantile(t_bwd, 0.5)
             << ", \"optimizer\": " << quantile(t_opt, 0.5) << ", \"steps\": " << t_step.size()
             << ", \"tokens_per_s\": " << tokens_per_s << ", \"peak_mb\": " << peak_mb << "}";
        first_json = false;

        if (profile) {
            // Separate pass: profiling synchronizes at every op and serializes backward, which skews
            // the timings above
            const int profiled_steps = 10;
            reset_profile();
            set_profiling(true);
            for (int s = 0; s < profiled_steps; ++s) {
                optimizer.zero_grad(params);
                scaler.scale(forward_loss())->backward();
            }
            set_profiling(false);
            std::string report;
            for (ProfilePhase phase : {ProfilePhase::Forward, ProfilePhase::Backward}) {
                auto entries = profile_report(phase);
                double total = 0.0;
                for (const auto& e : entries) total += e.total_ms;
                char title[256];
                std::snprintf(title, sizeof(title), "\n%s ops, %s (ms per step, %.2f ms total):\n",
                              phase == ProfilePhase::Forward ? "Forward" : "Backward", c.name.c_str(),
                              total / profiled_steps);
                report += title;
                int shown = 0;
                for (const auto& e : entries) {
                    if (shown++ == 12) break;
                    char line[256];
                    std::snprintf(line, sizeof(line), "  %8.2f ms  %5lld calls  %s\n", e.total_ms / profiled_steps,
                                  (long long)(e.calls / profiled_steps), e.name.c_str());
                    report += line;
                }
            }
            profiles.push_back(report);
        }
    }
    for (const auto& report : profiles) std::printf("%s", report.c_str());
    if (!json_path.empty()) {
        json << "]}\n";
        std::ofstream(json_path) << json.str();
    }
    return 0;
}
