// Training-step benchmark: ModernTransformer (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) at a
// few sizes, timed as forward / backward / optimizer, with host<->device traffic per step.
// bench/bench_torch.py runs the same architecture in PyTorch for comparison.
//
//   ./dg_bench                  all configurations
//   ./dg_bench small            one configuration (tiny, small, medium)
//   ./dg_bench --profile        also print the most expensive forward and backward ops per configuration
//   ./dg_bench --fp16           matmuls and attention on fp16 tensor cores (AutocastGuard + GradScaler)
//   ./dg_bench --bf16           matmuls and attention on bf16 tensor cores (Ampere or newer)
//   ./dg_bench --calling-thread run backward on the calling thread instead of the worker pool

#include "dummy_core.h"

#include <chrono>
#include <cstdio>
#include <iostream>
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
};

static void sync(Device device) {
#ifdef USE_CUDA
    if (device == Device::CUDA) cudaDeviceSynchronize();
#else
    (void)device;
#endif
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string only;
    bool profile = false;
    DType compute = DType::Float32;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--profile") profile = true;
        else if (arg == "--fp16") compute = DType::Float16;
        else if (arg == "--bf16") compute = DType::BFloat16;
        else if (arg == "--calling-thread") set_backward_execution(BackwardExecution::CallingThread);
        else only = arg;
    }
    const char* compute_name = compute == DType::Float16 ? "fp16" : compute == DType::BFloat16 ? "bf16" : "fp32";
    std::vector<std::string> profiles;
    Device device = Device::CPU;
#ifdef USE_CUDA
    int gpus = 0;
    if (cudaGetDeviceCount(&gpus) == cudaSuccess && gpus > 0) device = Device::CUDA;
#endif
    const int warmup = 3, steps = 10;
    std::printf("dummygrad %s benchmark on %s, %s autocast, backward on %s (%d timed steps after %d warmup)\n\n",
                DUMMYGRAD_VERSION, device == Device::CUDA ? "CUDA" : "CPU", compute_name,
                backward_execution() == BackwardExecution::CallingThread ? "the calling thread" : "worker threads",
                steps, warmup);
    std::printf("| config | params | ms/step | tokens/s | forward | backward | optimizer | copies/step | MB/step |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|\n");

    for (const auto& c : kConfigs) {
        if (!only.empty() && only != c.name) continue;
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

        double t_fwd = 0, t_bwd = 0, t_opt = 0;
        TransferStats traffic;
        for (int s = 0; s < warmup + steps; ++s) {
            bool timed = s >= warmup;
            if (s == warmup) reset_transfer_stats();
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
                t_fwd += t1 - t0;
                t_bwd += t2 - t1;
                t_opt += t3 - t2;
            }
        }
        traffic = transfer_stats();
        double per_step = (t_fwd + t_bwd + t_opt) / steps;
        double copies = (double)(traffic.to_device_count + traffic.to_host_count) / steps;
        double mb = (double)(traffic.to_device_bytes + traffic.to_host_bytes) / steps / 1e6;
        std::printf("| %s | %.1fM | %.1f | %.0f | %.1f | %.1f | %.1f | %.0f | %.1f |\n", c.name.c_str(),
                    model.num_parameters() / 1e6, per_step, c.batch * c.seq_len * 1000.0 / per_step,
                    t_fwd / steps, t_bwd / steps, t_opt / steps, copies, mb);
        std::fflush(stdout);

        if (profile) {
            // Separate pass: profiling synchronizes at every op and serializes backward, which skews
            // the timings above
            const int profiled_steps = 3;
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
    return 0;
}
