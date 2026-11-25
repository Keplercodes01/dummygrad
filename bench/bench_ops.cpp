// Single operations timed on the CPU or the GPU: what a research script leans on besides the training step that
// bench.cpp times. Each row takes a fraction of a second.
//
//   ./dg_bench_ops            on the GPU if there is one
//   ./dg_bench_ops --cpu

#include "dummy_core.h"
#include "allocator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

static Device g_device = Device::CPU;

static void finish() {
#ifdef USE_CUDA
    if (g_device == Device::CUDA) cudaDeviceSynchronize();
#endif
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// The median time of one call, after two that do not count
static double time_ms(const std::function<void()>& f, double budget_ms = 150.0) {
    for (int i = 0; i < 2; ++i) f();
    finish();
    std::vector<double> t;
    double start = now_ms();
    while (t.size() < 5 || (now_ms() - start < budget_ms && t.size() < 200)) {
        double a = now_ms();
        f();
        finish();
        t.push_back(now_ms() - a);
    }
    std::nth_element(t.begin(), t.begin() + (std::ptrdiff_t)t.size() / 2, t.end());
    return t[t.size() / 2];
}
static void row(const std::string& name, const std::function<void()>& f, const std::string& note = "") {
    reset_peak_memory_stats();
    size_t before = cuda_memory_stats().in_use;
    double ms = time_ms(f);
    std::printf("| %-52s | %9.3f ms | %7.1f MB | %s\n", name.c_str(), ms, (double)(cuda_memory_stats().peak_in_use - before) / 1e6, note.c_str());
    std::fflush(stdout);
}
static TensorPtr leaf(const std::vector<int64_t>& shape) {
    auto t = normal(shape, 0.0f, 1.0f, g_device);
    t->requires_grad = true;
    return t;
}

int main(int argc, char** argv) {
    bool cpu = argc > 1 && std::string(argv[1]) == "--cpu";
#ifdef USE_CUDA
    int n = 0;
    if (!cpu && cudaGetDeviceCount(&n) == cudaSuccess && n > 0) g_device = Device::CUDA;
#else
    (void)cpu;
#endif
    manual_seed(0);
    std::printf("device: %s\n", g_device == Device::CUDA ? "CUDA" : "CPU");
    const bool gpu = g_device == Device::CUDA;

    // Generation: with a cache one query per token, without one a whole forward pass
    {
        ModernTransformer model(8192, 512, 512, 8, 6);
        model.to(g_device);
        NoGradGuard no_grad;
        model.eval();
        for (int64_t batch : {1, 16}) {
            auto prompt = randint(0, 8192, {batch, 64}, g_device);
            GenerateOptions o;
            o.max_new_tokens = gpu ? 64 : 8;
            o.temperature = 0.0f;
            generate(model, prompt, o);
            double a = now_ms();
            generate(model, prompt, o);
            double cached = (now_ms() - a) / (double)o.max_new_tokens;
            auto ids = randint(0, 8192, {batch, 64 + o.max_new_tokens / 2}, g_device);  // half way through
            double full = time_ms([&] { model.forward(ids)->to_vector(); });
            std::printf("| generate, 23M parameters, batch %-2lld                    | %9.3f ms per token with a cache, %.3f ms without: %.0f tokens/s\n",
                        (long long)batch, cached, full, 1000.0 * (double)batch / cached);
        }
    }
    // Attention with a padding mask, forward and backward
    for (bool half : {false, true}) {
        if (half && !gpu) continue;
        AutocastGuard autocast(half ? DType::Float16 : DType::Float32);
        const std::string tag = half ? " fp16" : "";
        auto q = leaf({8, 8, 256, 64}), k = leaf({8, 8, 256, 64}), v = leaf({8, 8, 256, 64});
        auto mask = mul_scalar(bernoulli({8, 1, 1, 256}, 0.2f, g_device), -1e9f);
        row("attention, padding mask, one op" + tag, [&] { sum(scaled_dot_product_attention(q, k, v, mask))->backward(); });
        row("attention, padding mask, separate ops" + tag, [&] {
            TensorPtr w;
            sum(scaled_dot_product_attention(q, k, v, mask, 0.0f, false, true, &w))->backward();
        });
        auto kl = leaf({8, 8, 512, 64}), vl = leaf({8, 8, 512, 64});
        row("cross-attention, 256 queries, 512 keys, one op" + tag, [&] { sum(scaled_dot_product_attention(q, kl, vl))->backward(); });
    }
    // Sorting, scattering, transforms
    {
        auto x = normal({1000000}, 0, 1, g_device), logits = normal({64, 50000}, 0, 1, g_device);
        row("sort 1M", [&] { sort(x, 0); });
        row("topk 50 of [64, 50000]", [&] { topk(logits, 50); });
        auto at = randint(0, 10000, {1000000}, g_device);
        row("scatter_add 1M into 10K", [&] { scatter_add(zeros({10000}, g_device), 0, at, x); });
        auto rows = randint(0, 10000, {100000}, g_device), src = normal({100000, 64}, 0, 1, g_device);
        row("index_add 100K rows of 64 into 10K", [&] { index_add(zeros({10000, 64}, g_device), 0, rows, src); });
        auto signal = normal({256, 4096, 2}, 0, 1, g_device);
        row("fft [256, 4096] complex", [&] { fft::fft(signal); });
    }
    // Convolutions, forward and backward
    {
        auto x = leaf({32, 64, 56, 56}), w = leaf({64, 64, 3, 3});
        row("conv 3x3, 64 channels, [32, 64, 56, 56]", [&] { sum(conv_nd(x, w, nullptr, {1}, {1}, {1}, 1))->backward(); });
        auto w1 = leaf({64, 64, 1, 1});
        row("conv 1x1, 64 channels, [32, 64, 56, 56]", [&] { sum(conv_nd(x, w1, nullptr, {1}, {0}, {1}, 1))->backward(); });
    }
    // Sending a batch while the GPU is busy: how long the program waits
    if (gpu) {
        auto a = normal({2048, 2048}, 0, 1, g_device);
        auto batch = normal({64, 3, 64, 64});
        std::vector<double> t;
        for (int i = 0; i < 10; ++i) {
            NoGradGuard no_grad;
            TensorPtr y = a;
            for (int j = 0; j < 8; ++j) y = matmul(y, a);
            double s = now_ms();
            auto sent = batch->to(g_device);
            t.push_back(now_ms() - s);
            finish();
        }
        std::nth_element(t.begin(), t.begin() + 5, t.end());
        std::printf("| copy 3 MB to a busy GPU                              | %9.3f ms until the program goes on\n", t[5]);
    }
    return 0;
}
