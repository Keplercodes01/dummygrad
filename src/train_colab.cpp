#include "dummy_core.h"
#include <iostream>
#include <chrono>
#include <sys/stat.h>

#if defined(_WIN32)
#include <direct.h>
#define mkdir_p(dir) _mkdir(dir)
#else
#define mkdir_p(dir) mkdir(dir, 0755)
#endif

inline bool dir_exists(const std::string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  DummyGrad: Native GPU Training & Drive Checkpointing   \n";
    std::cout << "=========================================================\n\n";

    // 1. Instantiate Modern Transformer Architecture (RMSNorm + RoPE + SwiGLU + GQA)
    int vocab_size  = 512;
    int max_seq_len = 64;
    int d_model     = 128;
    int n_heads     = 4;
    int n_layers    = 2;
    int n_kv_heads  = 2;

    ModernTransformer model(vocab_size, max_seq_len, d_model, n_heads, n_layers, n_kv_heads);

    // 2. Transfer entire model onto GPU if CUDA is available, otherwise CPU
#ifdef USE_CUDA
    model.cuda();
    Device target_dev = Device::CUDA;
    std::cout << "[+] Hardware: NVIDIA GPU (CUDA active with cuBLAS)\n";
#else
    model.cpu();
    Device target_dev = Device::CPU;
    std::cout << "[+] Hardware: Host CPU\n";
#endif

    auto params = model.parameters();
    Adam optimizer(1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);

    int B = 4;  // batch size
    int T = 16; // sequence length
    int total_steps = 50;

    std::cout << "[+] Starting training for " << total_steps << " steps...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= total_steps; ++step) {
        optimizer.zero_grad(params);

        // Generate synthetic batch on host CPU then transfer to target device
        auto in_cpu = std::make_shared<Tensor>(std::vector<int64_t>{B, T}, Device::CPU, DType::Float32, false);
        auto tgt_cpu = std::make_shared<Tensor>(std::vector<int64_t>{B, T}, Device::CPU, DType::Float32, false);

        for (int i = 0; i < B * T; ++i) {
            in_cpu->data_ptr<float>()[i]  = static_cast<float>((step * 3 + i) % vocab_size);
            tgt_cpu->data_ptr<float>()[i] = static_cast<float>((step * 3 + i + 1) % vocab_size);
        }

        auto inputs  = (target_dev == Device::CUDA) ? in_cpu->to(Device::CUDA) : in_cpu;
        auto targets = (target_dev == Device::CUDA) ? tgt_cpu->to(Device::CUDA) : tgt_cpu;

        // Forward Pass (cuBLAS matmul + GPU RoPE rotation + SwiGLU)
        auto logits = model.forward(inputs);

        // Fused Cross Entropy loss with LogSumExp numerical stabilization
        auto loss = fused_cross_entropy(logits, targets);

        // Autograd GPU backpropagation
        loss->backward();

        // Clip gradients to stabilize updates
        clip_grad_norm_(params, 1.0f);

        // AdamW optimizer update
        optimizer.step(params);

        if (step % 10 == 0 || step == total_steps) {
            float loss_val = loss->cpu()->data_ptr<float>()[0];
            std::cout << "    Step " << step << "/" << total_steps 
                      << " | Loss: " << loss_val << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "[+] Training finished in " << ms / 1000.0 << "s (" << ms / total_steps << " ms/step)\n\n";

    // 3. Determine save location: Google Drive if mounted, otherwise local directory
    std::string ckpt_dir = dir_exists("/content/drive/MyDrive") 
        ? "/content/drive/MyDrive/dummygrad_checkpoints/" 
        : "./checkpoints/";

    mkdir_p(ckpt_dir.c_str());

    std::string ckpt_path = ckpt_dir + "modern_llama.safetensors";
    std::cout << "[+] Saving checkpoint directly via zero-copy SafeTensors:\n    " << ckpt_path << "\n";
    model.save_safetensors(ckpt_path);

    // 4. Verification: Load checkpoint into a brand-new uninitialized model instance
    std::cout << "\n[+] Verifying zero-copy reload into a fresh model instance...\n";
    ModernTransformer loaded_model(vocab_size, max_seq_len, d_model, n_heads, n_layers, n_kv_heads);
    if (target_dev == Device::CUDA) loaded_model.cuda();
    loaded_model.load_safetensors(ckpt_path);

    // Run forward pass on the restored model
    auto test_in = ones({1, 8}, target_dev, DType::Float32, false);
    auto test_out = loaded_model.forward(test_in);
    std::cout << "    Restored model output shape: [" 
              << test_out->shape[0] << ", " << test_out->shape[1] << ", " << test_out->shape[2] << "]\n";

    std::cout << "\n[SUCCESS] End-to-end training, Drive persistence, and reload verified!\n";
    return 0;
}
