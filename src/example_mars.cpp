#include "dummy_core.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Example: End-to-end MARS (Multimodal Action Reasoning System)
// Continuous joint angle next-frame prediction and autoregressive motion rollout
int main() {
    std::cout << "========================================================\n";
    std::cout << "  MARS: Multimodal Action Reasoning System (C++ Native) \n";
    std::cout << "========================================================\n\n";

    // 1. Initialize MARS architecture for CMU MoCap skeletal kinematics
    MARSConfig cfg;
    cfg.action_dim = 156;   // 156 continuous skeletal joint angle degrees of freedom
    cfg.d_model = 256;      // Latent transformer embedding dimension
    cfg.n_heads = 8;        // Causal attention heads
    cfg.n_layers = 4;       // 4 transformer decoder blocks
    cfg.max_seq_len = 128;  // Context length in frames

    MARS model(cfg);
    std::cout << "[+] Instantiated MARS Model:\n";
    std::cout << "    - Action Dimension : " << cfg.action_dim << " DOFs / frame\n";
    std::cout << "    - Latent Dimension : " << cfg.d_model << "\n";
    std::cout << "    - Attention Heads  : " << cfg.n_heads << "\n";
    std::cout << "    - Decoder Layers   : " << cfg.n_layers << "\n";
    std::cout << "    - Context Window   : " << cfg.max_seq_len << " frames\n\n";

    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& p : params) {
        if (p) total_params += p->size();
    }
    std::cout << "[+] Total Trainable Parameters: " << total_params << "\n\n";

    // 2. Synthesize a batch of realistic continuous motion trajectories
    // Shape: [batch_size=2, seq_len=32, action_dim=156]
    int B = 2;
    int T = 32;
    auto trajectory = std::make_shared<Tensor>(std::vector<int64_t>{B, T, cfg.action_dim}, false);
    float* data = trajectory->data_ptr<float>();
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < cfg.action_dim; d++) {
                // Smooth sinusoidal wave simulating skeletal joint trajectory
                float val = std::sin(0.1f * t + 0.05f * d + 0.2f * b);
                data[b * T * cfg.action_dim + t * cfg.action_dim + d] = val;
            }
        }
    }

    // 3. Training Loop Step: Forward Pass, Autograd Backward, and AdamW Optimization
    std::cout << "[+] Running Next-Frame Prediction Training Step...\n";
    Adam optimizer(1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);

    for (int step = 1; step <= 5; step++) {
        optimizer.zero_grad(params);

        // Compute autoregressive next-frame loss: A_{0:T-1} -> predicts A_{1:T}
        auto loss = model.compute_loss(trajectory, "mse");
        float loss_val = loss->data_ptr<float>()[0];

        // Autograd backward pass
        loss->backward();

        // Optimizer step
        optimizer.step(params);

        std::cout << "    Step " << step << " | Next-Frame MSE Loss: " 
                  << std::fixed << std::setprecision(6) << loss_val << "\n";
    }

    // 4. Autoregressive Motion Rollout (Continuous Generation)
    std::cout << "\n[+] Testing Autoregressive Motion Rollout...\n";
    // Provide a seed prompt of 5 initial frames
    int seed_len = 5;
    auto seed_frames = slice(trajectory, 1, 0, seed_len);
    std::cout << "    - Seed prompt sequence shape : [" 
              << seed_frames->shape[0] << ", " << seed_frames->shape[1] << ", " << seed_frames->shape[2] << "]\n";

    int future_steps = 15;
    auto generated_traj = model.rollout(seed_frames, future_steps);
    std::cout << "    - Rolled out trajectory shape: [" 
              << generated_traj->shape[0] << ", " << generated_traj->shape[1] << ", " << generated_traj->shape[2] << "]\n";
    std::cout << "    - Generated " << future_steps << " future frames successfully!\n\n";

    // 5. Zero-Copy SafeTensors Serialization Check
    std::cout << "[+] Exporting Model Weights to SafeTensors...\n";
    model.save_safetensors("mars_checkpoint.safetensors");
    std::cout << "    - Saved to mars_checkpoint.safetensors (Hugging Face / PyTorch interoperable)\n\n";

    std::cout << "[OK] MARS Engine Native C++ Verification Complete!\n";
    return 0;
}
