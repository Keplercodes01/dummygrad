#include <iostream>
#include <cassert>
#include "dummy.h"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "🚀 TRAINING A GPT TRANSFORMER MODEL IN PURE C++ 🚀" << std::endl;
    std::cout << "==================================================" << std::endl;

    int vocab_size = 10;
    int max_seq_len = 4;
    int d_model = 16;
    int n_heads = 2;
    int n_layers = 2;

    // Instantiate GPT model & Adam optimizer (lr = 0.001f)
    GPT model(vocab_size, max_seq_len, d_model, n_heads, n_layers);
    Adam optimizer(0.001f);

    auto params = model.parameters();
    std::cout << "Total Model Parameters: " << params.size() << std::endl;

    // Dummy input token IDs: [1, 3, 5, 7]
    auto input_ids = std::make_shared<Tensor>(std::vector<int>{4}, false);
    input_ids->fill({1.0f, 3.0f, 5.0f, 7.0f});

    // Positional IDs: [0, 1, 2, 3]
    auto pos_ids = std::make_shared<Tensor>(std::vector<int>{4}, false);
    pos_ids->fill({0.0f, 1.0f, 2.0f, 3.0f});

    // Target one-hot labels for sequence prediction
    auto targets = std::make_shared<Tensor>(std::vector<int>{4, 10}, false);
    targets->fill({
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0
    });

    std::cout << "\n--- Starting Training Loop (50 Steps) ---" << std::endl;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int step = 1; step <= 50; step++) {
        // 0. Zero gradients across all parameters
        for (auto& p : params) {
            p->zero_grad();
        }

        // 1. Forward pass
        auto logits = model.forward(input_ids, pos_ids);
        
        // Convert logits to probabilities
        auto probs = softmax(logits);

        // 2. Compute Cross-Entropy Loss
        auto loss = cross_entropy(probs, targets);
        float current_loss = loss->data_at(0);

        if (step == 1) initial_loss = current_loss;
        final_loss = current_loss;

        if (step % 5 == 0 || step == 1) {
            std::cout << "Step " << step << " | Loss: " << current_loss << std::endl;
        }

        // 3. Backward pass
        loss->backward();

        // 4. Optimizer update
        for (auto& p : params) {
            optimizer.step(p);
        }
    }

    std::cout << "\nSummary:" << std::endl;
    std::cout << "Initial Loss: " << initial_loss << std::endl;
    std::cout << "Final Loss:   " << final_loss << std::endl;

    assert(final_loss < initial_loss && "Training failed to reduce loss");
    std::cout << "\n🎉 GPT TRANSFORMER TRAINING SUCCESSFUL! LOSS DECREASED! 🎉" << std::endl;
    return 0;
}
