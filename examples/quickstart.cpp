// dummygrad quickstart: tensors, autograd, and training a small network.
//
// Build and run from the repository root:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
//   ./build/quickstart

#include "dummy_core.h"
#include <iostream>

int main() {
    manual_seed(0);

    // Use the GPU when dummygrad was built with CUDA and one is present
    Device dev = Device::CPU;
#ifdef USE_CUDA
    int gpus = 0;
    if (cudaGetDeviceCount(&gpus) == cudaSuccess && gpus > 0) dev = Device::CUDA;
#endif
    std::cout << "Running on " << (dev == Device::CUDA ? "CUDA" : "CPU") << "\n\n";

    // ------------------------------------------------------------------
    // 1. Tensors and autograd
    // ------------------------------------------------------------------
    // Tensors are shared_ptr<Tensor>. requires_grad=true makes autograd track them.
    auto x = ones({2, 3}, Device::CPU, DType::Float32, /*requires_grad=*/true);

    // y = sum(3 * x). sum() reduces one axis and keeps it, so two sums give a [1, 1] scalar.
    auto y = sum(sum(mul_scalar(x, 3.0f), -1), -2);
    y->backward();

    std::cout << "dy/dx (every element should be 3):\n";
    x->show_grad();

    // ------------------------------------------------------------------
    // 2. Train a two-layer network to learn XOR
    // ------------------------------------------------------------------
    auto X = std::make_shared<Tensor>(std::vector<int64_t>{4, 2}, Device::CPU, DType::Float32, false);
    auto Y = std::make_shared<Tensor>(std::vector<int64_t>{4, 1}, Device::CPU, DType::Float32, false);
    X->copy_from_vector({0, 0,
                         0, 1,
                         1, 0,
                         1, 1});
    Y->copy_from_vector({0, 1, 1, 0});
    X = X->to(dev);
    Y = Y->to(dev);

    Linear hidden(2, 8);
    Linear output(8, 1);
    hidden.to(dev);
    output.to(dev);

    // Collect parameters after moving the layers: to() replaces the tensors
    std::vector<std::shared_ptr<Tensor>> params = {hidden.W, hidden.b, output.W, output.b};
    Adam optimizer(/*lr=*/0.05f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.0f);

    std::cout << "\nTraining on XOR:\n";
    float final_loss = 0.0f;
    for (int step = 1; step <= 500; ++step) {
        optimizer.zero_grad(params);

        auto pred = output.forward(tanh(hidden.forward(X)));
        auto loss = mse(pred, Y);

        loss->backward();
        optimizer.step(params);

        final_loss = loss->cpu()->data_ptr<float>()[0];
        if (step % 100 == 0) std::cout << "  step " << step << "  loss " << final_loss << "\n";
    }

    auto pred = output.forward(tanh(hidden.forward(X)))->cpu();
    std::cout << "\nPredictions (targets 0 1 1 0):";
    for (int i = 0; i < 4; ++i) std::cout << " " << pred->data_ptr<float>()[i];
    std::cout << "\n";

    // Non-zero exit if training failed, so CTest can use this as a smoke test
    return final_loss < 0.05f ? 0 : 1;
}
