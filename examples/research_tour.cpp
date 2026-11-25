// A tour of the research API: higher-order gradients, einsum, linear algebra, distributions, layers.
// Build and run: cmake --build build --target research_tour && ./build/research_tour

#include "dummy_core.h"
#include <iostream>

int main() {
    manual_seed(0);

    // The README's first example: learn |x|^2, then differentiate what was learned
    {
        Sequential net;
        net.add<Linear>(2, 64).add<GELU>().add<Linear>(64, 1);
        optim::AdamW opt(1e-2f);

        auto x = normal({256, 2});
        auto y = sum(square(x), 1);
        float first = 0.0f, last = 0.0f;
        for (int step = 0; step < 2000; ++step) {
            net.zero_grad();
            auto loss = mse_loss(net.forward(x), y);
            loss->backward();
            opt.step(net.parameters());
            (step == 0 ? first : last) = loss->item();
        }
        auto point = tensor({1, 2}, {1, 2}, Device::CPU, /*requires_grad=*/true);
        auto g = autograd::grad({sum(net.forward(point))}, {point}, {}, true, /*create_graph=*/true)[0];
        // The network's gradient, checked against a central difference of the network itself
        NoGradGuard no_grad;
        auto at = [&](float a, float b) { return net.forward(tensor({a, b}, {1, 2}))->item(); };
        float h = 1e-2f, d0 = (at(1 + h, 2) - at(1 - h, 2)) / (2 * h), d1 = (at(1, 2 + h) - at(1, 2 - h)) / (2 * h);
        std::cout << "|x|^2 fit: loss " << first << " -> " << last << "; gradient at (1, 2): autograd " << g->to_vector()[0] << ", "
                  << g->to_vector()[1] << ", finite difference " << d0 << ", " << d1 << " (of |x|^2 itself: 2, 4)\n";
        if (!(last < 0.1f * first) || std::fabs(g->to_vector()[0] - d0) > 0.05f || std::fabs(g->to_vector()[1] - d1) > 0.05f) return 1;
    }

    // Second derivatives: the Hessian of sum(x^3) is diag(6x)
    auto f = [](const TensorPtr& x) { return sum(pow(x, 3.0f)); };
    std::cout << "Hessian of sum(x^3) at (1, 2):\n";
    autograd::hessian(f, tensor({1, 2}, {2}))->show();

    // A gradient penalty, as in WGAN-GP: differentiate through a gradient
    Linear critic(4, 1);
    auto x = randn({8, 4});
    auto score = sum(tanh(critic.forward(x)));
    auto dx = autograd::grad({score}, {x}, {}, /*retain_graph=*/true, /*create_graph=*/true)[0];
    auto penalty = mean(square(sub_scalar(norm(dx, 2.0f, 1), 1.0f)));
    penalty->backward();
    std::cout << "gradient penalty " << penalty->item() << ", its gradient reached the critic: " << (critic.W->grad ? "yes" : "no") << "\n";

    // einsum and linear algebra, both differentiable
    auto a = randn({3, 4, 4});
    auto spd = add(einsum("bij,bkj->bik", {a, a}), eye(4));
    auto nll = neg(sum(linalg::slogdet(spd).second));
    nll->backward();
    std::cout << "-log det = " << nll->item() << "\n";

    // A variational autoencoder step: reparameterized samples and a KL term
    Linear encoder(16, 8), decoder(4, 16);
    auto data = rand({32, 16});
    auto stats = chunk(encoder.forward(data), 2, 1);
    distributions::Normal posterior(stats[0], exp(stats[1])), prior(zeros_like(stats[0]), ones_like(stats[0]));
    auto z = posterior.rsample();
    auto reconstruction = binary_cross_entropy_with_logits(decoder.forward(z), data, nullptr, nullptr, Reduction::Sum);
    auto loss = div_scalar(add(reconstruction, sum(distributions::kl_divergence(posterior, prior))), 32.0f);
    loss->backward();
    std::cout << "VAE loss " << loss->item() << "\n";

    // A small convolutional network trained with Muon on its matrices and AdamW on the rest
    auto net = std::make_shared<Sequential>();
    net->add<Conv2d>(1, 8, 3, 1, 1).add<BatchNorm2d>(8).add<ReLU>().add<MaxPool2d>(2).add<Flatten>().add<Linear>(8 * 4 * 4, 2);
    std::vector<TensorPtr> matrices, rest;
    for (auto& p : net->parameters()) (p->ndim() >= 2 && p->shape[0] > 1 ? matrices : rest).push_back(p);
    optim::Muon muon(0.02f);
    optim::AdamW adamw(0.01f);
    auto images = normal({32, 1, 8, 8});
    auto labels = gt(sum(images, {1, 2, 3}, false), 0.0f);
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 60; ++step) {
        net->zero_grad();
        auto l = cross_entropy_loss(net->forward(images), labels);
        l->backward();
        muon.step(matrices);
        adamw.step(rest);
        (step == 0 ? first : last) = l->item();
    }
    std::cout << "CNN loss " << first << " -> " << last << "\n";
    return last < first ? 0 : 1;
}
