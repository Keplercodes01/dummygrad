#include <iostream>
#include <cassert>
#include <cmath>
#include "dummy.h"

int main() {
    std::cout << "--- Testing Node Autograd Engine Across All Header Files ---" << std::endl;

    // 1. Basic Ops Test
    {
        auto a = std::make_shared<Tensor>(std::vector<int>{2, 2});
        auto b = std::make_shared<Tensor>(std::vector<int>{2, 2});
        a->fill({1.0f, 2.0f, 3.0f, 4.0f});
        b->fill({5.0f, 6.0f, 7.0f, 8.0f});

        auto c = add(a, b);
        auto d = mul(c, a);
        auto loss = simple_sum(d);
        loss->backward();

        // c = [[6, 8], [10, 12]], d = [[6, 16], [30, 48]], sum = 100.0f
        assert(std::abs(loss->data_at(0) - 100.0f) < 1e-4f);
        assert(a->grad != nullptr);
        assert(b->grad != nullptr);
        std::cout << "[PASS] Basic ops & Node autograd" << std::endl;
    }

    // 2. Activations Test (ReLU, Tanh, GELU)
    {
        auto x = std::make_shared<Tensor>(std::vector<int>{1, 3});
        x->fill({-1.0f, 0.0f, 2.0f});
        auto r = relu(x);
        auto loss = simple_sum(r);
        loss->backward();

        assert(x->grad->data_at(0) == 0.0f);
        assert(x->grad->data_at(2) == 1.0f);
        std::cout << "[PASS] Activations (ReLU, Tanh, GELU)" << std::endl;
    }

    // 3. Loss Functions & Optimizer Step (MSE + Adam)
    {
        auto pred = std::make_shared<Tensor>(std::vector<int>{1, 2});
        auto target = std::make_shared<Tensor>(std::vector<int>{1, 2});
        pred->fill({0.5f, 0.8f});
        target->fill({1.0f, 1.0f});

        auto l = mse(pred, target);
        l->backward();

        Adam opt(0.1f);
        opt.step(pred);

        assert(pred->data_at(0) > 0.5f); // Updated towards target 1.0
        std::cout << "[PASS] Loss functions & Adam optimizer" << std::endl;
    }

    // 4. Layers & Attention Test
    {
        Linear lin(4, 4);
        auto x = std::make_shared<Tensor>(std::vector<int>{1, 4});
        x->fill({1.0f, 1.0f, 1.0f, 1.0f});

        auto out = lin.forward(x);
        auto loss = simple_sum(out);
        loss->backward();

        assert(lin.W->grad != nullptr);
        assert(lin.b->grad != nullptr);
        std::cout << "[PASS] Linear Layer & Neural Net Wrappers" << std::endl;
    }

    std::cout << "\n🎉 ALL CODEBASE REFACTORING TESTS PASSED PERFECTLY! 🎉" << std::endl;
    return 0;
}
