#include <iostream>
#include <cassert>
#include <cmath>
#include "engine.h"
#include "matmul.h"
#include "ops.h"

int main() {
    std::cout << "Running matmul broadcasting test..." << std::endl;

    // A shape [2, 1, 2, 2]
    auto a = std::make_shared<Tensor>(std::vector<int>{2, 1, 2, 2});
    a->fill({
        1.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 1.0f
    });

    // B shape [1, 2, 2, 2]
    auto b = std::make_shared<Tensor>(std::vector<int>{1, 2, 2, 2});
    b->fill({
        2.0f, 2.0f,
        2.0f, 2.0f,
        2.0f, 2.0f,
        2.0f, 2.0f
    });

    auto out = matmul(a, b);

    std::cout << "Output shape check: ";
    out->show(); // should show shape=(2,2,2,2)

    auto loss = simple_sum(out);
    loss->backward();

    std::cout << "A grad: " << std::endl;
    a->show_grad();

    std::cout << "B grad: " << std::endl;
    b->show_grad();

    // Verify values:
    // Out elements should be 4.0f
    for (int i = 0; i < out->size(); i++) {
        if (std::abs(out->data_at(i) - 4.0f) > 1e-5) {
            std::cerr << "Verification failed: out[" << i << "] = " << out->data_at(i) << " (expected 4.0)" << std::endl;
            return 1;
        }
    }

    // A grad elements should be 8.0f
    for (int i = 0; i < a->size(); i++) {
        if (std::abs(a->grad_at(i) - 8.0f) > 1e-5) {
            std::cerr << "Verification failed: a->grad[" << i << "] = " << a->grad_at(i) << " (expected 8.0)" << std::endl;
            return 1;
        }
    }

    // B grad elements should be 4.0f
    for (int i = 0; i < b->size(); i++) {
        if (std::abs(b->grad_at(i) - 4.0f) > 1e-5) {
            std::cerr << "Verification failed: b->grad[" << i << "] = " << b->grad_at(i) << " (expected 4.0)" << std::endl;
            return 1;
        }
    }

    std::cout << "All matmul broadcasting tests passed successfully!" << std::endl;
    return 0;
}
