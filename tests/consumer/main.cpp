#include "dummy_core.h"
#include <iostream>

int main() {
    auto x = randn({2, 3});
    auto w = randn({3, 4});
    auto loss = sum(sum(matmul(x, w), -1), -2);
    loss->backward();
    if (!w->grad || w->grad->size() != 12) {
        std::cerr << "my_app: expected a 3x4 gradient for w\n";
        return 1;
    }
    std::cout << "my_app: dummygrad works, dloss/dw has " << w->grad->size() << " entries\n";
    return 0;
}
