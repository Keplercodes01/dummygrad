#include <iostream>
#include "engine.h"
#include "activations.h"
#include "ops.h"

int main() {
    auto x = std::make_shared<Tensor>(std::vector<int>{4});
    x->fill({1.0f, 2.0f, 3.0f, 4.0f});

    auto out = softmax(x);
    auto loss = simple_sum(out);
    loss->show();
    return 0;
}
