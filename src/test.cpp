#include "engine.h"
#include "activations.h"
#include "broadcasting.h"
#include "loss.h"
#include "scalar_ops.h"
#include "ops.h"
#include "optimizers.h"
#include "init.h"
#include "wrappers.h"
#include<iostream>

int main() {
    auto x = kaiming({3,3});
    std::cout << "x:\n";
    x->show();

    auto xs = std_dev(x, 1);
    std::cout << "std_dev of x:\n";
    xs->show();

    auto loss = simple_sum(xs);
    loss->backward();

    std::cout << "grad of x:\n";
    x->show_grad();

    return 0;
}
