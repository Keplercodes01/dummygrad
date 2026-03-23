//lets test the thing  

#include "engine.h"
#include<iostream>

int main() {

    manual_seed(42);

    auto a = randn({3, 4});
    auto b = randn({4, 5});
    auto c = matmul(a, b);
    auto l = mean(c);
    l->backward();

    a->show();
    b->show();
    c->show();
    a->show_grad();
    b->show_grad();
    c->show_grad();

    return 0;
}













































































