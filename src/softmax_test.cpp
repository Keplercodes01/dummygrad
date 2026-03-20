#include<iostream>
#include "engine.h"

int main() {
    auto a = randn({3, 3}); 
    auto b = softmax(a);
    auto l = mean(b);
    l->backward();

    std::cout<<"a: "<<std::endl;
    a->show();
    std::cout<<"b: "<<std::endl;
    b->show();
    std::cout<<"l: "<<std::endl;
    l->show();

    std::cout<<"a.grad: "<<std::endl;
    a->show_grad();
    std::cout<<"b.grad: "<<std::endl;
    b->show_grad();
    std::cout<<"l.grad: "<<std::endl;
    l->show_grad();

    return 0;
}
