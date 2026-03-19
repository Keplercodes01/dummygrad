//lets test the thing  

#include "engine.h"
#include<iostream>

int main() {

    manual_seed(42);

    auto a = randn({3,3}); 
    auto b = randn({3,3});

    auto c = matmul(a, b); 

    auto d = randn({3,3});
    auto e = mul(c, d);

    auto l = sum(e);

    l->backward();

    std::cout<<"e: "<<std::endl;
    e->show();
    std::cout<<"d: "<<std::endl;
    d->show();
    std::cout<<"c: "<<std::endl;
    c->show();
    std::cout<<"b: "<<std::endl;
    b->show();
    std::cout<<"a: "<<std::endl;
    a->show();
    std::cout<<"l: "<<std::endl;
    l->show();

    std::cout<<"grad of e: "<<std::endl;
    e->show_grad();
    std::cout<<"grad of d: "<<std::endl;
    d->show_grad();
    std::cout<<"grad of c: "<<std::endl;
    c->show_grad();
    std::cout<<"grad of b: "<<std::endl;
    b->show_grad();
    std::cout<<"grad of a: "<<std::endl;
    a->show_grad();
    std::cout<<"grad of l: "<<std::endl;
    l->show_grad();
    
    return 0;
}













































































