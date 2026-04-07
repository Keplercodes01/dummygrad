#include "engine.h"
#include "ops.h"
#include "activations.h"
#include "loss.h"
#include "init.h"
#include "optimizers.h"
#include "scalar_ops.h"
#include "broadcasting.h"
#include "batch_norm.h"
#include<iostream>

int main() {
    auto x = kaiming({3, 3}); 
    auto y = kaiming({3, 4});
    auto z = matmul(x, y);
    auto meaned = mean(z);
    meaned->backward();

    std::cout<<"x: "<<std::endl;
    x->show();
    std::cout<<"y: "<<std::endl;
    y->show();
    std::cout<<"z: "<<std::endl;
    z->show();
    std::cout<<"meaned: "<<std::endl;
    meaned->show();

    std::cout<<"grad of x: "<<std::endl;
    x->show_grad();
    std::cout<<"grad of y: "<<std::endl;
    y->show_grad();
    std::cout<<"grad of z: "<<std::endl;
    z->show_grad();
    std::cout<<"grad of meaned: "<<std::endl;
    meaned->show_grad();

    return 0;
}














