//lets test the thing  

#include "engine.h"
#include<iostream>

int main() {

    manual_seed(42);

    auto w = randn({3, 3});
    auto y = pow(w, 4); 
    auto l = sum(y); 

    l->backward();

    std::cout<<"w: "<<std::endl;
    w->show();
    std::cout<<"y: "<<std::endl;
    y->show();
    std::cout<<"l: "<<std::endl;
    l->show();

    std::cout<<"grad of w: "<<std::endl;
    w->show_grad();
    std::cout<<"grad of y: "<<std::endl; 
    y->show_grad();
    std::cout<<"grad of l: "<<std::endl; 
    l->show_grad();
    
    return 0;
}













































































