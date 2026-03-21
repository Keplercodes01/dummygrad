//lets test the thing  

#include "engine.h"
#include<iostream>

int main() {

    manual_seed(42);

    auto w = randn({3, 3});
    auto optimizer = Adam(0.001f); 

    for(int i = 0; i<100; i++) {
        auto l = mean(w);
        l->backward();
        optimizer.step(w);
        w->zero_grad();

        if(i%10 == 0) {
            std::cout<<"step "<<i<<" loss: "<<l->data[0]<<std::endl;
        }
    }
    
    return 0;
}













































































