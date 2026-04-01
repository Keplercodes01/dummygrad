#include "engine.h"
#include "ops.h"
#include "activations.h"
#include "loss.h"
#include "init.h"
#include "optimizers.h"
#include "scalar_ops.h"
#include "broadcasting.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>

int main() {
    
    // hyperparameters
    int block_size = 3;
    int n_embd = 10;
    int n_hidden = 200;
    int vocab_size = 27;
    int batch_size = 32;
    
    std::cout << "dummygrad MLP" << std::endl;
    
    return 0;
}
