#include "engine.h"
#include "activations.h"
#include "broadcasting.h"
#include "norm.h"
#include "loss.h"
#include "scalar_ops.h"
#include "ops.h"
#include "optimizers.h"
#include "init.h"
#include "wrappers.h"
#include<iostream>

int main() {
    int batch_size = 2;
    int n_examples = 4;
    int n_seq = 8;
    int n_embd = 12;
    int n_head = 3;
    int head_size = n_embd / n_head;
    int n_hidden = 100;

    auto emb = kaiming({});
    auto L1 = Linear(4, 8, 1.0f);
    auto L2 = Linear(8, 
    auto out = L1.forward(a, 3);

    return 0;
}
