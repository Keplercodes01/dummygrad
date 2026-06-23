#pragma once 
#include "engine.h"

//view
inline std::shared_ptr<Tensor> view(const std::shared_ptr<Tensor>& a, std::vector<int> new_shape) {
    int total = 1;
    for(int d : new_shape) total *= d;
    if(total != a->size()) throw std::runtime_error("view: total elements must match. cmon man.");

    auto out = std::make_shared<Tensor>(new_shape);
    out->storage = a->storage;
    out->offset = a->offset;

    out->prev.push_back(a);

    return out;
}

