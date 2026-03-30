#pragma once
#include"engine.h"

inline std::mt19937 g_gen(std::random_device{}());

inline void manual_seed(unsigned int seed) {
    g_gen.seed(seed);
}

//random init
inline std::shared_ptr<Tensor> randn(std::vector<int> shape) {
    auto t = std::make_shared<Tensor>(shape);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for(int i=0; i<t->size(); i++) {
        t->data_at(i) = dis(g_gen);
    }
    return t;
}

//xavier init
inline std::shared_ptr<Tensor> xavier(std::vector<int> shape) {
    if(shape.size() != 2) throw std::runtime_error("xavier: only 2D tensors supported. cmon man.");

    auto t = std::make_shared<Tensor>(shape);
    float std = std::sqrt(1.0f / (float)shape[0]);
    std::normal_distribution<float> dis(0.0f, std);
    for(int i = 0; i < t->size(); i++) {
        t->data_at(i) = dis(g_gen);
    }
    return t;
}

//kaiming init
inline std::shared_ptr<Tensor> kaiming(std::vector<int> shape) {
    if(shape.size() != 2) throw std::runtime_error("kaiming: only 2D tensors supported. cmon man.");

    auto t = std::make_shared<Tensor>(shape);
    float std = std::sqrt(2.0f / (float)shape[0]);
    std::normal_distribution<float> dis(0.0f, std);
    for(int i = 0; i < t->size(); i++) {
        t->data_at(i) = dis(g_gen);
    }
    return t;
}

//one_hot
inline std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& indices, int num_classes) {
    int n = indices->size();
    auto out = std::make_shared<Tensor>(std::vector<int>{n, num_classes});

    for(int i = 0; i < n; i++) {
        int idx = (int)indices->data_at(i);
        if(idx < 0 || idx >= num_classes) throw std::runtime_error("one_hot: index out of range. cmon man.");
        out->data_at(i * num_classes + idx) = 1.0f;
    }
    return out;
}

//ones
inline std::shared_ptr<Tensor> ones(std::vector<int> shape) {
    auto out = std::make_shared<Tensor>(shape);

    for(int i = 0; i<out->size(); i++) {
        out->data_at(i) = 1.0f;
    }
    return out;
}

//zeros
inline std::shared_ptr<Tensor> zeros(std::vector<int> shape) {
    return std::make_shared<Tensor>(shape);
}
