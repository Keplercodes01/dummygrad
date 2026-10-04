#pragma once
#include "tensor.h"
#include "functional.h"
#include <random>

inline thread_local std::mt19937 g_gen(std::random_device{}());

inline void manual_seed(unsigned int seed) {
    g_gen.seed(seed);
}

// random init
inline std::shared_ptr<Tensor> randn(std::vector<int64_t> shape, Device device = Device::CPU, DType dtype = DType::Float32, bool requires_grad = true) {
    auto t = std::make_shared<Tensor>(shape, Device::CPU, DType::Float32, false);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    float* ptr = t->data_ptr<float>();
    int64_t size = t->size();
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = dis(g_gen);
    }
    if (device != Device::CPU || dtype != DType::Float32) {
        auto out = t->to(device);
        out->requires_grad = requires_grad;
        return out;
    }
    t->requires_grad = requires_grad;
    return t;
}

// xavier init
inline std::shared_ptr<Tensor> xavier(std::vector<int64_t> shape, Device device = Device::CPU, DType dtype = DType::Float32, bool requires_grad = true) {
    if (shape.size() != 2) throw std::runtime_error("xavier: only 2D tensors supported.");

    auto t = std::make_shared<Tensor>(shape, Device::CPU, DType::Float32, false);
    float std_val = std::sqrt(1.0f / (float)shape[0]);
    std::normal_distribution<float> dis(0.0f, std_val);
    float* ptr = t->data_ptr<float>();
    int64_t size = t->size();
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = dis(g_gen);
    }
    if (device != Device::CPU || dtype != DType::Float32) {
        auto out = t->to(device);
        out->requires_grad = requires_grad;
        return out;
    }
    t->requires_grad = requires_grad;
    return t;
}

// kaiming init
inline std::shared_ptr<Tensor> kaiming(std::vector<int64_t> shape, Device device = Device::CPU, DType dtype = DType::Float32, bool requires_grad = true) {
    if (shape.size() != 2) throw std::runtime_error("kaiming: only 2D tensors supported.");

    auto t = std::make_shared<Tensor>(shape, Device::CPU, DType::Float32, false);
    float std_val = std::sqrt(2.0f / (float)shape[0]);
    std::normal_distribution<float> dis(0.0f, std_val);
    float* ptr = t->data_ptr<float>();
    int64_t size = t->size();
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = dis(g_gen);
    }
    if (device != Device::CPU || dtype != DType::Float32) {
        auto out = t->to(device);
        out->requires_grad = requires_grad;
        return out;
    }
    t->requires_grad = requires_grad;
    return t;
}

// one_hot
inline std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& indices, int num_classes) {
    int64_t n = indices->size();
    auto idx_cpu = (indices->device == Device::CPU && indices->is_contiguous()) ? indices : indices->cpu();
    auto out = std::make_shared<Tensor>(std::vector<int64_t>{n, num_classes}, Device::CPU, DType::Float32, false);
    out->fill_(0.0f);
    const float* idx_ptr = idx_cpu->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int64_t i = 0; i < n; i++) {
        int idx = (int)idx_ptr[i];
        if (idx < 0 || idx >= num_classes) throw std::runtime_error("one_hot: index out of range.");
        out_ptr[i * num_classes + idx] = 1.0f;
    }
    return (indices->device != Device::CPU) ? out->to(indices->device) : out;
}

// ones
inline std::shared_ptr<Tensor> ones(std::vector<int64_t> shape, Device device = Device::CPU, DType dtype = DType::Float32, bool requires_grad = false) {
    auto out = std::make_shared<Tensor>(shape, device, dtype, requires_grad);
    out->fill_(1.0f);
    return out;
}

// zeros
inline std::shared_ptr<Tensor> zeros(std::vector<int64_t> shape, Device device = Device::CPU, DType dtype = DType::Float32, bool requires_grad = false) {
    auto out = std::make_shared<Tensor>(shape, device, dtype, requires_grad);
    out->fill_(0.0f);
    return out;
}
