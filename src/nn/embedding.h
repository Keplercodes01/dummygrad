#pragma once
#include "tensor.h"
#include "functional.h"
#include "init.h"

// Token / Positional Embedding Layer
class Embedding {
public:
    std::shared_ptr<Tensor> weight;
    int num_embeddings;
    int embedding_dim;

    Embedding(int num_embeddings, int embedding_dim)
        : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
        weight = xavier({num_embeddings, embedding_dim});
    }

    // Forward pass accepting 1D or multi-dimensional indices tensor
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& indices) {
        if (!indices) throw std::runtime_error("Embedding: indices tensor is null");
        auto oh = one_hot(indices, num_embeddings);
        if (oh->device != weight->device) {
            oh = oh->to(weight->device);
        }
        auto out = matmul(oh, weight);
        if (indices->ndim() > 1) {
            std::vector<int64_t> out_shape = indices->shape;
            out_shape.push_back(embedding_dim);
            return reshape(out, out_shape);
        } else if (indices->ndim() == 0) {
            return reshape(out, {embedding_dim});
        }
        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {weight};
    }

    void to(Device device, DType dtype = DType::Float32) {
        weight = weight->to(device, dtype);
        weight->requires_grad = true;
    }
};
