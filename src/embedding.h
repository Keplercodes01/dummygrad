#pragma once
#include "engine.h"
#include "init.h"
#include "matmul.h"

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

    // Forward pass accepting 1D indices tensor
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& indices) {
        auto oh = one_hot(indices, num_embeddings);
        return matmul(oh, weight);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {weight};
    }
};
