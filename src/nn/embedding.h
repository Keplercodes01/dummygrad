#pragma once
#include "tensor.h"
#include "functional.h"
#include "module.h"
#include "init.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace embedding_detail {

// Throws unless every id (stored as a float, truncated like the lookup does) is in [0, vocab)
inline void check_ids(const Tensor& ids, int64_t vocab) {
    const float* p = ids.data_ptr<float>();
    for (int64_t i = 0; i < ids.size(); ++i) {
        if (!(p[i] >= 0.0f && p[i] < (float)vocab)) {
            throw std::out_of_range("Embedding: index " + std::to_string(p[i]) + " is out of range for " +
                                    std::to_string(vocab) + " embeddings");
        }
    }
}

}  // namespace embedding_detail

// Gradient of a row lookup: grad_weight[id] = sum of the output gradients at positions holding id.
// Positions are grouped by id on the host and summed in position order, so the result is
// deterministic (no atomics on the GPU).
struct EmbeddingBackward : public Node {
    // On the host when forward was given them there (no copy back needed), else on the weight's device
    std::shared_ptr<Tensor> indices;
    int64_t vocab, dim;

    EmbeddingBackward(std::shared_ptr<Tensor> indices, int64_t vocab, int64_t dim)
        : indices(std::move(indices)), vocab(vocab), dim(dim) {}

    void release_variables() override { indices.reset(); }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !indices) return {nullptr};
        auto g = grads[0];
        Device dev = g->device;

        auto ids_host = indices->device == Device::CPU ? indices : indices->cpu();
        // Ids that stayed on the GPU were not checked in forward (out-of-range ones gave NaN rows);
        // the gradient below writes the row of every id, so a bad one must stop here
        embedding_detail::check_ids(*ids_host, vocab);
        const float* ids = ids_host->data_ptr<float>();
        int64_t n = ids_host->size();

#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            // Group positions by id (CSR): positions sorted stably by id, one run per unique id
            std::vector<int> positions(n);
            std::iota(positions.begin(), positions.end(), 0);
            std::stable_sort(positions.begin(), positions.end(), [&](int a, int b) { return (int64_t)ids[a] < (int64_t)ids[b]; });
            std::vector<int> unique_ids, starts;
            for (int64_t p = 0; p < n; ++p) {
                int id = (int)ids[positions[p]];
                if (unique_ids.empty() || unique_ids.back() != id) {
                    unique_ids.push_back(id);
                    starts.push_back((int)p);
                }
            }
            starts.push_back((int)n);
            auto gw = std::make_shared<Tensor>(std::vector<int64_t>{vocab, dim}, dev, g->dtype, false);
            cuda::embedding_backward(g->data_ptr<float>(), positions, unique_ids, starts, gw->data_ptr<float>(), vocab, dim);
            return {gw};
        }
#endif
        auto gh = (dev == Device::CPU) ? g : g->cpu();
        auto gw = std::make_shared<Tensor>(std::vector<int64_t>{vocab, dim}, Device::CPU, g->dtype, false);
        gw->fill_(0.0f);
        const float* gp = gh->data_ptr<float>();
        float* wp = gw->data_ptr<float>();
        for (int64_t p = 0; p < n; ++p) {
            int64_t id = (int64_t)ids[p];
            for (int64_t d = 0; d < dim; ++d) wp[id * dim + d] += gp[p * dim + d];
        }
        return {(dev == Device::CPU) ? gw : gw->to(dev)};
    }
};

// Token / positional embedding: a learned [num_embeddings, embedding_dim] table indexed by integer
// ids (stored as floats). The output has the indices' shape plus a trailing embedding dimension.
//
// Out-of-range ids throw std::out_of_range. Ids given on the host are checked in forward; ids
// already on the GPU are checked in backward (checking them in forward would cost a device sync),
// and until then their output rows are NaN.
class Embedding : public Module {
public:
    std::shared_ptr<Tensor> weight;
    int num_embeddings;
    int embedding_dim;

    Embedding(int num_embeddings, int embedding_dim)
        : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
        weight = xavier({num_embeddings, embedding_dim});
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& indices_in) {
        DG_PROFILE_OP("embedding");
        if (!indices_in) throw std::runtime_error("Embedding: indices tensor is null");
        Device dev = weight->device;
        int64_t vocab = weight->shape[0], dim = weight->shape[1];
        auto ids = make_contiguous(indices_in);

        // The ids on the host, checked, whenever getting them there costs no device sync
        std::shared_ptr<Tensor> host_ids;
        if (ids->device == Device::CPU || dev != Device::CUDA) {
            host_ids = ids->device == Device::CPU ? ids : ids->cpu();
            embedding_detail::check_ids(*host_ids, vocab);
        }
        auto indices = ids->device == dev ? ids : ids->to(dev);
        int64_t n = indices->size();

        std::vector<int64_t> out_shape = indices->shape;
        out_shape.push_back(dim);
        auto out = std::make_shared<Tensor>(out_shape, dev, weight->dtype, false);

#ifdef USE_CUDA
        if (dev == Device::CUDA) {
            // Out-of-range ids produce NaN rows (a kernel cannot throw)
            cuda::embedding_forward(indices->data_ptr<float>(), weight->data_ptr<float>(), out->data_ptr<float>(), n, vocab, dim);
        } else
#endif
        {
            auto wh = (dev == Device::CPU) ? weight : weight->cpu();
            auto oh = (dev == Device::CPU) ? out : std::make_shared<Tensor>(out_shape, Device::CPU, weight->dtype, false);
            const float* id_ptr = host_ids->data_ptr<float>();
            const float* w = wh->data_ptr<float>();
            float* o = oh->data_ptr<float>();
            for (int64_t p = 0; p < n; ++p) {
                int64_t id = (int64_t)id_ptr[p];
                std::copy(w + id * dim, w + (id + 1) * dim, o + p * dim);
            }
            if (dev != Device::CPU) out = oh->to(dev);
        }

        out->requires_grad = needs_grad(weight);
        if (needs_grad(weight)) {
            auto grad_fn = std::make_shared<EmbeddingBackward>(host_ids ? host_ids : indices, vocab, dim);
            grad_fn->add_next_edge(get_grad_edge(weight).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    void visit(ModuleVisitor& v) override { v.parameter("weight", weight); }
};
