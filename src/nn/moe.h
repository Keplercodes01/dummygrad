#pragma once
// Mixture of experts: every token goes through the top_k experts its router prefers, weighted by the router's
// probabilities (Shazeer et al. 2017, Mixtral).

#include "functional.h"
#include "linear.h"
#include "modern_layers.h"
#include "module.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Feed-forward layer of SwiGLU experts. After each forward, aux_loss holds the load-balancing loss (Fedus et al.
// 2021): 1 when tokens are spread evenly, up to num_experts when one expert gets them all. Add a small multiple of it
// to the training loss.
class MixtureOfExperts : public Module {
public:
    Linear router;
    std::vector<std::shared_ptr<SwiGLU>> experts;
    int num_experts, top_k;
    float router_noise;  // standard deviation of noise on the router's logits while training (0: none)

    std::shared_ptr<Tensor> aux_loss;        // of the last forward
    std::vector<int64_t> tokens_per_expert;  // of the last forward

    MixtureOfExperts(int d_model, int hidden_dim, int num_experts, int top_k = 2, bool bias = false, float router_noise = 0.0f)
        : router(d_model, num_experts, 0.0f, /*use_bias=*/false), num_experts(num_experts), top_k(top_k), router_noise(router_noise) {
        if (num_experts < 1 || top_k < 1 || top_k > num_experts) throw std::invalid_argument("MixtureOfExperts: needs 1 <= top_k <= num_experts");
        for (int e = 0; e < num_experts; ++e) experts.push_back(std::make_shared<SwiGLU>(d_model, hidden_dim, bias));
    }

    // x: [..., d_model]
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x_in) {
        int64_t D = x_in->shape.back(), N = x_in->size() / std::max<int64_t>(1, D);
        auto x = reshape(contiguous(float32_input(x_in)), {N, D});
        auto logits = router.forward(x);
        if (training_ && router_noise > 0.0f) logits = add(logits, mul_scalar(randn_like(logits), router_noise));
        auto probs = softmax(logits);                      // [N, experts]
        auto [weight, choice] = topk(probs, top_k, -1);    // [N, top_k]
        weight = div(weight, sum(weight, -1, true));       // the chosen experts' weights add up to 1

        // Which (token, rank) pairs each expert serves, in token order
        auto picked = choice->to_vector();
        std::vector<std::vector<float>> tokens((size_t)num_experts), ranks((size_t)num_experts);
        for (int64_t i = 0; i < N * top_k; ++i) {
            size_t e = (size_t)picked[(size_t)i];
            tokens[e].push_back((float)(i / top_k));
            ranks[e].push_back((float)i);
        }
        tokens_per_expert.assign((size_t)num_experts, 0);
        auto flat_weight = reshape(weight, {N * top_k, 1});
        auto out = op_detail::filled({N, D}, x->device, 0.0f);
        for (int e = 0; e < num_experts; ++e) {
            int64_t n = (int64_t)tokens[(size_t)e].size();
            tokens_per_expert[(size_t)e] = n;
            if (n == 0) continue;
            auto rows = op_detail::from_values(tokens[(size_t)e], {n}, x->device);
            auto pairs = op_detail::from_values(ranks[(size_t)e], {n}, x->device);
            auto y = experts[(size_t)e]->forward(index_select(x, 0, rows));
            out = index_add(out, 0, rows, mul(y, index_select(flat_weight, 0, pairs)));
        }

        // experts * sum over experts of (share of tokens) * (mean probability)
        std::vector<float> share((size_t)num_experts);
        for (int e = 0; e < num_experts; ++e) share[(size_t)e] = (float)tokens_per_expert[(size_t)e] / (float)std::max<int64_t>(1, N * top_k);
        aux_loss = mul_scalar(sum(mul(mean(probs, 0, false), op_detail::from_values(share, {num_experts}, x->device))), (float)num_experts);
        return reshape(out, x_in->shape);
    }

    void visit(ModuleVisitor& v) override {
        v.module("router", router);
        for (size_t e = 0; e < experts.size(); ++e) v.module("experts." + std::to_string(e), *experts[e]);
    }
};
