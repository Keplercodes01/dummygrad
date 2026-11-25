#pragma once
// softmax(scale * q k^T + mask) v as one op: on a GPU one kernel sequence that stores no score matrices, with a
// matching backward; elsewhere products and a softmax.

#include "functional.h"

// -1e9 where query i may not see key j: j > i + past
inline std::shared_ptr<Tensor> offset_causal_mask(int64_t queries, int64_t keys, int64_t past, Device device) {
    std::vector<float> m((size_t)(queries * keys), 0.0f);
    for (int64_t i = 0; i < queries; ++i)
        for (int64_t j = i + past + 1; j < keys; ++j) m[(size_t)(i * keys + j)] = -1e9f;
    return op_detail::from_values(m, {queries, keys}, device);
}

namespace attention_detail {

// The scores' causal mask. In past tokens are ones the keys have and the queries do not.
inline std::shared_ptr<Tensor> mask_scores(std::shared_ptr<Tensor> scores, const std::shared_ptr<Tensor>& mask, bool causal, int64_t past) {
    if (mask) scores = add(scores, mask);
    if (!causal) return scores;
    int64_t queries = scores->shape[scores->shape.size() - 2], keys = scores->shape.back();
    if (past == 0) return causal_mask(scores);
    return queries + past < keys || queries > 1 ? add(scores, offset_causal_mask(queries, keys, past, scores->device)) : scores;
}

}  // namespace attention_detail

#ifdef USE_CUDA
// q, k, v, out are contiguous [batch, heads, time, head_dim]
struct FusedAttentionBackward : public Node {
    std::shared_ptr<Tensor> q, k, v, out, lse, mask;
    cuda::AttentionShape shape;
    cuda::AttentionImpl impl;  // what the forward ran, so backward reads matching statistics
    DType compute;             // GEMM precision the forward ran with (autocast)
    uint64_t mask_version = 0;

    FusedAttentionBackward(std::shared_ptr<Tensor> q, std::shared_ptr<Tensor> k, std::shared_ptr<Tensor> v,
                           std::shared_ptr<Tensor> out, std::shared_ptr<Tensor> lse, std::shared_ptr<Tensor> mask_in,
                           const cuda::AttentionShape& shape, cuda::AttentionImpl impl, DType compute)
        : q(std::move(q)), k(std::move(k)), v(std::move(v)), out(std::move(out)), lse(std::move(lse)),
          mask(std::move(mask_in)), shape(shape), impl(impl), compute(compute) {
        if (mask) mask_version = mask->storage->version;
    }

    void release_variables() override { q = k = v = out = lse = mask = nullptr; }

    // The same gradients from explicit scores (memory quadratic in the sequence length)
    std::vector<std::shared_ptr<Tensor>> apply_graph(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto qv = graph_input(*this, 0, q), kv = graph_input(*this, 1, k), vv = graph_input(*this, 2, v);
        auto scores = mul_scalar(matmul(qv, transpose(kv, -2, -1)), shape.scale);
        auto p = softmax(attention_detail::mask_scores(scores, mask, shape.causal, shape.past));
        auto gp = matmul(grads[0], transpose(vv, -2, -1));
        auto gs = mul_scalar(mul(p, sub(gp, sum(mul(gp, p), -1, true))), shape.scale);
        return {matmul(gs, kv), matmul(transpose(gs, -2, -1), qv), matmul(transpose(p, -2, -1), grads[0])};
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !q || !k || !v || !out) return {nullptr, nullptr, nullptr};
        if (mask && mask->storage->version != mask_version) throw std::runtime_error("attention: the mask was modified in place after the forward pass");
        auto grad_out = make_contiguous(grads[0]->to(out->dtype));
        auto wanted = [&](size_t i) { return impl == cuda::AttentionImpl::Tiled || (i < next_edges.size() && next_edges[i].function); };
        auto like = [](const std::shared_ptr<Tensor>& t) { return std::make_shared<Tensor>(t->shape, t->device, t->dtype, false); };
        auto gq = wanted(0) ? like(q) : nullptr, gk = wanted(1) ? like(k) : nullptr, gv = wanted(2) ? like(v) : nullptr;
        auto data = [](const std::shared_ptr<Tensor>& t) { return t ? t->data_ptr<void>() : nullptr; };
        cuda::attention_backward(q->data_ptr<void>(), k->data_ptr<void>(), v->data_ptr<void>(), out->data_ptr<void>(),
                                 grad_out->data_ptr<void>(), lse->data_ptr<float>(), data(gq), data(gk), data(gv),
                                 shape, impl, compute, q->dtype);
        return {gq, gk, gv};
    }
};
#endif

// q is [..., queries, dim]; k and v are [..., keys, dim] with the same leading dimensions. mask broadcasts to
// [..., queries, keys] and is added to the scores (a large negative value blocks). With causal, query i sees the keys
// j <= i + past.
inline std::shared_ptr<Tensor> fused_attention(const std::shared_ptr<Tensor>& q_in, const std::shared_ptr<Tensor>& k_in,
                                               const std::shared_ptr<Tensor>& v_in, const std::shared_ptr<Tensor>& mask,
                                               bool causal, float scale, int64_t past = 0) {
    DG_PROFILE_OP("attention");
#ifdef USE_CUDA
    auto leading = [](const std::shared_ptr<Tensor>& t) { return std::vector<int64_t>(t->shape.begin(), t->shape.end() - 2); };
    bool fused = q_in->device == Device::CUDA && k_in->device == Device::CUDA && v_in->device == Device::CUDA &&
                 q_in->ndim() >= 2 && k_in->ndim() >= 2 && v_in->ndim() >= 2 && leading(q_in) == leading(k_in) &&
                 leading(q_in) == leading(v_in) && q_in->shape.back() == k_in->shape.back() && k_in->shape == v_in->shape &&
                 (!mask || (mask->device == Device::CUDA && !needs_grad(mask)));
    if (fused) {
        auto lead = leading(q_in);
        cuda::AttentionShape s;
        s.queries = (int)q_in->shape[lead.size()];
        s.keys = (int)k_in->shape[lead.size()];
        s.head_dim = (int)q_in->shape.back();
        s.scale = scale;
        s.causal = causal;
        s.past = (int)past;
        for (size_t d = 0; d < lead.size(); ++d) (lead.size() == 2 && d == 1 ? s.heads : s.batch) *= (int)lead[d];
        auto full = lead;
        full.push_back(s.queries);
        full.push_back(s.keys);
        std::shared_ptr<Tensor> m;
        if (mask) {
            m = op_detail::strided(mask);
            if (op_detail::broadcast_shapes(m->shape, full, "attention mask") != full) throw std::invalid_argument("attention: mask " + shape_string(m->shape) + " does not fit scores " + shape_string(full));
            auto strides = op_detail::broadcast_strides(*m, full);
            if (lead.size() > 2) {
                // More leading dimensions than the kernels know: one matrix per batch entry, unless it repeats anyway
                bool repeats = std::all_of(strides.begin(), strides.begin() + (std::ptrdiff_t)lead.size(), [](int64_t x) { return x == 0; });
                if (!repeats) m = contiguous(expand(m, full));
                s.mask_stride[0] = repeats ? 0 : (int64_t)s.queries * s.keys;
                s.mask_stride[2] = repeats ? strides[lead.size()] : s.keys;
                s.mask_stride[3] = repeats ? strides[lead.size() + 1] : 1;
            } else {
                for (size_t d = 0; d < lead.size(); ++d) s.mask_stride[d] = strides[d];
                s.mask_stride[2] = strides[lead.size()];
                s.mask_stride[3] = strides[lead.size() + 1];
            }
            s.mask = m->data_ptr<float>();
        }
        // The tiled kernels and a mix of dtypes run in fp32
        auto q = q_in, k = k_in, v = v_in;
        if (!(native_half(q) && native_half(k) && native_half(v) && cuda::attention_impl(s) == cuda::AttentionImpl::Gemm)) {
            q = float32_input(q);
            k = float32_input(k);
            v = float32_input(v);
        }
        auto qc = make_contiguous(q), kc = make_contiguous(k), vc = make_contiguous(v);
        // q, k and v carry the weights' requires_grad even when the model's input has none
        bool grad = grad_enabled() && (q->requires_grad || k->requires_grad || v->requires_grad);
        auto out = std::make_shared<Tensor>(q->shape, Device::CUDA, q->dtype, grad);
        auto lse_shape = lead;
        lse_shape.push_back(s.queries);
        auto lse = std::make_shared<Tensor>(lse_shape, Device::CUDA, DType::Float32, false);
        DType compute = autocast_dtype();
        auto impl = cuda::attention_forward(qc->data_ptr<void>(), kc->data_ptr<void>(), vc->data_ptr<void>(), out->data_ptr<void>(),
                                            lse->data_ptr<float>(), s, compute, q->dtype);
        if (grad) {
            auto grad_fn = std::make_shared<FusedAttentionBackward>(qc, kc, vc, saved_output(out), lse, m ? m->detach() : nullptr, s, impl, compute);
            grad_fn->add_next_edge(get_grad_edge(q).function);
            grad_fn->add_next_edge(get_grad_edge(k).function);
            grad_fn->add_next_edge(get_grad_edge(v).function);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    auto scores = mul_scalar(matmul(q_in, transpose(k_in, -2, -1)), scale);
    return matmul(softmax(attention_detail::mask_scores(scores, mask, causal, past)), v_in);
}
