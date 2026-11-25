#pragma once
// create_graph support for the backward nodes of functional.h: each returns the gradients its apply()
// computes, built from differentiable ops. Included last from functional.h.

#include "conv_ops.h"

inline TensorList ContiguousBackward::apply_graph(const TensorList& g) { return {g[0]}; }
inline TensorList CastBackward::apply_graph(const TensorList& g) { return {cast(g[0], input_dtype)}; }
inline TensorList ReluBackward::apply_graph(const TensorList& g) { return {mul(g[0], gt(a, 0.0f))}; }
inline TensorList TanhBackward::apply_graph(const TensorList& g) {
    return {mul(g[0], rsub_scalar(1.0f, square(graph_output(*this, out_val))))};
}
inline TensorList SigmoidBackward::apply_graph(const TensorList& g) {
    auto y = graph_output(*this, out_val);
    return {mul(g[0], mul(y, rsub_scalar(1.0f, y)))};
}
inline TensorList GeluBackward::apply_graph(const TensorList& g) {
    auto x = graph_input(*this, 0, a);
    auto t = tanh(mul_scalar(add(x, mul_scalar(pow(x, 3.0f), coeff)), sqrt_2_over_pi));
    auto d_inner = mul_scalar(add_scalar(mul_scalar(square(x), 3.0f * coeff), 1.0f), sqrt_2_over_pi);
    auto d = add(mul_scalar(add_scalar(t, 1.0f), 0.5f), mul(mul_scalar(x, 0.5f), mul(rsub_scalar(1.0f, square(t)), d_inner)));
    return {mul(g[0], d)};
}
inline TensorList SiluBackward::apply_graph(const TensorList& g) {
    auto x = graph_input(*this, 0, a);
    auto s = sigmoid(x);
    return {mul(g[0], mul(s, add_scalar(mul(x, rsub_scalar(1.0f, s)), 1.0f)))};
}
inline TensorList LeakyReluBackward::apply_graph(const TensorList& g) { return {mul(g[0], where(gt(a, 0.0f), 1.0f, full_like(a, negative_slope)))}; }
inline TensorList BroadcastBinaryBackward::apply_graph(const TensorList& g) {
    bool need_x = next_edges.size() > 0 && next_edges[0].function, need_y = next_edges.size() > 1 && next_edges[1].function;
    auto xv = graph_input(*this, 0, x), yv = graph_input(*this, 1, y);
    TensorPtr gx, gy;
    switch (op) {
        case BinaryOp::Add:
            if (need_x) gx = g[0];
            if (need_y) gy = g[0];
            break;
        case BinaryOp::Sub:
            if (need_x) gx = g[0];
            if (need_y) gy = neg(g[0]);
            break;
        case BinaryOp::Mul:
            if (need_x) gx = mul(g[0], yv);
            if (need_y) gy = mul(g[0], xv);
            break;
        default:
            if (need_x) gx = div(g[0], yv);
            if (need_y) gy = neg(div(mul(g[0], xv), square(yv)));
            break;
    }
    return {gx ? sum_to(gx, x_shape) : nullptr, gy ? sum_to(gy, y_shape) : nullptr};
}
inline TensorList ScalarOpBackward::apply_graph(const TensorList& g) {
    switch (op) {
        case ScalarOp::Add: case ScalarOp::Sub: return {g[0]};
        case ScalarOp::Mul: return {mul_scalar(g[0], scalar)};
        case ScalarOp::Div: return {div_scalar(g[0], scalar)};
        case ScalarOp::RSub: return {neg(g[0])};
        default: return {mul_scalar(div(g[0], square(graph_input(*this, 0, a))), -scalar)};
    }
}
inline TensorList ConcatBackward::apply_graph(const TensorList& g) {
    int nd = (int)a_shape.size();
    int dim = axis == 0 ? nd - 2 : nd - 1;
    int64_t na = a_shape[(size_t)dim], nb = b_shape[(size_t)dim];
    return {narrow(g[0], dim, 0, na), narrow(g[0], dim, na, nb)};
}
inline TensorList CrossEntropyBackward::apply_graph(const TensorList& g) {
    auto p = graph_input(*this, 0, pred);
    return {mul(reshape(g[0], {}), mul_scalar(div(target, add_scalar(p, 1e-8f)), -1.0f / (float)num_samples))};
}
inline TensorList FusedCrossEntropyBackward::apply_graph(const TensorList& g) {
    auto l = reshape(graph_input(*this, 0, logits), {rows, vocab});
    auto t = reshape(targets, {rows, 1});
    auto keep = ne(t, (float)ignore_index);
    auto one_hot = scatter(zeros_like(l->detach()), 1, where(keep, t, 0.0f), 1.0f);
    auto d = mul(sub(softmax(l), one_hot), keep);
    auto scale = div(reshape(g[0], {}), clamp_min(reshape(count, {}), 1.0f));
    return {reshape(mul(d, scale), logits->shape)};
}
inline TensorList MseBackward::apply_graph(const TensorList& g) {
    return {mul(mul_scalar(sub(graph_input(*this, 0, pred), target), 2.0f / (float)n), reshape(g[0], {}))};
}
inline TensorList L1LossBackward::apply_graph(const TensorList& g) {
    return {mul(div_scalar(sign(sub(pred, target)), (float)n), reshape(g[0], {}))};
}
inline TensorList BoolMaskBackward::apply_graph(const TensorList& g) { return {mul(g[0], ne(m, 0.0f))}; }
inline TensorList CausalMaskBackward::apply_graph(const TensorList& g) { return {tril(g[0])}; }
inline TensorList SliceBackward::apply_graph(const TensorList& g) {
    int d = dim;
    int64_t s = start;
    auto shape = g[0]->shape;
    return {op_detail::view_scatter("slice", g[0], in_shape, [shape, d, s](const std::vector<int64_t>& base) {
        return op_detail::ViewGeometry{shape, base, s * base[(size_t)d]};
    })};
}
inline TensorList SoftmaxBackward::apply_graph(const TensorList& g) {
    auto y = graph_output(*this, s);
    return {mul(y, sub(g[0], sum(mul(g[0], y), -1, true)))};
}
inline TensorList TransposeBackward::apply_graph(const TensorList& g) { return {transpose(g[0], dim0, dim1)}; }
inline TensorList ViewBackward::apply_graph(const TensorList& g) { return {reshape(g[0], in_shape)}; }
inline TensorList Im2ColBackward::apply_graph(const TensorList& g) {
    int64_t B = x_shape[0], C = x_shape[1];
    auto w = conv_detail::windows(x_shape, 2, {kH, kW}, {strideH, strideW}, {padH, padW}, {1, 1}, "im2col");
    return {conv_detail::fold_windows(reshape(g[0], {B * C, w.kernel_size(), w.out_size()}), x_shape, w)};
}
namespace op_detail {
// Gradients of a @ b (trans_b: a @ b^T, b 2-D) for the output gradient g, as differentiable ops
inline TensorList matmul_grads(const TensorPtr& g, const TensorPtr& a, const TensorPtr& b, const std::vector<int64_t>& a_shape,
                               const std::vector<int64_t>& b_shape, bool need_a, bool need_b, bool trans_b) {
    TensorPtr ga, gb;
    if (need_a) ga = sum_to(matmul(g, trans_b ? b : transpose(b, -2, -1)), a_shape);
    if (need_b) {
        if (trans_b) {
            int64_t N = b_shape[0], K = b_shape[1];
            gb = matmul(transpose(reshape(g, {g->size() / N, N}), 0, 1), reshape(a, {a->size() / K, K}));
        } else {
            gb = sum_to(matmul(transpose(a, -2, -1), g), b_shape);
        }
    }
    return {ga, gb};
}
inline TensorPtr bias_grad(const TensorPtr& g, const std::vector<int64_t>& bias_shape) {
    int64_t cols = g->shape.back();
    return reshape(sum(reshape(g, {g->size() / cols, cols}), 0, false), bias_shape);
}
}  // namespace op_detail
inline TensorList MatmulBackward::apply_graph(const TensorList& g) {
    bool need_a = next_edges.size() > 0 && next_edges[0].function, need_b = next_edges.size() > 1 && next_edges[1].function;
    auto out = op_detail::matmul_grads(g[0], graph_input(*this, 0, a), graph_input(*this, 1, b), a_shape, b_shape, need_a, need_b, trans_b);
    if (has_bias) out.push_back(next_edges.size() > 2 && next_edges[2].function ? op_detail::bias_grad(g[0], bias_shape) : nullptr);
    return out;
}
#ifdef USE_CUDA
inline TensorList CudaMatmulBackward::apply_graph(const TensorList& g) {
    // The saved operands hold the inputs' values as the GEMM read them. At fp32 those are the inputs'
    // own values, which is what differentiating again needs.
    if (compute != DType::Float32) throw std::runtime_error("create_graph=true needs matmuls computed in fp32 (no autocast)");
    auto live = [&](size_t i, const std::shared_ptr<Tensor>& saved) { return saved ? graph_input(*this, i, saved) : nullptr; };
    auto out = op_detail::matmul_grads(g[0], live(0, a), live(1, b), a_shape, b_shape, grad_a, grad_b, trans_b);
    if (!bias_shape.empty()) out.push_back(grad_bias ? op_detail::bias_grad(g[0], bias_shape) : nullptr);
    return out;
}
#endif
