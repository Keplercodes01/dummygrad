#pragma once
// Losses with PyTorch's definitions. Reduction defaults to the mean; Sum adds up and None keeps every
// element. cross-entropy targets are class indices stored as floats. Included from functional.h.

#include "contract_ops.h"

enum class Reduction { Mean, Sum, None };

namespace loss_detail {
inline TensorPtr reduce(const TensorPtr& x, Reduction r) {
    switch (r) {
        case Reduction::Mean: return mean(x);
        case Reduction::Sum: return sum(x);
        default: return x;
    }
}
inline void same_shape(const TensorPtr& a, const TensorPtr& b, const char* who) {
    if (a->shape != b->shape) throw std::invalid_argument(std::string(who) + ": shapes " + shape_string(a->shape) + " and " + shape_string(b->shape) + " differ");
}
}  // namespace loss_detail

inline TensorPtr mse_loss(const TensorPtr& x, const TensorPtr& y, Reduction r = Reduction::Mean) {
    loss_detail::same_shape(x, y, "mse_loss");
    return loss_detail::reduce(square(sub(x, y)), r);
}
inline TensorPtr l1_loss(const TensorPtr& x, const TensorPtr& y, Reduction r) {
    loss_detail::same_shape(x, y, "l1_loss");
    return loss_detail::reduce(abs(sub(x, y)), r);
}
inline TensorPtr huber_loss(const TensorPtr& x, const TensorPtr& y, float delta = 1.0f, Reduction r = Reduction::Mean) {
    loss_detail::same_shape(x, y, "huber_loss");
    auto d = sub(x, y);
    auto a = abs(d);
    auto quadratic = mul_scalar(square(d), 0.5f);
    auto linear_part = mul_scalar(sub_scalar(a, 0.5f * delta), delta);
    return loss_detail::reduce(where(le(a, delta), quadratic, linear_part), r);
}
inline TensorPtr smooth_l1_loss(const TensorPtr& x, const TensorPtr& y, float beta = 1.0f, Reduction r = Reduction::Mean) {
    if (beta == 0.0f) return l1_loss(x, y, r);
    return div_scalar(huber_loss(x, y, beta, r), beta);
}
// Negative log likelihood of class indices under log-probabilities [N, C] or [N, C, d1, ...].
// weight [C] scales classes; the mean divides by the total weight of the targets that count.
inline TensorPtr nll_loss(const TensorPtr& log_probs, const TensorPtr& target, const TensorPtr& weight = nullptr,
                          int ignore_index = -100, Reduction r = Reduction::Mean) {
    if (log_probs->ndim() < 2) throw std::invalid_argument("nll_loss: log-probabilities must be [N, C, ...]");
    auto t = op_detail::index_tensor(target, log_probs, "nll_loss");
    auto expected = log_probs->shape;
    expected.erase(expected.begin() + 1);
    if (t->shape != expected) throw std::invalid_argument("nll_loss: target " + shape_string(t->shape) + " does not match " + shape_string(expected));
    auto keep = ne(t, (float)ignore_index);
    auto safe = where(keep, t, 0.0f);
    auto picked = squeeze(gather(log_probs, 1, unsqueeze(safe, 1)), 1);
    auto w = keep;
    if (weight) w = mul(keep, reshape(index_select(weight, 0, reshape(safe, {safe->size()})), safe->shape));
    auto loss = neg(mul(picked, w));
    if (r == Reduction::None) return loss;
    if (r == Reduction::Sum) return sum(loss);
    return div(sum(loss), sum(w));
}
// Cross entropy of logits [N, C, ...] against class indices, with class weights, an ignored index and
// label smoothing. fused_cross_entropy is the single-kernel version for [..., vocab] logits.
inline TensorPtr cross_entropy_loss(const TensorPtr& logits, const TensorPtr& target, const TensorPtr& weight = nullptr,
                                    int ignore_index = -100, float label_smoothing = 0.0f, Reduction r = Reduction::Mean) {
    auto lp = log_softmax(logits, 1);
    auto nll = nll_loss(lp, target, weight, ignore_index, r);
    if (label_smoothing == 0.0f) return nll;
    auto t = op_detail::index_tensor(target, logits, "cross_entropy_loss");
    auto keep = ne(t, (float)ignore_index);
    auto weighted = lp;
    if (weight) {
        std::vector<int64_t> shape((size_t)lp->ndim(), 1);
        shape[1] = lp->shape[1];
        weighted = mul(lp, reshape(weight, shape));
    }
    auto smooth = mul(neg(sum(weighted, 1, false)), keep);
    if (r == Reduction::Sum) smooth = sum(smooth);
    else if (r == Reduction::Mean) {
        auto w = keep;
        if (weight) w = mul(keep, reshape(index_select(weight, 0, reshape(where(keep, t, 0.0f), {t->size()})), t->shape));
        smooth = div(sum(smooth), sum(w));
    }
    return add(mul_scalar(nll, 1.0f - label_smoothing), mul_scalar(smooth, label_smoothing / (float)logits->shape[1]));
}
// Cross entropy against probabilities of the logits' shape (dimension 1 holds classes)
inline TensorPtr soft_cross_entropy_loss(const TensorPtr& logits, const TensorPtr& probs, Reduction r = Reduction::Mean) {
    loss_detail::same_shape(logits, probs, "soft_cross_entropy_loss");
    return loss_detail::reduce(neg(sum(mul(log_softmax(logits, 1), probs), 1, false)), r);
}
inline TensorPtr binary_cross_entropy(const TensorPtr& p, const TensorPtr& y, const TensorPtr& weight = nullptr, Reduction r = Reduction::Mean) {
    loss_detail::same_shape(p, y, "binary_cross_entropy");
    auto log_p = clamp_min(log(p), -100.0f), log_q = clamp_min(log1p(neg(p)), -100.0f);
    auto loss = neg(add(mul(y, log_p), mul(rsub_scalar(1.0f, y), log_q)));
    if (weight) loss = mul(loss, weight);
    return loss_detail::reduce(loss, r);
}
// From logits, without computing a saturating sigmoid; pos_weight scales the positive term
inline TensorPtr binary_cross_entropy_with_logits(const TensorPtr& x, const TensorPtr& y, const TensorPtr& weight = nullptr,
                                                  const TensorPtr& pos_weight = nullptr, Reduction r = Reduction::Mean) {
    loss_detail::same_shape(x, y, "binary_cross_entropy_with_logits");
    TensorPtr loss;
    if (pos_weight) {
        auto scale = add_scalar(mul(y, sub_scalar(pos_weight, 1.0f)), 1.0f);
        loss = add(mul(rsub_scalar(1.0f, y), x), mul(scale, softplus(neg(x))));
    } else {
        loss = add(mul(rsub_scalar(1.0f, y), x), softplus(neg(x)));
    }
    if (weight) loss = mul(loss, weight);
    return loss_detail::reduce(loss, r);
}
// KL(target || input) for log-probability inputs; BatchMean divides the sum by the batch size
inline TensorPtr kl_div(const TensorPtr& log_q, const TensorPtr& p, bool log_target = false, bool batch_mean = true, Reduction r = Reduction::Mean) {
    loss_detail::same_shape(log_q, p, "kl_div");
    auto loss = log_target ? mul(exp(p), sub(p, log_q)) : sub(xlogy(p, p), mul(p, log_q));
    if (batch_mean && r == Reduction::Mean) return div_scalar(sum(loss), (float)log_q->shape[0]);
    return loss_detail::reduce(loss, r);
}
inline TensorPtr poisson_nll_loss(const TensorPtr& x, const TensorPtr& y, bool log_input = true, float eps = 1e-8f, Reduction r = Reduction::Mean) {
    auto loss = log_input ? sub(exp(x), mul(y, x)) : sub(x, mul(y, log(add_scalar(x, eps))));
    return loss_detail::reduce(loss, r);
}
inline TensorPtr gaussian_nll_loss(const TensorPtr& mu, const TensorPtr& y, const TensorPtr& variance, float eps = 1e-6f, Reduction r = Reduction::Mean) {
    auto v = clamp_min(variance, eps);
    return loss_detail::reduce(mul_scalar(add(log(v), div(square(sub(mu, y)), v)), 0.5f), r);
}
inline TensorPtr hinge_embedding_loss(const TensorPtr& x, const TensorPtr& y, float margin = 1.0f, Reduction r = Reduction::Mean) {
    return loss_detail::reduce(where(eq(y, 1.0f), x, relu(rsub_scalar(margin, x))), r);
}
inline TensorPtr soft_margin_loss(const TensorPtr& x, const TensorPtr& y, Reduction r = Reduction::Mean) {
    return loss_detail::reduce(softplus(neg(mul(y, x))), r);
}
inline TensorPtr margin_ranking_loss(const TensorPtr& a, const TensorPtr& b, const TensorPtr& y, float margin = 0.0f, Reduction r = Reduction::Mean) {
    return loss_detail::reduce(relu(add_scalar(neg(mul(y, sub(a, b))), margin)), r);
}
inline TensorPtr cosine_embedding_loss(const TensorPtr& a, const TensorPtr& b, const TensorPtr& y, float margin = 0.0f, Reduction r = Reduction::Mean) {
    auto cosine = cosine_similarity(a, b, (int)a->ndim() - 1);
    return loss_detail::reduce(where(eq(y, 1.0f), rsub_scalar(1.0f, cosine), relu(sub_scalar(cosine, margin))), r);
}
inline TensorPtr triplet_margin_loss(const TensorPtr& anchor, const TensorPtr& positive, const TensorPtr& negative,
                                     float margin = 1.0f, float p = 2.0f, bool swap = false, Reduction r = Reduction::Mean) {
    auto dp = pairwise_distance(anchor, positive, p), dn = pairwise_distance(anchor, negative, p);
    if (swap) dn = minimum(dn, pairwise_distance(positive, negative, p));
    return loss_detail::reduce(relu(add_scalar(sub(dp, dn), margin)), r);
}
// Multi-class hinge: scores [N, C] against class indices
inline TensorPtr multi_margin_loss(const TensorPtr& x, const TensorPtr& target, float p = 1.0f, float margin = 1.0f, Reduction r = Reduction::Mean) {
    auto t = unsqueeze(op_detail::index_tensor(target, x, "multi_margin_loss"), 1);
    auto own = gather(x, 1, t);
    auto violation = relu(add_scalar(sub(x, own), margin));
    if (p != 1.0f) violation = pow(violation, p);
    auto others = scatter(ones_like(x), 1, t, 0.0f);
    return loss_detail::reduce(div_scalar(sum(mul(violation, others), 1, false), (float)x->shape[1]), r);
}
inline TensorPtr multilabel_soft_margin_loss(const TensorPtr& x, const TensorPtr& y, Reduction r = Reduction::Mean) {
    auto loss = binary_cross_entropy_with_logits(x, y, nullptr, nullptr, Reduction::None);
    return loss_detail::reduce(mean(loss, -1, false), r);
}
inline TensorPtr sigmoid_focal_loss(const TensorPtr& x, const TensorPtr& y, float alpha = 0.25f, float gamma = 2.0f, Reduction r = Reduction::Mean) {
    auto p = sigmoid(x);
    auto ce = binary_cross_entropy_with_logits(x, y, nullptr, nullptr, Reduction::None);
    auto pt = add(mul(p, y), mul(rsub_scalar(1.0f, p), rsub_scalar(1.0f, y)));
    auto loss = mul(ce, pow(rsub_scalar(1.0f, pt), gamma));
    if (alpha >= 0.0f) loss = mul(loss, add(mul_scalar(y, alpha), mul_scalar(rsub_scalar(1.0f, y), 1.0f - alpha)));
    return loss_detail::reduce(loss, r);
}
// Connectionist temporal classification (Graves et al., 2006). log_probs [T, N, C]; targets [N, S] of
// class indices; input_lengths and target_lengths give each sequence's used part. With zero_infinity,
// impossible alignments count 0. The forward recursion runs in log space; its gradient comes from it.
inline TensorPtr ctc_loss(const TensorPtr& log_probs, const TensorPtr& targets, const std::vector<int64_t>& input_lengths,
                          const std::vector<int64_t>& target_lengths, int blank = 0, Reduction r = Reduction::Mean, bool zero_infinity = false) {
    DG_PROFILE_OP("ctc_loss");
    if (log_probs->ndim() != 3) throw std::invalid_argument("ctc_loss: log_probs must be [T, N, C]");
    int64_t N = log_probs->shape[1];
    if ((int64_t)input_lengths.size() != N || (int64_t)target_lengths.size() != N) throw std::invalid_argument("ctc_loss: one length per sequence is needed");
    auto ids = op_detail::host_floats(targets);
    int64_t S = N > 0 ? (int64_t)ids.size() / N : 0;
    const float kLogZero = -1e30f;
    Device dev = log_probs->device;
    TensorList losses;
    for (int64_t n = 0; n < N; ++n) {
        int64_t L = target_lengths[(size_t)n], T = input_lengths[(size_t)n];
        if (L < 0 || L > S || T < 1 || T > log_probs->shape[0]) throw std::invalid_argument("ctc_loss: lengths are out of range");
        // Labels with blanks between and around them, and the states each may be entered from
        int64_t M = 2 * L + 1;
        std::vector<float> label((size_t)M, (float)blank), skip((size_t)M, 0.0f);
        for (int64_t i = 0; i < L; ++i) label[(size_t)(2 * i + 1)] = ids[(size_t)(n * S + i)];
        for (int64_t m = 3; m < M; m += 2) skip[(size_t)m] = label[(size_t)m] != label[(size_t)(m - 2)] ? 1.0f : 0.0f;
        auto lab = op_detail::from_values(label, {M}, dev), can_skip = op_detail::from_values(skip, {M}, dev);
        auto emit = index_select(select(log_probs, 1, n), 1, lab);  // [T_max, M]
        std::vector<float> start((size_t)M, 0.0f);
        start[0] = 1.0f;
        if (M > 1) start[1] = 1.0f;
        auto alpha = where(op_detail::from_values(start, {M}, dev), select(emit, 0, 0), kLogZero);
        auto pad1 = op_detail::filled({std::min<int64_t>(1, M)}, dev, kLogZero), pad2 = op_detail::filled({std::min<int64_t>(2, M)}, dev, kLogZero);
        for (int64_t t = 1; t < T; ++t) {
            auto from1 = M > 1 ? cat({pad1, narrow(alpha, 0, 0, M - 1)}, 0) : pad1;
            auto from2 = M > 2 ? cat({pad2, narrow(alpha, 0, 0, M - 2)}, 0) : pad2;
            auto stay = logaddexp(alpha, from1);
            alpha = add(where(can_skip, logaddexp(stay, from2), stay), select(emit, 0, t));
        }
        auto total = M > 1 ? logaddexp(select(alpha, 0, M - 1), select(alpha, 0, M - 2)) : select(alpha, 0, 0);
        auto loss = neg(total);
        if (zero_infinity) loss = where(gt(loss, 1e20f), 0.0f, loss);
        if (r == Reduction::Mean) loss = div_scalar(loss, (float)std::max<int64_t>(1, L));
        losses.push_back(loss);
    }
    return loss_detail::reduce(stack(losses, 0), r);
}
