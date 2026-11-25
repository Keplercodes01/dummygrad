#pragma once
// Reductions over any dimensions, cumulative ops, softmax over any dimension, norms and statistics.
// keepdim defaults to true where a dimension is given (these ops predate this file) and full reductions
// return a 0-dimensional tensor. Included from functional.h.

#include "shape_ops.h"

namespace op_detail {

inline std::vector<int> wrap_dims(const std::vector<int>& dims, int64_t ndim, const char* who) {
    std::vector<int> out;
    for (int d : dims) out.push_back(wrap_dim(d, ndim, who));
    std::sort(out.begin(), out.end());
    if (std::adjacent_find(out.begin(), out.end()) != out.end()) throw std::invalid_argument(std::string(who) + ": repeated dimension");
    return out;
}
inline std::vector<int> all_dims(const TensorPtr& x) {
    std::vector<int> dims((size_t)x->ndim());
    std::iota(dims.begin(), dims.end(), 0);
    return dims;
}
inline std::vector<int64_t> reduced_shape(const std::vector<int64_t>& shape, const std::vector<int>& dims, bool keepdim) {
    std::vector<int64_t> out;
    for (int d = 0; d < (int)shape.size(); ++d) {
        bool gone = std::find(dims.begin(), dims.end(), d) != dims.end();
        if (!gone) out.push_back(shape[(size_t)d]);
        else if (keepdim) out.push_back(1);
    }
    return out;
}

// The value of a reduction over sorted dims, with those dims kept as 1. Neighbouring dims go in one
// pass; associative ops (and the map of SumSquares etc. in the first pass only) handle the rest.
inline TensorPtr reduce_value(kern::Reduce op, const TensorPtr& x_in, const std::vector<int>& dims) {
    auto x = make_contiguous(x_in->detach());
    auto shape = x->shape;
    if (dims.empty() && op == kern::Reduce::SumSquares) return square(x);
    if (dims.empty() && op == kern::Reduce::SumAbs) return abs(x);
    size_t end = dims.size();
    while (end > 0) {
        size_t begin = end - 1;
        while (begin > 0 && dims[begin - 1] == dims[begin] - 1) --begin;
        int64_t outer = 1, K = 1, inner = 1;
        for (int d = 0; d < (int)shape.size(); ++d) {
            if (d < dims[begin]) outer *= shape[(size_t)d];
            else if (d <= dims[end - 1]) K *= shape[(size_t)d];
            else inner *= shape[(size_t)d];
        }
        for (size_t i = begin; i < end; ++i) shape[(size_t)dims[i]] = 1;
        auto out = empty(shape, x->device);
        DG_KERNEL(x->device, reduce(op, x->data_ptr<float>(), out->data_ptr<float>(), outer, K, inner));
        x = out;
        if (op == kern::Reduce::SumSquares || op == kern::Reduce::SumAbs || op == kern::Reduce::CountNonzero) op = kern::Reduce::Sum;
        end = begin;
    }
    return x;
}

inline int64_t count_over(const TensorPtr& x, const std::vector<int>& dims) {
    int64_t n = 1;
    for (int d : dims) n *= x->shape[(size_t)d];
    return n;
}

}  // namespace op_detail

inline TensorPtr sum(const TensorPtr& x_in, const std::vector<int>& dims_in, bool keepdim = true) {
    DG_PROFILE_OP("sum");
    auto x = op_detail::strided(x_in);
    auto dims = op_detail::wrap_dims(dims_in, x->ndim(), "sum");
    auto kept = op_detail::reduce_value(kern::Reduce::Sum, x, dims);
    auto out = kept->_reshape(op_detail::reduced_shape(x->shape, dims, keepdim))->detach();
    auto x_shape = x->shape, kept_shape = kept->shape;
    return make_op("Sum", out, {x}, {}, [x_shape, kept_shape](const OpContext&, const TensorPtr& g) -> TensorList {
        return {contiguous(expand(reshape(g, kept_shape), x_shape))};
    });
}
inline TensorPtr sum(const TensorPtr& x, int dim, bool keepdim = true) { return sum(x, std::vector<int>{dim}, keepdim); }
inline TensorPtr sum(const TensorPtr& x) { return sum(x, op_detail::all_dims(x), false); }

inline TensorPtr mean(const TensorPtr& x, const std::vector<int>& dims, bool keepdim = true) {
    DG_PROFILE_OP("mean");
    int64_t n = op_detail::count_over(x, op_detail::wrap_dims(dims, x->ndim(), "mean"));
    return div_scalar(sum(x, dims, keepdim), (float)n);
}
inline TensorPtr mean(const TensorPtr& x, int dim, bool keepdim = true) { return mean(x, std::vector<int>{dim}, keepdim); }
inline TensorPtr mean(const TensorPtr& x) { return mean(x, op_detail::all_dims(x), false); }

// Reduces x to a shape it was broadcast from
inline TensorPtr sum_to(const TensorPtr& x, const std::vector<int64_t>& shape) {
    if (x->shape == shape) return x;
    int extra = (int)(x->shape.size() - shape.size());
    std::vector<int> dims;
    for (int d = 0; d < (int)x->shape.size(); ++d) {
        if (d < extra || (shape[(size_t)(d - extra)] == 1 && x->shape[(size_t)d] != 1)) dims.push_back(d);
    }
    auto y = dims.empty() ? x : sum(x, dims, true);
    return reshape(y, shape);
}

// Variance about the mean, divided by n - correction (0, the default: population variance)
inline TensorPtr var(const TensorPtr& x, const std::vector<int>& dims_in, bool keepdim = true, int correction = 0) {
    DG_PROFILE_OP("var");
    auto dims = op_detail::wrap_dims(dims_in, x->ndim(), "var");
    int64_t n = op_detail::count_over(x, dims);
    auto centered = sub(x, mean(x, dims, true));
    return div_scalar(sum(square(centered), dims, keepdim), (float)(n - correction));
}
inline TensorPtr var(const TensorPtr& x, int dim, bool keepdim = true, int correction = 0) { return var(x, std::vector<int>{dim}, keepdim, correction); }
inline TensorPtr var(const TensorPtr& x) { return var(x, op_detail::all_dims(x), false); }

namespace op_detail {
// sqrt whose gradient is 0 where the value is 0, for norms and standard deviations
inline TensorPtr safe_sqrt(const TensorPtr& x) {
    return unary("sqrt", kern::Unary::Sqrt, x, 0, 0, {kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto y = c.output();
        auto zero = eq(y, 0.0f);
        return {where(zero, 0.0f, div(mul_scalar(g, 0.5f), where(zero, 1.0f, y)))};
    });
}
}  // namespace op_detail

inline TensorPtr std_dev(const TensorPtr& x, const std::vector<int>& dims, bool keepdim = true, int correction = 0) {
    return op_detail::safe_sqrt(var(x, dims, keepdim, correction));
}
inline TensorPtr std_dev(const TensorPtr& x, int dim, bool keepdim = true, int correction = 0) { return std_dev(x, std::vector<int>{dim}, keepdim, correction); }
inline TensorPtr std_dev(const TensorPtr& x) { return std_dev(x, op_detail::all_dims(x), false); }

// Largest / smallest values over dims. Elements tied for the extreme share the gradient equally.
namespace op_detail {
inline TensorPtr extreme(const char* name, kern::Reduce op, const TensorPtr& x_in, const std::vector<int>& dims_in, bool keepdim) {
    DG_PROFILE_OP(name);
    auto x = strided(x_in);
    auto dims = wrap_dims(dims_in, x->ndim(), name);
    if (x->size() == 0) throw std::invalid_argument(std::string(name) + ": empty tensor");
    auto kept = reduce_value(op, x, dims);
    auto out = kept->_reshape(reduced_shape(x->shape, dims, keepdim))->detach();
    auto kept_shape = kept->shape;
    return make_op(name, out, {x}, {0}, [kept, kept_shape, dims](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto mask = eq(c.input(0), kept);
        return {mul(reshape(g, kept_shape), div(mask, sum(mask, dims, true)))};
    });
}
}  // namespace op_detail
inline TensorPtr amax(const TensorPtr& x, const std::vector<int>& dims, bool keepdim = true) { return op_detail::extreme("amax", kern::Reduce::Max, x, dims, keepdim); }
inline TensorPtr amin(const TensorPtr& x, const std::vector<int>& dims, bool keepdim = true) { return op_detail::extreme("amin", kern::Reduce::Min, x, dims, keepdim); }
inline TensorPtr amax(const TensorPtr& x, int dim, bool keepdim = true) { return amax(x, std::vector<int>{dim}, keepdim); }
inline TensorPtr amin(const TensorPtr& x, int dim, bool keepdim = true) { return amin(x, std::vector<int>{dim}, keepdim); }
inline TensorPtr amax(const TensorPtr& x) { return amax(x, op_detail::all_dims(x), false); }
inline TensorPtr amin(const TensorPtr& x) { return amin(x, op_detail::all_dims(x), false); }

namespace op_detail {
inline TensorPtr arg_extreme(const char* name, bool largest, const TensorPtr& x_in, int dim_in, bool keepdim) {
    DG_PROFILE_OP(name);
    NoGradGuard no_grad;
    auto x = dense(x_in);
    int dim = wrap_dim(dim_in, x->ndim(), name);
    auto s = split_at(x->shape, dim);
    if (s.K == 0) throw std::invalid_argument(std::string(name) + ": empty dimension");
    if (s.K > (1 << 24)) throw std::invalid_argument(std::string(name) + ": indices past 2^24 do not fit in float32");
    auto out = empty(reduced_shape(x->shape, {dim}, keepdim), x->device);
    DG_KERNEL(x->device, arg_reduce(largest, x->data_ptr<float>(), out->data_ptr<float>(), s.outer, s.K, s.inner));
    return out;
}
}  // namespace op_detail
// Position of the (first) largest / smallest element along dim, as floats
inline TensorPtr argmax(const TensorPtr& x, int dim = -1, bool keepdim = false) { return op_detail::arg_extreme("argmax", true, x, dim, keepdim); }
inline TensorPtr argmin(const TensorPtr& x, int dim = -1, bool keepdim = false) { return op_detail::arg_extreme("argmin", false, x, dim, keepdim); }

// Nonzero tests and counts (no gradient)
inline TensorPtr count_nonzero(const TensorPtr& x, const std::vector<int>& dims, bool keepdim = false) {
    NoGradGuard no_grad;
    auto f = op_detail::strided(x);
    auto d = op_detail::wrap_dims(dims, f->ndim(), "count_nonzero");
    if (d.empty()) return ne(f, 0.0f);
    return op_detail::reduce_value(kern::Reduce::CountNonzero, f, d)->_reshape(op_detail::reduced_shape(f->shape, d, keepdim));
}
inline bool any(const TensorPtr& x) { return x->size() > 0 && count_nonzero(x, op_detail::all_dims(x))->item() > 0.0f; }
inline bool all(const TensorPtr& x) { return !any(eq(x, 0.0f)); }

// ---------------------------------------------------------------------------------------------
// Cumulative ops
// ---------------------------------------------------------------------------------------------

namespace op_detail {
inline TensorPtr cumulate_value(kern::Cumulate op, const TensorPtr& x, int dim, TensorPtr* index = nullptr) {
    auto s = split_at(x->shape, dim);
    auto out = empty(x->shape, x->device);
    float* ip = nullptr;
    if (index) {
        if (s.K > (1 << 24)) throw std::invalid_argument("cummax/cummin: indices past 2^24 do not fit in float32");
        *index = empty(x->shape, x->device);
        ip = (*index)->data_ptr<float>();
    }
    if (x->size() > 0) DG_KERNEL(x->device, cumulate(op, x->data_ptr<float>(), out->data_ptr<float>(), ip, s.outer, s.K, s.inner));
    return out;
}
}  // namespace op_detail

inline TensorPtr cumsum(const TensorPtr& x_in, int dim_in) {
    DG_PROFILE_OP("cumsum");
    auto x = op_detail::dense(x_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "cumsum");
    return make_op("cumsum", op_detail::cumulate_value(kern::Cumulate::Sum, x, dim), {x}, {}, [dim](const OpContext&, const TensorPtr& g) -> TensorList {
        return {flip(cumsum(flip(g, {dim}), dim), {dim})};
    });
}
inline TensorPtr cumprod(const TensorPtr& x_in, int dim_in) {
    DG_PROFILE_OP("cumprod");
    auto x = op_detail::dense(x_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "cumprod");
    return make_op("cumprod", op_detail::cumulate_value(kern::Cumulate::Prod, x, dim), {x}, {0, kSaveOutput}, [dim](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        if (all(v)) return {div(flip(cumsum(flip(mul(g, c.output()), {dim}), dim), {dim}), v)};
        // With zeros: grad_i = (product before i) * r_i, r_i = g_i + x_{i+1} r_{i+1}
        int64_t K = v->shape[(size_t)dim];
        auto first = v->shape;
        first[(size_t)dim] = 1;
        auto before = op_detail::filled(first, v->device, 1.0f);
        if (K > 1) before = cat({before, narrow(cumprod(v, dim), dim, 0, K - 1)}, dim);
        TensorList r((size_t)K);
        r[(size_t)K - 1] = narrow(g, dim, K - 1, 1);
        for (int64_t i = K - 2; i >= 0; --i) r[(size_t)i] = add(narrow(g, dim, i, 1), mul(narrow(v, dim, i + 1, 1), r[(size_t)i + 1]));
        return {mul(before, cat(r, dim))};
    });
}
inline TensorPtr logcumsumexp(const TensorPtr& x_in, int dim_in) {
    DG_PROFILE_OP("logcumsumexp");
    auto x = op_detail::dense(x_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "logcumsumexp");
    // d out_j / d x_i = exp(x_i - out_j) for j >= i
    return make_op("logcumsumexp", op_detail::cumulate_value(kern::Cumulate::LogSumExp, x, dim), {x}, {0, kSaveOutput}, [dim](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto weighted = mul(g, exp(neg(c.output())));
        return {mul(exp(c.input(0)), flip(cumsum(flip(weighted, {dim}), dim), {dim}))};
    });
}

// Product over dims. The gradient is exact even where elements are 0.
inline TensorPtr prod(const TensorPtr& x_in, int dim_in, bool keepdim = true) {
    DG_PROFILE_OP("prod");
    auto x = op_detail::strided(x_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "prod");
    auto kept = op_detail::reduce_value(kern::Reduce::Prod, x, {dim});
    auto out = kept->_reshape(op_detail::reduced_shape(x->shape, {dim}, keepdim))->detach();
    auto kept_shape = kept->shape;
    return make_op("prod", out, {x}, {0}, [dim, kept_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        int64_t K = v->shape[(size_t)dim];
        auto one = op_detail::filled(kept_shape, v->device, 1.0f);
        auto others = one;
        if (K > 1) {
            auto before = cat({one, cumprod(narrow(v, dim, 0, K - 1), dim)}, dim);
            auto after = flip(cat({one, cumprod(narrow(flip(v, {dim}), dim, 0, K - 1), dim)}, dim), {dim});
            others = mul(before, after);
        }
        return {mul(reshape(g, kept_shape), others)};
    });
}
inline TensorPtr prod(const TensorPtr& x) { return prod(reshape(x, {x->size()}), 0, false); }

// ---------------------------------------------------------------------------------------------
// Softmax family
// ---------------------------------------------------------------------------------------------

namespace op_detail {
inline TensorPtr softmax_any(const TensorPtr& x, int dim, bool log) {
    auto s = split_at(x->shape, dim);
    auto out = empty(x->shape, x->device);
    if (x->size() > 0) DG_KERNEL(x->device, softmax(x->data_ptr<float>(), out->data_ptr<float>(), s.outer, s.K, s.inner, log));
    if (log) {
        return make_op("log_softmax", out, {x}, {kSaveOutput}, [dim](const OpContext& c, const TensorPtr& g) -> TensorList {
            return {sub(g, mul(exp(c.output()), sum(g, dim, true)))};
        });
    }
    return make_op("softmax", out, {x}, {kSaveOutput}, [dim](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto y = c.output();
        return {mul(y, sub(g, sum(mul(g, y), dim, true)))};
    });
}
}  // namespace op_detail

inline TensorPtr log_softmax(const TensorPtr& x_in, int dim = -1) {
    DG_PROFILE_OP("log_softmax");
    auto x = op_detail::dense(x_in);
    return op_detail::softmax_any(x, op_detail::wrap_dim(dim, x->ndim(), "log_softmax"), true);
}
inline TensorPtr softmin(const TensorPtr& x, int dim = -1) { return softmax(neg(x), dim); }

inline TensorPtr logsumexp(const TensorPtr& x, const std::vector<int>& dims_in, bool keepdim = false) {
    DG_PROFILE_OP("logsumexp");
    auto dims = op_detail::wrap_dims(dims_in, x->ndim(), "logsumexp");
    auto shift = amax(x->detach(), dims, true);
    shift = where(is_finite(shift), shift, 0.0f);
    auto kept = add(log(sum(exp(sub(x, shift)), dims, true)), shift);
    return reshape(kept, op_detail::reduced_shape(x->shape, dims, keepdim));
}
inline TensorPtr logsumexp(const TensorPtr& x, int dim, bool keepdim = false) { return logsumexp(x, std::vector<int>{dim}, keepdim); }

// ---------------------------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------------------------

// p-norm over dims; p = inf and -inf give the largest and smallest magnitude, p = 0 the count of nonzeros
inline TensorPtr norm(const TensorPtr& x, float p, const std::vector<int>& dims, bool keepdim = false) {
    DG_PROFILE_OP("norm");
    if (p == 2.0f) return op_detail::safe_sqrt(sum(square(x), dims, keepdim));
    if (p == 1.0f) return sum(abs(x), dims, keepdim);
    if (p == 0.0f) return count_nonzero(x, dims, keepdim);
    if (std::isinf(p)) return p > 0 ? amax(abs(x), dims, keepdim) : amin(abs(x), dims, keepdim);
    return pow(sum(pow(abs(x), p), dims, keepdim), 1.0f / p);
}
inline TensorPtr norm(const TensorPtr& x, float p, int dim, bool keepdim = false) { return norm(x, p, std::vector<int>{dim}, keepdim); }
inline TensorPtr norm(const TensorPtr& x, float p = 2.0f) { return norm(x, p, op_detail::all_dims(x), false); }

// x / max(||x||_p, eps) along dim
inline TensorPtr normalize(const TensorPtr& x, float p = 2.0f, int dim = 1, float eps = 1e-12f) {
    return div(x, clamp_min(norm(x, p, dim, true), eps));
}
inline TensorPtr cosine_similarity(const TensorPtr& a, const TensorPtr& b, int dim = 1, float eps = 1e-8f) {
    auto dot = sum(mul(a, b), dim, false);
    auto scale = mul(clamp_min(norm(a, 2.0f, dim), eps), clamp_min(norm(b, 2.0f, dim), eps));
    return div(dot, scale);
}
