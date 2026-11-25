#pragma once
// Elementwise ops: functions of one tensor, binary ops with full NumPy broadcasting (both operands may
// broadcast), comparisons (1.0 / 0.0 results), where and clamp. Included from functional.h.

#include "fn_node.h"
#include "functional.h"
#include "kernels/kernels.h"

#include <cmath>
#include <limits>

inline TensorPtr sum_to(const TensorPtr& x, const std::vector<int64_t>& shape);  // reduce_ops.h
inline TensorPtr expand(const TensorPtr& x, const std::vector<int64_t>& shape);

namespace op_detail {

// Dense float32 input, still connected to the graph
inline TensorPtr dense(const TensorPtr& x) {
    if (!x) throw std::invalid_argument("null tensor");
    auto f = float32_input(x);
    if (f->dtype != DType::Float32) throw std::invalid_argument("tensors must be Float32, Float16 or BFloat16");
    return contiguous(f);
}

// Float32 input that may be strided
inline TensorPtr strided(const TensorPtr& x) {
    if (!x) throw std::invalid_argument("null tensor");
    auto f = float32_input(x);
    if (f->dtype != DType::Float32) throw std::invalid_argument("tensors must be Float32, Float16 or BFloat16");
    return f;
}

inline TensorPtr empty(const std::vector<int64_t>& shape, Device device) {
    return std::make_shared<Tensor>(shape, device, DType::Float32, false);
}

inline TensorPtr filled(const std::vector<int64_t>& shape, Device device, float value) {
    auto t = empty(shape, device);
    if (t->size() > 0) t->fill_(value);
    return t;
}

// Host float values as a tensor on device
inline TensorPtr from_values(const std::vector<float>& values, const std::vector<int64_t>& shape, Device device) {
    auto t = empty(shape, Device::CPU);
    if (!values.empty()) t->copy_from_vector(values);
    return device == Device::CPU ? t : t->to(device);
}

inline int wrap_dim(int dim, int64_t ndim, const char* who) {
    int n = (int)std::max<int64_t>(ndim, 1);
    if (dim < -n || dim >= n) {
        throw std::invalid_argument(std::string(who) + ": dim " + std::to_string(dim) + " is out of range for " +
                                    std::to_string(ndim) + " dimensions");
    }
    return dim < 0 ? dim + n : dim;
}

// Sizes before, at and after a dimension
struct Split {
    int64_t outer = 1, K = 1, inner = 1;
};
inline Split split_at(const std::vector<int64_t>& shape, int dim) {
    Split s;
    for (int d = 0; d < (int)shape.size(); ++d) {
        if (d < dim) s.outer *= shape[d];
        else if (d == dim) s.K = shape[d];
        else s.inner *= shape[d];
    }
    return s;
}

inline std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b, const char* who) {
    size_t n = std::max(a.size(), b.size());
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t x = i < n - a.size() ? 1 : a[i - (n - a.size())];
        int64_t y = i < n - b.size() ? 1 : b[i - (n - b.size())];
        if (x != y && x != 1 && y != 1) {
            throw std::invalid_argument(std::string(who) + ": cannot broadcast shapes " + shape_string(a) + " and " + shape_string(b));
        }
        out[i] = x == 1 ? y : x;
    }
    return out;
}

// Strides walking t over a broadcast shape: 0 where t repeats
inline std::vector<int64_t> broadcast_strides(const Tensor& t, const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 0);
    size_t extra = shape.size() - t.shape.size();
    for (size_t d = 0; d < t.shape.size(); ++d) s[extra + d] = t.shape[d] == 1 ? 0 : t.strides[d];
    return s;
}

inline void check_same_device(const TensorPtr& a, const TensorPtr& b, const char* who) {
    if (a->device != b->device) throw std::invalid_argument(std::string(who) + ": tensors are on different devices");
}

inline TensorPtr unary(const char* name, kern::Unary op, const TensorPtr& x_in, float p, float q,
                       std::initializer_list<int> save, FnBackward::Vjp vjp) {
    DG_PROFILE_OP(name);
    auto x = dense(x_in);
    auto out = empty(x->shape, x->device);
    DG_KERNEL(x->device, unary(op, x->data_ptr<float>(), out->data_ptr<float>(), x->size(), p, q));
    return make_op(name, out, {x}, save, std::move(vjp));
}

// Same, for results that carry no gradient (comparisons, rounding)
inline TensorPtr unary_const(const char* name, kern::Unary op, const TensorPtr& x_in, float p = 0.0f, float q = 0.0f) {
    DG_PROFILE_OP(name);
    NoGradGuard no_grad;
    auto x = dense(x_in);
    auto out = empty(x->shape, x->device);
    DG_KERNEL(x->device, unary(op, x->data_ptr<float>(), out->data_ptr<float>(), x->size(), p, q));
    return out;
}

// out = op(a, b) over their broadcast shape, without materializing either
inline TensorPtr binary_value(kern::Binary op, const TensorPtr& a, const TensorPtr& b, const char* name) {
    check_same_device(a, b, name);
    auto shape = broadcast_shapes(a->shape, b->shape, name);
    auto out = empty(shape, a->device);
    if (out->size() == 0) return out;
    auto grid = kern::make_grid(shape, {broadcast_strides(*a, shape), broadcast_strides(*b, shape), out->strides});
    DG_KERNEL(a->device, binary(op, a->data_ptr<float>(), b->data_ptr<float>(), out->data_ptr<float>(), grid));
    return out;
}

inline TensorPtr binary(const char* name, kern::Binary op, const TensorPtr& a_in, const TensorPtr& b_in,
                        std::initializer_list<int> save, FnBackward::Vjp vjp) {
    DG_PROFILE_OP(name);
    auto a = strided(a_in), b = strided(b_in);
    auto out = binary_value(op, a, b, name);
    return make_op(name, out, {a, b}, save, std::move(vjp));
}

inline TensorPtr binary_const(const char* name, kern::Binary op, const TensorPtr& a_in, const TensorPtr& b_in) {
    DG_PROFILE_OP(name);
    NoGradGuard no_grad;
    return binary_value(op, strided(a_in), strided(b_in), name);
}

}  // namespace op_detail

// ---------------------------------------------------------------------------------------------
// Binary arithmetic
// ---------------------------------------------------------------------------------------------

inline TensorPtr sub(const TensorPtr& a, const TensorPtr& b) { return broadcast_binary(BinaryOp::Sub, a, b); }
inline TensorPtr div(const TensorPtr& a, const TensorPtr& b) { return broadcast_binary(BinaryOp::Div, a, b); }

// ---------------------------------------------------------------------------------------------
// Comparisons and logic: 1.0 where true, 0.0 where false, never part of the graph
// ---------------------------------------------------------------------------------------------

inline TensorPtr eq(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("eq", kern::Binary::Eq, a, b); }
inline TensorPtr ne(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("ne", kern::Binary::Ne, a, b); }
inline TensorPtr lt(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("lt", kern::Binary::Lt, a, b); }
inline TensorPtr le(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("le", kern::Binary::Le, a, b); }
inline TensorPtr gt(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("gt", kern::Binary::Gt, a, b); }
inline TensorPtr ge(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("ge", kern::Binary::Ge, a, b); }
inline TensorPtr eq(const TensorPtr& a, float s) { return op_detail::unary_const("eq", kern::Unary::EqScalar, a, s); }
inline TensorPtr ne(const TensorPtr& a, float s) { return op_detail::unary_const("ne", kern::Unary::NeScalar, a, s); }
inline TensorPtr lt(const TensorPtr& a, float s) { return op_detail::unary_const("lt", kern::Unary::LtScalar, a, s); }
inline TensorPtr le(const TensorPtr& a, float s) { return op_detail::unary_const("le", kern::Unary::LeScalar, a, s); }
inline TensorPtr gt(const TensorPtr& a, float s) { return op_detail::unary_const("gt", kern::Unary::GtScalar, a, s); }
inline TensorPtr ge(const TensorPtr& a, float s) { return op_detail::unary_const("ge", kern::Unary::GeScalar, a, s); }
inline TensorPtr logical_and(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("logical_and", kern::Binary::And, a, b); }
inline TensorPtr logical_or(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("logical_or", kern::Binary::Or, a, b); }
inline TensorPtr logical_xor(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("logical_xor", kern::Binary::Xor, a, b); }
inline TensorPtr logical_not(const TensorPtr& a) { return op_detail::unary_const("logical_not", kern::Unary::LogicalNot, a); }
inline TensorPtr is_nan(const TensorPtr& a) { return op_detail::unary_const("is_nan", kern::Unary::IsNan, a); }
inline TensorPtr is_inf(const TensorPtr& a) { return op_detail::unary_const("is_inf", kern::Unary::IsInf, a); }
inline TensorPtr is_finite(const TensorPtr& a) { return op_detail::unary_const("is_finite", kern::Unary::IsFinite, a); }

// Element-wise selection with broadcasting: a where cond is nonzero, else b
inline TensorPtr where(const TensorPtr& cond_in, const TensorPtr& a_in, const TensorPtr& b_in) {
    DG_PROFILE_OP("where");
    auto cond = op_detail::strided(cond_in)->detach();
    auto a = op_detail::strided(a_in), b = op_detail::strided(b_in);
    op_detail::check_same_device(a, b, "where");
    op_detail::check_same_device(a, cond, "where");
    auto shape = op_detail::broadcast_shapes(op_detail::broadcast_shapes(a->shape, b->shape, "where"), cond->shape, "where");
    auto out = op_detail::empty(shape, a->device);
    if (out->size() > 0) {
        auto grid = kern::make_grid(shape, {op_detail::broadcast_strides(*cond, shape), op_detail::broadcast_strides(*a, shape),
                                            op_detail::broadcast_strides(*b, shape), out->strides});
        DG_KERNEL(a->device, where(cond->data_ptr<float>(), a->data_ptr<float>(), b->data_ptr<float>(), out->data_ptr<float>(), grid));
    }
    auto a_shape = a->shape, b_shape = b->shape;
    return make_op("where", out, {a, b}, {}, [cond, a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto zero = op_detail::filled({}, g->device, 0.0f);
        return {c.needs(0) ? sum_to(where(cond, g, zero), a_shape) : nullptr,
                c.needs(1) ? sum_to(where(cond, zero, g), b_shape) : nullptr};
    });
}
inline TensorPtr where(const TensorPtr& cond, const TensorPtr& a, float b) {
    return where(cond, a, op_detail::filled({}, a->device, b));
}
inline TensorPtr where(const TensorPtr& cond, float a, const TensorPtr& b) {
    return where(cond, op_detail::filled({}, b->device, a), b);
}
// x with value wherever mask is nonzero
inline TensorPtr masked_fill(const TensorPtr& x, const TensorPtr& mask, float value) { return where(mask, value, x); }

// ---------------------------------------------------------------------------------------------
// Functions of one tensor
// ---------------------------------------------------------------------------------------------

inline TensorPtr exp(const TensorPtr& x);
inline TensorPtr log(const TensorPtr& x);
inline TensorPtr sin(const TensorPtr& x);
inline TensorPtr cos(const TensorPtr& x);
inline TensorPtr sinh(const TensorPtr& x);
inline TensorPtr cosh(const TensorPtr& x);
inline TensorPtr sqrt(const TensorPtr& x);
inline TensorPtr square(const TensorPtr& x);
inline TensorPtr pow(const TensorPtr& x, float exponent);
inline TensorPtr digamma(const TensorPtr& x);
inline TensorPtr i1(const TensorPtr& x);

inline TensorPtr sign(const TensorPtr& x) { return op_detail::unary_const("sign", kern::Unary::Sign, x); }
inline TensorPtr floor(const TensorPtr& x) { return op_detail::unary_const("floor", kern::Unary::Floor, x); }
inline TensorPtr ceil(const TensorPtr& x) { return op_detail::unary_const("ceil", kern::Unary::Ceil, x); }
inline TensorPtr round(const TensorPtr& x) { return op_detail::unary_const("round", kern::Unary::Round, x); }  // half to even
inline TensorPtr trunc(const TensorPtr& x) { return op_detail::unary_const("trunc", kern::Unary::Trunc, x); }

inline TensorPtr reciprocal(const TensorPtr& x) {
    return op_detail::unary("reciprocal", kern::Unary::Reciprocal, x, 0, 0, {kSaveOutput},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {neg(mul(g, square(c.output())))}; });
}
inline TensorPtr abs(const TensorPtr& x) {
    return op_detail::unary("abs", kern::Unary::Abs, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, sign(c.input(0)))}; });
}
inline TensorPtr exp(const TensorPtr& x) {
    return op_detail::unary("exp", kern::Unary::Exp, x, 0, 0, {kSaveOutput},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, c.output())}; });
}
inline TensorPtr exp2(const TensorPtr& x) { return exp(mul_scalar(x, 0.69314718055994530942f)); }
inline TensorPtr expm1(const TensorPtr& x) {
    return op_detail::unary("expm1", kern::Unary::Expm1, x, 0, 0, {kSaveOutput},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, add_scalar(c.output(), 1.0f))}; });
}
inline TensorPtr log(const TensorPtr& x) {
    return op_detail::unary("log", kern::Unary::Log, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(g, c.input(0))}; });
}
inline TensorPtr log2(const TensorPtr& x) {
    return op_detail::unary("log2", kern::Unary::Log2, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(mul_scalar(g, 1.44269504088896341f), c.input(0))}; });
}
inline TensorPtr log10(const TensorPtr& x) {
    return op_detail::unary("log10", kern::Unary::Log10, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(mul_scalar(g, 0.43429448190325183f), c.input(0))}; });
}
inline TensorPtr log1p(const TensorPtr& x) {
    return op_detail::unary("log1p", kern::Unary::Log1p, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(g, add_scalar(c.input(0), 1.0f))}; });
}
inline TensorPtr sqrt(const TensorPtr& x) {
    return op_detail::unary("sqrt", kern::Unary::Sqrt, x, 0, 0, {kSaveOutput},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(mul_scalar(g, 0.5f), c.output())}; });
}
inline TensorPtr rsqrt(const TensorPtr& x) {
    return op_detail::unary("rsqrt", kern::Unary::Rsqrt, x, 0, 0, {kSaveOutput},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(mul_scalar(g, -0.5f), pow(c.output(), 3.0f))}; });
}
inline TensorPtr square(const TensorPtr& x) {
    return op_detail::unary("square", kern::Unary::Square, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(mul_scalar(g, 2.0f), c.input(0))}; });
}
inline TensorPtr sin(const TensorPtr& x) {
    return op_detail::unary("sin", kern::Unary::Sin, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, cos(c.input(0)))}; });
}
inline TensorPtr cos(const TensorPtr& x) {
    return op_detail::unary("cos", kern::Unary::Cos, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {neg(mul(g, sin(c.input(0))))}; });
}
inline TensorPtr tan(const TensorPtr& x) {
    return op_detail::unary("tan", kern::Unary::Tan, x, 0, 0, {kSaveOutput},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, add_scalar(square(c.output()), 1.0f))}; });
}
inline TensorPtr asin(const TensorPtr& x) {
    return op_detail::unary("asin", kern::Unary::Asin, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, rsqrt(rsub_scalar(1.0f, square(c.input(0)))))}; });
}
inline TensorPtr acos(const TensorPtr& x) {
    return op_detail::unary("acos", kern::Unary::Acos, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {neg(mul(g, rsqrt(rsub_scalar(1.0f, square(c.input(0))))))}; });
}
inline TensorPtr atan(const TensorPtr& x) {
    return op_detail::unary("atan", kern::Unary::Atan, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(g, add_scalar(square(c.input(0)), 1.0f))}; });
}
inline TensorPtr sinh(const TensorPtr& x) {
    return op_detail::unary("sinh", kern::Unary::Sinh, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, cosh(c.input(0)))}; });
}
inline TensorPtr cosh(const TensorPtr& x) {
    return op_detail::unary("cosh", kern::Unary::Cosh, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, sinh(c.input(0)))}; });
}
inline TensorPtr asinh(const TensorPtr& x) {
    return op_detail::unary("asinh", kern::Unary::Asinh, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, rsqrt(add_scalar(square(c.input(0)), 1.0f)))}; });
}
inline TensorPtr acosh(const TensorPtr& x) {
    return op_detail::unary("acosh", kern::Unary::Acosh, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, rsqrt(sub_scalar(square(c.input(0)), 1.0f)))}; });
}
inline TensorPtr atanh(const TensorPtr& x) {
    return op_detail::unary("atanh", kern::Unary::Atanh, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {div(g, rsub_scalar(1.0f, square(c.input(0))))}; });
}
inline TensorPtr erf(const TensorPtr& x) {
    return op_detail::unary("erf", kern::Unary::Erf, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {mul(g, mul_scalar(exp(neg(square(c.input(0)))), 1.12837916709551257f))};
    });
}
inline TensorPtr erfc(const TensorPtr& x) {
    return op_detail::unary("erfc", kern::Unary::Erfc, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {mul(g, mul_scalar(exp(neg(square(c.input(0)))), -1.12837916709551257f))};
    });
}
inline TensorPtr erfinv(const TensorPtr& x) {
    return op_detail::unary("erfinv", kern::Unary::Erfinv, x, 0, 0, {kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {mul(g, mul_scalar(exp(square(c.output())), 0.88622692545275801f))};
    });
}
inline TensorPtr lgamma(const TensorPtr& x) {
    return op_detail::unary("lgamma", kern::Unary::Lgamma, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, digamma(c.input(0)))}; });
}
// Derivatives of lgamma: digamma, trigamma, tetragamma. The last one cannot be differentiated further.
inline TensorPtr tetragamma(const TensorPtr& x) {
    return op_detail::unary("tetragamma", kern::Unary::Tetragamma, x, 0, 0, {}, [](const OpContext&, const TensorPtr&) -> TensorList {
        throw std::runtime_error("tetragamma: derivatives of lgamma past the third are not implemented");
    });
}
inline TensorPtr trigamma(const TensorPtr& x) {
    return op_detail::unary("trigamma", kern::Unary::Trigamma, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, tetragamma(c.input(0)))}; });
}
inline TensorPtr digamma(const TensorPtr& x) {
    return op_detail::unary("digamma", kern::Unary::Digamma, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, trigamma(c.input(0)))}; });
}
inline TensorPtr i0(const TensorPtr& x) {
    return op_detail::unary("i0", kern::Unary::I0, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, i1(c.input(0)))}; });
}
inline TensorPtr i1(const TensorPtr& x) {
    return op_detail::unary("i1", kern::Unary::I1, x, 0, 0, {0, kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);  // I1'(x) = I0(x) - I1(x) / x, which tends to 1/2 at 0
        auto safe = where(eq(v, 0.0f), 1.0f, v);
        return {mul(g, where(eq(v, 0.0f), 0.5f, sub(i0(v), div(c.output(), safe))))};
    });
}
// sin(pi x) / (pi x)
inline TensorPtr sinc(const TensorPtr& x) {
    return op_detail::unary("sinc", kern::Unary::Sinc, x, 0, 0, {0, kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        auto safe = where(eq(v, 0.0f), 1.0f, v);
        auto d = div(sub(cos(mul_scalar(v, 3.14159265358979324f)), c.output()), safe);
        return {mul(g, where(eq(v, 0.0f), 0.0f, d))};
    });
}
inline TensorPtr frac(const TensorPtr& x) { return sub(x, trunc(x)); }

inline TensorPtr pow(const TensorPtr& x, float exponent) {
    return op_detail::unary("pow", kern::Unary::PowScalar, x, exponent, 0, {0}, [exponent](const OpContext& c, const TensorPtr& g) -> TensorList {
        if (exponent == 0.0f) return {op_detail::filled(g->shape, g->device, 0.0f)};
        if (exponent == 1.0f) return {g};
        if (exponent == 2.0f) return {mul(g, mul_scalar(c.input(0), 2.0f))};
        return {mul(g, mul_scalar(pow(c.input(0), exponent - 1.0f), exponent))};
    });
}
// base^x
inline TensorPtr pow(float base, const TensorPtr& x) {
    float log_base = std::log(base);
    return op_detail::unary("scalar_pow", kern::Unary::ScalarPow, x, base, 0, {kSaveOutput},
        [log_base](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, mul_scalar(c.output(), log_base))}; });
}
inline TensorPtr pow(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape, b_shape = b->shape;
    return op_detail::binary("pow", kern::Binary::Pow, a, b, {0, 1, kSaveOutput}, [a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto x = c.input(0), y = c.input(1);
        TensorPtr gx, gy;
        if (c.needs(0)) {
            auto d = where(eq(y, 0.0f), 0.0f, mul(pow(x, sub_scalar(y, 1.0f)), y));
            gx = sum_to(mul(g, d), a_shape);
        }
        if (c.needs(1)) {
            // d/dy x^y = x^y log x, taken as 0 at x = 0 (the limit for y > 0)
            auto safe = where(eq(x, 0.0f), 1.0f, x);
            gy = sum_to(mul(g, mul(c.output(), log(safe))), b_shape);
        }
        return {gx, gy};
    });
}

inline TensorPtr maximum(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape, b_shape = b->shape;
    return op_detail::binary("maximum", kern::Binary::Max, a, b, {0, 1}, [a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto x = c.input(0), y = c.input(1);
        auto tie = mul_scalar(eq(x, y), 0.5f);  // ties share the gradient
        return {c.needs(0) ? sum_to(mul(g, add(gt(x, y), tie)), a_shape) : nullptr,
                c.needs(1) ? sum_to(mul(g, add(lt(x, y), tie)), b_shape) : nullptr};
    });
}
inline TensorPtr minimum(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape, b_shape = b->shape;
    return op_detail::binary("minimum", kern::Binary::Min, a, b, {0, 1}, [a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto x = c.input(0), y = c.input(1);
        auto tie = mul_scalar(eq(x, y), 0.5f);
        return {c.needs(0) ? sum_to(mul(g, add(lt(x, y), tie)), a_shape) : nullptr,
                c.needs(1) ? sum_to(mul(g, add(gt(x, y), tie)), b_shape) : nullptr};
    });
}
inline TensorPtr atan2(const TensorPtr& y, const TensorPtr& x) {
    auto y_shape = y->shape, x_shape = x->shape;
    return op_detail::binary("atan2", kern::Binary::Atan2, y, x, {0, 1}, [y_shape, x_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto b = c.input(0), a = c.input(1);
        auto scaled = div(g, add(square(a), square(b)));
        return {c.needs(0) ? sum_to(mul(scaled, a), y_shape) : nullptr,
                c.needs(1) ? sum_to(neg(mul(scaled, b)), x_shape) : nullptr};
    });
}
inline TensorPtr hypot(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape, b_shape = b->shape;
    return op_detail::binary("hypot", kern::Binary::Hypot, a, b, {0, 1, kSaveOutput}, [a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto scaled = div(g, c.output());
        return {c.needs(0) ? sum_to(mul(scaled, c.input(0)), a_shape) : nullptr,
                c.needs(1) ? sum_to(mul(scaled, c.input(1)), b_shape) : nullptr};
    });
}
inline TensorPtr floor_divide(const TensorPtr& a, const TensorPtr& b) { return op_detail::binary_const("floor_divide", kern::Binary::FloorDivide, a, b); }
// C-style remainder (sign of a)
inline TensorPtr fmod(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape, b_shape = b->shape;
    return op_detail::binary("fmod", kern::Binary::Fmod, a, b, {0, 1}, [a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {c.needs(0) ? sum_to(g, a_shape) : nullptr,
                c.needs(1) ? sum_to(neg(mul(g, trunc(div(c.input(0), c.input(1))))), b_shape) : nullptr};
    });
}
// Python-style remainder (sign of b)
inline TensorPtr remainder(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape, b_shape = b->shape;
    return op_detail::binary("remainder", kern::Binary::Remainder, a, b, {0, 1}, [a_shape, b_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {c.needs(0) ? sum_to(g, a_shape) : nullptr,
                c.needs(1) ? sum_to(neg(mul(g, floor_divide(c.input(0), c.input(1)))), b_shape) : nullptr};
    });
}
inline TensorPtr fmod(const TensorPtr& a, float s) {
    return op_detail::unary("fmod", kern::Unary::FmodScalar, a, s, 0, {}, [](const OpContext&, const TensorPtr& g) -> TensorList { return {g}; });
}
inline TensorPtr remainder(const TensorPtr& a, float s) {
    return op_detail::unary("remainder", kern::Unary::RemainderScalar, a, s, 0, {}, [](const OpContext&, const TensorPtr& g) -> TensorList { return {g}; });
}
inline TensorPtr copysign(const TensorPtr& a, const TensorPtr& b) {
    auto a_shape = a->shape;
    return op_detail::binary("copysign", kern::Binary::Copysign, a, b, {0, kSaveOutput}, [a_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto x = c.input(0);
        auto ratio = where(eq(x, 0.0f), 0.0f, mul(sign(x), sign(c.output())));
        return {c.needs(0) ? sum_to(mul(g, ratio), a_shape) : nullptr, nullptr};
    });
}

inline TensorPtr clamp(const TensorPtr& x, float lo, float hi) {
    return op_detail::unary("clamp", kern::Unary::Clamp, x, lo, hi, {0}, [lo, hi](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {mul(g, op_detail::unary_const("in_range", kern::Unary::InRange, c.input(0), lo, hi))};
    });
}
inline TensorPtr clamp_min(const TensorPtr& x, float lo) { return clamp(x, lo, std::numeric_limits<float>::infinity()); }
inline TensorPtr clamp_max(const TensorPtr& x, float hi) { return clamp(x, -std::numeric_limits<float>::infinity(), hi); }
// NaN becomes nan, infinities the largest finite values
inline TensorPtr nan_to_num(const TensorPtr& x, float nan = 0.0f) {
    return op_detail::unary("nan_to_num", kern::Unary::NanToNum, x, nan, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, is_finite(c.input(0)))}; });
}
inline TensorPtr lerp(const TensorPtr& a, const TensorPtr& b, float weight) { return add(a, mul_scalar(sub(b, a), weight)); }
inline TensorPtr lerp(const TensorPtr& a, const TensorPtr& b, const TensorPtr& weight) { return add(a, mul(weight, sub(b, a))); }
// x + value * a * b, x + value * a / b
inline TensorPtr addcmul(const TensorPtr& x, const TensorPtr& a, const TensorPtr& b, float value = 1.0f) { return add(x, mul_scalar(mul(a, b), value)); }
inline TensorPtr addcdiv(const TensorPtr& x, const TensorPtr& a, const TensorPtr& b, float value = 1.0f) { return add(x, mul_scalar(div(a, b), value)); }
// x log y, with 0 wherever x is 0
inline TensorPtr xlogy(const TensorPtr& x, const TensorPtr& y) {
    auto zero = eq(x, 0.0f);
    return where(zero, 0.0f, mul(x, log(where(zero, 1.0f, y))));
}
inline TensorPtr logaddexp(const TensorPtr& a, const TensorPtr& b) {
    auto hi = maximum(a, b), lo = minimum(a, b);
    return add(hi, log1p(exp(sub(lo, hi))));
}
inline TensorPtr logit(const TensorPtr& x, float eps = 0.0f) {
    auto p = eps > 0.0f ? clamp(x, eps, 1.0f - eps) : x;
    return sub(log(p), log1p(neg(p)));
}

// ---------------------------------------------------------------------------------------------
// Activations (relu, leaky_relu, gelu (tanh form), silu, sigmoid, tanh are in functional.h)
// ---------------------------------------------------------------------------------------------

inline TensorPtr gelu_erf(const TensorPtr& x) {
    return op_detail::unary("gelu_erf", kern::Unary::Gelu, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        auto cdf = mul_scalar(erfc(mul_scalar(v, -0.70710678118654752f)), 0.5f);
        auto pdf = mul_scalar(exp(mul_scalar(square(v), -0.5f)), 0.39894228040143268f);
        return {mul(g, add(cdf, mul(v, pdf)))};
    });
}
inline TensorPtr softplus(const TensorPtr& x, float beta = 1.0f, float threshold = 20.0f) {
    return op_detail::unary("softplus", kern::Unary::Softplus, x, beta, threshold, {0},
        [beta](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, sigmoid(mul_scalar(c.input(0), beta)))}; });
}
inline TensorPtr log_sigmoid(const TensorPtr& x) {
    return op_detail::unary("log_sigmoid", kern::Unary::LogSigmoid, x, 0, 0, {0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, sigmoid(neg(c.input(0))))}; });
}
inline TensorPtr elu(const TensorPtr& x, float alpha = 1.0f) {
    return op_detail::unary("elu", kern::Unary::Elu, x, alpha, 0, {0}, [alpha](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        return {mul(g, where(gt(v, 0.0f), 1.0f, mul_scalar(exp(clamp_max(v, 0.0f)), alpha)))};
    });
}
inline TensorPtr selu(const TensorPtr& x) { return mul_scalar(elu(x, 1.6732632423543772f), 1.0507009873554805f); }
inline TensorPtr celu(const TensorPtr& x, float alpha = 1.0f) { return mul_scalar(elu(div_scalar(x, alpha), 1.0f), alpha); }
inline TensorPtr mish(const TensorPtr& x) {
    return op_detail::unary("mish", kern::Unary::Mish, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        auto t = tanh(softplus(v));
        return {mul(g, add(t, mul(v, mul(sigmoid(v), rsub_scalar(1.0f, square(t))))))};
    });
}
inline TensorPtr relu6(const TensorPtr& x) { return clamp(x, 0.0f, 6.0f); }
inline TensorPtr hardtanh(const TensorPtr& x, float lo = -1.0f, float hi = 1.0f) { return clamp(x, lo, hi); }
inline TensorPtr hardswish(const TensorPtr& x) {
    return op_detail::unary("hardswish", kern::Unary::Hardswish, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        auto mid = add_scalar(mul_scalar(v, 1.0f / 3.0f), 0.5f);
        return {mul(g, where(le(v, -3.0f), 0.0f, where(ge(v, 3.0f), 1.0f, mid)))};
    });
}
inline TensorPtr hardsigmoid(const TensorPtr& x) {
    return op_detail::unary("hardsigmoid", kern::Unary::Hardsigmoid, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto v = c.input(0);
        return {mul(g, mul_scalar(mul(gt(v, -3.0f), lt(v, 3.0f)), 1.0f / 6.0f))};
    });
}
inline TensorPtr softsign(const TensorPtr& x) {
    return op_detail::unary("softsign", kern::Unary::Softsign, x, 0, 0, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {div(g, square(add_scalar(abs(c.input(0)), 1.0f)))};
    });
}
inline TensorPtr tanhshrink(const TensorPtr& x) { return sub(x, tanh(x)); }
inline TensorPtr softshrink(const TensorPtr& x, float lambda = 0.5f) {
    return op_detail::unary("softshrink", kern::Unary::Softshrink, x, lambda, 0, {0},
        [lambda](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, gt(abs(c.input(0)), lambda))}; });
}
inline TensorPtr hardshrink(const TensorPtr& x, float lambda = 0.5f) {
    return op_detail::unary("hardshrink", kern::Unary::Hardshrink, x, lambda, 0, {0},
        [lambda](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, gt(abs(c.input(0)), lambda))}; });
}
// x where x > threshold, else value
inline TensorPtr threshold(const TensorPtr& x, float limit, float value) {
    return op_detail::unary("threshold", kern::Unary::Threshold, x, limit, value, {0},
        [limit](const OpContext& c, const TensorPtr& g) -> TensorList { return {mul(g, gt(c.input(0), limit))}; });
}
// Leaky ReLU with a learned slope: one value, or one per channel (dimension 1)
inline TensorPtr prelu(const TensorPtr& x, const TensorPtr& slope) {
    auto a = slope;
    if (slope->size() > 1) {
        if (x->ndim() < 2 || slope->size() != x->shape[1]) throw std::invalid_argument("prelu: the slope must have one value or one per channel");
        std::vector<int64_t> shape((size_t)x->ndim(), 1);
        shape[1] = slope->size();
        a = reshape(slope, shape);
    }
    auto zero = op_detail::filled({}, x->device, 0.0f);
    return add(maximum(x, zero), mul(a, minimum(x, zero)));
}
