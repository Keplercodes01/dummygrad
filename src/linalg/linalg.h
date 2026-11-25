#pragma once
// Differentiable linear algebra (torch.linalg) on batches of matrices [..., m, n]. Factorizations run in
// double precision on the host (GPU tensors are copied there and back); everything composed from matrix
// products runs on the device. Gradients are the standard ones (Giles 2008; Seeger et al. 2017) and
// support create_graph.

#include "functional.h"
#include "linalg/dense.h"

#include <tuple>

namespace linalg {

namespace detail {

inline void check_square(const TensorPtr& a, const char* who) {
    if (a->ndim() < 2 || a->shape[(size_t)a->ndim() - 2] != a->shape.back()) throw std::invalid_argument(std::string(who) + ": square matrices are needed, got " + shape_string(a->shape));
}
inline std::vector<int64_t> batch_shape(const TensorPtr& a) { return {a->shape.begin(), a->shape.end() - 2}; }
inline std::vector<int64_t> with(std::vector<int64_t> batch, std::initializer_list<int64_t> tail) {
    batch.insert(batch.end(), tail.begin(), tail.end());
    return batch;
}
inline TensorPtr T(const TensorPtr& a) { return transpose(a, -2, -1); }

// Runs f(matrix b as doubles, index b) for every matrix of a
template <typename F>
inline void for_each_matrix(const TensorPtr& a, const F& f) {
    auto v = a->to_vector();
    int64_t size = a->shape[(size_t)a->ndim() - 2] * a->shape.back(), batch = size ? a->size() / size : 0;
    parallel_for(0, batch, std::max<int64_t>(1, 4096 / std::max<int64_t>(1, size)), [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) f(dense::Matrix(v.begin() + b * size, v.begin() + (b + 1) * size), b);
    });
}
// Where for_each_matrix puts results: out is one float per element of the whole batch
inline void put(std::vector<float>& out, const std::vector<double>& m, int64_t b) {
    for (size_t i = 0; i < m.size(); ++i) out[(size_t)b * m.size() + i] = (float)m[i];
}
inline TensorPtr make(const std::vector<float>& v, const std::vector<int64_t>& shape, Device device) { return op_detail::from_values(v, shape, device); }

}  // namespace detail

inline TensorPtr inv(const TensorPtr& a);
inline TensorPtr solve(const TensorPtr& a, const TensorPtr& b);

// Solves T X = B for triangular T [..., n, n] and B [..., n, m]; only T's triangle is read
inline TensorPtr solve_triangular(const TensorPtr& t_in, const TensorPtr& b_in, bool upper, bool transposed = false, bool unit_diagonal = false) {
    DG_PROFILE_OP("solve_triangular");
    detail::check_square(t_in, "solve_triangular");
    auto t = op_detail::dense(t_in), b = op_detail::dense(b_in);
    if (b->ndim() != t->ndim() || detail::batch_shape(t) != detail::batch_shape(b)) throw std::invalid_argument("solve_triangular: batch dimensions must match");
    int64_t n = t->shape.back(), m = b->shape.back();
    if (b->shape[(size_t)b->ndim() - 2] != n) throw std::invalid_argument("solve_triangular: B must have as many rows as T");
    auto bv = b->to_vector();
    std::vector<float> out(bv.size());
    detail::for_each_matrix(t, [&](const dense::Matrix& tm, int64_t i) {
        dense::Matrix rhs(bv.begin() + i * n * m, bv.begin() + (i + 1) * n * m);
        dense::triangular_solve(tm, n, rhs, m, upper, transposed, unit_diagonal);
        detail::put(out, rhs, i);
    });
    return make_op("solve_triangular", detail::make(out, b->shape, b->device), {t, b}, {0, kSaveOutput}, [upper, transposed, unit_diagonal](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto tv = c.input(0);
        auto gb = solve_triangular(tv, g, upper, !transposed, unit_diagonal);
        TensorPtr gt;
        if (c.needs(0)) {
            auto x = c.output();
            gt = neg(transposed ? matmul(x, detail::T(gb)) : matmul(gb, detail::T(x)));
            gt = upper ? triu(gt, unit_diagonal ? 1 : 0) : tril(gt, unit_diagonal ? -1 : 0);
        }
        return {gt, gb};
    });
}

// Lower-triangular L with A = L L^T
inline TensorPtr cholesky(const TensorPtr& a_in) {
    DG_PROFILE_OP("cholesky");
    detail::check_square(a_in, "cholesky");
    auto a = op_detail::dense(a_in);
    int64_t n = a->shape.back();
    std::vector<float> out((size_t)a->size());
    std::atomic<bool> ok{true};
    detail::for_each_matrix(a, [&](dense::Matrix m, int64_t i) {
        if (!dense::cholesky(m, n)) ok = false;
        detail::put(out, m, i);
    });
    if (!ok) throw std::runtime_error("cholesky: a matrix is not positive definite");
    return make_op("cholesky", detail::make(out, a->shape, a->device), {a}, {kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto L = c.output();
        // Phi(L^T G): the lower triangle with a halved diagonal
        auto s = tril(matmul(detail::T(L), g));
        s = sub(s, mul_scalar(diag_embed(diagonal(s, 0, -2, -1)), 0.5f));
        auto left = solve_triangular(L, s, false, true);                                // L^-T Phi
        auto r = detail::T(solve_triangular(L, contiguous(detail::T(left)), false, true));  // ... L^-1
        return {mul_scalar(add(r, detail::T(r)), 0.5f)};
    });
}
// Solves A X = B given A's Cholesky factor
inline TensorPtr cholesky_solve(const TensorPtr& b, const TensorPtr& L) { return solve_triangular(L, solve_triangular(L, b, false), false, true); }

// X = A^-1 B for B [..., n, m] or, with one fewer dimension than A, a vector per matrix
inline TensorPtr solve(const TensorPtr& a_in, const TensorPtr& b_in) {
    DG_PROFILE_OP("solve");
    detail::check_square(a_in, "solve");
    if (b_in->ndim() == a_in->ndim() - 1) return squeeze(solve(a_in, unsqueeze(b_in, -1)), -1);
    auto a = op_detail::dense(a_in), b = op_detail::dense(b_in);
    int64_t n = a->shape.back(), m = b->shape.back();
    if (b->ndim() != a->ndim() || detail::batch_shape(a) != detail::batch_shape(b) || b->shape[(size_t)b->ndim() - 2] != n) throw std::invalid_argument("solve: B must be [..., n, m] with A's batch dimensions");
    auto bv = b->to_vector();
    std::vector<float> out(bv.size());
    std::atomic<bool> ok{true};
    detail::for_each_matrix(a, [&](dense::Matrix am, int64_t i) {
        dense::Matrix rhs(bv.begin() + i * n * m, bv.begin() + (i + 1) * n * m);
        std::vector<int64_t> pivots;
        if (dense::lu(am, n, pivots) == 0) ok = false;
        dense::lu_solve(am, n, pivots, rhs, m);
        detail::put(out, rhs, i);
    });
    if (!ok) throw std::runtime_error("solve: a matrix is singular");
    return make_op("solve", detail::make(out, b->shape, b->device), {a, b}, {0, kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto gb = solve(contiguous(detail::T(c.input(0))), g);
        return {c.needs(0) ? neg(matmul(gb, detail::T(c.output()))) : nullptr, gb};
    });
}

inline TensorPtr inv(const TensorPtr& a_in) {
    DG_PROFILE_OP("inv");
    detail::check_square(a_in, "inv");
    auto a = op_detail::dense(a_in);
    int64_t n = a->shape.back();
    std::vector<float> out((size_t)a->size());
    std::atomic<bool> ok{true};
    detail::for_each_matrix(a, [&](dense::Matrix m, int64_t i) {
        dense::Matrix rhs((size_t)(n * n), 0.0);
        for (int64_t d = 0; d < n; ++d) rhs[(size_t)(d * n + d)] = 1.0;
        std::vector<int64_t> pivots;
        if (dense::lu(m, n, pivots) == 0) ok = false;
        dense::lu_solve(m, n, pivots, rhs, n);
        detail::put(out, rhs, i);
    });
    if (!ok) throw std::runtime_error("inv: a matrix is singular");
    return make_op("inv", detail::make(out, a->shape, a->device), {a}, {kSaveOutput}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto t = detail::T(c.output());
        return {neg(matmul(t, matmul(g, t)))};
    });
}

// Sign and log |det A|
inline std::pair<TensorPtr, TensorPtr> slogdet(const TensorPtr& a_in) {
    DG_PROFILE_OP("slogdet");
    detail::check_square(a_in, "slogdet");
    auto a = op_detail::dense(a_in);
    int64_t n = a->shape.back();
    auto batch = detail::batch_shape(a);
    int64_t count = 1;
    for (int64_t d : batch) count *= d;
    std::vector<float> sign((size_t)count), logabs((size_t)count);
    detail::for_each_matrix(a, [&](dense::Matrix m, int64_t i) {
        std::vector<int64_t> pivots;
        int s = dense::lu(m, n, pivots);
        double total = 0.0;
        for (int64_t d = 0; d < n; ++d) {
            double v = m[(size_t)(d * n + d)];
            if (v < 0.0) s = -s;
            total += std::log(std::fabs(v));
        }
        sign[(size_t)i] = (float)s;
        logabs[(size_t)i] = (float)total;
    });
    auto out = make_op("slogdet", detail::make(logabs, batch, a->device), {a}, {0}, [](const OpContext& c, const TensorPtr& g) -> TensorList {
        auto scale = unsqueeze(unsqueeze(g, -1), -1);
        return {mul(contiguous(detail::T(inv(c.input(0)))), scale)};
    });
    return {detail::make(sign, batch, a->device), out};
}
inline TensorPtr det(const TensorPtr& a) {
    auto [sign, logabs] = slogdet(a);
    return mul(exp(logabs), sign);
}
inline TensorPtr logdet(const TensorPtr& a) {
    auto [sign, logabs] = slogdet(a);
    return where(gt(sign, 0.0f), logabs, std::numeric_limits<float>::quiet_NaN());
}

// Reduced QR: A = Q R with R's diagonal non-negative. Gradients need rows >= columns.
inline std::pair<TensorPtr, TensorPtr> qr(const TensorPtr& a_in) {
    DG_PROFILE_OP("qr");
    if (a_in->ndim() < 2) throw std::invalid_argument("qr: matrices are needed");
    auto a = op_detail::dense(a_in);
    int64_t m = a->shape[(size_t)a->ndim() - 2], n = a->shape.back(), k = std::min(m, n);
    auto batch = detail::batch_shape(a);
    auto q_shape = detail::with(batch, {m, k}), r_shape = detail::with(batch, {k, n});
    int64_t count = a->size() / std::max<int64_t>(1, m * n);
    std::vector<float> qv((size_t)(count * m * k)), rv((size_t)(count * k * n));
    detail::for_each_matrix(a, [&](const dense::Matrix& am, int64_t i) {
        dense::Matrix q, r;
        dense::qr(am, m, n, q, r);
        detail::put(qv, q, i);
        detail::put(rv, r, i);
    });
    auto Q = detail::make(qv, q_shape, a->device), R = detail::make(rv, r_shape, a->device);
    // Each output gets its own node with its part of the (linear) gradient
    auto Qc = Q->detach(), Rc = R->detach();
    auto pull = [m, n, Qc, Rc](const OpContext& c, const TensorPtr& gq, const TensorPtr& gr) -> TensorPtr {
        if (m < n) throw std::runtime_error("qr: gradients need at least as many rows as columns");
        auto q = Qc, r = Rc;
        if (c.graph()) std::tie(q, r) = qr(c.input(0));  // connected to A
        TensorPtr M;
        if (gr) M = matmul(r, detail::T(gr));
        if (gq) M = M ? sub(M, matmul(detail::T(gq), q)) : neg(matmul(detail::T(gq), q));
        auto low = tril(M, -1);
        auto sym = add(tril(M), detail::T(low));  // copyltu: the lower triangle mirrored up
        auto rhs = matmul(q, sym);
        if (gq) rhs = add(rhs, gq);
        // rhs R^-T: solve R Y = rhs^T
        return contiguous(detail::T(solve_triangular(r, contiguous(detail::T(rhs)), true)));
    };
    auto Qo = make_op("qr_q", Q, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, g, nullptr)}; });
    auto Ro = make_op("qr_r", R, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, nullptr, g)}; });
    return {Qo, Ro};
}

// Eigenvalues (ascending) and orthonormal eigenvectors (columns) of symmetric matrices. Eigenvector
// gradients need distinct eigenvalues.
inline std::pair<TensorPtr, TensorPtr> eigh(const TensorPtr& a_in) {
    DG_PROFILE_OP("eigh");
    detail::check_square(a_in, "eigh");
    auto a = op_detail::dense(a_in);
    int64_t n = a->shape.back();
    auto batch = detail::batch_shape(a);
    int64_t count = a->size() / std::max<int64_t>(1, n * n);
    std::vector<float> wv((size_t)(count * n)), vv((size_t)a->size());
    detail::for_each_matrix(a, [&](const dense::Matrix& am, int64_t i) {
        std::vector<double> w;
        dense::Matrix v;
        dense::eigh(am, n, w, v);
        detail::put(wv, w, i);
        detail::put(vv, v, i);
    });
    auto W = detail::make(wv, detail::with(batch, {n}), a->device), V = detail::make(vv, a->shape, a->device);
    auto Wc = W->detach(), Vc = V->detach();
    auto pull = [n, Wc, Vc](const OpContext& c, const TensorPtr& gw, const TensorPtr& gv) -> TensorPtr {
        auto w = Wc, v = Vc;
        if (c.graph()) std::tie(w, v) = eigh(c.input(0));
        TensorPtr mid;
        if (gw) mid = diag_embed(gw);
        if (gv) {
            auto gap = sub(unsqueeze(w, -2), unsqueeze(w, -1));  // w_j - w_i
            auto off = rsub_scalar(1.0f, eye(n, n, w->device));
            auto F = div(off, where(off, gap, 1.0f));
            auto part = mul(F, matmul(detail::T(v), gv));
            mid = mid ? add(mid, part) : part;
        }
        auto g = matmul(matmul(v, mid), detail::T(v));
        return mul_scalar(add(g, detail::T(g)), 0.5f);
    };
    auto Wo = make_op("eigh_w", W, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, g, nullptr)}; });
    auto Vo = make_op("eigh_v", V, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, nullptr, g)}; });
    return {Wo, Vo};
}
inline TensorPtr eigvalsh(const TensorPtr& a) { return eigh(a).first; }

// Thin SVD A = U diag(S) Vt, S descending. Gradients of U and Vt need distinct nonzero singular values.
inline std::tuple<TensorPtr, TensorPtr, TensorPtr> svd(const TensorPtr& a_in) {
    DG_PROFILE_OP("svd");
    if (a_in->ndim() < 2) throw std::invalid_argument("svd: matrices are needed");
    auto a = op_detail::dense(a_in);
    int64_t m = a->shape[(size_t)a->ndim() - 2], n = a->shape.back(), k = std::min(m, n);
    auto batch = detail::batch_shape(a);
    int64_t count = a->size() / std::max<int64_t>(1, m * n);
    std::vector<float> uv((size_t)(count * m * k)), sv((size_t)(count * k)), vv((size_t)(count * k * n));
    detail::for_each_matrix(a, [&](const dense::Matrix& am, int64_t i) {
        dense::Matrix u, vt;
        std::vector<double> s;
        dense::svd(am, m, n, u, s, vt);
        detail::put(uv, u, i);
        detail::put(sv, s, i);
        detail::put(vv, vt, i);
    });
    auto U = detail::make(uv, detail::with(batch, {m, k}), a->device), S = detail::make(sv, detail::with(batch, {k}), a->device);
    auto Vt = detail::make(vv, detail::with(batch, {k, n}), a->device);
    auto Uc = U->detach(), Sc = S->detach(), Vc = Vt->detach();
    auto pull = [m, n, k, Uc, Sc, Vc](const OpContext& c, const TensorPtr& gu, const TensorPtr& gs, const TensorPtr& gvt) -> TensorPtr {
        auto u = Uc, s = Sc, vt = Vc;
        if (c.graph()) std::tie(u, s, vt) = svd(c.input(0));
        auto v = detail::T(vt);
        auto row = unsqueeze(s, -2), col = unsqueeze(s, -1);  // scaling columns and rows by s
        TensorPtr total;
        auto accumulate = [&](const TensorPtr& t) { total = total ? add(total, t) : t; };
        if (gs) accumulate(matmul(mul(u, unsqueeze(gs, -2)), vt));
        if (gu || gvt) {
            auto off = rsub_scalar(1.0f, eye(k, k, s->device));
            auto gap = sub(square(row), square(col));  // s_j^2 - s_i^2
            auto F = div(off, where(off, gap, 1.0f));
            if (gu) {
                auto J = mul(F, matmul(detail::T(u), gu));
                accumulate(matmul(mul(matmul(u, add(J, detail::T(J))), row), vt));
                if (m > k) {
                    auto rest = sub(gu, matmul(u, matmul(detail::T(u), gu)));
                    accumulate(matmul(div(rest, row), vt));
                }
            }
            if (gvt) {
                auto gv = detail::T(gvt);
                auto K = mul(F, matmul(vt, gv));
                accumulate(matmul(u, mul(col, matmul(add(K, detail::T(K)), vt))));
                if (n > k) {
                    auto rest = sub(gvt, matmul(matmul(gvt, v), vt));
                    accumulate(matmul(u, div(rest, col)));
                }
            }
        }
        return total;
    };
    auto Uo = make_op("svd_u", U, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, g, nullptr, nullptr)}; });
    auto So = make_op("svd_s", S, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, nullptr, g, nullptr)}; });
    auto Vo = make_op("svd_v", Vt, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, nullptr, nullptr, g)}; });
    return {Uo, So, Vo};
}
inline TensorPtr svdvals(const TensorPtr& a) { return std::get<1>(svd(a)); }

// Eigenvalues of general square matrices, as (real, imaginary) pairs [..., n, 2]. No gradient.
inline TensorPtr eigvals(const TensorPtr& a_in) {
    DG_PROFILE_OP("eigvals");
    detail::check_square(a_in, "eigvals");
    auto a = op_detail::dense(a_in);
    int64_t n = a->shape.back();
    std::vector<float> out((size_t)(2 * (a->size() / std::max<int64_t>(1, n))));
    detail::for_each_matrix(a, [&](const dense::Matrix& m, int64_t i) {
        std::vector<double> re, im;
        dense::eigvals(m, n, re, im);
        for (int64_t j = 0; j < n; ++j) {
            out[(size_t)(2 * (i * n + j))] = (float)re[(size_t)j];
            out[(size_t)(2 * (i * n + j) + 1)] = (float)im[(size_t)j];
        }
    });
    return detail::make(out, detail::with(detail::batch_shape(a), {n, 2}), a->device);
}

// Least squares min |A X - B| for full-rank A with rows >= columns
inline TensorPtr lstsq(const TensorPtr& a, const TensorPtr& b) {
    auto [q, r] = qr(a);
    return solve_triangular(r, matmul(detail::T(q), b), true);
}
// Moore-Penrose inverse; singular values below rcond * the largest count as zero
inline TensorPtr pinv(const TensorPtr& a, float rcond = 1e-6f) {
    auto [u, s, vt] = svd(a);
    auto cut = mul_scalar(amax(s->detach(), -1, true), rcond);
    auto keep = gt(s, cut);
    auto inverse = mul(keep, reciprocal(where(keep, s, 1.0f)));
    return matmul(mul(detail::T(vt), unsqueeze(inverse, -2)), detail::T(u));
}
inline TensorPtr matrix_rank(const TensorPtr& a, float rcond = 1e-6f) {
    NoGradGuard no_grad;
    auto s = svdvals(a);
    return sum(gt(s, mul_scalar(amax(s, -1, true), rcond)), -1, false);
}
inline TensorPtr cond(const TensorPtr& a) {
    auto s = svdvals(a);
    return div(amax(s, -1, false), amin(s, -1, false));
}
inline TensorPtr matrix_power(const TensorPtr& a, int64_t n) {
    detail::check_square(a, "matrix_power");
    if (n < 0) return matrix_power(inv(a), -n);
    if (n == 0) return contiguous(expand(eye(a->shape.back(), a->shape.back(), a->device), a->shape));
    auto base = a;
    TensorPtr result;
    for (; n > 0; n >>= 1) {
        if (n & 1) result = result ? matmul(result, base) : base;
        if (n > 1) base = matmul(base, base);
    }
    return result;
}
// e^A, by scaling and squaring a Taylor polynomial
inline TensorPtr matrix_exp(const TensorPtr& a) {
    detail::check_square(a, "matrix_exp");
    double largest = 0.0;
    {
        NoGradGuard no_grad;
        largest = amax(sum(abs(a), -1, false))->item();
    }
    int squarings = largest > 0.5 ? (int)std::ceil(std::log2(largest / 0.5)) : 0;
    auto x = mul_scalar(a, (float)std::ldexp(1.0, -squarings));
    auto term = x;
    auto result = add(eye(a->shape.back(), a->shape.back(), a->device), x);
    for (int k = 2; k <= 10; ++k) {
        term = mul_scalar(matmul(term, x), 1.0f / (float)k);
        result = add(result, term);
    }
    for (int i = 0; i < squarings; ++i) result = matmul(result, result);
    return result;
}

enum class MatrixNorm { Frobenius, Nuclear, Spectral, One, Inf };
inline TensorPtr matrix_norm(const TensorPtr& a, MatrixNorm kind = MatrixNorm::Frobenius) {
    switch (kind) {
        case MatrixNorm::Frobenius: return norm(a, 2.0f, std::vector<int>{-2, -1});
        case MatrixNorm::Nuclear: return sum(svdvals(a), -1, false);
        case MatrixNorm::Spectral: return amax(svdvals(a), -1, false);
        case MatrixNorm::One: return amax(sum(abs(a), -2, false), -1, false);
        default: return amax(sum(abs(a), -1, false), -1, false);
    }
}

// P, L, U with A = P L U. Gradients through L and U.
inline std::tuple<TensorPtr, TensorPtr, TensorPtr> lu(const TensorPtr& a_in) {
    DG_PROFILE_OP("lu");
    detail::check_square(a_in, "lu");
    auto a = op_detail::dense(a_in);
    int64_t n = a->shape.back();
    std::vector<float> pv((size_t)a->size(), 0.0f), lv((size_t)a->size(), 0.0f), uv((size_t)a->size(), 0.0f);
    detail::for_each_matrix(a, [&](dense::Matrix m, int64_t i) {
        std::vector<int64_t> pivots;
        dense::lu(m, n, pivots);
        std::vector<int64_t> order((size_t)n);
        for (int64_t r = 0; r < n; ++r) order[(size_t)r] = r;
        for (int64_t r = 0; r < n; ++r) std::swap(order[(size_t)r], order[(size_t)pivots[(size_t)r]]);
        size_t base = (size_t)(i * n * n);
        for (int64_t r = 0; r < n; ++r) {
            pv[base + (size_t)(order[(size_t)r] * n + r)] = 1.0f;
            for (int64_t c = 0; c < n; ++c) {
                if (c < r) lv[base + (size_t)(r * n + c)] = (float)m[(size_t)(r * n + c)];
                else uv[base + (size_t)(r * n + c)] = (float)m[(size_t)(r * n + c)];
            }
            lv[base + (size_t)(r * n + r)] = 1.0f;
        }
    });
    auto P = detail::make(pv, a->shape, a->device), L = detail::make(lv, a->shape, a->device), U = detail::make(uv, a->shape, a->device);
    auto Lc = L->detach(), Uc = U->detach();
    auto pull = [P, Lc, Uc](const OpContext& c, const TensorPtr& gl, const TensorPtr& gu) -> TensorPtr {
        auto l = Lc, u = Uc;
        if (c.graph()) {
            auto again = lu(c.input(0));
            l = std::get<1>(again);
            u = std::get<2>(again);
        }
        TensorPtr mid;
        if (gl) mid = tril(matmul(detail::T(l), gl), -1);
        if (gu) {
            auto part = triu(matmul(gu, detail::T(u)));
            mid = mid ? add(mid, part) : part;
        }
        auto left = solve_triangular(l, mid, false, true, true);                                      // L^-T mid
        auto right = detail::T(solve_triangular(u, contiguous(detail::T(left)), true, false));        // ... U^-T
        return matmul(P, right);
    };
    auto Lo = make_op("lu_l", L, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, g, nullptr)}; });
    auto Uo = make_op("lu_u", U, {a}, {0}, [pull](const OpContext& c, const TensorPtr& g) -> TensorList { return {pull(c, nullptr, g)}; });
    return {P, Lo, Uo};
}

}  // namespace linalg
