#pragma once
// Products and contractions: einsum, tensordot, outer, kron, cdist, ... and tensor constructors that
// take ranges or grids. Included from functional.h.

#include "random_ops.h"

#include <map>
#include <set>
#include <string>

// ---------------------------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------------------------

inline TensorPtr full(const std::vector<int64_t>& shape, float value, Device device = Device::CPU) { return op_detail::filled(shape, device, value); }
inline TensorPtr full_like(const TensorPtr& x, float value) { return op_detail::filled(x->shape, x->device, value); }
inline TensorPtr zeros_like(const TensorPtr& x) { return full_like(x, 0.0f); }
inline TensorPtr ones_like(const TensorPtr& x) { return full_like(x, 1.0f); }
inline TensorPtr scalar_tensor(float value, Device device = Device::CPU) { return op_detail::filled({}, device, value); }
// Values in row-major order
inline TensorPtr tensor(const std::vector<float>& values, const std::vector<int64_t>& shape, Device device = Device::CPU, bool requires_grad = false) {
    auto t = op_detail::from_values(values, shape, device);
    t->requires_grad = requires_grad;
    return t;
}
inline TensorPtr tensor(const std::vector<float>& values, Device device = Device::CPU, bool requires_grad = false) {
    return tensor(values, {(int64_t)values.size()}, device, requires_grad);
}
// start, start + step, ... before end
inline TensorPtr arange(double start, double end, double step = 1.0, Device device = Device::CPU) {
    if (step == 0.0 || (end - start) * step < 0.0) throw std::invalid_argument("arange: step must lead from start to end");
    int64_t n = (int64_t)std::ceil((end - start) / step);
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)(start + (double)i * step);
    return op_detail::from_values(v, {n}, device);
}
inline TensorPtr arange(double end) { return arange(0.0, end); }
// n evenly spaced values from start to end, both included
inline TensorPtr linspace(double start, double end, int64_t n, Device device = Device::CPU) {
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)(n == 1 ? start : start + (end - start) * (double)i / (double)(n - 1));
    return op_detail::from_values(v, {n}, device);
}
inline TensorPtr logspace(double start, double end, int64_t n, double base = 10.0, Device device = Device::CPU) {
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (float)std::pow(base, n == 1 ? start : start + (end - start) * (double)i / (double)(n - 1));
    return op_detail::from_values(v, {n}, device);
}
inline TensorPtr eye(int64_t n, int64_t m = -1, Device device = Device::CPU) {
    if (m < 0) m = n;
    std::vector<float> v((size_t)(n * m), 0.0f);
    for (int64_t i = 0; i < std::min(n, m); ++i) v[(size_t)(i * m + i)] = 1.0f;
    return op_detail::from_values(v, {n, m}, device);
}
// Coordinate grids of 1-D tensors, indexed like a matrix ("ij")
inline TensorList meshgrid(const TensorList& axes) {
    std::vector<int64_t> shape;
    for (const auto& a : axes) shape.push_back(a->size());
    TensorList out;
    for (size_t i = 0; i < axes.size(); ++i) {
        std::vector<int64_t> s(axes.size(), 1);
        s[i] = shape[i];
        out.push_back(contiguous(expand(reshape(axes[i], s), shape)));
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// einsum
// ---------------------------------------------------------------------------------------------

namespace op_detail {

// An operand of an einsum: its tensor and one label per dimension
struct Labeled {
    TensorPtr t;
    std::string labels;
};

// Keeps one copy of a label that repeats (its diagonal)
inline Labeled take_diagonals(Labeled x) {
    for (size_t i = 0; i < x.labels.size(); ++i) {
        size_t j = x.labels.find(x.labels[i], i + 1);
        if (j == std::string::npos) continue;
        if (x.t->shape[i] != x.t->shape[j]) throw std::invalid_argument("einsum: a repeated subscript has two sizes");
        char label = x.labels[i];
        x.t = diagonal(x.t, 0, (int)i, (int)j);  // the diagonal becomes the last dimension
        x.labels.erase(j, 1);
        x.labels.erase(i, 1);
        x.labels.push_back(label);
        return take_diagonals(std::move(x));
    }
    return x;
}

// Sums out the labels that nothing later needs
inline Labeled drop_labels(Labeled x, const std::string& needed) {
    std::vector<int> dims;
    std::string kept;
    for (size_t i = 0; i < x.labels.size(); ++i) {
        if (needed.find(x.labels[i]) == std::string::npos) dims.push_back((int)i);
        else kept.push_back(x.labels[i]);
    }
    if (!dims.empty()) x.t = sum(x.t, dims, false);
    x.labels = kept;
    return x;
}

inline Labeled contract_pair(Labeled a, Labeled b, const std::string& needed) {
    a = drop_labels(std::move(a), needed + b.labels);
    b = drop_labels(std::move(b), needed + a.labels);
    std::string batch, inner, left, right;
    for (char c : a.labels) {
        bool in_b = b.labels.find(c) != std::string::npos;
        if (!in_b) left.push_back(c);
        else if (needed.find(c) != std::string::npos) batch.push_back(c);
        else inner.push_back(c);
    }
    for (char c : b.labels) if (a.labels.find(c) == std::string::npos) right.push_back(c);

    // Both operands get every shared label's full size (a size-1 dimension broadcasts)
    std::map<char, int64_t> size;
    for (const Labeled* x : {&a, &b}) {
        for (size_t i = 0; i < x->labels.size(); ++i) {
            int64_t n = x->t->shape[i];
            auto it = size.find(x->labels[i]);
            if (it == size.end()) size[x->labels[i]] = n;
            else if (it->second != n) {
                if (it->second != 1 && n != 1) throw std::invalid_argument("einsum: a subscript has two sizes");
                it->second = std::max(it->second, n);
            }
        }
    }
    auto arrange = [&](const Labeled& x, const std::string& first, const std::string& second, const std::string& third,
                       int64_t& n2, int64_t& n3) {
        std::vector<int> order;
        std::vector<int64_t> full_shape;
        int64_t n1 = 1;
        n2 = n3 = 1;
        for (const std::string* group : {&first, &second, &third}) {
            for (char c : *group) {
                order.push_back((int)x.labels.find(c));
                int64_t n = size[c];
                full_shape.push_back(n);
                (group == &first ? n1 : group == &second ? n2 : n3) *= n;
            }
        }
        auto t = expand(permute(x.t, order), full_shape);
        return batch.empty() ? reshape(t, {n2, n3}) : reshape(t, {n1, n2, n3});
    };
    int64_t rows, k1, k2, cols;
    auto l = arrange(a, batch, left, inner, rows, k1);
    auto r = arrange(b, batch, inner, right, k2, cols);
    Labeled out;
    out.labels = batch + left + right;
    std::vector<int64_t> shape;
    for (char c : out.labels) shape.push_back(size[c]);
    out.t = reshape(matmul(l, r), shape);
    return out;
}

}  // namespace op_detail

// Einstein summation, as in NumPy and PyTorch: einsum("bij,bjk->bik", {a, b}). Any number of operands
// (contracted left to right), repeated subscripts (diagonals), implicit outputs and "...".
inline TensorPtr einsum(const std::string& equation_in, const TensorList& operands) {
    DG_PROFILE_OP("einsum");
    std::string eq_text;
    for (char c : equation_in) if (c != ' ') eq_text.push_back(c);
    std::string lhs = eq_text, rhs;
    size_t arrow = eq_text.find("->");
    bool explicit_output = arrow != std::string::npos;
    if (explicit_output) {
        lhs = eq_text.substr(0, arrow);
        rhs = eq_text.substr(arrow + 2);
    }
    std::vector<std::string> terms(1);
    for (char c : lhs) {
        if (c == ',') terms.emplace_back();
        else terms.back().push_back(c);
    }
    if (terms.size() != operands.size()) throw std::invalid_argument("einsum: the equation names " + std::to_string(terms.size()) + " operands, " + std::to_string(operands.size()) + " given");

    // "..." becomes digit labels, aligned from the right across operands
    size_t ellipsis_rank = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        size_t at = terms[i].find("...");
        size_t named = terms[i].size() - (at == std::string::npos ? 0 : 3);
        size_t nd = operands[i]->shape.size();
        if (named > nd || (at == std::string::npos && named != nd)) throw std::invalid_argument("einsum: \"" + terms[i] + "\" does not match a tensor with " + std::to_string(nd) + " dimensions");
        if (at != std::string::npos) ellipsis_rank = std::max(ellipsis_rank, nd - named);
    }
    if (ellipsis_rank > 10) throw std::invalid_argument("einsum: too many dimensions under \"...\"");
    auto ellipsis_labels = [&](size_t rank) {
        std::string s;
        for (size_t d = ellipsis_rank - rank; d < ellipsis_rank; ++d) s.push_back((char)('0' + d));
        return s;
    };
    std::vector<op_detail::Labeled> xs;
    for (size_t i = 0; i < terms.size(); ++i) {
        size_t at = terms[i].find("...");
        std::string labels = terms[i];
        if (at != std::string::npos) labels.replace(at, 3, ellipsis_labels(operands[i]->shape.size() - (terms[i].size() - 3)));
        for (char c : labels) if (!std::isalnum((unsigned char)c)) throw std::invalid_argument("einsum: subscripts must be letters");
        xs.push_back({operands[i], labels});
    }
    if (explicit_output) {
        size_t at = rhs.find("...");
        if (at != std::string::npos) rhs.replace(at, 3, ellipsis_labels(ellipsis_rank));
    } else {
        // Implicit output: the ellipsis, then labels that occur once, in alphabetical order
        std::map<char, int> count;
        for (const auto& t : terms) for (char c : t) if (std::isalpha((unsigned char)c)) count[c]++;
        rhs = ellipsis_labels(ellipsis_rank);
        for (const auto& [c, n] : count) if (n == 1) rhs.push_back(c);
    }
    {
        std::set<char> seen;
        for (char c : rhs) {
            bool known = false;
            for (const auto& x : xs) known = known || x.labels.find(c) != std::string::npos;
            if (!known || !seen.insert(c).second) throw std::invalid_argument("einsum: output subscripts must be distinct and appear in an operand");
        }
    }

    for (auto& x : xs) x = op_detail::take_diagonals(std::move(x));
    op_detail::Labeled acc = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) {
        std::string needed = rhs;
        for (size_t j = i + 1; j < xs.size(); ++j) needed += xs[j].labels;
        acc = op_detail::contract_pair(std::move(acc), xs[i], needed);
    }
    acc = op_detail::drop_labels(std::move(acc), rhs);
    std::vector<int> order;
    for (char c : rhs) order.push_back((int)acc.labels.find(c));
    return permute(acc.t, order);
}

// ---------------------------------------------------------------------------------------------
// Products
// ---------------------------------------------------------------------------------------------

inline TensorPtr dot(const TensorPtr& a, const TensorPtr& b) { return sum(mul(reshape(a, {a->size()}), reshape(b, {b->size()}))); }
inline TensorPtr outer(const TensorPtr& a, const TensorPtr& b) { return mul(contiguous(expand(reshape(a, {a->size(), 1}), {a->size(), b->size()})), reshape(b, {1, b->size()})); }
inline TensorPtr mv(const TensorPtr& m, const TensorPtr& v) { return reshape(matmul(m, reshape(v, {v->size(), 1})), {m->shape[0]}); }
inline TensorPtr bmm(const TensorPtr& a, const TensorPtr& b) { return matmul(a, b); }
inline TensorPtr trace(const TensorPtr& x) { return sum(diagonal(x, 0, -2, -1), -1, false); }
// beta * m + alpha * a @ b
inline TensorPtr addmm(const TensorPtr& m, const TensorPtr& a, const TensorPtr& b, float beta = 1.0f, float alpha = 1.0f) {
    return add(mul_scalar(matmul(a, b), alpha), mul_scalar(m, beta));
}
// Sums over the last dims dimensions of a and the first dims of b
inline TensorPtr tensordot(const TensorPtr& a, const TensorPtr& b, int dims = 2) {
    if (dims < 0 || dims > a->ndim() || dims > b->ndim()) throw std::invalid_argument("tensordot: too many dimensions to contract");
    std::vector<int64_t> out(a->shape.begin(), a->shape.end() - dims);
    int64_t rows = 1, k = 1, cols = 1;
    for (int64_t d : out) rows *= d;
    for (int d = 0; d < dims; ++d) k *= b->shape[(size_t)d];
    for (size_t d = (size_t)dims; d < b->shape.size(); ++d) {
        out.push_back(b->shape[d]);
        cols *= b->shape[d];
    }
    return reshape(matmul(reshape(a, {rows, k}), reshape(b, {k, cols})), out);
}
inline TensorPtr kron(const TensorPtr& a, const TensorPtr& b) {
    if (a->ndim() != 2 || b->ndim() != 2) throw std::invalid_argument("kron: matrices are needed");
    int64_t m = a->shape[0], n = a->shape[1], p = b->shape[0], q = b->shape[1];
    auto t = outer(a, b);
    return reshape(permute(reshape(t, {m, n, p, q}), {0, 2, 1, 3}), {m * p, n * q});
}
inline TensorPtr cross(const TensorPtr& a, const TensorPtr& b, int dim = -1) {
    auto part = [&](const TensorPtr& t, int64_t i) { return select(t, dim, i); };
    auto x = sub(mul(part(a, 1), part(b, 2)), mul(part(a, 2), part(b, 1)));
    auto y = sub(mul(part(a, 2), part(b, 0)), mul(part(a, 0), part(b, 2)));
    auto z = sub(mul(part(a, 0), part(b, 1)), mul(part(a, 1), part(b, 0)));
    return stack({x, y, z}, op_detail::wrap_dim(dim, a->ndim(), "cross"));
}
// Pairwise p-distances between rows: [..., n, d], [..., m, d] -> [..., n, m]
inline TensorPtr cdist(const TensorPtr& a, const TensorPtr& b, float p = 2.0f) {
    return norm(sub(unsqueeze(a, -2), unsqueeze(b, -3)), p, -1);
}
inline TensorPtr pairwise_distance(const TensorPtr& a, const TensorPtr& b, float p = 2.0f, float eps = 1e-6f) {
    return norm(add_scalar(sub(a, b), eps), p, -1);
}
// Differences of neighbours along dim
inline TensorPtr diff(const TensorPtr& x, int dim = -1) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "diff");
    int64_t n = x->shape[(size_t)d];
    return sub(narrow(x, d, 1, n - 1), narrow(x, d, 0, n - 1));
}
// Trapezoid-rule integral along the last dimension with spacing dx
inline TensorPtr trapezoid(const TensorPtr& y, float dx = 1.0f) {
    int64_t n = y->shape.back();
    return mul_scalar(sum(add(narrow(y, -1, 1, n - 1), narrow(y, -1, 0, n - 1)), -1, false), 0.5f * dx);
}
