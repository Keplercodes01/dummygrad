#pragma once
// Shapes and views: expand, permute, squeeze, slicing with steps, diagonals, joining, splitting,
// padding, flipping. Views share memory with their input. Included from functional.h.

#include "elementwise.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace op_detail {

// The part of a tensor a view shows: shape, strides and offset as functions of the base's strides
struct ViewGeometry {
    std::vector<int64_t> shape, strides;
    int64_t offset = 0;  // relative to the base's first element
};
using ViewFn = std::function<ViewGeometry(const std::vector<int64_t>& base_strides)>;

inline TensorPtr view_gather(const char* name, const TensorPtr& x, const ViewFn& geometry);

// Zeros of base_shape with g written where the view reads (the gradient of view_gather)
inline TensorPtr view_scatter(const char* name, const TensorPtr& g_in, const std::vector<int64_t>& base_shape, const ViewFn& geometry) {
    auto g = strided(g_in);
    auto out = filled(base_shape, g->device, 0.0f);
    auto v = geometry(out->strides);
    if (g->size() > 0) {
        auto grid = kern::make_grid(v.shape, {g->strides, v.strides});
        DG_KERNEL(g->device, copy(g->data_ptr<float>(), out->data_ptr<float>() + v.offset, grid));
    }
    return make_op("view_scatter", out, {g}, {}, [name, geometry](const OpContext&, const TensorPtr& gg) -> TensorList {
        return {view_gather(name, gg, geometry)};
    });
}

// A view of x that reads every element at most once
inline TensorPtr view_gather(const char* name, const TensorPtr& x, const ViewFn& geometry) {
    DG_PROFILE_OP(name);
    if (!x) throw std::invalid_argument(std::string(name) + ": null tensor");
    auto v = geometry(x->strides);
    auto out = std::make_shared<Tensor>(x->storage, v.shape, v.strides, x->global_offset + v.offset, x->device, x->dtype, false);
    auto base_shape = x->shape;
    return make_op(name, out, {x}, {}, [name, base_shape, geometry](const OpContext&, const TensorPtr& g) -> TensorList {
        return {view_scatter(name, g, base_shape, geometry)};
    });
}

}  // namespace op_detail

// x repeated along size-1 (or new leading) dimensions, without copying. -1 keeps a dimension.
inline TensorPtr expand(const TensorPtr& x, const std::vector<int64_t>& shape_in) {
    DG_PROFILE_OP("expand");
    auto shape = shape_in;
    if (shape.size() < x->shape.size()) throw std::invalid_argument("expand: fewer dimensions than the input has");
    size_t extra = shape.size() - x->shape.size();
    for (size_t d = 0; d < shape.size(); ++d) {
        int64_t have = d < extra ? 1 : x->shape[d - extra];
        if (shape[d] == -1) shape[d] = have;
        if (have != 1 && have != shape[d]) {
            throw std::invalid_argument("expand: cannot expand " + shape_string(x->shape) + " to " + shape_string(shape_in));
        }
    }
    if (shape == x->shape) return x;
    auto out = std::make_shared<Tensor>(x->storage, shape, op_detail::broadcast_strides(*x, shape), x->global_offset, x->device, x->dtype, false);
    auto x_shape = x->shape;
    return make_op("expand", out, {x}, {}, [x_shape](const OpContext&, const TensorPtr& g) -> TensorList { return {sum_to(g, x_shape)}; });
}
inline TensorPtr broadcast_to(const TensorPtr& x, const std::vector<int64_t>& shape) { return expand(x, shape); }
inline TensorPtr expand_as(const TensorPtr& x, const TensorPtr& other) { return expand(x, other->shape); }

inline TensorPtr permute(const TensorPtr& x, std::vector<int> dims) {
    DG_PROFILE_OP("permute");
    if (dims.size() != x->shape.size()) throw std::invalid_argument("permute: one entry per dimension is needed");
    std::vector<int> inverse(dims.size(), -1);
    std::vector<int64_t> shape, strides;
    for (size_t i = 0; i < dims.size(); ++i) {
        dims[i] = op_detail::wrap_dim(dims[i], x->ndim(), "permute");
        if (inverse[(size_t)dims[i]] != -1) throw std::invalid_argument("permute: repeated dimension");
        inverse[(size_t)dims[i]] = (int)i;
        shape.push_back(x->shape[(size_t)dims[i]]);
        strides.push_back(x->strides[(size_t)dims[i]]);
    }
    auto out = std::make_shared<Tensor>(x->storage, shape, strides, x->global_offset, x->device, x->dtype, false);
    return make_op("permute", out, {x}, {}, [inverse](const OpContext&, const TensorPtr& g) -> TensorList { return {permute(g, inverse)}; });
}

inline TensorPtr unsqueeze(const TensorPtr& x, int dim) {
    auto shape = x->shape;
    int d = dim < 0 ? dim + (int)x->ndim() + 1 : dim;
    if (d < 0 || d > (int)x->ndim()) throw std::invalid_argument("unsqueeze: dim out of range");
    shape.insert(shape.begin() + d, 1);
    auto strides = x->strides;
    strides.insert(strides.begin() + d, d < (int)x->ndim() ? x->strides[(size_t)d] * x->shape[(size_t)d] : 1);
    auto out = std::make_shared<Tensor>(x->storage, shape, strides, x->global_offset, x->device, x->dtype, false);
    auto x_shape = x->shape;
    return make_op("unsqueeze", out, {x}, {}, [x_shape](const OpContext&, const TensorPtr& g) -> TensorList { return {reshape(g, x_shape)}; });
}
// Drops dimension dim, which must have size 1
inline TensorPtr squeeze(const TensorPtr& x, int dim) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "squeeze");
    if (x->ndim() == 0 || x->shape[(size_t)d] != 1) return x;
    auto shape = x->shape, strides = x->strides;
    shape.erase(shape.begin() + d);
    strides.erase(strides.begin() + d);
    auto out = std::make_shared<Tensor>(x->storage, shape, strides, x->global_offset, x->device, x->dtype, false);
    auto x_shape = x->shape;
    return make_op("squeeze", out, {x}, {}, [x_shape](const OpContext&, const TensorPtr& g) -> TensorList { return {reshape(g, x_shape)}; });
}
// Drops every dimension of size 1
inline TensorPtr squeeze(const TensorPtr& x) {
    auto y = x;
    for (int d = (int)x->ndim() - 1; d >= 0; --d) y = squeeze(y, d);
    return y;
}
inline TensorPtr flatten(const TensorPtr& x, int start = 0, int end = -1) {
    if (x->ndim() == 0) return reshape(x, {1});
    int s = op_detail::wrap_dim(start, x->ndim(), "flatten"), e = op_detail::wrap_dim(end, x->ndim(), "flatten");
    std::vector<int64_t> shape(x->shape.begin(), x->shape.begin() + s);
    int64_t n = 1;
    for (int d = s; d <= e; ++d) n *= x->shape[(size_t)d];
    shape.push_back(n);
    shape.insert(shape.end(), x->shape.begin() + e + 1, x->shape.end());
    return reshape(x, shape);
}
inline TensorPtr unflatten(const TensorPtr& x, int dim, const std::vector<int64_t>& sizes) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "unflatten");
    auto shape = x->shape;
    shape.erase(shape.begin() + d);
    shape.insert(shape.begin() + d, sizes.begin(), sizes.end());
    return reshape(x, shape);
}
inline TensorPtr movedim(const TensorPtr& x, int from, int to) {
    int n = (int)x->ndim();
    int f = op_detail::wrap_dim(from, n, "movedim"), t = op_detail::wrap_dim(to, n, "movedim");
    std::vector<int> order;
    for (int d = 0; d < n; ++d) if (d != f) order.push_back(d);
    order.insert(order.begin() + t, f);
    return permute(x, order);
}

// Elements start, start + step, ... before end along dim
inline TensorPtr slice_step(const TensorPtr& x, int dim_in, int64_t start, int64_t end, int64_t step) {
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "slice_step");
    int64_t size = x->shape[(size_t)dim];
    if (start < 0) start += size;
    if (end < 0) end += size;
    end = std::min(end, size);
    if (step <= 0 || start < 0 || start > end) throw std::invalid_argument("slice_step: invalid bounds or step");
    auto shape = x->shape;
    shape[(size_t)dim] = (end - start + step - 1) / step;
    return op_detail::view_gather("slice_step", x, [shape, dim, start, step](const std::vector<int64_t>& base) {
        op_detail::ViewGeometry v{shape, base, start * base[(size_t)dim]};
        v.strides[(size_t)dim] *= step;
        return v;
    });
}
inline TensorPtr narrow(const TensorPtr& x, int dim, int64_t start, int64_t length) { return slice_step(x, dim, start, start + length, 1); }
// Index i along dim; the dimension disappears
inline TensorPtr select(const TensorPtr& x, int dim_in, int64_t i) {
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "select");
    if (x->ndim() == 0) throw std::invalid_argument("select: tensor has no dimensions");
    if (i < 0) i += x->shape[(size_t)dim];
    if (i < 0 || i >= x->shape[(size_t)dim]) throw std::out_of_range("select: index out of range");
    auto shape = x->shape;
    shape.erase(shape.begin() + dim);
    return op_detail::view_gather("select", x, [shape, dim, i](const std::vector<int64_t>& base) {
        op_detail::ViewGeometry v{shape, base, i * base[(size_t)dim]};
        v.strides.erase(v.strides.begin() + dim);
        return v;
    });
}
// View of a diagonal of dims (dim1, dim2), as a new last dimension
inline TensorPtr diagonal(const TensorPtr& x, int64_t offset = 0, int dim1 = 0, int dim2 = 1) {
    int a = op_detail::wrap_dim(dim1, x->ndim(), "diagonal"), b = op_detail::wrap_dim(dim2, x->ndim(), "diagonal");
    if (x->ndim() < 2 || a == b) throw std::invalid_argument("diagonal: two different dimensions are needed");
    int64_t rows = x->shape[(size_t)a], cols = x->shape[(size_t)b];
    int64_t len = offset >= 0 ? std::min(rows, cols - offset) : std::min(rows + offset, cols);
    len = std::max<int64_t>(len, 0);
    std::vector<int64_t> shape;
    for (int d = 0; d < (int)x->ndim(); ++d) if (d != a && d != b) shape.push_back(x->shape[(size_t)d]);
    shape.push_back(len);
    return op_detail::view_gather("diagonal", x, [shape, a, b, offset](const std::vector<int64_t>& base) {
        op_detail::ViewGeometry v;
        v.shape = shape;
        for (int d = 0; d < (int)base.size(); ++d) if (d != a && d != b) v.strides.push_back(base[(size_t)d]);
        v.strides.push_back(base[(size_t)a] + base[(size_t)b]);
        v.offset = offset >= 0 ? offset * base[(size_t)b] : -offset * base[(size_t)a];
        return v;
    });
}
// Square matrices [..., n, n] with x [..., n] on their diagonal
inline TensorPtr diag_embed(const TensorPtr& x) {
    if (x->ndim() < 1) throw std::invalid_argument("diag_embed: at least one dimension is needed");
    int64_t n = x->shape.back();
    auto shape = x->shape;
    shape.push_back(n);
    auto view_shape = x->shape;
    return op_detail::view_scatter("diag_embed", x, shape, [view_shape](const std::vector<int64_t>& base) {
        op_detail::ViewGeometry v;
        v.shape = view_shape;
        v.strides.assign(base.begin(), base.end() - 2);
        v.strides.push_back(base[base.size() - 2] + base[base.size() - 1]);
        return v;
    });
}
// A vector's diagonal matrix, or a matrix's diagonal
inline TensorPtr diag(const TensorPtr& x) {
    if (x->ndim() == 1) return diag_embed(x);
    if (x->ndim() == 2) return diagonal(x);
    throw std::invalid_argument("diag: a vector or a matrix is needed");
}

// Concatenation of any number of tensors along any dimension
inline TensorPtr cat(const TensorList& xs_in, int dim_in = 0) {
    DG_PROFILE_OP("cat");
    if (xs_in.empty()) throw std::invalid_argument("cat: no tensors");
    TensorList xs;
    for (const auto& t : xs_in) xs.push_back(op_detail::strided(t));
    int dim = op_detail::wrap_dim(dim_in, xs[0]->ndim(), "cat");
    auto shape = xs[0]->shape;
    if (shape.empty()) throw std::invalid_argument("cat: tensors must have at least one dimension");
    shape[(size_t)dim] = 0;
    for (const auto& t : xs) {
        op_detail::check_same_device(xs[0], t, "cat");
        auto expected = shape;
        if (t->shape.size() == shape.size()) expected[(size_t)dim] = t->shape[(size_t)dim];
        if (t->shape != expected) {
            throw std::invalid_argument("cat: shapes " + shape_string(xs[0]->shape) + " and " + shape_string(t->shape) + " differ outside dim " + std::to_string(dim));
        }
    }
    std::vector<int64_t> sizes;
    for (const auto& t : xs) {
        sizes.push_back(t->shape[(size_t)dim]);
        shape[(size_t)dim] += sizes.back();
    }
    auto out = op_detail::empty(shape, xs[0]->device);
    int64_t at = 0;
    for (const auto& t : xs) {
        if (t->size() > 0) {
            auto grid = kern::make_grid(t->shape, {t->strides, out->strides});
            DG_KERNEL(t->device, copy(t->data_ptr<float>(), out->data_ptr<float>() + at * out->strides[(size_t)dim], grid));
        }
        at += t->shape[(size_t)dim];
    }
    return make_op("cat", out, xs, {}, [sizes, dim](const OpContext& c, const TensorPtr& g) -> TensorList {
        TensorList grads(sizes.size());
        int64_t start = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (c.needs(i)) grads[i] = narrow(g, dim, start, sizes[i]);
            start += sizes[i];
        }
        return grads;
    });
}
inline TensorPtr stack(const TensorList& xs, int dim = 0) {
    TensorList parts;
    for (const auto& t : xs) parts.push_back(unsqueeze(t, dim));
    return cat(parts, dim < 0 && !xs.empty() ? dim + (int)xs[0]->ndim() + 1 : dim);
}
inline TensorPtr hstack(const TensorList& xs) { return cat(xs, !xs.empty() && xs[0]->ndim() <= 1 ? 0 : 1); }
inline TensorPtr vstack(const TensorList& xs) {
    TensorList rows;
    for (const auto& t : xs) rows.push_back(t->ndim() == 1 ? unsqueeze(t, 0) : t);
    return cat(rows, 0);
}

// Views of consecutive pieces of the given sizes along dim
inline TensorList split(const TensorPtr& x, const std::vector<int64_t>& sizes, int dim = 0) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "split");
    if (std::accumulate(sizes.begin(), sizes.end(), (int64_t)0) != x->shape[(size_t)d]) throw std::invalid_argument("split: sizes must add up to the dimension");
    TensorList out;
    int64_t start = 0;
    for (int64_t s : sizes) {
        out.push_back(narrow(x, d, start, s));
        start += s;
    }
    return out;
}
inline TensorList split(const TensorPtr& x, int64_t size, int dim = 0) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "split");
    if (size <= 0) throw std::invalid_argument("split: size must be positive");
    std::vector<int64_t> sizes;
    for (int64_t left = x->shape[(size_t)d]; left > 0; left -= size) sizes.push_back(std::min(left, size));
    return split(x, sizes, d);
}
inline TensorList chunk(const TensorPtr& x, int64_t chunks, int dim = 0) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "chunk");
    if (chunks <= 0) throw std::invalid_argument("chunk: the number of chunks must be positive");
    return split(x, std::max<int64_t>(1, (x->shape[(size_t)d] + chunks - 1) / chunks), d);
}
inline TensorList unbind(const TensorPtr& x, int dim = 0) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "unbind");
    TensorList out;
    for (int64_t i = 0; i < x->shape[(size_t)d]; ++i) out.push_back(select(x, d, i));
    return out;
}

// x tiled repeats[d] times along each dimension (extra leading entries add dimensions)
inline TensorPtr repeat(const TensorPtr& x, const std::vector<int64_t>& repeats) {
    if (repeats.size() < x->shape.size()) throw std::invalid_argument("repeat: one count per dimension is needed");
    size_t extra = repeats.size() - x->shape.size();
    std::vector<int64_t> spaced, tiled, shape;
    for (size_t d = 0; d < repeats.size(); ++d) {
        int64_t size = d < extra ? 1 : x->shape[d - extra];
        spaced.push_back(1);
        spaced.push_back(size);
        tiled.push_back(repeats[d]);
        tiled.push_back(size);
        shape.push_back(repeats[d] * size);
    }
    return reshape(expand(reshape(x, spaced), tiled), shape);
}
inline TensorPtr tile(const TensorPtr& x, std::vector<int64_t> repeats) {
    while (repeats.size() < x->shape.size()) repeats.insert(repeats.begin(), 1);
    return repeat(x, repeats);
}
// Each element repeated n times along dim: [a, b] -> [a, a, b, b]
inline TensorPtr repeat_interleave(const TensorPtr& x, int64_t n, int dim) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "repeat_interleave");
    auto shape = x->shape;
    shape[(size_t)d] *= n;
    auto y = unsqueeze(x, d + 1);
    auto expanded = y->shape;
    expanded[(size_t)d + 1] = n;
    return reshape(expand(y, expanded), shape);
}

inline TensorPtr flip(const TensorPtr& x_in, const std::vector<int>& dims_in) {
    DG_PROFILE_OP("flip");
    auto x = op_detail::strided(x_in);
    auto strides = x->strides;
    int64_t offset = 0;
    std::vector<int> dims;
    for (int dim : dims_in) {
        int d = op_detail::wrap_dim(dim, x->ndim(), "flip");
        dims.push_back(d);
        offset += strides[(size_t)d] * (x->shape[(size_t)d] - 1);
        strides[(size_t)d] = -strides[(size_t)d];
    }
    auto out = op_detail::empty(x->shape, x->device);
    if (x->size() > 0) {
        auto grid = kern::make_grid(x->shape, {strides, out->strides});
        DG_KERNEL(x->device, copy(x->data_ptr<float>() + offset, out->data_ptr<float>(), grid));
    }
    return make_op("flip", out, {x}, {}, [dims](const OpContext&, const TensorPtr& g) -> TensorList { return {flip(g, dims)}; });
}
// Cyclic shift along dim: element i moves to i + shift
inline TensorPtr roll(const TensorPtr& x, int64_t shift, int dim) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "roll");
    int64_t n = x->shape[(size_t)d];
    if (n == 0) return x;
    shift = ((shift % n) + n) % n;
    if (shift == 0) return x;
    return cat({narrow(x, d, n - shift, shift), narrow(x, d, 0, n - shift)}, d);
}

enum class PadMode { Constant, Reflect, Replicate, Circular };

// PyTorch's padding order: (left, right) of the last dimension, then of the one before it, ...
// Negative amounts crop (constant mode).
inline TensorPtr pad(const TensorPtr& x_in, const std::vector<int64_t>& amounts, PadMode mode = PadMode::Constant, float value = 0.0f) {
    if (amounts.size() % 2 != 0 || (int64_t)amounts.size() / 2 > x_in->ndim()) throw std::invalid_argument("pad: amounts come in (left, right) pairs, from the last dimension");
    auto x = op_detail::strided(x_in);
    int nd = (int)x->ndim();
    for (size_t pair = 0; pair < amounts.size() / 2; ++pair) {
        int d = nd - 1 - (int)pair;
        int64_t left = amounts[2 * pair], right = amounts[2 * pair + 1], n = x->shape[(size_t)d];
        if (left == 0 && right == 0) continue;
        if (left < 0 || right < 0) {
            int64_t l = std::max<int64_t>(0, -left), r = std::max<int64_t>(0, -right);
            x = narrow(x, d, l, n - l - r);
            left = std::max<int64_t>(left, 0);
            right = std::max<int64_t>(right, 0);
            n = x->shape[(size_t)d];
            if (left == 0 && right == 0) continue;
        }
        auto side = [&](int64_t count, bool is_left) -> TensorPtr {
            switch (mode) {
                case PadMode::Constant: {
                    auto shape = x->shape;
                    shape[(size_t)d] = count;
                    return op_detail::filled(shape, x->device, value);
                }
                case PadMode::Reflect:
                    if (count >= n) throw std::invalid_argument("pad: reflection must be smaller than the dimension");
                    return flip(is_left ? narrow(x, d, 1, count) : narrow(x, d, n - 1 - count, count), {d});
                case PadMode::Replicate: {
                    auto shape = x->shape;
                    shape[(size_t)d] = count;
                    return expand(narrow(x, d, is_left ? 0 : n - 1, 1), shape);
                }
                default:
                    if (count > n) throw std::invalid_argument("pad: circular padding cannot exceed the dimension");
                    return is_left ? narrow(x, d, n - count, count) : narrow(x, d, 0, count);
            }
        };
        TensorList parts;
        if (left > 0) parts.push_back(side(left, true));
        parts.push_back(x);
        if (right > 0) parts.push_back(side(right, false));
        x = cat(parts, d);
    }
    return x;
}

// ---------------------------------------------------------------------------------------------
// Triangles (of the last two dimensions)
// ---------------------------------------------------------------------------------------------

namespace op_detail {
inline TensorPtr triangle_mask(int64_t rows, int64_t cols, int64_t diagonal, bool lower, Device device) {
    std::vector<float> m((size_t)(rows * cols));
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j) m[(size_t)(i * cols + j)] = (lower ? j - i <= diagonal : j - i >= diagonal) ? 1.0f : 0.0f;
    return from_values(m, {rows, cols}, device);
}
}  // namespace op_detail

inline TensorPtr tril(const TensorPtr& x, int64_t diagonal = 0) {
    if (x->ndim() < 2) throw std::invalid_argument("tril: at least two dimensions are needed");
    return mul(x, op_detail::triangle_mask(x->shape[(size_t)x->ndim() - 2], x->shape.back(), diagonal, true, x->device));
}
inline TensorPtr triu(const TensorPtr& x, int64_t diagonal = 0) {
    if (x->ndim() < 2) throw std::invalid_argument("triu: at least two dimensions are needed");
    return mul(x, op_detail::triangle_mask(x->shape[(size_t)x->ndim() - 2], x->shape.back(), diagonal, false, x->device));
}

// Moves between devices inside the graph (Tensor::to copies without history)
inline TensorPtr to_device(const TensorPtr& x, Device device) {
    if (x->device == device) return x;
    auto out = make_contiguous(x)->to(device)->detach();
    Device from = x->device;
    return make_op("to_device", out, {x}, {}, [from](const OpContext&, const TensorPtr& g) -> TensorList { return {to_device(g, from)}; });
}
