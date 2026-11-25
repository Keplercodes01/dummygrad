#pragma once
// a + b, x->index({Slice(1, 3), 0}) and std::cout << x. Included from functional.h.
//
// Comparisons have no operators: on shared_ptr, == and < already compare pointers. Use eq, lt, ...

#include "conv_ops.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <optional>
#include <ostream>
#include <sstream>

// ---------------------------------------------------------------------------------------------
// Arithmetic: elementwise, with broadcasting. Matrix products are matmul(a, b).
// ---------------------------------------------------------------------------------------------

inline TensorPtr operator+(const TensorPtr& a, const TensorPtr& b) { return add(a, b); }
inline TensorPtr operator-(const TensorPtr& a, const TensorPtr& b) { return sub(a, b); }
inline TensorPtr operator*(const TensorPtr& a, const TensorPtr& b) { return mul(a, b); }
inline TensorPtr operator/(const TensorPtr& a, const TensorPtr& b) { return div(a, b); }
inline TensorPtr operator+(const TensorPtr& a, float s) { return add_scalar(a, s); }
inline TensorPtr operator-(const TensorPtr& a, float s) { return sub_scalar(a, s); }
inline TensorPtr operator*(const TensorPtr& a, float s) { return mul_scalar(a, s); }
inline TensorPtr operator/(const TensorPtr& a, float s) { return div_scalar(a, s); }
inline TensorPtr operator+(float s, const TensorPtr& a) { return add_scalar(a, s); }
inline TensorPtr operator-(float s, const TensorPtr& a) { return rsub_scalar(s, a); }
inline TensorPtr operator*(float s, const TensorPtr& a) { return mul_scalar(a, s); }
inline TensorPtr operator/(float s, const TensorPtr& a) { return rdiv_scalar(s, a); }
inline TensorPtr operator-(const TensorPtr& a) { return neg(a); }

// These rebind the variable to a new tensor (a = a + b). Nothing is modified in place, so they are safe
// inside a graph, and other variables holding the old tensor keep it.
inline TensorPtr& operator+=(TensorPtr& a, const TensorPtr& b) { return a = add(a, b); }
inline TensorPtr& operator-=(TensorPtr& a, const TensorPtr& b) { return a = sub(a, b); }
inline TensorPtr& operator*=(TensorPtr& a, const TensorPtr& b) { return a = mul(a, b); }
inline TensorPtr& operator/=(TensorPtr& a, const TensorPtr& b) { return a = div(a, b); }
inline TensorPtr& operator+=(TensorPtr& a, float s) { return a = add_scalar(a, s); }
inline TensorPtr& operator-=(TensorPtr& a, float s) { return a = sub_scalar(a, s); }
inline TensorPtr& operator*=(TensorPtr& a, float s) { return a = mul_scalar(a, s); }
inline TensorPtr& operator/=(TensorPtr& a, float s) { return a = div_scalar(a, s); }

// ---------------------------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------------------------

// start:stop:step, with Python's rules: negative positions count from the end, out-of-range ones are
// clamped, a missing one means "from the end it starts at", and step may be negative
struct Slice {
    std::optional<int64_t> start, stop;
    int64_t step = 1;
    Slice() = default;
    Slice(std::optional<int64_t> start, std::optional<int64_t> stop = std::nullopt, int64_t step = 1)
        : start(start), stop(stop), step(step) {}
};
struct EllipsisType {};
struct NewAxisType {};
inline constexpr EllipsisType Ellipsis{};  // as many full slices as it takes
inline constexpr NewAxisType NewAxis{};    // a new dimension of size 1
// A mask of one or more dimensions (nonzero = take). Index tensors are floats here, so a mask must say it is one.
struct Mask {
    TensorPtr mask;
    explicit Mask(TensorPtr m) : mask(std::move(m)) {}
};

// One entry of an index
class IndexItem {
public:
    enum class Kind { Integer, Slice, Ellipsis, NewAxis, Tensor, Mask };
    Kind kind;
    int64_t integer = 0;
    ::Slice slice;
    TensorPtr tensor;  // an index tensor or a mask

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    IndexItem(I i) : kind(Kind::Integer), integer((int64_t)i) {}
    IndexItem(::Slice s) : kind(Kind::Slice), slice(s) {}
    IndexItem(EllipsisType) : kind(Kind::Ellipsis) {}
    IndexItem(NewAxisType) : kind(Kind::NewAxis) {}
    IndexItem(TensorPtr t) : kind(Kind::Tensor), tensor(std::move(t)) {}
    IndexItem(::Mask m) : kind(Kind::Mask), tensor(std::move(m.mask)) {}
};

namespace op_detail {

inline TensorPtr apply_slice(const TensorPtr& x, int dim, const Slice& s) {
    int64_t n = x->shape[(size_t)dim];
    if (s.step == 0) throw std::invalid_argument("index: a slice step cannot be 0");
    auto resolve = [n](int64_t p, int64_t lo, int64_t hi) { return std::max(lo, std::min(hi, p < 0 ? p + n : p)); };
    if (s.step > 0) {
        int64_t start = s.start ? resolve(*s.start, 0, n) : 0, stop = s.stop ? resolve(*s.stop, 0, n) : n;
        if (start == 0 && stop == n && s.step == 1) return x;
        return slice_step(x, dim, start, std::max(start, stop), s.step);
    }
    // Downwards: the same elements upwards, then flipped
    int64_t start = s.start ? resolve(*s.start, -1, n - 1) : n - 1, stop = s.stop ? resolve(*s.stop, -1, n - 1) : -1;
    int64_t count = start > stop ? (start - stop + (-s.step) - 1) / (-s.step) : 0;
    if (count == 0) return slice_step(x, dim, 0, 0, 1);
    return flip(slice_step(x, dim, start + (count - 1) * s.step, start + 1, -s.step), {dim});
}

// x indexed along dims by integer tensors that broadcast together (NumPy's advanced indexing). Their
// broadcast shape replaces the dims if those are neighbours, else it comes first.
inline TensorPtr advanced_index(const TensorPtr& x, const std::vector<int>& dims, const TensorList& indices) {
    std::vector<int64_t> shape;
    for (const auto& t : indices) shape = broadcast_shapes(shape, t->shape, "index");
    int64_t total = 1, count = 1;
    for (int d : dims) total *= x->shape[(size_t)d];
    for (int64_t d : shape) count *= d;
    // One flat vector of non-negative positions per dimension
    TensorList at;
    {
        NoGradGuard no_grad;
        for (size_t k = 0; k < dims.size(); ++k) {
            int64_t size = x->shape[(size_t)dims[k]];
            auto t = strided(indices[k]->device == x->device ? indices[k] : indices[k]->to(x->device))->detach();
            t = where(lt(t, 0.0f), add_scalar(t, (float)size), t);
            check_index(t, size, "index");
            at.push_back(reshape(contiguous(expand(t, shape)), {count}));
        }
    }
    std::vector<int> order(dims.begin(), dims.end());
    std::vector<int64_t> indexed, rest;
    for (int d : dims) indexed.push_back(x->shape[(size_t)d]);
    for (int d = 0; d < (int)x->ndim(); ++d) {
        if (std::find(dims.begin(), dims.end(), d) != dims.end()) continue;
        order.push_back(d);
        rest.push_back(x->shape[(size_t)d]);
    }
    auto with = [](std::vector<int64_t> a, const std::vector<int64_t>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    auto moved = permute(x, order);  // [indexed..., rest...]
    TensorPtr picked;                // [count, rest...]
    if (dims.size() == 1 || total <= (1 << 24)) {
        // Float32 counts exactly up to 2^24, so one combined position per element will do
        NoGradGuard no_grad;
        auto flat = at[0];
        for (size_t k = 1; k < dims.size(); ++k) flat = add(mul_scalar(flat, (float)indexed[k]), at[k]);
        at = {flat};
        indexed = {total};
    }
    picked = index_select(reshape(moved, with(indexed, rest)), 0, at[0]);
    for (size_t k = 1; k < at.size(); ++k) {
        // [count, size of this dimension, others..., rest...] -> one entry of this dimension per row
        auto one = picked->shape, keep = picked->shape;
        one[1] = 1;
        keep.erase(keep.begin() + 1);
        std::vector<int64_t> column(one.size(), 1);
        column[0] = count;
        picked = reshape(gather(picked, 1, contiguous(expand(reshape(at[k], column), one))), keep);
    }
    bool adjacent = true;
    for (size_t k = 1; k < dims.size(); ++k) adjacent = adjacent && dims[k] == dims[k - 1] + 1;
    size_t front = adjacent ? (size_t)dims[0] : 0;  // dimensions before the indexed ones in the result
    auto y = reshape(picked, with(shape, rest));
    if (front == 0) return y;
    // [index shape, rest] -> [rest before, index shape, rest after]
    std::vector<int> back;
    int ni = (int)shape.size();
    for (size_t d = 0; d < front; ++d) back.push_back(ni + (int)d);
    for (int d = 0; d < ni; ++d) back.push_back(d);
    for (size_t d = front; d < rest.size(); ++d) back.push_back(ni + (int)d);
    return permute(y, back);
}

// Whether the index only looks at part of the tensor: no index tensors, masks or negative steps, which copy
inline bool is_view_index(const std::vector<IndexItem>& items) {
    for (const auto& i : items) {
        if (i.kind == IndexItem::Kind::Tensor || i.kind == IndexItem::Kind::Mask) return false;
        if (i.kind == IndexItem::Kind::Slice && i.slice.step < 0) return false;
    }
    return true;
}

}  // namespace op_detail

// x[items], with NumPy's rules: integers drop a dimension, slices keep it, Ellipsis fills in, NewAxis adds
// one, index tensors (integers as floats) and masks pick elements. Integers, slices, Ellipsis and NewAxis give a
// view (negative steps copy); index tensors and masks copy.
inline TensorPtr index(const TensorPtr& x, const std::vector<IndexItem>& items_in) {
    DG_PROFILE_OP("index");
    if (!x) throw std::invalid_argument("index: null tensor");
    int64_t used = 0;
    bool ellipsis = false;
    for (const auto& item : items_in) {
        switch (item.kind) {
            case IndexItem::Kind::Ellipsis:
                if (ellipsis) throw std::invalid_argument("index: only one Ellipsis is allowed");
                ellipsis = true;
                break;
            case IndexItem::Kind::NewAxis: break;
            case IndexItem::Kind::Mask:
                if (!item.tensor) throw std::invalid_argument("index: null mask");
                used += item.tensor->ndim();
                break;
            case IndexItem::Kind::Tensor:
                if (!item.tensor) throw std::invalid_argument("index: null index tensor");
                ++used;
                break;
            default: ++used;
        }
    }
    if (used > x->ndim()) throw std::invalid_argument("index: " + std::to_string(used) + " indices for a tensor with " + std::to_string(x->ndim()) + " dimensions");
    auto items = items_in;
    if (!ellipsis) items.emplace_back(Ellipsis);

    auto y = x;
    int d = 0;  // the dimension of y the next item applies to
    std::vector<int> dims;
    TensorList indices;
    for (const auto& item : items) {
        switch (item.kind) {
            case IndexItem::Kind::Integer:
                y = select(y, d, item.integer);
                break;
            case IndexItem::Kind::Slice:
                y = op_detail::apply_slice(y, d, item.slice);
                ++d;
                break;
            case IndexItem::Kind::NewAxis:
                y = unsqueeze(y, d);
                ++d;
                break;
            case IndexItem::Kind::Ellipsis: d += (int)(x->ndim() - used); break;
            case IndexItem::Kind::Tensor:
                dims.push_back(d++);
                indices.push_back(item.tensor);
                break;
            case IndexItem::Kind::Mask: {
                const auto& m = item.tensor;
                for (int k = 0; k < (int)m->ndim(); ++k) {
                    if (m->shape[(size_t)k] != y->shape[(size_t)(d + k)]) throw std::invalid_argument("index: mask " + shape_string(m->shape) + " does not match the dimensions it covers");
                }
                auto where_set = nonzero(m);  // [count, ndim]
                for (int k = 0; k < (int)m->ndim(); ++k) {
                    dims.push_back(d++);
                    indices.push_back(contiguous(select(where_set, 1, k)));
                }
                break;
            }
        }
    }
    return dims.empty() ? y : op_detail::advanced_index(y, dims, indices);
}

// A copy of x with value (broadcast to x[items]'s shape) written at items, or added with accumulate (every
// one of several additions to the same element counts). Both x and value get gradients. An index that is not a
// view (index tensors, masks, negative steps) needs x to have at most 2^24 elements.
inline TensorPtr index_put(const TensorPtr& x_in, const std::vector<IndexItem>& items, const TensorPtr& value_in, bool accumulate = false) {
    DG_PROFILE_OP("index_put");
    auto x = op_detail::dense(x_in), value = op_detail::strided(value_in);
    op_detail::check_same_device(x, value, "index_put");
    if (op_detail::is_view_index(items)) {
        if (accumulate) return add(x, index_put(op_detail::filled(x->shape, x->device, 0.0f), items, value));
        auto out = op_detail::empty(x->shape, x->device);
        {
            NoGradGuard no_grad;
            ops::make_contiguous(*out, *x);
            auto to = index(out, items);  // out's own memory
            auto from = expand(value->detach(), to->shape);
            if (to->size() > 0) {
                auto grid = kern::make_grid(to->shape, {from->strides, to->strides});
                DG_KERNEL(x->device, copy(from->data_ptr<float>(), to->data_ptr<float>(), grid));
            }
        }
        auto value_shape = value->shape;
        return make_op("index_put", out, {x, value}, {}, [items, value_shape](const OpContext& c, const TensorPtr& g) -> TensorList {
            return {c.needs(0) ? index_put(g, items, op_detail::filled({}, g->device, 0.0f)) : nullptr,
                    c.needs(1) ? sum_to(index(g, items), value_shape) : nullptr};
        });
    }
    if (x->size() > (1 << 24)) throw std::invalid_argument("index_put: tensors past 2^24 elements only take integers and slices");
    TensorPtr at;
    {
        // Where each selected element lives: the same index applied to the positions themselves
        NoGradGuard no_grad;
        at = contiguous(index(reshape(arange(0, (double)x->size(), 1.0, x->device), x->shape), items));
    }
    auto flat_at = reshape(at, {at->size()});
    auto flat_x = reshape(x, {x->size()}), flat_v = reshape(contiguous(expand(value, at->shape)), {at->size()});
    auto out = accumulate ? index_add(flat_x, 0, flat_at, flat_v) : scatter(flat_x, 0, flat_at, flat_v);
    return reshape(out, x->shape);
}
inline TensorPtr index_put(const TensorPtr& x, const std::vector<IndexItem>& items, float value, bool accumulate = false) {
    return index_put(x, items, op_detail::filled({}, x->device, value), accumulate);
}

inline std::shared_ptr<Tensor> Tensor::index(const std::vector<IndexItem>& items) { return ::index(shared_from_this(), items); }
inline std::shared_ptr<Tensor> Tensor::index_put(const std::vector<IndexItem>& items, const std::shared_ptr<Tensor>& value, bool accumulate) {
    return ::index_put(shared_from_this(), items, value, accumulate);
}

// ---------------------------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------------------------

// The values, nested by dimension, with the shape; long dimensions show their first and last three entries
inline std::string tensor_string(const TensorPtr& t, int precision = 4) {
    if (!t) return "tensor(null)";
    auto v = t->to_vector();
    auto strides = make_strides(t->shape);
    bool summarize = t->size() > 1000, integers = true;
    for (float f : v) integers = integers && std::isfinite(f) && f == std::floor(f) && std::fabs(f) < 1e7f;
    std::ostringstream os;
    os << std::setprecision(precision);
    auto print_value = [&](float f) {
        if (integers) os << (long long)f;
        else os << f;
    };
    std::function<void(size_t, int64_t)> print = [&](size_t dim, int64_t offset) {
        if (dim == t->shape.size()) {
            print_value(v[(size_t)offset]);
            return;
        }
        int64_t n = t->shape[dim];
        bool last = dim + 1 == t->shape.size();
        os << "[";
        for (int64_t i = 0; i < n; ++i) {
            if (summarize && n > 6 && i == 3) {
                os << (last ? ", ..." : ",\n" + std::string(dim + 1, ' ') + "...");
                i = n - 3;
            }
            if (i > 0) os << (last ? ", " : ",\n" + std::string(dim + 1, ' '));
            print(dim + 1, offset + i * strides[dim]);
        }
        os << "]";
    };
    print(0, 0);
    os << ", shape=" << shape_string(t->shape);
    if (t->device != Device::CPU) os << (t->device == Device::CUDA ? ", cuda" : t->device == Device::MPS ? ", mps" : ", tpu");
    if (t->dtype != DType::Float32) os << (t->dtype == DType::Float16 ? ", fp16" : t->dtype == DType::BFloat16 ? ", bf16" : ", int8");
    return os.str();
}
inline std::ostream& operator<<(std::ostream& os, const TensorPtr& t) { return os << tensor_string(t); }
