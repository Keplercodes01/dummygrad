#pragma once
// Indexing, sorting, searching. Index tensors hold integers as floats, like every other id in
// dummygrad, on the same device as the data (exact up to 2^24). Included from functional.h.

#include "reduce_ops.h"

#include <atomic>
#include <utility>

// Indices are range-checked before any kernel reads through them; on a GPU that costs two small
// reductions and a sync per op. Turn it off only for code whose indices are known to be good.
inline std::atomic<bool> g_index_checks{true};
inline void set_index_checks(bool enabled) { g_index_checks = enabled; }

namespace op_detail {

inline void check_index(const TensorPtr& index, int64_t limit, const char* who) {
    if (limit > (1 << 24)) throw std::invalid_argument(std::string(who) + ": dimensions past 2^24 cannot be indexed by float32 indices");
    if (!g_index_checks || index->size() == 0) return;
    float lo, hi;
    if (index->device == Device::CUDA) {
        NoGradGuard no_grad;
        lo = amin(index)->item();
        hi = amax(index)->item();
    } else {
        auto h = make_contiguous(index);
        const float* p = h->data_ptr<float>();
        lo = hi = p[0];
        for (int64_t i = 0; i < h->size(); ++i) {
            if (!(p[i] >= lo)) lo = p[i];
            if (!(p[i] <= hi)) hi = p[i];
        }
    }
    if (!(lo >= 0.0f && hi < (float)limit)) {
        throw std::out_of_range(std::string(who) + ": index " + std::to_string(lo < 0.0f ? lo : hi) + " is out of range for size " + std::to_string(limit));
    }
}

// Dense, detached indices on the data's device
inline TensorPtr index_tensor(const TensorPtr& index, const TensorPtr& data, const char* who) {
    if (!index) throw std::invalid_argument(std::string(who) + ": null index");
    auto i = index->device == data->device ? index : index->to(data->device);
    return make_contiguous(strided(i)->detach());
}

inline std::vector<float> host_floats(const TensorPtr& t) { return t->to_vector(); }

inline kern::GatherPlan gather_plan(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& index_shape, int dim, const char* who) {
    if (x_shape.size() != index_shape.size()) throw std::invalid_argument(std::string(who) + ": the index must have as many dimensions as the tensor");
    if (x_shape.size() > (size_t)kern::kMaxDims) throw std::invalid_argument(std::string(who) + ": at most 8 dimensions");
    kern::GatherPlan g;
    g.ndim = (int)std::max<size_t>(x_shape.size(), 1);
    g.dim = dim;
    auto xs = make_strides(x_shape), is = make_strides(index_shape);
    for (size_t d = 0; d < x_shape.size(); ++d) {
        if ((int)d != dim && index_shape[d] > x_shape[d]) throw std::invalid_argument(std::string(who) + ": the index is larger than the tensor outside dim");
        g.shape[d] = index_shape[d];
        g.x_shape[d] = x_shape[d];
        g.x_stride[d] = xs[d];
        g.index_stride[d] = is[d];
        g.n *= index_shape[d];
    }
    g.limit = x_shape.empty() ? 1 : x_shape[(size_t)dim];
    return g;
}

}  // namespace op_detail

inline TensorPtr gather(const TensorPtr& x, int dim, const TensorPtr& index);

// x with src added at the positions gather(x, dim, index) reads. Deterministic. On CUDA, an index
// longer than 64 along dim is summed on the host.
inline TensorPtr scatter_add(const TensorPtr& x_in, int dim_in, const TensorPtr& index_in, const TensorPtr& src_in) {
    DG_PROFILE_OP("scatter_add");
    auto x = op_detail::dense(x_in);
    auto index = op_detail::index_tensor(index_in, x, "scatter_add");
    auto src = op_detail::dense(src_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "scatter_add");
    if (src->shape != index->shape) {
        // src may be larger than the index: its leading corner is used
        for (size_t d = 0; d < index->shape.size() && d < src->shape.size(); ++d) src = narrow(src, (int)d, 0, index->shape[d]);
        src = op_detail::dense(src);
        if (src->shape != index->shape) throw std::invalid_argument("scatter_add: src must cover the index");
    }
    auto plan = op_detail::gather_plan(x->shape, index->shape, dim, "scatter_add");
    op_detail::check_index(index, plan.limit, "scatter_add");
    auto out = op_detail::empty(x->shape, x->device);
    if (x->device == Device::CUDA && plan.n > 0 && index->shape[(size_t)dim] > 64) {
        auto h = make_contiguous(x)->to(Device::CPU)->detach();
        kern::host::scatter_add_scan(h->data_ptr<float>(), index->to(Device::CPU)->data_ptr<float>(), src->to(Device::CPU)->data_ptr<float>(), plan);
        out = h->to(x->device)->detach();
    } else {
        ops::make_contiguous(*out, *x);
        if (plan.n > 0) DG_KERNEL(x->device, scatter_add_scan(out->data_ptr<float>(), index->data_ptr<float>(), src->data_ptr<float>(), plan));
    }
    return make_op("scatter_add", out, {x, src}, {}, [dim, index](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {g, c.needs(1) ? gather(g, dim, index) : nullptr};
    });
}

// out[..., i, ...] = x[..., index[..., i, ...], ...] along dim
inline TensorPtr gather(const TensorPtr& x_in, int dim_in, const TensorPtr& index_in) {
    DG_PROFILE_OP("gather");
    auto x = op_detail::dense(x_in);
    auto index = op_detail::index_tensor(index_in, x, "gather");
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "gather");
    auto plan = op_detail::gather_plan(x->shape, index->shape, dim, "gather");
    op_detail::check_index(index, plan.limit, "gather");
    auto out = op_detail::empty(index->shape, x->device);
    if (plan.n > 0) DG_KERNEL(x->device, gather(x->data_ptr<float>(), index->data_ptr<float>(), out->data_ptr<float>(), plan));
    auto x_shape = x->shape;
    return make_op("gather", out, {x}, {}, [dim, index, x_shape](const OpContext&, const TensorPtr& g) -> TensorList {
        return {scatter_add(op_detail::filled(x_shape, g->device, 0.0f), dim, index, g)};
    });
}
inline TensorPtr take_along_dim(const TensorPtr& x, const TensorPtr& index, int dim) { return gather(x, dim, index); }

// x with src written at the positions of index along dim. Where an index repeats, the last one wins.
inline TensorPtr scatter(const TensorPtr& x_in, int dim_in, const TensorPtr& index_in, const TensorPtr& src_in) {
    DG_PROFILE_OP("scatter");
    auto x = op_detail::dense(x_in);
    auto index = op_detail::index_tensor(index_in, x, "scatter");
    auto src = op_detail::dense(src_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "scatter");
    if (src->shape != index->shape) throw std::invalid_argument("scatter: src and index must have the same shape");
    auto plan = op_detail::gather_plan(x->shape, index->shape, dim, "scatter");
    op_detail::check_index(index, plan.limit, "scatter");
    // Last write wins only if writes happen in order, so this runs on the host
    auto h = std::make_shared<Tensor>(x->shape, Device::CPU, DType::Float32, false);
    auto xh = x->device == Device::CPU ? x : x->to(Device::CPU);
    ops::make_contiguous(*h, *xh);
    auto ih = index->to(Device::CPU), sh = src->to(Device::CPU);
    const float* ip = ih->data_ptr<float>();
    const float* sp = sh->data_ptr<float>();
    float* op = h->data_ptr<float>();
    std::vector<int64_t> coord(x->shape.size(), 0);
    for (int64_t i = 0; i < plan.n; ++i) {
        int64_t off = (int64_t)ip[i] * plan.x_stride[dim];
        for (int d = 0; d < (int)x->ndim(); ++d) if (d != dim) off += coord[(size_t)d] * plan.x_stride[d];
        op[off] = sp[i];
        for (int d = (int)x->ndim() - 1; d >= 0; --d) {
            if (++coord[(size_t)d] < index->shape[(size_t)d]) break;
            coord[(size_t)d] = 0;
        }
    }
    auto out = x->device == Device::CPU ? h : h->to(x->device)->detach();
    return make_op("scatter", out, {x, src}, {}, [dim, index](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {c.needs(0) ? scatter(g, dim, index, op_detail::filled(index->shape, g->device, 0.0f)) : nullptr,
                c.needs(1) ? gather(g, dim, index) : nullptr};
    });
}
inline TensorPtr scatter(const TensorPtr& x, int dim, const TensorPtr& index, float value) {
    return scatter(x, dim, index, op_detail::filled(index->shape, x->device, value));
}

// Rows (along dim) listed by a 1-D index
inline TensorPtr index_add(const TensorPtr& x, int dim, const TensorPtr& index, const TensorPtr& src, float alpha = 1.0f);
inline TensorPtr index_select(const TensorPtr& x_in, int dim_in, const TensorPtr& index_in) {
    DG_PROFILE_OP("index_select");
    auto x = op_detail::dense(x_in);
    auto index = op_detail::index_tensor(index_in, x, "index_select");
    if (index->ndim() != 1) throw std::invalid_argument("index_select: the index must be 1-D");
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "index_select");
    auto s = op_detail::split_at(x->shape, dim);
    op_detail::check_index(index, s.K, "index_select");
    auto shape = x->shape;
    shape[(size_t)dim] = index->size();
    auto out = op_detail::empty(shape, x->device);
    if (out->size() > 0) DG_KERNEL(x->device, index_select(x->data_ptr<float>(), index->data_ptr<float>(), out->data_ptr<float>(), s.outer, s.K, s.inner, index->size()));
    auto x_shape = x->shape;
    return make_op("index_select", out, {x}, {}, [dim, index, x_shape](const OpContext&, const TensorPtr& g) -> TensorList {
        return {index_add(op_detail::filled(x_shape, g->device, 0.0f), dim, index, g)};
    });
}
// x with alpha * src's rows added at the positions of index along dim; repeated indices accumulate
inline TensorPtr index_add(const TensorPtr& x_in, int dim_in, const TensorPtr& index_in, const TensorPtr& src_in, float alpha) {
    DG_PROFILE_OP("index_add");
    auto x = op_detail::dense(x_in);
    auto index = op_detail::index_tensor(index_in, x, "index_add");
    auto src = op_detail::dense(alpha == 1.0f ? src_in : mul_scalar(src_in, alpha));
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "index_add");
    auto s = op_detail::split_at(x->shape, dim);
    int64_t M = index->size();
    auto expected = x->shape;
    if (!expected.empty()) expected[(size_t)dim] = M;
    if (index->ndim() != 1 || src->shape != expected) throw std::invalid_argument("index_add: src must match x except along dim, where it has one row per index");
    // Positions grouped by the row they add to (stable, so sums are in index order)
    auto ids = op_detail::host_floats(index);
    std::vector<int> starts((size_t)s.K + 1, 0), positions((size_t)M);
    for (float f : ids) {
        if (!(f >= 0.0f && f < (float)s.K)) throw std::out_of_range("index_add: index " + std::to_string(f) + " is out of range for size " + std::to_string(s.K));
        starts[(size_t)f + 1]++;
    }
    for (size_t k = 0; k < (size_t)s.K; ++k) starts[k + 1] += starts[k];
    auto next = starts;
    for (int64_t m = 0; m < M; ++m) positions[(size_t)next[(size_t)ids[(size_t)m]]++] = (int)m;
    auto out = op_detail::empty(x->shape, x->device);
    ops::make_contiguous(*out, *x);
    if (M > 0 && x->size() > 0) DG_KERNEL(x->device, index_add(out->data_ptr<float>(), src->data_ptr<float>(), positions, starts, s.outer, s.K, s.inner, M));
    return make_op("index_add", out, {x, src}, {}, [dim, index](const OpContext& c, const TensorPtr& g) -> TensorList {
        return {g, c.needs(1) ? index_select(g, dim, index) : nullptr};
    });
}
// x with value in the rows listed by index
inline TensorPtr index_fill(const TensorPtr& x, int dim_in, const TensorPtr& index, float value) {
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "index_fill");
    int64_t K = x->shape[(size_t)dim];
    std::vector<float> mask((size_t)K, 0.0f);
    for (float f : op_detail::host_floats(index)) {
        if (!(f >= 0.0f && f < (float)K)) throw std::out_of_range("index_fill: index out of range");
        mask[(size_t)f] = 1.0f;
    }
    std::vector<int64_t> shape((size_t)x->ndim(), 1);
    shape[(size_t)dim] = K;
    return masked_fill(x, op_detail::from_values(mask, shape, x->device), value);
}
// x with src's rows written at (distinct) indices
inline TensorPtr index_copy(const TensorPtr& x, int dim, const TensorPtr& index, const TensorPtr& src) {
    return index_add(index_fill(x, dim, index, 0.0f), dim, index, src);
}
// Elements of x, seen as one long vector, at the positions of index; the result has index's shape
inline TensorPtr take(const TensorPtr& x, const TensorPtr& index) {
    return reshape(index_select(reshape(x, {x->size()}), 0, reshape(index, {index->size()})), index->shape);
}
// Positions of nonzero elements as rows of coordinates: [count, ndim]
inline TensorPtr nonzero(const TensorPtr& x) {
    auto v = op_detail::host_floats(x);
    std::vector<float> coords;
    int64_t count = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == 0.0f) continue;
        ++count;
        for (int64_t c : unravel((int64_t)i, x->shape)) coords.push_back((float)c);
    }
    return op_detail::from_values(coords, {count, x->ndim()}, x->device);
}
// The elements where mask (x's shape) is nonzero, as a vector
inline TensorPtr masked_select(const TensorPtr& x, const TensorPtr& mask) {
    if (mask->shape != x->shape) throw std::invalid_argument("masked_select: mask and tensor must have the same shape");
    auto m = op_detail::host_floats(mask);
    std::vector<float> at;
    for (size_t i = 0; i < m.size(); ++i) if (m[i] != 0.0f) at.push_back((float)i);
    return index_select(reshape(x, {x->size()}), 0, op_detail::from_values(at, {(int64_t)at.size()}, x->device));
}

// ---------------------------------------------------------------------------------------------
// Sorting and searching. The order is found on the host; values flow through gather, so gradients do too.
// ---------------------------------------------------------------------------------------------

namespace op_detail {
// The first k positions of every line along dim in sorted order (stable; NaN counts as largest)
inline TensorPtr sorted_index(const TensorPtr& x, int dim, int64_t k, bool descending) {
    auto s = split_at(x->shape, dim);
    if (s.K > (1 << 24)) throw std::invalid_argument("sort: dimensions past 2^24 cannot be indexed by float32 indices");
    auto v = host_floats(x);
    auto shape = x->shape;
    if (!shape.empty()) shape[(size_t)dim] = k;
    auto out = empty(shape, Device::CPU);
    float* op = out->data_ptr<float>();
    auto before = [descending](float a, float b) {
        bool na = std::isnan(a), nb = std::isnan(b);
        if (na || nb) return descending ? (na && !nb) : (!na && nb);
        return descending ? a > b : a < b;
    };
    std::vector<int> order((size_t)s.K);
    for (int64_t o = 0; o < s.outer; ++o) {
        for (int64_t i = 0; i < s.inner; ++i) {
            const float* line = v.data() + o * s.K * s.inner + i;
            std::iota(order.begin(), order.end(), 0);
            auto less = [&](int a, int b) {
                float x0 = line[a * s.inner], x1 = line[b * s.inner];
                if (before(x0, x1)) return true;
                if (before(x1, x0)) return false;
                return a < b;
            };
            if (k < s.K) std::partial_sort(order.begin(), order.begin() + k, order.end(), less);
            else std::sort(order.begin(), order.end(), less);
            for (int64_t r = 0; r < k; ++r) op[(o * k + r) * s.inner + i] = (float)order[(size_t)r];
        }
    }
    return x->device == Device::CPU ? out : out->to(x->device)->detach();
}
}  // namespace op_detail

using ValuesAndIndices = std::pair<TensorPtr, TensorPtr>;

inline TensorPtr argsort(const TensorPtr& x, int dim = -1, bool descending = false) {
    DG_PROFILE_OP("argsort");
    int d = op_detail::wrap_dim(dim, x->ndim(), "argsort");
    return op_detail::sorted_index(x, d, x->shape[(size_t)d], descending);
}
inline ValuesAndIndices sort(const TensorPtr& x, int dim = -1, bool descending = false) {
    auto index = argsort(x, dim, descending);
    return {gather(x, dim, index), index};
}
inline ValuesAndIndices topk(const TensorPtr& x, int64_t k, int dim = -1, bool largest = true) {
    DG_PROFILE_OP("topk");
    int d = op_detail::wrap_dim(dim, x->ndim(), "topk");
    if (k < 0 || k > x->shape[(size_t)d]) throw std::invalid_argument("topk: k is larger than the dimension");
    auto index = op_detail::sorted_index(x, d, k, largest);
    return {gather(x, d, index), index};
}
// The k-th smallest (1-based) along dim
inline ValuesAndIndices kthvalue(const TensorPtr& x, int64_t k, int dim = -1, bool keepdim = false) {
    auto [values, index] = topk(x, k, dim, false);
    int d = op_detail::wrap_dim(dim, x->ndim(), "kthvalue");
    auto v = narrow(values, d, k - 1, 1), i = narrow(index, d, k - 1, 1);
    return keepdim ? ValuesAndIndices{v, i} : ValuesAndIndices{squeeze(v, d), squeeze(i, d)};
}
// Lower median, as in PyTorch
inline ValuesAndIndices median(const TensorPtr& x, int dim, bool keepdim = false) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "median");
    return kthvalue(x, (x->shape[(size_t)d] + 1) / 2, d, keepdim);
}
inline TensorPtr median(const TensorPtr& x) { return median(reshape(x, {x->size()}), 0).first; }
// Values at quantile q along dim, interpolating linearly between neighbours
inline TensorPtr quantile(const TensorPtr& x, float q, int dim = -1, bool keepdim = false) {
    if (!(q >= 0.0f && q <= 1.0f)) throw std::invalid_argument("quantile: q must be in [0, 1]");
    int d = op_detail::wrap_dim(dim, x->ndim(), "quantile");
    auto sorted = sort(x, d).first;
    double pos = (double)q * (double)(x->shape[(size_t)d] - 1);
    int64_t lo = (int64_t)std::floor(pos), hi = (int64_t)std::ceil(pos);
    auto v = lerp(narrow(sorted, d, lo, 1), narrow(sorted, d, hi, 1), (float)(pos - (double)lo));
    return keepdim ? v : squeeze(v, d);
}

// Largest / smallest along dim with its position. The gradient goes to that position.
inline ValuesAndIndices max(const TensorPtr& x, int dim, bool keepdim = false) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "max");
    auto index = argmax(x, d, true);
    auto v = gather(x, d, index);
    return keepdim ? ValuesAndIndices{v, index} : ValuesAndIndices{squeeze(v, d), squeeze(index, d)};
}
inline ValuesAndIndices min(const TensorPtr& x, int dim, bool keepdim = false) {
    int d = op_detail::wrap_dim(dim, x->ndim(), "min");
    auto index = argmin(x, d, true);
    auto v = gather(x, d, index);
    return keepdim ? ValuesAndIndices{v, index} : ValuesAndIndices{squeeze(v, d), squeeze(index, d)};
}
inline TensorPtr max(const TensorPtr& x) { return amax(x); }
inline TensorPtr min(const TensorPtr& x) { return amin(x); }

inline ValuesAndIndices cummax(const TensorPtr& x_in, int dim_in) {
    DG_PROFILE_OP("cummax");
    auto x = op_detail::dense(x_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "cummax");
    TensorPtr index;
    op_detail::cumulate_value(kern::Cumulate::Max, x, dim, &index);
    return {gather(x, dim, index), index};
}
inline ValuesAndIndices cummin(const TensorPtr& x_in, int dim_in) {
    DG_PROFILE_OP("cummin");
    auto x = op_detail::dense(x_in);
    int dim = op_detail::wrap_dim(dim_in, x->ndim(), "cummin");
    TensorPtr index;
    op_detail::cumulate_value(kern::Cumulate::Min, x, dim, &index);
    return {gather(x, dim, index), index};
}

struct UniqueResult {
    TensorPtr values;   // sorted distinct values
    TensorPtr inverse;  // for every element, its position in values
    TensorPtr counts;
};
inline UniqueResult unique(const TensorPtr& x) {
    auto v = op_detail::host_floats(x);
    auto values = v;
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    std::vector<float> inverse(v.size()), counts(values.size(), 0.0f);
    for (size_t i = 0; i < v.size(); ++i) {
        size_t at = (size_t)(std::lower_bound(values.begin(), values.end(), v[i]) - values.begin());
        inverse[i] = (float)at;
        counts[at] += 1.0f;
    }
    int64_t n = (int64_t)values.size();
    return {op_detail::from_values(values, {n}, x->device), op_detail::from_values(inverse, x->shape, x->device), op_detail::from_values(counts, {n}, x->device)};
}
// Where each value would go in a sorted 1-D sequence: before equal entries, or after them with right
inline TensorPtr searchsorted(const TensorPtr& sorted, const TensorPtr& values, bool right = false) {
    if (sorted->ndim() != 1) throw std::invalid_argument("searchsorted: the sorted sequence must be 1-D");
    auto s = op_detail::host_floats(sorted), v = op_detail::host_floats(values);
    std::vector<float> at(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        auto it = right ? std::upper_bound(s.begin(), s.end(), v[i]) : std::lower_bound(s.begin(), s.end(), v[i]);
        at[i] = (float)(it - s.begin());
    }
    return op_detail::from_values(at, values->shape, values->device);
}
inline TensorPtr bucketize(const TensorPtr& values, const TensorPtr& boundaries, bool right = false) { return searchsorted(boundaries, values, right); }
// How often each integer 0, 1, ... occurs (or the sum of its weights)
inline TensorPtr bincount(const TensorPtr& x, const TensorPtr& weights = nullptr, int64_t minlength = 0) {
    auto v = op_detail::host_floats(x);
    std::vector<float> w;
    if (weights) w = op_detail::host_floats(weights);
    if (weights && w.size() != v.size()) throw std::invalid_argument("bincount: one weight per element is needed");
    int64_t n = minlength;
    for (float f : v) {
        if (f < 0.0f) throw std::invalid_argument("bincount: values must be non-negative integers");
        n = std::max(n, (int64_t)f + 1);
    }
    std::vector<float> counts((size_t)n, 0.0f);
    for (size_t i = 0; i < v.size(); ++i) counts[(size_t)v[i]] += weights ? w[i] : 1.0f;
    return op_detail::from_values(counts, {n}, x->device);
}
// Counts of values in equal-width bins over [lo, hi]
inline TensorPtr histc(const TensorPtr& x, int64_t bins, float lo, float hi) {
    if (bins <= 0 || !(hi > lo)) throw std::invalid_argument("histc: needs positive bins and lo < hi");
    std::vector<float> counts((size_t)bins, 0.0f);
    for (float f : op_detail::host_floats(x)) {
        if (!(f >= lo && f <= hi)) continue;
        int64_t b = std::min(bins - 1, (int64_t)((f - lo) / (hi - lo) * (float)bins));
        counts[(size_t)b] += 1.0f;
    }
    return op_detail::from_values(counts, {bins}, x->device);
}

enum class ScatterReduce { Sum, Mean, Amax, Amin };

// x combined with src at the positions of index along dim; x's own values take part
inline TensorPtr scatter_reduce(const TensorPtr& x, int dim, const TensorPtr& index, const TensorPtr& src, ScatterReduce how) {
    if (how == ScatterReduce::Sum) return scatter_add(x, dim, index, src);
    if (how == ScatterReduce::Mean) {
        auto ones = op_detail::filled(src->shape, src->device, 1.0f);
        auto counts = scatter_add(op_detail::filled(x->shape, x->device, 1.0f), dim, index, ones);
        return div(scatter_add(x, dim, index, src), counts);
    }
    // The extreme per position, found on the host, then the winners' mean carries the gradient
    bool largest = how == ScatterReduce::Amax;
    TensorPtr best;
    {
        NoGradGuard no_grad;
        auto xd = op_detail::dense(x);
        auto id = op_detail::index_tensor(index, xd, "scatter_reduce");
        int d = op_detail::wrap_dim(dim, xd->ndim(), "scatter_reduce");
        if (id->shape != src->shape) throw std::invalid_argument("scatter_reduce: src and index must have the same shape");
        auto plan = op_detail::gather_plan(xd->shape, id->shape, d, "scatter_reduce");
        op_detail::check_index(id, plan.limit, "scatter_reduce");
        auto v = op_detail::host_floats(xd), iv = op_detail::host_floats(id), sv = op_detail::host_floats(src);
        for (int64_t i = 0; i < plan.n; ++i) {
            auto c = unravel(i, id->shape);
            int64_t off = 0;
            for (int k = 0; k < (int)c.size(); ++k) off += (k == d ? (int64_t)iv[(size_t)i] : c[(size_t)k]) * plan.x_stride[k];
            float& slot = v[(size_t)off];
            slot = largest ? std::max(slot, sv[(size_t)i]) : std::min(slot, sv[(size_t)i]);
        }
        best = op_detail::from_values(v, xd->shape, xd->device);
    }
    auto x_wins = eq(x, best), src_wins = eq(src, gather(best, dim, index));
    auto winners = scatter_add(x_wins, dim, index, src_wins);
    return div(scatter_add(mul(x, x_wins), dim, index, mul(src, src_wins)), winners);
}
