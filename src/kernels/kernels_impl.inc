// Kernel bodies, included by kernels_host.cpp and kernels_cuda.cu inside namespace kern::host or
// kern::gpu, after each defines
//   launch(n, functor, cost)   run functor(i) for i in [0, n); cost ~ work per i relative to an add
// The host backend replaces a few (binary, where, copy, reduce, softmax, index_select, index_add,
// scatter_add_scan) with loops that suit caches and SIMD better.

#define DG_SWITCH(list, Enum, op, CASE)  \
    switch (op) { list(CASE) default: throw std::invalid_argument("unknown kernel op"); }

// ---------------------------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------------------------

template <Unary OP>
struct UnaryFn {
    const float* x;
    float* y;
    float p, q;
    DG_HD void operator()(int64_t i) const { y[i] = unary_eval<OP>(x[i], p, q); }
};

// Ops that are a few instructions, as opposed to a transcendental function
inline bool cheap_unary(Unary op) {
    switch (op) {
        case Unary::Abs: case Unary::Neg: case Unary::Sign: case Unary::Sqrt: case Unary::Square: case Unary::Reciprocal:
        case Unary::Floor: case Unary::Ceil: case Unary::Round: case Unary::Trunc: case Unary::IsNan: case Unary::IsInf:
        case Unary::IsFinite: case Unary::LogicalNot: case Unary::Clamp: case Unary::Relu: case Unary::LeakyRelu:
        case Unary::Hardswish: case Unary::Hardsigmoid: case Unary::Softsign: case Unary::Softshrink: case Unary::Hardshrink:
        case Unary::Threshold: case Unary::EqScalar: case Unary::NeScalar: case Unary::LtScalar: case Unary::LeScalar:
        case Unary::GtScalar: case Unary::GeScalar: case Unary::InRange:
            return true;
        default: return false;
    }
}

void unary(Unary op, const float* x, float* y, int64_t n, float p, float q) {
    const int64_t cost = cheap_unary(op) ? 2 : 16;
#define DG_CASE(name)                                                   \
    case Unary::name:                                                   \
        launch(n, UnaryFn<Unary::name>{x, y, p, q}, cost);              \
        return;
    DG_SWITCH(DG_UNARY_LIST, Unary, op, DG_CASE)
#undef DG_CASE
}

// Offsets of element i for the first OPS operands of a grid. 32-bit division when it suffices: 64-bit
// division is several times slower on GPUs.
template <int OPS>
DG_HD void grid_offsets(const Grid& g, int64_t i, int64_t* off) {
    for (int k = 0; k < OPS; ++k) off[k] = 0;
    if (g.n <= 0x7fffffffLL) {
        uint32_t r = (uint32_t)i;
        for (int d = g.ndim - 1; d > 0; --d) {
            uint32_t s = (uint32_t)g.shape[d];
            uint32_t quot = r / s;
            int64_t c = (int64_t)(r - quot * s);
            r = quot;
            for (int k = 0; k < OPS; ++k) off[k] += c * g.stride[k][d];
        }
        for (int k = 0; k < OPS; ++k) off[k] += (int64_t)r * g.stride[k][0];
    } else {
        for (int d = g.ndim - 1; d > 0; --d) {
            int64_t quot = i / g.shape[d];
            int64_t c = i - quot * g.shape[d];
            i = quot;
            for (int k = 0; k < OPS; ++k) off[k] += c * g.stride[k][d];
        }
        for (int k = 0; k < OPS; ++k) off[k] += i * g.stride[k][0];
    }
}

#ifndef DG_KERNELS_HOST

template <Binary OP>
struct BinaryFn {
    const float* a;
    const float* b;
    float* out;
    Grid g;
    DG_HD void operator()(int64_t i) const {
        int64_t off[3];
        grid_offsets<3>(g, i, off);
        out[off[2]] = binary_eval<OP>(a[off[0]], b[off[1]]);
    }
};

void binary(Binary op, const float* a, const float* b, float* out, const Grid& g) {
#define DG_CASE(name)                                                   \
    case Binary::name:                                                  \
        launch(g.n, BinaryFn<Binary::name>{a, b, out, g}, 8);           \
        return;
    DG_SWITCH(DG_BINARY_LIST, Binary, op, DG_CASE)
#undef DG_CASE
}

struct WhereFn {
    const float* cond;
    const float* a;
    const float* b;
    float* out;
    Grid g;
    DG_HD void operator()(int64_t i) const {
        int64_t off[4];
        grid_offsets<4>(g, i, off);
        out[off[3]] = cond[off[0]] != 0.0f ? a[off[1]] : b[off[2]];
    }
};

void where(const float* cond, const float* a, const float* b, float* out, const Grid& g) {
    launch(g.n, WhereFn{cond, a, b, out, g}, 8);
}

struct CopyFn {
    const float* src;
    float* dst;
    Grid g;
    DG_HD void operator()(int64_t i) const {
        int64_t off[2];
        grid_offsets<2>(g, i, off);
        dst[off[1]] = src[off[0]];
    }
};

void copy(const float* src, float* dst, const Grid& g) { launch(g.n, CopyFn{src, dst, g}, 8); }

// ---------------------------------------------------------------------------------------------
// Reductions: K is cut into a number of chunks that depends only on the shapes; chunk results are
// combined in order
// ---------------------------------------------------------------------------------------------

template <Reduce OP>
struct ReducePartialFn {
    const float* x;
    float* partial;
    int64_t outer, K, inner, splits;
    DG_HD void operator()(int64_t w) const {
        int64_t n_out = outer * inner;
        int64_t t = w % n_out, s = w / n_out;
        int64_t o = t / inner, i = t % inner;
        int64_t k0 = K * s / splits, k1 = K * (s + 1) / splits;
        float acc = reduce_identity<OP>();
        for (int64_t k = k0; k < k1; ++k) acc = reduce_combine<OP>(acc, reduce_map<OP>(x[(o * K + k) * inner + i]));
        partial[w] = acc;
    }
};

template <Reduce OP>
struct ReduceCombineFn {
    const float* partial;
    float* out;
    int64_t n_out, splits;
    DG_HD void operator()(int64_t t) const {
        float acc = reduce_identity<OP>();
        for (int64_t s = 0; s < splits; ++s) acc = reduce_combine<OP>(acc, partial[s * n_out + t]);
        out[t] = acc;
    }
};

inline int64_t reduce_splits(int64_t n_out, int64_t K) {
    int64_t wanted = (131072 + n_out - 1) / n_out;
    int64_t most = K / 32;
    int64_t s = wanted < most ? wanted : most;
    return s < 1 ? 1 : (s > 4096 ? 4096 : s);
}

template <Reduce OP>
void reduce_impl(const float* x, float* out, int64_t outer, int64_t K, int64_t inner) {
    int64_t n_out = outer * inner;
    int64_t splits = reduce_splits(n_out, K);
    if (splits == 1) {
        launch(n_out, ReducePartialFn<OP>{x, out, outer, K, inner, 1}, K);
        return;
    }
    Scratch partial((size_t)(n_out * splits) * sizeof(float));
    launch(n_out * splits, ReducePartialFn<OP>{x, partial.as<float>(), outer, K, inner, splits}, K / splits);
    launch(n_out, ReduceCombineFn<OP>{partial.as<float>(), out, n_out, splits}, splits);
}

void reduce(Reduce op, const float* x, float* out, int64_t outer, int64_t K, int64_t inner) {
    if (outer * inner == 0) return;
    switch (op) {
        case Reduce::Sum: reduce_impl<Reduce::Sum>(x, out, outer, K, inner); return;
        case Reduce::Prod: reduce_impl<Reduce::Prod>(x, out, outer, K, inner); return;
        case Reduce::Max: reduce_impl<Reduce::Max>(x, out, outer, K, inner); return;
        case Reduce::Min: reduce_impl<Reduce::Min>(x, out, outer, K, inner); return;
        case Reduce::SumSquares: reduce_impl<Reduce::SumSquares>(x, out, outer, K, inner); return;
        case Reduce::SumAbs: reduce_impl<Reduce::SumAbs>(x, out, outer, K, inner); return;
        case Reduce::CountNonzero: reduce_impl<Reduce::CountNonzero>(x, out, outer, K, inner); return;
    }
}

#endif  // !DG_KERNELS_HOST

struct ArgReduceFn {
    const float* x;
    float* out;
    int64_t K, inner;
    bool largest;
    DG_HD void operator()(int64_t t) const {
        int64_t o = t / inner, i = t % inner;
        const float* base = x + o * K * inner + i;
        float best = base[0];
        int64_t at = 0;
        for (int64_t k = 1; k < K && !m_isnan(best); ++k) {
            float v = base[k * inner];
            if (m_isnan(v) || (largest ? v > best : v < best)) {
                best = v;
                at = k;
            }
        }
        out[t] = (float)at;
    }
};

void arg_reduce(bool largest, const float* x, float* out, int64_t outer, int64_t K, int64_t inner) {
    launch(outer * inner, ArgReduceFn{x, out, K, inner, largest}, 2 * K);
}

struct CumulateFn {
    Cumulate op;
    const float* x;
    float* out;
    float* index;
    int64_t K, inner;
    DG_HD void operator()(int64_t t) const {
        int64_t o = t / inner, i = t % inner;
        int64_t base = o * K * inner + i;
        if (K == 0) return;
        float acc = x[base];
        int64_t at = 0;
        out[base] = acc;
        if (index) index[base] = 0.0f;
        for (int64_t k = 1; k < K; ++k) {
            int64_t e = base + k * inner;
            float v = x[e];
            switch (op) {
                case Cumulate::Sum: acc += v; break;
                case Cumulate::Prod: acc *= v; break;
                case Cumulate::Max:
                    if (m_isnan(v) || (!m_isnan(acc) && v >= acc)) { acc = v; at = k; }
                    break;
                case Cumulate::Min:
                    if (m_isnan(v) || (!m_isnan(acc) && v <= acc)) { acc = v; at = k; }
                    break;
                case Cumulate::LogSumExp: {
                    float hi = acc > v ? acc : v, lo = acc > v ? v : acc;
                    acc = m_isinf(hi) ? hi : hi + log1pf(expf(lo - hi));
                    break;
                }
            }
            out[e] = acc;
            if (index) index[e] = (float)at;
        }
    }
};

void cumulate(Cumulate op, const float* x, float* out, float* index, int64_t outer, int64_t K, int64_t inner) {
    launch(outer * inner, CumulateFn{op, x, out, index, K, inner}, 2 * K);
}

// ---------------------------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------------------------

#ifndef DG_KERNELS_HOST

struct IndexSelectFn {
    const float* x;
    const float* index;
    float* out;
    int64_t K, inner, M;
    DG_HD void operator()(int64_t t) const {
        int64_t i = t % inner, rest = t / inner;
        int64_t m = rest % M, o = rest / M;
        out[t] = x[(o * K + (int64_t)index[m]) * inner + i];
    }
};

void index_select(const float* x, const float* index, float* out, int64_t outer, int64_t K, int64_t inner, int64_t M) {
    launch(outer * M * inner, IndexSelectFn{x, index, out, K, inner, M}, 8);
}

struct IndexAddFn {
    float* out;
    const float* src;
    const int* positions;
    const int* starts;
    int64_t K, inner, M;
    DG_HD void operator()(int64_t t) const {
        int64_t i = t % inner, rest = t / inner;
        int64_t k = rest % K, o = rest / K;
        float acc = out[t];
        for (int p = starts[k]; p < starts[k + 1]; ++p) acc += src[(o * M + positions[p]) * inner + i];
        out[t] = acc;
    }
};

void index_add(float* out, const float* src, const std::vector<int>& positions, const std::vector<int>& starts,
               int64_t outer, int64_t K, int64_t inner, int64_t M) {
    Scratch pos((positions.empty() ? 1 : positions.size()) * sizeof(int)), st(starts.size() * sizeof(int));
    pos.upload(positions.data(), positions.size() * sizeof(int));
    st.upload(starts.data(), starts.size() * sizeof(int));
    launch(outer * K * inner, IndexAddFn{out, src, pos.as<int>(), st.as<int>(), K, inner, M}, 8);
}

#endif

DG_HD int64_t gather_offset(const GatherPlan& g, int64_t i, int64_t along) {
    int64_t off = along * g.x_stride[g.dim];
    for (int d = g.ndim - 1; d >= 0; --d) {
        int64_t c = i % g.shape[d];
        i /= g.shape[d];
        if (d != g.dim) off += c * g.x_stride[d];
    }
    return off;
}

struct GatherFn {
    const float* x;
    const float* index;
    float* out;
    GatherPlan g;
    DG_HD void operator()(int64_t i) const { out[i] = x[gather_offset(g, i, (int64_t)index[i])]; }
};

void gather(const float* x, const float* index, float* out, const GatherPlan& g) {
    launch(g.n, GatherFn{x, index, out, g}, 16);
}

#ifndef DG_KERNELS_HOST

struct ScatterAddFn {
    float* out;
    const float* index;
    const float* src;
    GatherPlan g;
    DG_HD void operator()(int64_t j) const {
        int64_t rest = j, base = 0, along = 0;
        for (int d = g.ndim - 1; d >= 0; --d) {
            int64_t c = rest % g.x_shape[d];
            rest /= g.x_shape[d];
            if (d == g.dim) along = c;
            else if (c >= g.shape[d]) return;  // outside the index tensor: nothing scatters here
            else base += c * g.index_stride[d];
        }
        float acc = out[j];
        for (int64_t m = 0; m < g.shape[g.dim]; ++m) {
            int64_t e = base + m * g.index_stride[g.dim];
            if ((int64_t)index[e] == along) acc += src[e];
        }
        out[j] = acc;
    }
};

void scatter_add_scan(float* out, const float* index, const float* src, const GatherPlan& g) {
    int64_t n = 1;
    for (int d = 0; d < g.ndim; ++d) n *= g.x_shape[d];
    launch(n, ScatterAddFn{out, index, src, g}, 16 * g.shape[g.dim]);
}

#endif

// ---------------------------------------------------------------------------------------------
// Sliding windows
// ---------------------------------------------------------------------------------------------

#ifndef DG_KERNELS_HOST

struct UnfoldFn {
    const float* x;
    float* col;
    Windows w;
    DG_HD void operator()(int64_t t) const {
        int64_t n_out = w.out_size(), n_k = w.kernel_size();
        int64_t o = t % n_out, rest = t / n_out;
        int64_t k = rest % n_k, plane = rest / n_k;
        int64_t flat = 0, scale = 1;
        uint32_t o32 = (uint32_t)o, k32 = (uint32_t)k;  // 32-bit division is much cheaper on GPUs
        for (int d = 2; d >= 0; --d) {
            uint32_t od = o32 % (uint32_t)w.out[d], kd = k32 % (uint32_t)w.kernel[d];
            o32 /= (uint32_t)w.out[d];
            k32 /= (uint32_t)w.kernel[d];
            int64_t pos = od * w.stride[d] - w.pad[d] + kd * w.dilation[d];
            if (pos < 0 || pos >= w.in[d]) {
                col[t] = 0.0f;
                return;
            }
            flat += pos * scale;
            scale *= w.in[d];
        }
        col[t] = x[plane * scale + flat];
    }
};

void unfold(const float* x, float* col, const Windows& w) {
    launch(w.planes * w.kernel_size() * w.out_size(), UnfoldFn{x, col, w}, 24);
}

#endif

// Output position reading input position `in` through kernel element kd along dimension d, or -1
DG_HD int64_t window_over(const Windows& w, int d, int64_t in, int64_t kd) {
    int64_t p = in + w.pad[d] - kd * w.dilation[d];
    if (p < 0 || p % w.stride[d] != 0) return -1;
    p /= w.stride[d];
    return p < w.out[d] ? p : -1;
}

#ifndef DG_KERNELS_HOST

struct FoldFn {
    const float* col;
    float* x;
    Windows w;
    DG_HD void operator()(int64_t t) const {
        int64_t in2 = t % w.in[2], r = t / w.in[2];
        int64_t in1 = r % w.in[1];
        r /= w.in[1];
        int64_t in0 = r % w.in[0], plane = r / w.in[0];
        int64_t n_out = w.out_size(), n_k = w.kernel_size();
        float acc = 0.0f;
        for (int64_t k0 = 0; k0 < w.kernel[0]; ++k0) {
            int64_t o0 = window_over(w, 0, in0, k0);
            if (o0 < 0) continue;
            for (int64_t k1 = 0; k1 < w.kernel[1]; ++k1) {
                int64_t o1 = window_over(w, 1, in1, k1);
                if (o1 < 0) continue;
                for (int64_t k2 = 0; k2 < w.kernel[2]; ++k2) {
                    int64_t o2 = window_over(w, 2, in2, k2);
                    if (o2 < 0) continue;
                    int64_t k = (k0 * w.kernel[1] + k1) * w.kernel[2] + k2;
                    int64_t o = (o0 * w.out[1] + o1) * w.out[2] + o2;
                    acc += col[(plane * n_k + k) * n_out + o];
                }
            }
        }
        x[t] = acc;
    }
};

void fold(const float* col, float* x, const Windows& w) {
    launch(w.planes * w.in_size(), FoldFn{col, x, w}, 8 * w.kernel_size());
}

#endif

struct MaxPoolFn {
    const float* x;
    float* out;
    float* argmax;
    Windows w;
    DG_HD void operator()(int64_t t) const {
        int64_t o2 = t % w.out[2], r = t / w.out[2];
        int64_t o1 = r % w.out[1];
        r /= w.out[1];
        int64_t o0 = r % w.out[0], plane = r / w.out[0];
        const float* xp = x + plane * w.in_size();
        float best = -INFINITY;
        int64_t at = -1;
        for (int64_t k0 = 0; k0 < w.kernel[0]; ++k0) {
            int64_t i0 = o0 * w.stride[0] - w.pad[0] + k0 * w.dilation[0];
            if (i0 < 0 || i0 >= w.in[0]) continue;
            for (int64_t k1 = 0; k1 < w.kernel[1]; ++k1) {
                int64_t i1 = o1 * w.stride[1] - w.pad[1] + k1 * w.dilation[1];
                if (i1 < 0 || i1 >= w.in[1]) continue;
                for (int64_t k2 = 0; k2 < w.kernel[2]; ++k2) {
                    int64_t i2 = o2 * w.stride[2] - w.pad[2] + k2 * w.dilation[2];
                    if (i2 < 0 || i2 >= w.in[2]) continue;
                    int64_t flat = (i0 * w.in[1] + i1) * w.in[2] + i2;
                    float v = xp[flat];
                    if (v > best || (at < 0 && v == best)) {
                        best = v;
                        at = flat;
                    }
                }
            }
        }
        out[t] = best;
        argmax[t] = (float)at;
    }
};

void max_pool(const float* x, float* out, float* argmax, const Windows& w) {
    launch(w.planes * w.out_size(), MaxPoolFn{x, out, argmax, w}, 4 * w.kernel_size());
}

struct MaxPoolBackwardFn {
    const float* g;
    const float* argmax;
    float* gx;
    Windows w;
    DG_HD void operator()(int64_t t) const {
        int64_t flat = t % w.in_size(), plane = t / w.in_size();
        int64_t in2 = flat % w.in[2], r = flat / w.in[2];
        int64_t in1 = r % w.in[1], in0 = r / w.in[1];
        float acc = 0.0f;
        for (int64_t k0 = 0; k0 < w.kernel[0]; ++k0) {
            int64_t o0 = window_over(w, 0, in0, k0);
            if (o0 < 0) continue;
            for (int64_t k1 = 0; k1 < w.kernel[1]; ++k1) {
                int64_t o1 = window_over(w, 1, in1, k1);
                if (o1 < 0) continue;
                for (int64_t k2 = 0; k2 < w.kernel[2]; ++k2) {
                    int64_t o2 = window_over(w, 2, in2, k2);
                    if (o2 < 0) continue;
                    int64_t o = plane * w.out_size() + (o0 * w.out[1] + o1) * w.out[2] + o2;
                    if ((int64_t)argmax[o] == flat) acc += g[o];
                }
            }
        }
        gx[t] = acc;
    }
};

void max_pool_backward(const float* g, const float* argmax, float* gx, const Windows& w) {
    launch(w.planes * w.in_size(), MaxPoolBackwardFn{g, argmax, gx, w}, 8 * w.kernel_size());
}

// What average pooling divides output (o0, o1, o2) by
DG_HD float avg_divisor(const Windows& w, int64_t o0, int64_t o1, int64_t o2) {
    if (w.divisor != 0.0f) return w.divisor;
    int64_t o[3] = {o0, o1, o2};
    float count = 1.0f;
    for (int d = 0; d < 3; ++d) {
        int64_t begin = o[d] * w.stride[d] - w.pad[d], end = begin + w.kernel[d];
        int64_t lo = w.count_include_pad ? -w.pad[d] : 0, hi = w.count_include_pad ? w.in[d] + w.pad[d] : w.in[d];
        if (begin < lo) begin = lo;
        if (end > hi) end = hi;
        count *= (float)(end - begin);
    }
    return count;
}

struct AvgPoolFn {
    const float* x;
    float* out;
    Windows w;
    DG_HD void operator()(int64_t t) const {
        int64_t o2 = t % w.out[2], r = t / w.out[2];
        int64_t o1 = r % w.out[1];
        r /= w.out[1];
        int64_t o0 = r % w.out[0], plane = r / w.out[0];
        const float* xp = x + plane * w.in_size();
        float acc = 0.0f;
        for (int64_t k0 = 0; k0 < w.kernel[0]; ++k0) {
            int64_t i0 = o0 * w.stride[0] - w.pad[0] + k0;
            if (i0 < 0 || i0 >= w.in[0]) continue;
            for (int64_t k1 = 0; k1 < w.kernel[1]; ++k1) {
                int64_t i1 = o1 * w.stride[1] - w.pad[1] + k1;
                if (i1 < 0 || i1 >= w.in[1]) continue;
                for (int64_t k2 = 0; k2 < w.kernel[2]; ++k2) {
                    int64_t i2 = o2 * w.stride[2] - w.pad[2] + k2;
                    if (i2 >= 0 && i2 < w.in[2]) acc += xp[(i0 * w.in[1] + i1) * w.in[2] + i2];
                }
            }
        }
        out[t] = acc / avg_divisor(w, o0, o1, o2);
    }
};

void avg_pool(const float* x, float* out, const Windows& w) {
    launch(w.planes * w.out_size(), AvgPoolFn{x, out, w}, 4 * w.kernel_size());
}

struct AvgPoolBackwardFn {
    const float* g;
    float* gx;
    Windows w;
    DG_HD void operator()(int64_t t) const {
        int64_t flat = t % w.in_size(), plane = t / w.in_size();
        int64_t in2 = flat % w.in[2], r = flat / w.in[2];
        int64_t in1 = r % w.in[1], in0 = r / w.in[1];
        float acc = 0.0f;
        for (int64_t k0 = 0; k0 < w.kernel[0]; ++k0) {
            int64_t o0 = window_over(w, 0, in0, k0);
            if (o0 < 0) continue;
            for (int64_t k1 = 0; k1 < w.kernel[1]; ++k1) {
                int64_t o1 = window_over(w, 1, in1, k1);
                if (o1 < 0) continue;
                for (int64_t k2 = 0; k2 < w.kernel[2]; ++k2) {
                    int64_t o2 = window_over(w, 2, in2, k2);
                    if (o2 < 0) continue;
                    acc += g[plane * w.out_size() + (o0 * w.out[1] + o1) * w.out[2] + o2] / avg_divisor(w, o0, o1, o2);
                }
            }
        }
        gx[t] = acc;
    }
};

void avg_pool_backward(const float* g, float* gx, const Windows& w) {
    launch(w.planes * w.in_size(), AvgPoolBackwardFn{g, gx, w}, 8 * w.kernel_size());
}

// ---------------------------------------------------------------------------------------------
// Random numbers: one functor call per Philox block of four
// ---------------------------------------------------------------------------------------------

struct UniformFn {
    float* out;
    int64_t n;
    uint64_t seed, stream;
    DG_HD void operator()(int64_t b) const {
        Philox s = Philox::block(seed, stream, (uint64_t)b);
        for (int lane = 0; lane < 4 && 4 * b + lane < n; ++lane) out[4 * b + lane] = s.uniform(lane);
    }
};

void random_uniform(float* out, int64_t n, uint64_t seed, uint64_t stream) {
    launch((n + 3) / 4, UniformFn{out, n, seed, stream}, 64);
}

struct NormalFn {
    float* out;
    int64_t n;
    uint64_t seed, stream;
    DG_HD void operator()(int64_t b) const {
        Philox s = Philox::block(seed, stream, (uint64_t)b);
        float z[4];
        s.normals(0, z[0], z[1]);
        s.normals(1, z[2], z[3]);
        for (int lane = 0; lane < 4 && 4 * b + lane < n; ++lane) out[4 * b + lane] = z[lane];
    }
};

void random_normal(float* out, int64_t n, uint64_t seed, uint64_t stream) {
    launch((n + 3) / 4, NormalFn{out, n, seed, stream}, 256);
}

struct DropoutFn {
    const float* x;
    float* y;
    int64_t n;
    float keep, scale;
    uint64_t seed, stream;
    DG_HD void operator()(int64_t b) const {
        Philox s = Philox::block(seed, stream, (uint64_t)b);
        for (int lane = 0; lane < 4 && 4 * b + lane < n; ++lane) {
            int64_t i = 4 * b + lane;
            y[i] = s.uniform(lane) < keep ? x[i] * scale : 0.0f;
        }
    }
};

void dropout(const float* x, float* y, int64_t n, float keep, uint64_t seed, uint64_t stream) {
    launch((n + 3) / 4, DropoutFn{x, y, n, keep, keep > 0.0f ? 1.0f / keep : 0.0f, seed, stream}, 64);
}

// ---------------------------------------------------------------------------------------------
// Softmax over a middle dimension
// ---------------------------------------------------------------------------------------------

#ifndef DG_KERNELS_HOST

struct SoftmaxFn {
    const float* x;
    float* out;
    int64_t K, inner;
    bool log;
    DG_HD void operator()(int64_t t) const {
        int64_t o = t / inner, i = t % inner;
        const float* xp = x + o * K * inner + i;
        float* op = out + o * K * inner + i;
        float hi = -INFINITY;
        for (int64_t k = 0; k < K; ++k) hi = xp[k * inner] > hi ? xp[k * inner] : hi;
        float total = 0.0f;
        for (int64_t k = 0; k < K; ++k) total += m_exp(xp[k * inner] - hi);
        if (log) {
            float lse = hi + m_log(total);
            for (int64_t k = 0; k < K; ++k) op[k * inner] = xp[k * inner] - lse;
        } else {
            for (int64_t k = 0; k < K; ++k) op[k * inner] = m_exp(xp[k * inner] - hi) / total;
        }
    }
};

void softmax(const float* x, float* out, int64_t outer, int64_t K, int64_t inner, bool log) {
    launch(outer * inner, SoftmaxFn{x, out, K, inner, log}, 16 * K);
}

#endif

// ---------------------------------------------------------------------------------------------
// Optimizers (PyTorch's update rules)
// ---------------------------------------------------------------------------------------------

struct SgdFn {
    float* p;
    const float* g;
    float* buf;
    float lr, momentum, dampening, wd;
    bool nesterov, first;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        if (buf) {
            float b = first ? d : momentum * buf[i] + (1.0f - dampening) * d;
            buf[i] = b;
            d = nesterov ? d + momentum * b : b;
        }
        p[i] -= lr * d;
    }
};
void sgd_step(float* p, const float* g, float* buf, int64_t n, float lr, float momentum, float dampening,
              float weight_decay, bool nesterov, bool first) {
    launch(n, SgdFn{p, g, buf, lr, momentum, dampening, weight_decay, nesterov, first}, 4);
}

struct AdamFn {
    float* p;
    const float* g;
    float* m;
    float* v;
    float* vmax;
    float lr, beta1, beta2, eps, wd, bias1, bias2;
    bool decoupled;
    DG_HD void operator()(int64_t i) const {
        float w = p[i], d = g[i];
        if (decoupled) w -= lr * wd * w;
        else d += wd * w;
        float mi = beta1 * m[i] + (1.0f - beta1) * d;
        float vi = beta2 * v[i] + (1.0f - beta2) * d * d;
        m[i] = mi;
        v[i] = vi;
        if (vmax) {
            float top = vmax[i] > vi ? vmax[i] : vi;
            vmax[i] = top;
            vi = top;
        }
        p[i] = w - lr * (mi / bias1) / (sqrtf(vi / bias2) + eps);
    }
};
void adam_step(float* p, const float* g, float* m, float* v, float* vmax, int64_t n, float lr, float beta1, float beta2,
               float eps, float weight_decay, bool decoupled, float bias1, float bias2) {
    launch(n, AdamFn{p, g, m, v, vmax, lr, beta1, beta2, eps, weight_decay, bias1, bias2, decoupled}, 8);
}

struct NAdamFn {
    float* p;
    const float* g;
    float* m;
    float* v;
    float lr, beta1, beta2, eps, wd, mu_grad, mu_avg, bias2;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * d;
        float vi = beta2 * v[i] + (1.0f - beta2) * d * d;
        m[i] = mi;
        v[i] = vi;
        float denom = sqrtf(vi / bias2) + eps;
        p[i] -= lr * (mu_grad * d + mu_avg * mi) / denom;
    }
};
void nadam_step(float* p, const float* g, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps,
                float weight_decay, float mu_grad, float mu_avg, float bias2) {
    launch(n, NAdamFn{p, g, m, v, lr, beta1, beta2, eps, weight_decay, mu_grad, mu_avg, bias2}, 8);
}

struct RAdamFn {
    float* p;
    const float* g;
    float* m;
    float* v;
    float lr, beta1, beta2, eps, wd, bias1, bias2, rect;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * d;
        float vi = beta2 * v[i] + (1.0f - beta2) * d * d;
        m[i] = mi;
        v[i] = vi;
        float mh = mi / bias1;
        p[i] -= rect > 0.0f ? lr * mh * rect * sqrtf(bias2) / (sqrtf(vi) + eps) : lr * mh;
    }
};
void radam_step(float* p, const float* g, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias1, float bias2, float rect) {
    launch(n, RAdamFn{p, g, m, v, lr, beta1, beta2, eps, weight_decay, bias1, bias2, rect}, 8);
}

struct AdamaxFn {
    float* p;
    const float* g;
    float* m;
    float* u;
    float lr, beta1, beta2, eps, wd, bias1;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * d;
        float a = beta2 * u[i], b = fabsf(d) + eps;
        float ui = a > b ? a : b;
        m[i] = mi;
        u[i] = ui;
        p[i] -= lr / bias1 * mi / ui;
    }
};
void adamax_step(float* p, const float* g, float* m, float* u, int64_t n, float lr, float beta1, float beta2, float eps,
                 float weight_decay, float bias1) {
    launch(n, AdamaxFn{p, g, m, u, lr, beta1, beta2, eps, weight_decay, bias1}, 8);
}

struct AdagradFn {
    float* p;
    const float* g;
    float* sum;
    float lr, eps, wd;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        float s = sum[i] + d * d;
        sum[i] = s;
        p[i] -= lr * d / (sqrtf(s) + eps);
    }
};
void adagrad_step(float* p, const float* g, float* sum, int64_t n, float lr, float eps, float weight_decay) {
    launch(n, AdagradFn{p, g, sum, lr, eps, weight_decay}, 8);
}

struct AdadeltaFn {
    float* p;
    const float* g;
    float* sq;
    float* acc;
    float lr, rho, eps, wd;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        float s = rho * sq[i] + (1.0f - rho) * d * d;
        float step = sqrtf(acc[i] + eps) / sqrtf(s + eps) * d;
        sq[i] = s;
        acc[i] = rho * acc[i] + (1.0f - rho) * step * step;
        p[i] -= lr * step;
    }
};
void adadelta_step(float* p, const float* g, float* sq, float* acc, int64_t n, float lr, float rho, float eps,
                   float weight_decay) {
    launch(n, AdadeltaFn{p, g, sq, acc, lr, rho, eps, weight_decay}, 8);
}

struct RmspropFn {
    float* p;
    const float* g;
    float* sq;
    float* avg;
    float* buf;
    float lr, alpha, eps, wd, momentum;
    DG_HD void operator()(int64_t i) const {
        float d = g[i] + wd * p[i];
        float s = alpha * sq[i] + (1.0f - alpha) * d * d;
        sq[i] = s;
        if (avg) {
            float a = alpha * avg[i] + (1.0f - alpha) * d;
            avg[i] = a;
            s -= a * a;
        }
        float step = d / (sqrtf(s) + eps);
        if (buf) {
            step += momentum * buf[i];
            buf[i] = step;
        }
        p[i] -= lr * step;
    }
};
void rmsprop_step(float* p, const float* g, float* sq, float* avg, float* buf, int64_t n, float lr, float alpha, float eps,
                  float weight_decay, float momentum) {
    launch(n, RmspropFn{p, g, sq, avg, buf, lr, alpha, eps, weight_decay, momentum}, 8);
}

struct RpropFn {
    float* p;
    const float* g;
    float* prev;
    float* step;
    float eta_minus, eta_plus, step_min, step_max;
    DG_HD void operator()(int64_t i) const {
        float d = g[i], s = step[i];
        float agree = d * prev[i];
        if (agree > 0.0f) s *= eta_plus;
        else if (agree < 0.0f) s *= eta_minus;
        s = s < step_min ? step_min : (s > step_max ? step_max : s);
        step[i] = s;
        if (agree < 0.0f) d = 0.0f;  // the sign flipped: stay put this step
        prev[i] = d;
        p[i] -= d > 0.0f ? s : (d < 0.0f ? -s : 0.0f);
    }
};
void rprop_step(float* p, const float* g, float* prev, float* step, int64_t n, float eta_minus, float eta_plus,
                float step_min, float step_max) {
    launch(n, RpropFn{p, g, prev, step, eta_minus, eta_plus, step_min, step_max}, 8);
}

struct LionFn {
    float* p;
    const float* g;
    float* m;
    float lr, beta1, beta2, wd;
    DG_HD void operator()(int64_t i) const {
        float d = beta1 * m[i] + (1.0f - beta1) * g[i];
        float s = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        m[i] = beta2 * m[i] + (1.0f - beta2) * g[i];
        p[i] -= lr * (s + wd * p[i]);
    }
};
void lion_step(float* p, const float* g, float* m, int64_t n, float lr, float beta1, float beta2, float weight_decay) {
    launch(n, LionFn{p, g, m, lr, beta1, beta2, weight_decay}, 4);
}

struct EmaFn {
    float* avg;
    const float* p;
    float decay;
    DG_HD void operator()(int64_t i) const { avg[i] = decay * avg[i] + (1.0f - decay) * p[i]; }
};
void ema_step(float* avg, const float* p, int64_t n, float decay) { launch(n, EmaFn{avg, p, decay}, 2); }
