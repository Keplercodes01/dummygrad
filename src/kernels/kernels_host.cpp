// CPU backend of the kernels in kernels.h
#include "kernels.h"
#include "kernel_math.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace kern {
namespace host {

namespace {

int64_t grain_for(int64_t cost) { return std::max<int64_t>(1, 131072 / std::max<int64_t>(1, cost)); }

template <typename F>
void launch(int64_t n, const F& functor, int64_t cost) {
    parallel_for(0, n, grain_for(cost), [&](int64_t begin, int64_t end) {
        const F f = functor;  // a local copy the optimizer can keep in registers
        for (int64_t i = begin; i < end; ++i) f(i);
    });
}

// Calls body(offsets, run) for consecutive runs along the innermost dimension; offsets[k] is operand
// k's offset of the run's first element
template <int OPS, typename Body>
void for_each_run(const Grid& g, int64_t cost, const Body& body) {
    if (g.n == 0) return;
    parallel_for(0, g.n, grain_for(cost), [&](int64_t begin, int64_t end) {
        const int last = g.ndim - 1;
        const int64_t inner = g.shape[last];
        int64_t coord[kMaxDims];
        int64_t rest = begin;
        for (int d = last; d >= 0; --d) {
            coord[d] = rest % g.shape[d];
            rest /= g.shape[d];
        }
        for (int64_t i = begin; i < end;) {
            int64_t off[OPS];
            for (int k = 0; k < OPS; ++k) {
                off[k] = 0;
                for (int d = 0; d <= last; ++d) off[k] += coord[d] * g.stride[k][d];
            }
            int64_t run = std::min(inner - coord[last], end - i);
            body(off, run);
            i += run;
            coord[last] = 0;
            for (int d = last - 1; d >= 0; --d) {
                if (++coord[d] < g.shape[d]) break;
                coord[d] = 0;
            }
        }
    });
}

template <Binary OP>
void binary_impl(const float* a, const float* b, float* out, const Grid& g) {
    const int last = g.ndim - 1;
    const int64_t sa = g.stride[0][last], sb = g.stride[1][last], so = g.stride[2][last];
    for_each_run<3>(g, 4, [&](const int64_t* off, int64_t run) {
        const float* ap = a + off[0];
        const float* bp = b + off[1];
        float* op = out + off[2];
        if (so == 1 && sa == 1 && sb == 1) {
            for (int64_t j = 0; j < run; ++j) op[j] = binary_eval<OP>(ap[j], bp[j]);
        } else if (so == 1 && sa == 1 && sb == 0) {
            const float bv = *bp;
            for (int64_t j = 0; j < run; ++j) op[j] = binary_eval<OP>(ap[j], bv);
        } else if (so == 1 && sa == 0 && sb == 1) {
            const float av = *ap;
            for (int64_t j = 0; j < run; ++j) op[j] = binary_eval<OP>(av, bp[j]);
        } else {
            for (int64_t j = 0; j < run; ++j) op[j * so] = binary_eval<OP>(ap[j * sa], bp[j * sb]);
        }
    });
}

// One chunk of a line reduction, in 16 interleaved lanes so it vectorizes
template <Reduce OP>
float reduce_chunk(const float* x, int64_t n) {
    constexpr int kLanes = 16;
    float lane[kLanes];
    for (int l = 0; l < kLanes; ++l) lane[l] = reduce_identity<OP>();
    int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        for (int l = 0; l < kLanes; ++l) lane[l] = reduce_combine<OP>(lane[l], reduce_map<OP>(x[i + l]));
    }
    float acc = reduce_identity<OP>();
    for (int l = 0; l < kLanes; ++l) acc = reduce_combine<OP>(acc, lane[l]);
    for (; i < n; ++i) acc = reduce_combine<OP>(acc, reduce_map<OP>(x[i]));
    return acc;
}

constexpr int64_t kReduceChunk = 8192;

template <Reduce OP>
double combine_chunks(const std::vector<float>& partial) {
    if constexpr (OP == Reduce::Prod || OP == Reduce::Max || OP == Reduce::Min) {
        float acc = reduce_identity<OP>();
        for (float p : partial) acc = reduce_combine<OP>(acc, p);
        return acc;
    } else {
        double acc = 0.0;
        for (float p : partial) acc += p;
        return acc;
    }
}

// A contiguous line, cut into fixed chunks (so the result does not depend on the thread count)
template <Reduce OP>
float reduce_line(const float* x, int64_t n) {
    if (n <= kReduceChunk) return reduce_chunk<OP>(x, n);
    int64_t chunks = (n + kReduceChunk - 1) / kReduceChunk;
    std::vector<float> partial((size_t)chunks);
    parallel_for(0, chunks, 4, [&](int64_t b, int64_t e) {
        for (int64_t c = b; c < e; ++c)
            partial[(size_t)c] = reduce_chunk<OP>(x + c * kReduceChunk, std::min(kReduceChunk, n - c * kReduceChunk));
    });
    return (float)combine_chunks<OP>(partial);
}

template <Reduce OP>
void reduce_impl(const float* x, float* out, int64_t outer, int64_t K, int64_t inner) {
    if (inner == 1) {
        if (outer == 1) {
            out[0] = reduce_line<OP>(x, K);
            return;
        }
        parallel_for(0, outer, grain_for(K), [&](int64_t b, int64_t e) {
            for (int64_t o = b; o < e; ++o) out[o] = reduce_line<OP>(x + o * K, K);
        });
        return;
    }
    // Whole inner rows at a time; parallel over [outer] when there are several, else over blocks of inner
    auto block = [&](int64_t o, int64_t i0, int64_t i1) {
        float* acc = out + o * inner;
        for (int64_t i = i0; i < i1; ++i) acc[i] = reduce_identity<OP>();
        for (int64_t k = 0; k < K; ++k) {
            const float* row = x + (o * K + k) * inner;
            for (int64_t i = i0; i < i1; ++i) acc[i] = reduce_combine<OP>(acc[i], reduce_map<OP>(row[i]));
        }
    };
    if (outer >= get_num_threads()) {
        parallel_for(0, outer, grain_for(K * inner), [&](int64_t b, int64_t e) {
            for (int64_t o = b; o < e; ++o) block(o, 0, inner);
        });
    } else {
        for (int64_t o = 0; o < outer; ++o) {
            parallel_for(0, inner, std::max<int64_t>(16, grain_for(K)), [&](int64_t b, int64_t e) { block(o, b, e); });
        }
    }
}

}  // namespace

#define DG_KERNELS_HOST 1
#include "kernels_impl.inc"

void binary(Binary op, const float* a, const float* b, float* out, const Grid& g) {
#define DG_CASE(name)                                        \
    case Binary::name:                                       \
        binary_impl<Binary::name>(a, b, out, g);             \
        return;
    DG_SWITCH(DG_BINARY_LIST, Binary, op, DG_CASE)
#undef DG_CASE
}

void where(const float* cond, const float* a, const float* b, float* out, const Grid& g) {
    const int last = g.ndim - 1;
    const int64_t sc = g.stride[0][last], sa = g.stride[1][last], sb = g.stride[2][last], so = g.stride[3][last];
    for_each_run<4>(g, 4, [&](const int64_t* off, int64_t run) {
        const float* cp = cond + off[0];
        const float* ap = a + off[1];
        const float* bp = b + off[2];
        float* op = out + off[3];
        for (int64_t j = 0; j < run; ++j) op[j * so] = cp[j * sc] != 0.0f ? ap[j * sa] : bp[j * sb];
    });
}

void copy(const float* src, float* dst, const Grid& g) {
    const int last = g.ndim - 1;
    const int64_t ss = g.stride[0][last], sd = g.stride[1][last];
    for_each_run<2>(g, 2, [&](const int64_t* off, int64_t run) {
        const float* sp = src + off[0];
        float* dp = dst + off[1];
        if (ss == 1 && sd == 1) std::memcpy(dp, sp, (size_t)run * sizeof(float));
        else if (ss == 0 && sd == 1) std::fill(dp, dp + run, *sp);
        else for (int64_t j = 0; j < run; ++j) dp[j * sd] = sp[j * ss];
    });
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

void index_select(const float* x, const float* index, float* out, int64_t outer, int64_t K, int64_t inner, int64_t M) {
    parallel_for(0, outer * M, grain_for(inner), [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; ++t) {
            int64_t o = t / M, m = t % M;
            std::memcpy(out + t * inner, x + (o * K + (int64_t)index[m]) * inner, (size_t)inner * sizeof(float));
        }
    });
}

void index_add(float* out, const float* src, const std::vector<int>& positions, const std::vector<int>& starts,
               int64_t outer, int64_t K, int64_t inner, int64_t M) {
    parallel_for(0, outer * K, grain_for(inner * (1 + M / std::max<int64_t>(1, K))), [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; ++t) {
            int64_t o = t / K, k = t % K;
            float* dst = out + t * inner;
            for (int p = starts[(size_t)k]; p < starts[(size_t)k + 1]; ++p) {
                const float* s = src + (o * M + positions[(size_t)p]) * inner;
                for (int64_t i = 0; i < inner; ++i) dst[i] += s[i];
            }
        }
    });
}

// The direct loop: no scanning needed when one thread walks the index in order
void scatter_add_scan(float* out, const float* index, const float* src, const GatherPlan& g) {
    for (int64_t i = 0; i < g.n; ++i) out[gather_offset(g, i, (int64_t)index[i])] += src[i];
}

void softmax(const float* x, float* out, int64_t outer, int64_t K, int64_t inner, bool log) {
    parallel_for(0, outer, grain_for(16 * K * inner), [&](int64_t b, int64_t e) {
        std::vector<float> hi((size_t)inner), total((size_t)inner);
        for (int64_t o = b; o < e; ++o) {
            const float* xp = x + o * K * inner;
            float* op = out + o * K * inner;
            if (inner == 1) {
                float h = reduce_chunk<Reduce::Max>(xp, K);
                for (int64_t k = 0; k < K; ++k) op[k] = simd::exp(xp[k] - h);
                float t = reduce_chunk<Reduce::Sum>(op, K);
                if (log) {
                    float lse = h + simd::log(t);
                    for (int64_t k = 0; k < K; ++k) op[k] = xp[k] - lse;
                } else {
                    float inv = 1.0f / t;
                    for (int64_t k = 0; k < K; ++k) op[k] *= inv;
                }
                continue;
            }
            std::fill(hi.begin(), hi.end(), -INFINITY);
            std::fill(total.begin(), total.end(), 0.0f);
            for (int64_t k = 0; k < K; ++k)
                for (int64_t i = 0; i < inner; ++i) hi[i] = std::max(hi[i], xp[k * inner + i]);
            for (int64_t k = 0; k < K; ++k)
                for (int64_t i = 0; i < inner; ++i) {
                    float v = simd::exp(xp[k * inner + i] - hi[i]);
                    op[k * inner + i] = v;
                    total[i] += v;
                }
            for (int64_t k = 0; k < K; ++k)
                for (int64_t i = 0; i < inner; ++i)
                    op[k * inner + i] = log ? xp[k * inner + i] - (hi[i] + simd::log(total[i])) : op[k * inner + i] / total[i];
        }
    });
}

namespace {

// Calls body(k, o, in, count, in_step) for runs of output positions along the last dimension whose input
// positions lie inside the plane: output o + i reads (or adds to) input in + i * in_step through kernel
// element k. With outside set, it also calls outside(k, o, count) for runs that fall off the input.
template <typename Body, typename Outside>
void window_runs(const Windows& w, const Body& body, const Outside& outside) {
    const int64_t s2 = w.stride[2];
    int64_t k = 0;
    for (int64_t k0 = 0; k0 < w.kernel[0]; ++k0) {
        for (int64_t k1 = 0; k1 < w.kernel[1]; ++k1) {
            for (int64_t k2 = 0; k2 < w.kernel[2]; ++k2, ++k) {
                // Outputs [lo, hi) of a row read inside the input
                int64_t first = k2 * w.dilation[2] - w.pad[2];  // input position of output 0
                int64_t lo = first >= 0 ? 0 : (-first + s2 - 1) / s2;
                int64_t hi = w.in[2] - 1 - first >= 0 ? (w.in[2] - 1 - first) / s2 + 1 : 0;
                lo = std::min(lo, w.out[2]);
                hi = std::max(lo, std::min(hi, w.out[2]));
                for (int64_t o0 = 0; o0 < w.out[0]; ++o0) {
                    int64_t i0 = o0 * w.stride[0] - w.pad[0] + k0 * w.dilation[0];
                    for (int64_t o1 = 0; o1 < w.out[1]; ++o1) {
                        int64_t i1 = o1 * w.stride[1] - w.pad[1] + k1 * w.dilation[1];
                        int64_t o = (o0 * w.out[1] + o1) * w.out[2];
                        if (i0 < 0 || i0 >= w.in[0] || i1 < 0 || i1 >= w.in[1]) {
                            outside(k, o, w.out[2]);
                            continue;
                        }
                        if (lo > 0) outside(k, o, lo);
                        if (hi > lo) body(k, o + lo, (i0 * w.in[1] + i1) * w.in[2] + first + lo * s2, hi - lo, s2);
                        if (hi < w.out[2]) outside(k, o + hi, w.out[2] - hi);
                    }
                }
            }
        }
    }
}

}  // namespace

void unfold(const float* x, float* col, const Windows& w) {
    const int64_t n_out = w.out_size(), per_plane = w.kernel_size() * n_out, in_size = w.in_size();
    parallel_for(0, w.planes, grain_for(per_plane), [&](int64_t b, int64_t e) {
        for (int64_t p = b; p < e; ++p) {
            const float* xp = x + p * in_size;
            float* cp = col + p * per_plane;
            window_runs(
                w,
                [&](int64_t k, int64_t o, int64_t in, int64_t count, int64_t step) {
                    float* dst = cp + k * n_out + o;
                    if (step == 1) std::memcpy(dst, xp + in, (size_t)count * sizeof(float));
                    else for (int64_t i = 0; i < count; ++i) dst[i] = xp[in + i * step];
                },
                [&](int64_t k, int64_t o, int64_t count) { std::fill(cp + k * n_out + o, cp + k * n_out + o + count, 0.0f); });
        }
    });
}

void fold(const float* col, float* x, const Windows& w) {
    const int64_t n_out = w.out_size(), per_plane = w.kernel_size() * n_out, in_size = w.in_size();
    parallel_for(0, w.planes, grain_for(per_plane), [&](int64_t b, int64_t e) {
        for (int64_t p = b; p < e; ++p) {
            float* xp = x + p * in_size;
            const float* cp = col + p * per_plane;
            std::fill(xp, xp + in_size, 0.0f);
            window_runs(
                w,
                [&](int64_t k, int64_t o, int64_t in, int64_t count, int64_t step) {
                    const float* src = cp + k * n_out + o;
                    if (step == 1) for (int64_t i = 0; i < count; ++i) xp[in + i] += src[i];
                    else for (int64_t i = 0; i < count; ++i) xp[in + i * step] += src[i];
                },
                [](int64_t, int64_t, int64_t) {});
        }
    });
}

}  // namespace host
}  // namespace kern
