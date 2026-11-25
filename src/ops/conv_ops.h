#pragma once
// Convolution, transposed convolution, pooling and resampling in 1, 2 and 3 spatial dimensions, with
// PyTorch's argument conventions (weights [out, in / groups, ...]). Convolutions unfold the input into
// columns and multiply; transposed convolutions multiply and fold. Included from functional.h.

#include "loss_ops.h"

namespace conv_detail {

using Dims = std::vector<int64_t>;

// One value per spatial dimension
inline Dims spread(const Dims& v, int n, const char* who) {
    if (v.size() == 1) return Dims((size_t)n, v[0]);
    if ((int)v.size() != n) throw std::invalid_argument(std::string(who) + ": expected 1 or " + std::to_string(n) + " values");
    return v;
}

// Windows over the last n dimensions of x [B, C, ...]
inline kern::Windows windows(const std::vector<int64_t>& x_shape, int n, const Dims& kernel, const Dims& stride, const Dims& pad,
                             const Dims& dilation, const char* who, bool ceil_mode = false) {
    kern::Windows w;
    w.planes = x_shape[0] * x_shape[1];
    for (int i = 0; i < n; ++i) {
        int d = 3 - n + i;
        w.in[d] = x_shape[(size_t)(2 + i)];
        w.kernel[d] = kernel[(size_t)i];
        w.stride[d] = stride[(size_t)i];
        w.pad[d] = pad[(size_t)i];
        w.dilation[d] = dilation[(size_t)i];
        if (w.kernel[d] < 1 || w.stride[d] < 1 || w.dilation[d] < 1 || w.pad[d] < 0) throw std::invalid_argument(std::string(who) + ": kernel, stride and dilation must be positive, padding non-negative");
        int64_t room = w.in[d] + 2 * w.pad[d] - w.dilation[d] * (w.kernel[d] - 1) - 1;
        if (room < 0) throw std::invalid_argument(std::string(who) + ": the kernel is larger than the padded input");
        w.out[d] = (ceil_mode ? (room + w.stride[d] - 1) / w.stride[d] : room / w.stride[d]) + 1;
        // With ceil_mode, the last window must still start inside the input or its left padding
        if (ceil_mode && (w.out[d] - 1) * w.stride[d] >= w.in[d] + w.pad[d]) --w.out[d];
    }
    return w;
}

inline TensorPtr fold_windows(const TensorPtr& col, const std::vector<int64_t>& x_shape, const kern::Windows& w);

// x -> [planes, kernel elements, output positions]
inline TensorPtr unfold_windows(const TensorPtr& x_in, const kern::Windows& w) {
    auto x = op_detail::dense(x_in);
    auto out = op_detail::empty({w.planes, w.kernel_size(), w.out_size()}, x->device);
    if (out->size() > 0) DG_KERNEL(x->device, unfold(x->data_ptr<float>(), out->data_ptr<float>(), w));
    auto x_shape = x->shape;
    return make_op("unfold", out, {x}, {}, [x_shape, w](const OpContext&, const TensorPtr& g) -> TensorList { return {fold_windows(g, x_shape, w)}; });
}
// Columns summed back into an input of shape x_shape
inline TensorPtr fold_windows(const TensorPtr& col_in, const std::vector<int64_t>& x_shape, const kern::Windows& w) {
    auto col = op_detail::dense(col_in);
    auto out = op_detail::empty(x_shape, col->device);
    if (out->size() > 0) DG_KERNEL(col->device, fold(col->data_ptr<float>(), out->data_ptr<float>(), w));
    return make_op("fold", out, {col}, {}, [w](const OpContext&, const TensorPtr& g) -> TensorList { return {unfold_windows(g, w)}; });
}

inline std::vector<int64_t> spatial_out(const std::vector<int64_t>& head, const kern::Windows& w, int n) {
    auto shape = head;
    for (int d = 3 - n; d < 3; ++d) shape.push_back(w.out[d]);
    return shape;
}

inline int64_t product(const std::vector<int64_t>& v, size_t from) {
    int64_t p = 1;
    for (size_t i = from; i < v.size(); ++i) p *= v[i];
    return p;
}

inline TensorPtr add_channel_bias(const TensorPtr& x, const TensorPtr& bias, int n) {
    if (!bias) return x;
    std::vector<int64_t> shape((size_t)n + 2, 1);
    shape[1] = bias->size();
    return add(x, reshape(bias, shape));
}

}  // namespace conv_detail

// Cross-correlation of x [B, C, d1, ..] with weight [out, C / groups, k1, ..], for 1 to 3 spatial dims
inline TensorPtr conv_nd(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias, const std::vector<int64_t>& stride_in,
                         const std::vector<int64_t>& padding_in, const std::vector<int64_t>& dilation_in, int64_t groups) {
    DG_PROFILE_OP("conv");
    int n = (int)x->ndim() - 2;
    if (n < 1 || n > 3 || weight->ndim() != x->ndim()) throw std::invalid_argument("conv: input [B, C, ...] and weight must both have 1 to 3 spatial dimensions");
    auto stride = conv_detail::spread(stride_in, n, "conv"), padding = conv_detail::spread(padding_in, n, "conv"), dilation = conv_detail::spread(dilation_in, n, "conv");
    int64_t B = x->shape[0], C = x->shape[1], O = weight->shape[0];
    if (groups < 1 || C % groups != 0 || O % groups != 0) throw std::invalid_argument("conv: groups must divide the input and output channels");
    if (weight->shape[1] != C / groups) throw std::invalid_argument("conv: weight " + shape_string(weight->shape) + " does not match input " + shape_string(x->shape) + " with " + std::to_string(groups) + " groups");
    std::vector<int64_t> kernel(weight->shape.begin() + 2, weight->shape.end());
    auto w = conv_detail::windows(x->shape, n, kernel, stride, padding, dilation, "conv");
    int64_t K = w.kernel_size(), L = w.out_size(), per_group = C / groups;
    // [B, groups, C / groups * K, L]; a group's weight [O / groups, C / groups * K] multiplies it
    auto col = conv_detail::unfold_windows(x, w);
    TensorPtr out;
    if (groups == 1) {
        out = matmul(reshape(weight, {O, C * K}), reshape(col, {B, C * K, L}));
    } else {
        out = matmul(reshape(weight, {1, groups, O / groups, per_group * K}), reshape(col, {B, groups, per_group * K, L}));
    }
    out = reshape(out, conv_detail::spatial_out({B, O}, w, n));
    return conv_detail::add_channel_bias(out, bias, n);
}

// Gradient of convolution with respect to its input, as an op. weight [C, out / groups, k1, ..].
inline TensorPtr conv_transpose_nd(const TensorPtr& x, const TensorPtr& weight, const TensorPtr& bias, const std::vector<int64_t>& stride_in,
                                   const std::vector<int64_t>& padding_in, const std::vector<int64_t>& output_padding_in,
                                   const std::vector<int64_t>& dilation_in, int64_t groups) {
    DG_PROFILE_OP("conv_transpose");
    int n = (int)x->ndim() - 2;
    if (n < 1 || n > 3 || weight->ndim() != x->ndim()) throw std::invalid_argument("conv_transpose: input [B, C, ...] and weight must both have 1 to 3 spatial dimensions");
    auto stride = conv_detail::spread(stride_in, n, "conv_transpose"), padding = conv_detail::spread(padding_in, n, "conv_transpose");
    auto dilation = conv_detail::spread(dilation_in, n, "conv_transpose"), extra = conv_detail::spread(output_padding_in, n, "conv_transpose");
    int64_t B = x->shape[0], C = x->shape[1];
    if (groups < 1 || C % groups != 0 || weight->shape[0] != C) throw std::invalid_argument("conv_transpose: weight must be [in channels, out channels / groups, ...]");
    int64_t O = weight->shape[1] * groups;
    std::vector<int64_t> kernel(weight->shape.begin() + 2, weight->shape.end());
    std::vector<int64_t> out_shape{B, O};
    for (int i = 0; i < n; ++i) {
        size_t s = (size_t)i;
        if (extra[s] >= stride[s] && extra[s] >= dilation[s]) throw std::invalid_argument("conv_transpose: output_padding must be smaller than stride or dilation");
        out_shape.push_back((x->shape[2 + s] - 1) * stride[s] - 2 * padding[s] + dilation[s] * (kernel[s] - 1) + extra[s] + 1);
        if (out_shape.back() < 1) throw std::invalid_argument("conv_transpose: the output would be empty");
    }
    auto w = conv_detail::windows(out_shape, n, kernel, stride, padding, dilation, "conv_transpose");
    int64_t K = w.kernel_size(), L = conv_detail::product(x->shape, 2);
    if (w.out_size() != L) throw std::logic_error("conv_transpose: inconsistent output size");
    TensorPtr col;
    if (groups == 1) {
        col = matmul(transpose(reshape(weight, {C, O * K}), 0, 1), reshape(x, {B, C, L}));
    } else {
        col = matmul(transpose(reshape(weight, {1, groups, C / groups, O / groups * K}), -2, -1), reshape(x, {B, groups, C / groups, L}));
    }
    auto out = conv_detail::fold_windows(reshape(col, {B * O, K, L}), out_shape, w);
    return conv_detail::add_channel_bias(out, bias, n);
}

// Sliding blocks of x [B, C, d1, ..] as columns [B, C * kernel elements, blocks], and their sum back
inline TensorPtr unfold(const TensorPtr& x, const std::vector<int64_t>& kernel, const std::vector<int64_t>& stride = {1},
                        const std::vector<int64_t>& padding = {0}, const std::vector<int64_t>& dilation = {1}) {
    int n = (int)x->ndim() - 2;
    if (n < 1 || n > 3) throw std::invalid_argument("unfold: input must be [B, C, ...] with 1 to 3 spatial dimensions");
    auto w = conv_detail::windows(x->shape, n, conv_detail::spread(kernel, n, "unfold"), conv_detail::spread(stride, n, "unfold"),
                                  conv_detail::spread(padding, n, "unfold"), conv_detail::spread(dilation, n, "unfold"), "unfold");
    return reshape(conv_detail::unfold_windows(x, w), {x->shape[0], x->shape[1] * w.kernel_size(), w.out_size()});
}
inline TensorPtr fold(const TensorPtr& col, const std::vector<int64_t>& output_size, const std::vector<int64_t>& kernel,
                      const std::vector<int64_t>& stride = {1}, const std::vector<int64_t>& padding = {0}, const std::vector<int64_t>& dilation = {1}) {
    int n = (int)output_size.size();
    if (n < 1 || n > 3 || col->ndim() != 3) throw std::invalid_argument("fold: columns must be [B, C * kernel elements, blocks]");
    auto k = conv_detail::spread(kernel, n, "fold");
    int64_t K = conv_detail::product(k, 0);
    if (col->shape[1] % K != 0) throw std::invalid_argument("fold: column length must be a multiple of the kernel size");
    std::vector<int64_t> shape{col->shape[0], col->shape[1] / K};
    shape.insert(shape.end(), output_size.begin(), output_size.end());
    auto w = conv_detail::windows(shape, n, k, conv_detail::spread(stride, n, "fold"), conv_detail::spread(padding, n, "fold"), conv_detail::spread(dilation, n, "fold"), "fold");
    if (w.out_size() != col->shape[2]) throw std::invalid_argument("fold: expected " + std::to_string(w.out_size()) + " blocks");
    return conv_detail::fold_windows(reshape(col, {w.planes, K, w.out_size()}), shape, w);
}

// ---------------------------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------------------------

namespace conv_detail {

inline TensorPtr max_pool_gather(const TensorPtr& x, const TensorPtr& argmax, const kern::Windows& w);

// Output gradients routed to the winning inputs
inline TensorPtr max_pool_route(const TensorPtr& g_in, const TensorPtr& argmax, const std::vector<int64_t>& x_shape, const kern::Windows& w) {
    auto g = op_detail::dense(g_in);
    auto gx = op_detail::empty(x_shape, g->device);
    DG_KERNEL(g->device, max_pool_backward(g->data_ptr<float>(), argmax->data_ptr<float>(), gx->data_ptr<float>(), w));
    return make_op("max_pool_route", gx, {g}, {}, [argmax, w](const OpContext&, const TensorPtr& gg) -> TensorList { return {max_pool_gather(gg, argmax, w)}; });
}
// The values at the winning positions: the transpose of max_pool_route
inline TensorPtr max_pool_gather(const TensorPtr& x, const TensorPtr& argmax, const kern::Windows& w) {
    auto flat = reshape(x, {w.planes, w.in_size()});
    auto at = clamp_min(reshape(argmax, {w.planes, w.out_size()}), 0.0f);
    auto picked = mul(gather(flat, 1, at), ge(reshape(argmax, {w.planes, w.out_size()}), 0.0f));
    return reshape(picked, argmax->shape);
}

}  // namespace conv_detail

// Maxima over windows of x [B, C, ...] and, for max_unpool, the flat position of each within its plane
inline ValuesAndIndices max_pool_nd(const TensorPtr& x_in, const std::vector<int64_t>& kernel, const std::vector<int64_t>& stride = {},
                                    const std::vector<int64_t>& padding = {0}, const std::vector<int64_t>& dilation = {1}, bool ceil_mode = false) {
    DG_PROFILE_OP("max_pool");
    auto x = op_detail::dense(x_in);
    int n = (int)x->ndim() - 2;
    if (n < 1 || n > 3) throw std::invalid_argument("max_pool: input must be [B, C, ...] with 1 to 3 spatial dimensions");
    auto k = conv_detail::spread(kernel, n, "max_pool"), p = conv_detail::spread(padding, n, "max_pool");
    for (int i = 0; i < n; ++i) if (2 * p[(size_t)i] > k[(size_t)i]) throw std::invalid_argument("max_pool: padding must be at most half the kernel");
    auto w = conv_detail::windows(x->shape, n, k, stride.empty() ? k : conv_detail::spread(stride, n, "max_pool"), p, conv_detail::spread(dilation, n, "max_pool"), "max_pool", ceil_mode);
    if (w.in_size() > (1 << 24)) throw std::invalid_argument("max_pool: planes past 2^24 elements are not supported");
    auto shape = conv_detail::spatial_out({x->shape[0], x->shape[1]}, w, n);
    auto out = op_detail::empty(shape, x->device), argmax = op_detail::empty(shape, x->device);
    DG_KERNEL(x->device, max_pool(x->data_ptr<float>(), out->data_ptr<float>(), argmax->data_ptr<float>(), w));
    auto x_shape = x->shape;
    auto result = make_op("max_pool", out, {x}, {}, [argmax, x_shape, w](const OpContext&, const TensorPtr& g) -> TensorList {
        return {conv_detail::max_pool_route(g, argmax, x_shape, w)};
    });
    return {result, argmax};
}
// Values put back at their max-pool positions in planes of the given spatial size; zeros elsewhere
inline TensorPtr max_unpool_nd(const TensorPtr& x, const TensorPtr& indices, const std::vector<int64_t>& output_size) {
    if (indices->shape != x->shape) throw std::invalid_argument("max_unpool: indices must match the input");
    int64_t planes = x->shape[0] * x->shape[1], n_out = conv_detail::product(output_size, 0), n_in = conv_detail::product(x->shape, 2);
    auto flat = scatter(op_detail::filled({planes, n_out}, x->device, 0.0f), 1, reshape(indices, {planes, n_in}), reshape(x, {planes, n_in}));
    std::vector<int64_t> shape{x->shape[0], x->shape[1]};
    shape.insert(shape.end(), output_size.begin(), output_size.end());
    return reshape(flat, shape);
}

namespace conv_detail {
inline TensorPtr avg_pool_windows(const TensorPtr& x, const std::vector<int64_t>& shape, const kern::Windows& w);
inline TensorPtr avg_pool_spread(const TensorPtr& g_in, const std::vector<int64_t>& x_shape, const kern::Windows& w) {
    auto g = op_detail::dense(g_in);
    auto gx = op_detail::empty(x_shape, g->device);
    DG_KERNEL(g->device, avg_pool_backward(g->data_ptr<float>(), gx->data_ptr<float>(), w));
    auto shape = g->shape;
    return make_op("avg_pool_spread", gx, {g}, {}, [shape, w](const OpContext&, const TensorPtr& gg) -> TensorList { return {avg_pool_windows(gg, shape, w)}; });
}
inline TensorPtr avg_pool_windows(const TensorPtr& x_in, const std::vector<int64_t>& shape, const kern::Windows& w) {
    auto x = op_detail::dense(x_in);
    auto out = op_detail::empty(shape, x->device);
    DG_KERNEL(x->device, avg_pool(x->data_ptr<float>(), out->data_ptr<float>(), w));
    auto x_shape = x->shape;
    return make_op("avg_pool", out, {x}, {}, [x_shape, w](const OpContext&, const TensorPtr& g) -> TensorList { return {avg_pool_spread(g, x_shape, w)}; });
}
}  // namespace conv_detail

inline TensorPtr avg_pool_nd(const TensorPtr& x, const std::vector<int64_t>& kernel, const std::vector<int64_t>& stride = {},
                             const std::vector<int64_t>& padding = {0}, bool ceil_mode = false, bool count_include_pad = true) {
    DG_PROFILE_OP("avg_pool");
    int n = (int)x->ndim() - 2;
    if (n < 1 || n > 3) throw std::invalid_argument("avg_pool: input must be [B, C, ...] with 1 to 3 spatial dimensions");
    auto k = conv_detail::spread(kernel, n, "avg_pool"), p = conv_detail::spread(padding, n, "avg_pool");
    for (int i = 0; i < n; ++i) if (2 * p[(size_t)i] > k[(size_t)i]) throw std::invalid_argument("avg_pool: padding must be at most half the kernel");
    auto w = conv_detail::windows(x->shape, n, k, stride.empty() ? k : conv_detail::spread(stride, n, "avg_pool"), p, std::vector<int64_t>((size_t)n, 1), "avg_pool", ceil_mode);
    w.count_include_pad = count_include_pad;
    return conv_detail::avg_pool_windows(x, conv_detail::spatial_out({x->shape[0], x->shape[1]}, w, n), w);
}
// (mean of |x|^p over the window times its size)^(1/p)
inline TensorPtr lp_pool_nd(const TensorPtr& x, float p, const std::vector<int64_t>& kernel, const std::vector<int64_t>& stride = {}) {
    int n = (int)x->ndim() - 2;
    int64_t K = conv_detail::product(conv_detail::spread(kernel, n, "lp_pool"), 0);
    return pow(mul_scalar(avg_pool_nd(pow(x, p), kernel, stride), (float)K), 1.0f / p);
}

namespace conv_detail {
// x with its last n dimensions each multiplied by a matrix [in, out]: what separable resampling is
inline TensorPtr apply_per_dim(const TensorPtr& x, const std::vector<std::vector<float>>& matrices, const std::vector<int64_t>& out_sizes) {
    auto y = x;
    int nd = (int)x->ndim(), n = (int)out_sizes.size();
    for (int i = 0; i < n; ++i) {
        int dim = nd - n + i;
        int64_t in = y->shape[(size_t)dim];
        auto m = op_detail::from_values(matrices[(size_t)i], {in, out_sizes[(size_t)i]}, x->device);
        auto moved = dim == nd - 1 ? y : transpose(y, dim, nd - 1);
        auto r = matmul(moved, m);
        y = dim == nd - 1 ? r : transpose(r, dim, nd - 1);
    }
    return contiguous(y);
}
}  // namespace conv_detail

// Average pooling to a given output size, with PyTorch's bin boundaries
inline TensorPtr adaptive_avg_pool_nd(const TensorPtr& x, const std::vector<int64_t>& output_size) {
    DG_PROFILE_OP("adaptive_avg_pool");
    int n = (int)x->ndim() - 2;
    auto sizes = conv_detail::spread(output_size, n, "adaptive_avg_pool");
    std::vector<std::vector<float>> m;
    for (int i = 0; i < n; ++i) {
        int64_t in = x->shape[(size_t)(2 + i)], out = sizes[(size_t)i];
        std::vector<float> a((size_t)(in * out), 0.0f);
        for (int64_t o = 0; o < out; ++o) {
            int64_t lo = o * in / out, hi = ((o + 1) * in + out - 1) / out;
            for (int64_t j = lo; j < hi; ++j) a[(size_t)(j * out + o)] = 1.0f / (float)(hi - lo);
        }
        m.push_back(std::move(a));
    }
    return conv_detail::apply_per_dim(x, m, sizes);
}
// Maxima over adaptive bins: one spatial dimension at a time
inline TensorPtr adaptive_max_pool_nd(const TensorPtr& x, const std::vector<int64_t>& output_size) {
    DG_PROFILE_OP("adaptive_max_pool");
    int n = (int)x->ndim() - 2;
    auto sizes = conv_detail::spread(output_size, n, "adaptive_max_pool");
    auto y = x;
    for (int i = 0; i < n; ++i) {
        int dim = 2 + i;
        int64_t in = y->shape[(size_t)dim], out = sizes[(size_t)i];
        TensorList bins;
        for (int64_t o = 0; o < out; ++o) {
            int64_t lo = o * in / out, hi = ((o + 1) * in + out - 1) / out;
            bins.push_back(amax(narrow(y, dim, lo, hi - lo), dim, true));
        }
        y = cat(bins, dim);
    }
    return y;
}

// ---------------------------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------------------------

enum class InterpolateMode { Nearest, Linear, Cubic, Area };

// Resizes the spatial dimensions of x [B, C, ...]. Linear covers linear, bilinear and trilinear.
inline TensorPtr interpolate(const TensorPtr& x, const std::vector<int64_t>& size, InterpolateMode mode = InterpolateMode::Nearest, bool align_corners = false) {
    DG_PROFILE_OP("interpolate");
    int n = (int)x->ndim() - 2;
    if (n < 1 || n > 3) throw std::invalid_argument("interpolate: input must be [B, C, ...] with 1 to 3 spatial dimensions");
    if (mode == InterpolateMode::Area) return adaptive_avg_pool_nd(x, size);
    auto sizes = conv_detail::spread(size, n, "interpolate");
    auto cubic = [](double t) {  // Keys, a = -0.75, as in PyTorch
        const double a = -0.75;
        t = std::fabs(t);
        if (t <= 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
        if (t < 2.0) return (((t - 5.0) * t + 8.0) * t - 4.0) * a;
        return 0.0;
    };
    std::vector<std::vector<float>> m;
    for (int i = 0; i < n; ++i) {
        int64_t in = x->shape[(size_t)(2 + i)], out = sizes[(size_t)i];
        if (out < 1) throw std::invalid_argument("interpolate: sizes must be positive");
        std::vector<float> a((size_t)(in * out), 0.0f);
        auto put = [&](int64_t src, int64_t o, double w) {
            src = std::max<int64_t>(0, std::min(in - 1, src));
            a[(size_t)(src * out + o)] += (float)w;
        };
        for (int64_t o = 0; o < out; ++o) {
            if (mode == InterpolateMode::Nearest) {
                put(o * in / out, o, 1.0);
                continue;
            }
            double pos = align_corners ? (out > 1 ? (double)o * (double)(in - 1) / (double)(out - 1) : 0.0)
                                       : ((double)o + 0.5) * (double)in / (double)out - 0.5;
            if (mode == InterpolateMode::Linear) {
                pos = std::max(0.0, pos);
                int64_t lo = (int64_t)std::floor(pos);
                double t = pos - (double)lo;
                put(lo, o, 1.0 - t);
                put(lo + 1, o, t);
            } else {
                int64_t lo = (int64_t)std::floor(pos);
                for (int64_t k = -1; k <= 2; ++k) put(lo + k, o, cubic(pos - (double)(lo + k)));
            }
        }
        m.push_back(std::move(a));
    }
    return conv_detail::apply_per_dim(x, m, sizes);
}
// The same by a scale factor
inline TensorPtr upsample(const TensorPtr& x, double scale, InterpolateMode mode = InterpolateMode::Nearest, bool align_corners = false) {
    std::vector<int64_t> size;
    for (size_t d = 2; d < x->shape.size(); ++d) size.push_back((int64_t)std::floor((double)x->shape[d] * scale));
    return interpolate(x, size, mode, align_corners);
}
// [B, C r^2, H, W] -> [B, C, H r, W r] and back
inline TensorPtr pixel_shuffle(const TensorPtr& x, int64_t r) {
    if (x->ndim() != 4 || x->shape[1] % (r * r) != 0) throw std::invalid_argument("pixel_shuffle: input must be [B, C * r * r, H, W]");
    int64_t B = x->shape[0], C = x->shape[1] / (r * r), H = x->shape[2], W = x->shape[3];
    return reshape(permute(reshape(x, {B, C, r, r, H, W}), {0, 1, 4, 2, 5, 3}), {B, C, H * r, W * r});
}
inline TensorPtr pixel_unshuffle(const TensorPtr& x, int64_t r) {
    if (x->ndim() != 4 || x->shape[2] % r != 0 || x->shape[3] % r != 0) throw std::invalid_argument("pixel_unshuffle: height and width must be multiples of r");
    int64_t B = x->shape[0], C = x->shape[1], H = x->shape[2] / r, W = x->shape[3] / r;
    return reshape(permute(reshape(x, {B, C, H, r, W, r}), {0, 1, 3, 5, 2, 4}), {B, C * r * r, H, W});
}
// Samples x [B, C, H, W] bilinearly at grid [B, H', W', 2] of (x, y) in [-1, 1]; zeros outside
inline TensorPtr grid_sample(const TensorPtr& x, const TensorPtr& grid, bool align_corners = false) {
    DG_PROFILE_OP("grid_sample");
    if (x->ndim() != 4 || grid->ndim() != 4 || grid->shape[3] != 2 || grid->shape[0] != x->shape[0]) throw std::invalid_argument("grid_sample: needs x [B, C, H, W] and grid [B, H', W', 2]");
    int64_t B = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3], P = grid->shape[1] * grid->shape[2];
    auto to_pixel = [&](const TensorPtr& g, int64_t size) {
        auto unit = mul_scalar(add_scalar(g, 1.0f), 0.5f);
        return align_corners ? mul_scalar(unit, (float)(size - 1)) : sub_scalar(mul_scalar(unit, (float)size), 0.5f);
    };
    auto gx = reshape(to_pixel(select(grid, 3, 0), W), {B, P}), gy = reshape(to_pixel(select(grid, 3, 1), H), {B, P});
    auto x0 = floor(gx), y0 = floor(gy);
    auto flat = reshape(x, {B, C, H * W});
    TensorPtr out;
    for (int corner = 0; corner < 4; ++corner) {
        auto cx = add_scalar(x0, (float)(corner & 1)), cy = add_scalar(y0, (float)(corner >> 1));
        auto wx = rsub_scalar(1.0f, abs(sub(gx, cx))), wy = rsub_scalar(1.0f, abs(sub(gy, cy)));
        auto inside = logical_and(logical_and(ge(cx, 0.0f), le(cx, (float)(W - 1))), logical_and(ge(cy, 0.0f), le(cy, (float)(H - 1))));
        auto at = add(mul_scalar(clamp(cy, 0.0f, (float)(H - 1)), (float)W), clamp(cx, 0.0f, (float)(W - 1)));
        auto values = gather(flat, 2, contiguous(expand(unsqueeze(at, 1), {B, C, P})));
        auto part = mul(values, unsqueeze(mul(mul(wx, wy), inside), 1));
        out = out ? add(out, part) : part;
    }
    return reshape(out, {B, C, grid->shape[1], grid->shape[2]});
}
// Sampling grid of a 2-D affine map theta [B, 2, 3] for grid_sample
inline TensorPtr affine_grid(const TensorPtr& theta, int64_t H, int64_t W, bool align_corners = false) {
    std::vector<float> base((size_t)(H * W * 3));
    auto coord = [&](int64_t i, int64_t n) {
        if (align_corners) return n > 1 ? -1.0f + 2.0f * (float)i / (float)(n - 1) : 0.0f;
        return (2.0f * (float)i + 1.0f) / (float)n - 1.0f;
    };
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x) {
            size_t i = (size_t)(y * W + x) * 3;
            base[i] = coord(x, W);
            base[i + 1] = coord(y, H);
            base[i + 2] = 1.0f;
        }
    int64_t B = theta->shape[0];
    auto b = contiguous(expand(op_detail::from_values(base, {1, H * W, 3}, theta->device), {B, H * W, 3}));
    return reshape(matmul(b, transpose(theta, -2, -1)), {B, H, W, 2});
}
