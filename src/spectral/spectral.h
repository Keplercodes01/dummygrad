#pragma once
// Fourier transforms (torch.fft) and the short-time Fourier transform. Complex tensors are real tensors
// with a trailing dimension of 2 holding (real, imaginary); dims count the dimensions before it. Like
// PyTorch's, forward transforms are unnormalized and inverse ones divide by n. Transforms run on the
// host in double precision (GPU tensors are copied there and back) and are differentiable to any order.

#include "functional.h"
#include "linalg/dense.h"

namespace fft {

namespace detail {

// Unnormalized transform over the second-to-last dimension of [..., n, 2]. It is linear, and its adjoint
// on (re, im) pairs is the transform in the other direction.
inline TensorPtr transform_last(const TensorPtr& x_in, bool inverse) {
    auto x = op_detail::dense(x_in);
    if (x->ndim() < 2 || x->shape.back() != 2) throw std::invalid_argument("fft: complex input must have a trailing dimension of 2");
    int64_t n = x->shape[(size_t)x->ndim() - 2], rows = x->size() / std::max<int64_t>(1, 2 * n);
    auto v = x->to_vector();
    std::vector<float> out(v.size());
    parallel_for(0, rows, std::max<int64_t>(1, 2048 / std::max<int64_t>(1, n)), [&](int64_t b, int64_t e) {
        std::vector<double> line((size_t)(2 * n));
        for (int64_t r = b; r < e; ++r) {
            for (int64_t i = 0; i < 2 * n; ++i) line[(size_t)i] = v[(size_t)(r * 2 * n + i)];
            dense::fft(line, n, inverse);
            for (int64_t i = 0; i < 2 * n; ++i) out[(size_t)(r * 2 * n + i)] = (float)line[(size_t)i];
        }
    });
    return make_op("fft", op_detail::from_values(out, x->shape, x->device), {x}, {}, [inverse](const OpContext&, const TensorPtr& g) -> TensorList {
        return {transform_last(g, !inverse)};
    });
}
inline TensorPtr transform(const TensorPtr& x, int dim_in, bool inverse) {
    int nd = (int)x->ndim() - 1;  // dimensions before the complex one
    int dim = op_detail::wrap_dim(dim_in, nd, "fft");
    int64_t n = x->shape[(size_t)dim];
    auto y = dim == nd - 1 ? x : movedim(x, dim, nd - 1);
    y = transform_last(y, inverse);
    if (inverse) y = div_scalar(y, (float)n);
    return dim == nd - 1 ? y : movedim(y, nd - 1, dim);
}
// Crops or zero-pads dim to n
inline TensorPtr resize(const TensorPtr& x, int dim, int64_t n) {
    int64_t have = x->shape[(size_t)dim];
    if (n == have) return x;
    if (n < have) return narrow(x, dim, 0, n);
    auto shape = x->shape;
    shape[(size_t)dim] = n - have;
    return cat({x, op_detail::filled(shape, x->device, 0.0f)}, dim);
}

}  // namespace detail

inline TensorPtr view_as_complex_parts(const TensorPtr& re, const TensorPtr& im) { return stack({re, im}, -1); }
inline TensorPtr to_complex(const TensorPtr& re) { return stack({re, zeros_like(re)}, -1); }
inline TensorPtr real(const TensorPtr& z) { return select(z, -1, 0); }
inline TensorPtr imag(const TensorPtr& z) { return select(z, -1, 1); }
inline TensorPtr conj(const TensorPtr& z) { return view_as_complex_parts(real(z), neg(imag(z))); }
inline TensorPtr complex_abs(const TensorPtr& z) { return norm(z, 2.0f, -1); }
inline TensorPtr complex_angle(const TensorPtr& z) { return atan2(imag(z), real(z)); }
// Elementwise complex product
inline TensorPtr complex_mul(const TensorPtr& a, const TensorPtr& b) {
    auto ar = real(a), ai = imag(a), br = real(b), bi = imag(b);
    return view_as_complex_parts(sub(mul(ar, br), mul(ai, bi)), add(mul(ar, bi), mul(ai, br)));
}

inline TensorPtr fft(const TensorPtr& z, int dim = -1) { DG_PROFILE_OP("fft"); return detail::transform(z, dim, false); }
inline TensorPtr ifft(const TensorPtr& z, int dim = -1) { DG_PROFILE_OP("ifft"); return detail::transform(z, dim, true); }
inline TensorPtr fftn(TensorPtr z, const std::vector<int>& dims) {
    for (int d : dims) z = fft(z, d);
    return z;
}
inline TensorPtr ifftn(TensorPtr z, const std::vector<int>& dims) {
    for (int d : dims) z = ifft(z, d);
    return z;
}
inline TensorPtr fft2(const TensorPtr& z) { return fftn(z, {-1, -2}); }
inline TensorPtr ifft2(const TensorPtr& z) { return ifftn(z, {-1, -2}); }

// The n / 2 + 1 non-negative frequencies of a real signal along dim (zero-padded or cropped to n)
inline TensorPtr rfft(const TensorPtr& x_in, int dim_in = -1, int64_t n = -1) {
    DG_PROFILE_OP("rfft");
    int dim = op_detail::wrap_dim(dim_in, x_in->ndim(), "rfft");
    auto x = n > 0 ? detail::resize(x_in, dim, n) : x_in;
    return narrow(fft(to_complex(x), dim), dim, 0, x->shape[(size_t)dim] / 2 + 1);
}
// The real signal of length n (default 2 (m - 1)) with the given m non-negative frequencies
inline TensorPtr irfft(const TensorPtr& z_in, int dim_in = -1, int64_t n = -1) {
    DG_PROFILE_OP("irfft");
    int dim = op_detail::wrap_dim(dim_in, z_in->ndim() - 1, "irfft");
    if (n < 0) n = 2 * (z_in->shape[(size_t)dim] - 1);
    if (n < 1) throw std::invalid_argument("irfft: the output would be empty");
    auto z = detail::resize(z_in, dim, n / 2 + 1);
    // Negative frequencies are the conjugates of the positive ones; the mean's and Nyquist bins are real
    std::vector<float> keep((size_t)(n / 2 + 1), 1.0f);
    keep[0] = 0.0f;
    if (n % 2 == 0) keep[(size_t)(n / 2)] = 0.0f;
    std::vector<int64_t> shape((size_t)z->ndim() - 1, 1);
    shape[(size_t)dim] = n / 2 + 1;
    z = view_as_complex_parts(real(z), mul(imag(z), op_detail::from_values(keep, shape, z->device)));
    int64_t mirrored = (n + 1) / 2 - 1;
    if (mirrored > 0) z = cat({z, conj(flip(narrow(z, dim, 1, mirrored), {dim}))}, dim);
    return contiguous(real(ifft(z, dim)));
}
inline TensorPtr rfft2(const TensorPtr& x) { return fft(rfft(x, -1), -2); }
inline TensorPtr irfft2(const TensorPtr& z, int64_t n_last = -1) { return irfft(ifft(z, -2), -1, n_last); }

// Frequencies of the bins, in cycles per unit of the sample spacing d
inline TensorPtr fftfreq(int64_t n, double d = 1.0) {
    std::vector<float> f((size_t)n);
    for (int64_t i = 0; i < n; ++i) f[(size_t)i] = (float)((double)(i < (n + 1) / 2 ? i : i - n) / ((double)n * d));
    return op_detail::from_values(f, {n}, Device::CPU);
}
inline TensorPtr rfftfreq(int64_t n, double d = 1.0) {
    std::vector<float> f((size_t)(n / 2 + 1));
    for (int64_t i = 0; i <= n / 2; ++i) f[(size_t)i] = (float)((double)i / ((double)n * d));
    return op_detail::from_values(f, {n / 2 + 1}, Device::CPU);
}
// Zero frequency to the middle of dim, and back
inline TensorPtr fftshift(const TensorPtr& x, int dim) { return roll(x, x->shape[(size_t)op_detail::wrap_dim(dim, x->ndim(), "fftshift")] / 2, dim); }
inline TensorPtr ifftshift(const TensorPtr& x, int dim) { return roll(x, -(x->shape[(size_t)op_detail::wrap_dim(dim, x->ndim(), "ifftshift")] / 2), dim); }

// Convolution of real signals along the last dimension through the Fourier domain: length n + m - 1
inline TensorPtr fft_convolve(const TensorPtr& a, const TensorPtr& b) {
    int64_t n = a->shape.back() + b->shape.back() - 1;
    return irfft(complex_mul(rfft(a, -1, n), rfft(b, -1, n)), -1, n);
}

// Windows (periodic, as torch.hann_window etc. default to)
inline TensorPtr hann_window(int64_t n, bool periodic = true) {
    std::vector<float> w((size_t)n);
    double d = periodic ? (double)n : (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) w[(size_t)i] = (float)(0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * (double)i / d));
    return op_detail::from_values(w, {n}, Device::CPU);
}
inline TensorPtr hamming_window(int64_t n, bool periodic = true) {
    std::vector<float> w((size_t)n);
    double d = periodic ? (double)n : (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) w[(size_t)i] = (float)(0.54 - 0.46 * std::cos(2.0 * 3.14159265358979323846 * (double)i / d));
    return op_detail::from_values(w, {n}, Device::CPU);
}

// Short-time Fourier transform of [batch, time]: [batch, n_fft / 2 + 1, frames, 2]. center pads by
// reflection so frame t is centered on sample t * hop.
inline TensorPtr stft(const TensorPtr& x_in, int64_t n_fft, int64_t hop, const TensorPtr& window_in = nullptr, bool center = true) {
    DG_PROFILE_OP("stft");
    if (x_in->ndim() != 2) throw std::invalid_argument("stft: input must be [batch, time]");
    auto x = center ? pad(x_in, {n_fft / 2, n_fft / 2}, PadMode::Reflect) : x_in;
    auto window = window_in ? window_in : op_detail::filled({n_fft}, x->device, 1.0f);
    int64_t B = x->shape[0], T = x->shape[1];
    if (T < n_fft) throw std::invalid_argument("stft: the signal is shorter than one frame");
    auto frames = transpose(unfold(reshape(x, {B, 1, T}), {n_fft}, {hop}), 1, 2);  // [B, frames, n_fft]
    return permute(rfft(mul(frames, window), -1), {0, 2, 1, 3});
}
// Inverse of stft by windowed overlap-add
inline TensorPtr istft(const TensorPtr& z, int64_t n_fft, int64_t hop, const TensorPtr& window_in = nullptr, bool center = true) {
    DG_PROFILE_OP("istft");
    if (z->ndim() != 4) throw std::invalid_argument("istft: input must be [batch, bins, frames, 2]");
    auto window = window_in ? window_in : op_detail::filled({n_fft}, z->device, 1.0f);
    auto frames = mul(irfft(permute(z, {0, 2, 1, 3}), -1, n_fft), window);  // [B, frames, n_fft]
    int64_t B = frames->shape[0], F = frames->shape[1], T = (F - 1) * hop + n_fft;
    auto overlap = [&](const TensorPtr& f) { return reshape(fold(transpose(f, 1, 2), {T}, {n_fft}, {hop}), {f->shape[0], T}); };
    auto envelope = overlap(contiguous(expand(reshape(square(window), {1, 1, n_fft}), {1, F, n_fft})));
    auto y = div(overlap(frames), clamp_min(envelope, 1e-11f));
    (void)B;
    return center ? narrow(y, 1, n_fft / 2, T - 2 * (n_fft / 2)) : y;
}

}  // namespace fft
