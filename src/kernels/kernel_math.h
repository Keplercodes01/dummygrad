#pragma once
// Scalar math for kernels that compile for both the CPU and the GPU

#include <cmath>
#include <cstdint>
#include <cstring>
#include "kernels.h"
#include "simd_math.h"

#ifdef __CUDACC__
#define DG_HD __host__ __device__ __forceinline__
#else
#define DG_HD inline __attribute__((always_inline))
#endif

namespace kern {

// exp, log and tanh: vectorizable versions on the host, the hardware-backed ones on the GPU
DG_HD float m_exp(float x) {
#ifdef __CUDA_ARCH__
    return expf(x);
#else
    return simd::exp(x);
#endif
}
DG_HD float m_log(float x) {
#ifdef __CUDA_ARCH__
    return logf(x);
#else
    return simd::log(x);
#endif
}
DG_HD float m_tanh(float x) {
#ifdef __CUDA_ARCH__
    return tanhf(x);
#else
    return simd::tanh(x);
#endif
}
DG_HD uint32_t m_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    return simd::to_bits(x);
#endif
}
// Bit tests, so that fast-math flags cannot optimize them away
DG_HD bool m_isnan(float x) { return (m_bits(x) & 0x7fffffffu) > 0x7f800000u; }
DG_HD bool m_isinf(float x) { return (m_bits(x) & 0x7fffffffu) == 0x7f800000u; }

DG_HD float m_log1pexp(float z) {  // log(1 + e^z)
    return z > 0.0f ? z + log1pf(m_exp(-z)) : log1pf(m_exp(z));
}

// Giles (2012), single precision
DG_HD float m_erfinv(float x) {
    if (x == 1.0f) return INFINITY;
    if (x == -1.0f) return -INFINITY;
    float w = -logf((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w -= 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrtf(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

constexpr float kPi = 3.14159265358979323846f;

DG_HD float m_digamma(float x) {
    float r = 0.0f;
    if (x <= 0.0f) {
        if (x == floorf(x)) return NAN;
        r = -kPi / tanf(kPi * x);
        x = 1.0f - x;
    }
    while (x < 6.0f) {
        r -= 1.0f / x;
        x += 1.0f;
    }
    float f = 1.0f / (x * x);
    return r + logf(x) - 0.5f / x - f * (1.0f / 12.0f - f * (1.0f / 120.0f - f * (1.0f / 252.0f - f / 240.0f)));
}

DG_HD float m_trigamma(float x) {
    float sign = 1.0f, r = 0.0f;
    if (x < 0.5f) {
        float s = sinf(kPi * x);
        r = -(kPi * kPi) / (s * s);
        x = 1.0f - x;
        sign = -1.0f;
    }
    while (x < 6.0f) {
        r += 1.0f / (x * x);
        x += 1.0f;
    }
    float f = 1.0f / (x * x);
    r += f / 2.0f + (1.0f + f * (1.0f / 6.0f - f * (1.0f / 30.0f - f * (1.0f / 42.0f - f / 30.0f)))) / x;
    return sign * r;
}

DG_HD float m_tetragamma(float x) {  // second derivative of digamma
    float r = 0.0f;
    if (x < 0.5f) {
        float s = sinf(kPi * x), c = cosf(kPi * x);
        r = -2.0f * kPi * kPi * kPi * c / (s * s * s);
        x = 1.0f - x;
    }
    while (x < 6.0f) {
        r -= 2.0f / (x * x * x);
        x += 1.0f;
    }
    float f = 1.0f / (x * x);
    return r - f * (1.0f + 1.0f / x + f * (0.5f - f * (1.0f / 6.0f - f * (1.0f / 6.0f - f * 0.3f))));
}

// Modified Bessel functions of the first kind (Abramowitz & Stegun 9.8.1 - 9.8.4)
DG_HD float m_i0(float x) {
    float a = fabsf(x);
    if (a < 3.75f) {
        float t = (x / 3.75f) * (x / 3.75f);
        return 1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
    }
    float t = 3.75f / a;
    float p = 0.39894228f + t * (0.01328592f + t * (0.00225319f + t * (-0.00157565f + t * (0.00916281f +
              t * (-0.02057706f + t * (0.02635537f + t * (-0.01647633f + t * 0.00392377f)))))));
    return expf(a) / sqrtf(a) * p;
}
DG_HD float m_i1(float x) {
    float a = fabsf(x);
    float r;
    if (a < 3.75f) {
        float t = (x / 3.75f) * (x / 3.75f);
        r = a * (0.5f + t * (0.87890594f + t * (0.51498869f + t * (0.15084934f + t * (0.02658733f + t * (0.00301532f + t * 0.00032411f))))));
    } else {
        float t = 3.75f / a;
        float p = 0.02282967f + t * (-0.02895312f + t * (0.01787654f - t * 0.00420059f));
        p = 0.39894228f + t * (-0.03988024f + t * (-0.00362018f + t * (0.00163801f + t * (-0.01031555f + t * p))));
        r = expf(a) / sqrtf(a) * p;
    }
    return x < 0.0f ? -r : r;
}

template <Unary OP>
DG_HD float unary_eval(float x, float p, float q) {
    (void)p;
    (void)q;
    if constexpr (OP == Unary::Abs) return fabsf(x);
    else if constexpr (OP == Unary::Neg) return -x;
    else if constexpr (OP == Unary::Sign) return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : x);
    else if constexpr (OP == Unary::Exp) return m_exp(x);
    else if constexpr (OP == Unary::Expm1) return expm1f(x);
    else if constexpr (OP == Unary::Log) return m_log(x);
    else if constexpr (OP == Unary::Log2) return log2f(x);
    else if constexpr (OP == Unary::Log10) return log10f(x);
    else if constexpr (OP == Unary::Log1p) return log1pf(x);
    else if constexpr (OP == Unary::Sqrt) return sqrtf(x);
    else if constexpr (OP == Unary::Rsqrt) return 1.0f / sqrtf(x);
    else if constexpr (OP == Unary::Square) return x * x;
    else if constexpr (OP == Unary::Reciprocal) return 1.0f / x;
    else if constexpr (OP == Unary::Sin) return sinf(x);
    else if constexpr (OP == Unary::Cos) return cosf(x);
    else if constexpr (OP == Unary::Tan) return tanf(x);
    else if constexpr (OP == Unary::Asin) return asinf(x);
    else if constexpr (OP == Unary::Acos) return acosf(x);
    else if constexpr (OP == Unary::Atan) return atanf(x);
    else if constexpr (OP == Unary::Sinh) return sinhf(x);
    else if constexpr (OP == Unary::Cosh) return coshf(x);
    else if constexpr (OP == Unary::Tanh) return m_tanh(x);
    else if constexpr (OP == Unary::Asinh) return asinhf(x);
    else if constexpr (OP == Unary::Acosh) return acoshf(x);
    else if constexpr (OP == Unary::Atanh) return atanhf(x);
    else if constexpr (OP == Unary::Erf) return erff(x);
    else if constexpr (OP == Unary::Erfc) return erfcf(x);
    else if constexpr (OP == Unary::Erfinv) return m_erfinv(x);
    else if constexpr (OP == Unary::Lgamma) return lgammaf(x);
    else if constexpr (OP == Unary::Digamma) return m_digamma(x);
    else if constexpr (OP == Unary::Trigamma) return m_trigamma(x);
    else if constexpr (OP == Unary::Tetragamma) return m_tetragamma(x);
    else if constexpr (OP == Unary::Floor) return floorf(x);
    else if constexpr (OP == Unary::Ceil) return ceilf(x);
    else if constexpr (OP == Unary::Round) return rintf(x);
    else if constexpr (OP == Unary::Trunc) return truncf(x);
    else if constexpr (OP == Unary::IsNan) return m_isnan(x) ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::IsInf) return m_isinf(x) ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::IsFinite) return (m_bits(x) & 0x7f800000u) != 0x7f800000u ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::LogicalNot) return x == 0.0f ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::Clamp) return x < p ? p : (x > q ? q : x);
    else if constexpr (OP == Unary::PowScalar) return powf(x, p);
    else if constexpr (OP == Unary::ScalarPow) return powf(p, x);
    else if constexpr (OP == Unary::Sigmoid) return 1.0f / (1.0f + m_exp(-x));
    else if constexpr (OP == Unary::LogSigmoid) return -m_log1pexp(-x);
    else if constexpr (OP == Unary::Softplus) return x * p > q ? x : m_log1pexp(x * p) / p;
    else if constexpr (OP == Unary::Relu) return x > 0.0f ? x : 0.0f;
    else if constexpr (OP == Unary::LeakyRelu) return x > 0.0f ? x : p * x;
    else if constexpr (OP == Unary::Elu) return x > 0.0f ? x : p * expm1f(x);
    else if constexpr (OP == Unary::Gelu) return 0.5f * x * erfcf(-0.70710678118654752f * x);
    else if constexpr (OP == Unary::GeluTanh)
        return 0.5f * x * (1.0f + m_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    else if constexpr (OP == Unary::Silu) return x / (1.0f + m_exp(-x));
    else if constexpr (OP == Unary::Mish) return x * m_tanh(m_log1pexp(x));
    else if constexpr (OP == Unary::Hardswish) return x <= -3.0f ? 0.0f : (x >= 3.0f ? x : x * (x + 3.0f) / 6.0f);
    else if constexpr (OP == Unary::Hardsigmoid) return x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : x / 6.0f + 0.5f);
    else if constexpr (OP == Unary::Softsign) return x / (1.0f + fabsf(x));
    else if constexpr (OP == Unary::Softshrink) return x > p ? x - p : (x < -p ? x + p : 0.0f);
    else if constexpr (OP == Unary::Hardshrink) return (x > p || x < -p) ? x : 0.0f;
    else if constexpr (OP == Unary::Threshold) return x > p ? x : q;
    else if constexpr (OP == Unary::EqScalar) return x == p ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::NeScalar) return x != p ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::LtScalar) return x < p ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::LeScalar) return x <= p ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::GtScalar) return x > p ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::GeScalar) return x >= p ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::FmodScalar) return fmodf(x, p);
    else if constexpr (OP == Unary::RemainderScalar) {
        float r = fmodf(x, p);
        return (r != 0.0f && ((r < 0.0f) != (p < 0.0f))) ? r + p : r;
    } else if constexpr (OP == Unary::InRange) return (x >= p && x <= q) ? 1.0f : 0.0f;
    else if constexpr (OP == Unary::Sinc) return x == 0.0f ? 1.0f : sinf(kPi * x) / (kPi * x);
    else if constexpr (OP == Unary::I0) return m_i0(x);
    else if constexpr (OP == Unary::I1) return m_i1(x);
    else {
        static_assert(OP == Unary::NanToNum, "unary_eval: unhandled op");
        if (m_isnan(x)) return p;
        if (m_isinf(x)) return x > 0.0f ? 3.402823466e+38f : -3.402823466e+38f;
        return x;
    }
}

template <Binary OP>
DG_HD float binary_eval(float a, float b) {
    if constexpr (OP == Binary::Add) return a + b;
    else if constexpr (OP == Binary::Sub) return a - b;
    else if constexpr (OP == Binary::Mul) return a * b;
    else if constexpr (OP == Binary::Div) return a / b;
    else if constexpr (OP == Binary::Pow) return powf(a, b);
    else if constexpr (OP == Binary::Max) return (m_isnan(a) || a > b) ? a : b;
    else if constexpr (OP == Binary::Min) return (m_isnan(a) || a < b) ? a : b;
    else if constexpr (OP == Binary::Atan2) return atan2f(a, b);
    else if constexpr (OP == Binary::Fmod) return fmodf(a, b);
    else if constexpr (OP == Binary::Remainder) {
        float r = fmodf(a, b);
        return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r;
    } else if constexpr (OP == Binary::FloorDivide) {
        float r = fmodf(a, b);
        float d = (a - r) / b;
        if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) d -= 1.0f;
        float f = floorf(d);
        return d - f > 0.5f ? f + 1.0f : f;
    } else if constexpr (OP == Binary::Hypot) return hypotf(a, b);
    else if constexpr (OP == Binary::Copysign) return copysignf(a, b);
    else if constexpr (OP == Binary::Eq) return a == b ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::Ne) return a != b ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::Lt) return a < b ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::Le) return a <= b ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::Gt) return a > b ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::Ge) return a >= b ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::And) return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
    else if constexpr (OP == Binary::Or) return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
    else {
        static_assert(OP == Binary::Xor, "binary_eval: unhandled op");
        return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f;
    }
}

template <Reduce OP>
DG_HD float reduce_identity() {
    if constexpr (OP == Reduce::Prod) return 1.0f;
    else if constexpr (OP == Reduce::Max) return -INFINITY;
    else if constexpr (OP == Reduce::Min) return INFINITY;
    else return 0.0f;
}
// The value an element contributes
template <Reduce OP>
DG_HD float reduce_map(float x) {
    if constexpr (OP == Reduce::SumSquares) return x * x;
    else if constexpr (OP == Reduce::SumAbs) return fabsf(x);
    else if constexpr (OP == Reduce::CountNonzero) return x != 0.0f ? 1.0f : 0.0f;
    else return x;
}
template <Reduce OP>
DG_HD float reduce_combine(float a, float b) {
    if constexpr (OP == Reduce::Prod) return a * b;
    else if constexpr (OP == Reduce::Max) return (m_isnan(a) || a > b) ? a : b;
    else if constexpr (OP == Reduce::Min) return (m_isnan(a) || a < b) ? a : b;
    else return a + b;
}

// Philox 4x32-10 (Salmon et al., 2011): a keyed bijection of a 128-bit counter
struct Philox {
    uint32_t v[4];
    DG_HD static uint32_t mulhi(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }
    DG_HD static Philox block(uint64_t seed, uint64_t stream, uint64_t counter) {
        Philox s;
        s.v[0] = (uint32_t)counter;
        s.v[1] = (uint32_t)(counter >> 32);
        s.v[2] = (uint32_t)stream;
        s.v[3] = (uint32_t)(stream >> 32);
        uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
        for (int round = 0; round < 10; ++round) {
            uint32_t hi0 = mulhi(0xD2511F53u, s.v[0]), lo0 = 0xD2511F53u * s.v[0];
            uint32_t hi1 = mulhi(0xCD9E8D57u, s.v[2]), lo1 = 0xCD9E8D57u * s.v[2];
            uint32_t n0 = hi1 ^ s.v[1] ^ k0, n2 = hi0 ^ s.v[3] ^ k1;
            s.v[0] = n0;
            s.v[1] = lo1;
            s.v[2] = n2;
            s.v[3] = lo0;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        return s;
    }
    DG_HD float uniform(int lane) const { return (float)(v[lane] >> 8) * 5.9604644775390625e-08f; }
    // Two independent standard normals from lanes 2 * pair and 2 * pair + 1 (Box-Muller)
    DG_HD void normals(int pair, float& z0, float& z1) const {
        float u1 = ((float)v[2 * pair] + 1.0f) * 2.3283064365386963e-10f;  // (0, 1]
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * kPi * uniform(2 * pair + 1);
        z0 = r * cosf(theta);
        z1 = r * sinf(theta);
    }
};

}  // namespace kern
