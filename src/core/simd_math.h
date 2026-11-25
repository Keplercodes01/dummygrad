#pragma once
// Branch-free single-precision exp, log and tanh for CPU loops. Plain C++ that compilers vectorize
// (unlike calls into libm), so a loop over them runs 4 to 16 elements at a time. Coefficients are
// from Cephes; errors stay within a few ulp, and infinities, NaN and subnormals behave as in libm.

#include <cmath>
#include <cstdint>
#include <cstring>

#define DG_FORCE_INLINE inline __attribute__((always_inline))

namespace simd {

DG_FORCE_INLINE float from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

DG_FORCE_INLINE uint32_t to_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

DG_FORCE_INLINE float exp(float x) {
    // Past these bounds the result is inf or 0 anyway; NaN passes both comparisons through
    float c = x > 88.8f ? 88.8f : x;
    c = c < -104.0f ? -104.0f : c;
    float n = floorf(c * 1.44269504088896341f + 0.5f);
    float r = c - n * 0.693359375f;
    r = r + n * 2.12194440e-4f;
    float y = 1.9875691500e-4f;
    y = y * r + 1.3981999507e-3f;
    y = y * r + 8.3334519073e-3f;
    y = y * r + 4.1665795894e-2f;
    y = y * r + 1.6666665459e-1f;
    y = y * r + 5.0000001201e-1f;
    y = y * (r * r) + r + 1.0f;
    // 2^n as two factors, so n = 128 and subnormal results come out right
    int32_t ni = (int32_t)n;
    int32_t n1 = ni >> 1, n2 = ni - n1;
    return y * from_bits((uint32_t)(n1 + 127) << 23) * from_bits((uint32_t)(n2 + 127) << 23);
}

DG_FORCE_INLINE float log(float x) {
    float scaled = x < 1.17549435e-38f ? x * 8388608.0f : x;  // subnormals: scale by 2^23
    float e_adjust = x < 1.17549435e-38f ? -23.0f : 0.0f;
    uint32_t u = to_bits(scaled);
    float e = (float)((int32_t)((u >> 23) & 0xff) - 126) + e_adjust;
    float m = from_bits((u & 0x007fffffu) | 0x3f000000u);  // [0.5, 1)
    float low = m < 0.707106781186547524f ? 1.0f : 0.0f;
    e = e - low;
    m = m + low * m - 1.0f;
    float z = m * m;
    float y = 7.0376836292e-2f;
    y = y * m - 1.1514610310e-1f;
    y = y * m + 1.1676998740e-1f;
    y = y * m - 1.2420140846e-1f;
    y = y * m + 1.4249322787e-1f;
    y = y * m - 1.6668057665e-1f;
    y = y * m + 2.0000714765e-1f;
    y = y * m - 2.4999993993e-1f;
    y = y * m + 3.3333331174e-1f;
    y = y * m * z;
    y = y - 2.12194440e-4f * e - 0.5f * z;
    float result = m + y + 0.693359375f * e;
    float inf = from_bits(0x7f800000u);
    result = x == inf ? inf : result;
    result = x == 0.0f ? -inf : result;
    return (x < 0.0f || x != x) ? from_bits(0x7fc00000u) : result;
}

DG_FORCE_INLINE float tanh(float x) {
    float a = fabsf(x);
    float z = x * x;
    float p = -5.70498872745e-3f;
    p = p * z + 2.06390887954e-2f;
    p = p * z - 5.37397155531e-2f;
    p = p * z + 1.33314422036e-1f;
    p = p * z - 3.33332819422e-1f;
    float small = p * z * x + x;
    float large = 1.0f - 2.0f / (simd::exp(2.0f * a) + 1.0f);
    float signed_large = x < 0.0f ? -large : large;
    return a < 0.625f ? small : signed_large;
}

DG_FORCE_INLINE float sigmoid(float x) { return 1.0f / (1.0f + simd::exp(-x)); }

}  // namespace simd
