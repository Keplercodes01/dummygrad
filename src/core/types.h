#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

enum class Device { CPU, CUDA, MPS, TPU };
enum class DType { Float32, Float16, BFloat16, Int8 };

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:  return 4;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int8:     return 1;
        default: return 4;
    }
}

// -------------------------------------------------------------
// IEEE 754 FP16 and Brain Floating Point BF16 bitwise helpers
// -------------------------------------------------------------

// Round to nearest even, like CUDA's __float2half_rn: finite values past the fp16 range become
// +-inf, NaN stays a (quiet) NaN, and values below the smallest normal become subnormals.
static inline uint16_t float_to_half_bits(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000;
    uint32_t mag = f & 0x7fffffff;
    if (mag > 0x7f800000) return (uint16_t)(sign | 0x7e00);   // NaN
    if (mag >= 0x477ff000) return (uint16_t)(sign | 0x7c00);  // >= 65520 rounds past 65504: inf
    if (mag < 0x38800000) {
        // Subnormal: adding 0.5 aligns the value so the float adder rounds it at fp16's
        // subnormal step (2^-24), leaving the result in the low mantissa bits
        float a;
        std::memcpy(&a, &mag, sizeof(a));
        a += 0.5f;
        uint32_t bits;
        std::memcpy(&bits, &a, sizeof(bits));
        return (uint16_t)(sign | (bits - 0x3f000000));
    }
    // Normal: rebias the exponent (127 -> 15) and round the 13 dropped mantissa bits to even.
    // A carry out of the mantissa correctly bumps the exponent.
    uint32_t odd = (mag >> 13) & 1;
    mag += 0xc8000fff + odd;
    return (uint16_t)(sign | (mag >> 13));
}

static inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x03ff;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign;
        } else {
            exp = 0;
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp++;
            }
            frac &= 0x03ff;
            exp = 127 - 14 - exp;
            f = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000 | (frac << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

static inline uint16_t float_to_bfloat16_bits(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    // Preserve NaN without adding rounding bias
    if ((f & 0x7f800000) == 0x7f800000 && (f & 0x007fffff) != 0) {
        return static_cast<uint16_t>((f >> 16) | 0x0040);
    }
    uint32_t lsb = (f >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    f += rounding_bias;
    return static_cast<uint16_t>(f >> 16);
}

static inline float bfloat16_bits_to_float(uint16_t b) {
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

struct float16 { 
    uint16_t bits; 
    float16() : bits(0) {}
    float16(float f) : bits(float_to_half_bits(f)) {}
    operator float() const { return half_bits_to_float(bits); }

    float16& operator+=(const float16& rhs) { *this = float16((float)*this + (float)rhs); return *this; }
    float16& operator-=(const float16& rhs) { *this = float16((float)*this - (float)rhs); return *this; }
    float16& operator*=(const float16& rhs) { *this = float16((float)*this * (float)rhs); return *this; }
    float16& operator/=(const float16& rhs) { *this = float16((float)*this / (float)rhs); return *this; }

    bool operator==(const float16& rhs) const { return (float)*this == (float)rhs; }
    bool operator!=(const float16& rhs) const { return (float)*this != (float)rhs; }
    bool operator<(const float16& rhs) const { return (float)*this < (float)rhs; }
    bool operator<=(const float16& rhs) const { return (float)*this <= (float)rhs; }
    bool operator>(const float16& rhs) const { return (float)*this > (float)rhs; }
    bool operator>=(const float16& rhs) const { return (float)*this >= (float)rhs; }
};

inline float16 operator+(float16 a, float16 b) { return float16((float)a + (float)b); }
inline float16 operator-(float16 a, float16 b) { return float16((float)a - (float)b); }
inline float16 operator*(float16 a, float16 b) { return float16((float)a * (float)b); }
inline float16 operator/(float16 a, float16 b) { return float16((float)a / (float)b); }

struct bfloat16 { 
    uint16_t bits; 
    bfloat16() : bits(0) {}
    bfloat16(float f) : bits(float_to_bfloat16_bits(f)) {}
    operator float() const { return bfloat16_bits_to_float(bits); }

    bfloat16& operator+=(const bfloat16& rhs) { *this = bfloat16((float)*this + (float)rhs); return *this; }
    bfloat16& operator-=(const bfloat16& rhs) { *this = bfloat16((float)*this - (float)rhs); return *this; }
    bfloat16& operator*=(const bfloat16& rhs) { *this = bfloat16((float)*this * (float)rhs); return *this; }
    bfloat16& operator/=(const bfloat16& rhs) { *this = bfloat16((float)*this / (float)rhs); return *this; }

    bool operator==(const bfloat16& rhs) const { return (float)*this == (float)rhs; }
    bool operator!=(const bfloat16& rhs) const { return (float)*this != (float)rhs; }
    bool operator<(const bfloat16& rhs) const { return (float)*this < (float)rhs; }
    bool operator<=(const bfloat16& rhs) const { return (float)*this <= (float)rhs; }
    bool operator>(const bfloat16& rhs) const { return (float)*this > (float)rhs; }
    bool operator>=(const bfloat16& rhs) const { return (float)*this >= (float)rhs; }
};

inline bfloat16 operator+(bfloat16 a, bfloat16 b) { return bfloat16((float)a + (float)b); }
inline bfloat16 operator-(bfloat16 a, bfloat16 b) { return bfloat16((float)a - (float)b); }
inline bfloat16 operator*(bfloat16 a, bfloat16 b) { return bfloat16((float)a * (float)b); }
inline bfloat16 operator/(bfloat16 a, bfloat16 b) { return bfloat16((float)a / (float)b); }

// The DType a C++ element type reads, so typed data access can reject a mismatch: fp16 storage
// read as float is garbage, not an error. Raw types (void, char, uint16_t, ...) are not checked.
template <typename T>
inline constexpr bool has_dtype_v = std::is_same_v<T, float> || std::is_same_v<T, float16> ||
                                    std::is_same_v<T, bfloat16> || std::is_same_v<T, int8_t>;

template <typename T>
constexpr DType dtype_of() {
    if constexpr (std::is_same_v<T, float>) return DType::Float32;
    else if constexpr (std::is_same_v<T, float16>) return DType::Float16;
    else if constexpr (std::is_same_v<T, bfloat16>) return DType::BFloat16;
    else return DType::Int8;
}

#define Dispatch_DType(type, ...) \
    switch(type) { \
        case DType::Float32:  { using scalar_type = float; __VA_ARGS__; break; } \
        case DType::Float16:  { using scalar_type = float16; __VA_ARGS__; break; } \
        case DType::BFloat16: { using scalar_type = bfloat16; __VA_ARGS__; break; } \
        case DType::Int8:     { using scalar_type = int8_t; __VA_ARGS__; break; } \
        default: throw std::runtime_error("Unsupported dtype"); \
    }
