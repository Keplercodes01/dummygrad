#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>

enum class Device { CPU, CUDA, MPS };
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

static inline uint16_t float_to_half_bits(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0x00ff) - 127 + 15;
    uint32_t frac = f & 0x007fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        frac |= 0x00800000;
        frac >>= (1 - exp);
        return (uint16_t)(sign | (frac >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00 | (frac ? 1 : 0));
    }
    return (uint16_t)(sign | (exp << 10) | (frac >> 13));
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
            exp = 1;
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp++;
            }
            frac &= 0x03ff;
            exp = 127 - 15 - exp + 1;
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
    explicit float16(float f) : bits(float_to_half_bits(f)) {}
    operator float() const { return half_bits_to_float(bits); }

    float16& operator+=(const float16& rhs) { *this = float16((float)*this + (float)rhs); return *this; }
    float16& operator-=(const float16& rhs) { *this = float16((float)*this - (float)rhs); return *this; }
    float16& operator*=(const float16& rhs) { *this = float16((float)*this * (float)rhs); return *this; }
    float16& operator/=(const float16& rhs) { *this = float16((float)*this / (float)rhs); return *this; }

    bool operator==(const float16& rhs) const { return bits == rhs.bits; }
    bool operator!=(const float16& rhs) const { return bits != rhs.bits; }
};

struct bfloat16 { 
    uint16_t bits; 
    bfloat16() : bits(0) {}
    explicit bfloat16(float f) : bits(float_to_bfloat16_bits(f)) {}
    operator float() const { return bfloat16_bits_to_float(bits); }

    bfloat16& operator+=(const bfloat16& rhs) { *this = bfloat16((float)*this + (float)rhs); return *this; }
    bfloat16& operator-=(const bfloat16& rhs) { *this = bfloat16((float)*this - (float)rhs); return *this; }
    bfloat16& operator*=(const bfloat16& rhs) { *this = bfloat16((float)*this * (float)rhs); return *this; }
    bfloat16& operator/=(const bfloat16& rhs) { *this = bfloat16((float)*this / (float)rhs); return *this; }

    bool operator==(const bfloat16& rhs) const { return bits == rhs.bits; }
    bool operator!=(const bfloat16& rhs) const { return bits != rhs.bits; }
};

#define Dispatch_DType(type, MATH_CODE) \
    switch(type) { \
        case DType::Float32:  { using scalar_type = float; MATH_CODE; break; } \
        case DType::Float16:  { using scalar_type = float16; MATH_CODE; break; } \
        case DType::BFloat16: { using scalar_type = bfloat16; MATH_CODE; break; } \
        case DType::Int8:     { using scalar_type = int8_t; MATH_CODE; break; } \
        default: throw std::runtime_error("Unsupported dtype"); \
    }
