#pragma once
#include <cstdint>
#include <stdexcept>

enum class Device { CPU, CUDA, MPS };
enum class DType { Float32, Float16, BFloat16, Int8 };

struct float16 { 
    uint16_t bits; 
    explicit float16(float f = 0.0f) : bits(0) {}
    float16& operator+=(const float16& rhs) { return *this; }
};

struct bfloat16 { 
    uint16_t bits; 
    explicit bfloat16(float f = 0.0f) : bits(0) {}
    bfloat16& operator+=(const bfloat16& rhs) { return *this; }
};

#define Dispatch_DType(type, MATH_CODE) \
    switch(type) { \
        case DType::Float32:  { using scalar_type = float; MATH_CODE; break; } \
        case DType::Float16:  { using scalar_type = float16; MATH_CODE; break; } \
        case DType::BFloat16: { using scalar_type = bfloat16; MATH_CODE; break; } \
        case DType::Int8:     { using scalar_type = int8_t; MATH_CODE; break; } \
        default: throw std::runtime_error("Unsupported dtype"); \
    }
