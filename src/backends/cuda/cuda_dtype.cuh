#pragma once
// Device-side element types for the float dtypes, for .cu files only (cuda_common.h is also
// compiled by the host compiler, so device code stays out of it). Kernels written once over
// scalar_t load with to_float and store with from_float<scalar_t>, doing their arithmetic in fp32.

#include "types.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace cuda {

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

// Round to nearest even, like the CPU conversions in types.h
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

inline bool is_float_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}

}  // namespace cuda

// Runs the statements with scalar_t bound to the device element type of a float dtype; throws
// for any other dtype, naming who
#define DG_CUDA_DISPATCH_FLOAT(dtype, who, ...)                                                       \
    switch (dtype) {                                                                                  \
        case DType::Float32: { using scalar_t = float; __VA_ARGS__; break; }                          \
        case DType::Float16: { using scalar_t = __half; __VA_ARGS__; break; }                         \
        case DType::BFloat16: { using scalar_t = __nv_bfloat16; __VA_ARGS__; break; }                 \
        default: throw std::runtime_error(std::string(who) + ": expected Float32, Float16 or BFloat16"); \
    }
