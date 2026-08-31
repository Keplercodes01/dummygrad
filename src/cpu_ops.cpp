#include "ops.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX__) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cpu {

    void fill(Tensor& tensor, float value) {
        if (tensor.dtype == DType::Float32 && tensor.is_contiguous()) {
            float* ptr = tensor.data_ptr<float>();
            int64_t size = tensor.size();
            int64_t i = 0;
#if defined(__ARM_NEON)
            float32x4_t val_vec = vdupq_n_f32(value);
            for (; i <= size - 16; i += 16) {
                vst1q_f32(ptr + i, val_vec);
                vst1q_f32(ptr + i + 4, val_vec);
                vst1q_f32(ptr + i + 8, val_vec);
                vst1q_f32(ptr + i + 12, val_vec);
            }
            for (; i <= size - 4; i += 4) {
                vst1q_f32(ptr + i, val_vec);
            }
#elif defined(__AVX__) || defined(__x86_64__) || defined(_M_X64)
            __m256 val_vec = _mm256_set1_ps(value);
            for (; i <= size - 32; i += 32) {
                _mm256_storeu_ps(ptr + i, val_vec);
                _mm256_storeu_ps(ptr + i + 8, val_vec);
                _mm256_storeu_ps(ptr + i + 16, val_vec);
                _mm256_storeu_ps(ptr + i + 24, val_vec);
            }
            for (; i <= size - 8; i += 8) {
                _mm256_storeu_ps(ptr + i, val_vec);
            }
#endif
            for (; i < size; i++) ptr[i] = value;
            return;
        }

        Dispatch_DType(tensor.dtype, {
            scalar_type* ptr = tensor.data_ptr<scalar_type>();
            scalar_type cast_val = static_cast<scalar_type>(value);
            int64_t size = tensor.size();
            
            if (tensor.is_contiguous()) {
                for (int64_t i = 0; i < size; i++) ptr[i] = cast_val;
            } else {
                int64_t ndim = tensor.ndim();
                std::vector<int64_t> coords(ndim, 0);
                for (int64_t i = 0; i < size; i++) {
                    int64_t flat_idx = 0;
                    for (int64_t d = 0; d < ndim; d++) flat_idx += coords[d] * tensor.strides[d];
                    ptr[flat_idx] = cast_val;
                    
                    for (int64_t d = ndim - 1; d >= 0; d--) {
                        coords[d]++;
                        if (coords[d] < tensor.shape[d]) break;
                        coords[d] = 0;
                    }
                }
            }
        });
    }

    void add_inplace(Tensor& dst, const Tensor& src) {
        if (!dst.is_contiguous() || !src.is_contiguous()) 
            throw std::runtime_error("add_inplace requires contiguous tensors");

        if (dst.dtype == DType::Float32) {
            float* d_ptr = dst.data_ptr<float>();
            const float* s_ptr = src.data_ptr<float>();
            int64_t size = dst.size();
            int64_t i = 0;
#if defined(__ARM_NEON)
            for (; i <= size - 16; i += 16) {
                float32x4_t a0 = vld1q_f32(d_ptr + i);
                float32x4_t b0 = vld1q_f32(s_ptr + i);
                vst1q_f32(d_ptr + i, vaddq_f32(a0, b0));

                float32x4_t a1 = vld1q_f32(d_ptr + i + 4);
                float32x4_t b1 = vld1q_f32(s_ptr + i + 4);
                vst1q_f32(d_ptr + i + 4, vaddq_f32(a1, b1));

                float32x4_t a2 = vld1q_f32(d_ptr + i + 8);
                float32x4_t b2 = vld1q_f32(s_ptr + i + 8);
                vst1q_f32(d_ptr + i + 8, vaddq_f32(a2, b2));

                float32x4_t a3 = vld1q_f32(d_ptr + i + 12);
                float32x4_t b3 = vld1q_f32(s_ptr + i + 12);
                vst1q_f32(d_ptr + i + 12, vaddq_f32(a3, b3));
            }
            for (; i <= size - 4; i += 4) {
                float32x4_t a = vld1q_f32(d_ptr + i);
                float32x4_t b = vld1q_f32(s_ptr + i);
                vst1q_f32(d_ptr + i, vaddq_f32(a, b));
            }
#elif defined(__AVX__) || defined(__x86_64__) || defined(_M_X64)
            for (; i <= size - 32; i += 32) {
                __m256 a0 = _mm256_loadu_ps(d_ptr + i);
                __m256 b0 = _mm256_loadu_ps(s_ptr + i);
                _mm256_storeu_ps(d_ptr + i, _mm256_add_ps(a0, b0));

                __m256 a1 = _mm256_loadu_ps(d_ptr + i + 8);
                __m256 b1 = _mm256_loadu_ps(s_ptr + i + 8);
                _mm256_storeu_ps(d_ptr + i + 8, _mm256_add_ps(a1, b1));

                __m256 a2 = _mm256_loadu_ps(d_ptr + i + 16);
                __m256 b2 = _mm256_loadu_ps(s_ptr + i + 16);
                _mm256_storeu_ps(d_ptr + i + 16, _mm256_add_ps(a2, b2));

                __m256 a3 = _mm256_loadu_ps(d_ptr + i + 24);
                __m256 b3 = _mm256_loadu_ps(s_ptr + i + 24);
                _mm256_storeu_ps(d_ptr + i + 24, _mm256_add_ps(a3, b3));
            }
            for (; i <= size - 8; i += 8) {
                __m256 a = _mm256_loadu_ps(d_ptr + i);
                __m256 b = _mm256_loadu_ps(s_ptr + i);
                _mm256_storeu_ps(d_ptr + i, _mm256_add_ps(a, b));
            }
#endif
            for (; i < size; ++i) d_ptr[i] += s_ptr[i];
            return;
        }

        Dispatch_DType(dst.dtype, {
            scalar_type* d_ptr = dst.data_ptr<scalar_type>();
            const scalar_type* s_ptr = src.data_ptr<scalar_type>();
            for (int64_t i = 0; i < dst.size(); ++i) d_ptr[i] += s_ptr[i];
        });
    }

    void copy_from_vector(Tensor& tensor, const std::vector<float>& values) {
        if (tensor.dtype == DType::Float32 && tensor.is_contiguous()) {
            std::memcpy(tensor.data_ptr<float>(), values.data(), tensor.size() * sizeof(float));
            return;
        }

        Dispatch_DType(tensor.dtype, {
            scalar_type* ptr = tensor.data_ptr<scalar_type>();
            int64_t size = tensor.size();

            if (tensor.is_contiguous()) {
                for (int64_t i = 0; i < size; i++) ptr[i] = static_cast<scalar_type>(values[i]);
            } else {
                int64_t ndim = tensor.ndim();
                std::vector<int64_t> coords(ndim, 0);
                for (int64_t i = 0; i < size; i++) {
                    int64_t flat_idx = 0;
                    for (int64_t d = 0; d < ndim; d++) flat_idx += coords[d] * tensor.strides[d];
                    ptr[flat_idx] = static_cast<scalar_type>(values[i]);
                    
                    for (int64_t d = ndim - 1; d >= 0; d--) {
                        coords[d]++;
                        if (coords[d] < tensor.shape[d]) break;
                        coords[d] = 0;
                    }
                }
            }
        });
    }

    void make_contiguous(Tensor& dst, const Tensor& src) {
        Dispatch_DType(dst.dtype, {
            scalar_type* out_ptr = dst.data_ptr<scalar_type>();
            const scalar_type* src_data = src.data_ptr<scalar_type>();
            int64_t size = src.size();
            
            if (src.is_contiguous()) {
                std::memcpy(out_ptr, src_data, size * sizeof(scalar_type));
                return;
            }

            int64_t ndim = src.ndim();
            std::vector<int64_t> coords(ndim, 0);
            for (int64_t i = 0; i < size; i++) {
                int64_t src_idx = 0;
                for (int64_t d = 0; d < ndim; d++) src_idx += coords[d] * src.strides[d];
                out_ptr[i] = src_data[src_idx];
                
                for (int64_t d = ndim - 1; d >= 0; d--) {
                    coords[d]++;
                    if (coords[d] < src.shape[d]) break;
                    coords[d] = 0;
                }
            }
        });
    }
}
