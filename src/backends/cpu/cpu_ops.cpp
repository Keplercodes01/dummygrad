#include "ops.h"
#include "kernels/kernels.h"
#include "parallel.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace cpu {

    void fill(Tensor& tensor, float value) {
        if (tensor.dtype == DType::Float32 && tensor.is_contiguous()) {
            float* ptr = tensor.data_ptr<float>();
            parallel_for(0, tensor.size(), 65536, [=](int64_t begin, int64_t end) { std::fill(ptr + begin, ptr + end, value); });
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
        if (dst.dtype != src.dtype)
            throw std::runtime_error("add_inplace requires identical dtypes");
        if (dst.size() != src.size())
            throw std::runtime_error("add_inplace requires identical sizes");
        if (!dst.is_contiguous() || !src.is_contiguous()) 
            throw std::runtime_error("add_inplace requires contiguous tensors");

        if (dst.dtype == DType::Float32) {
            float* d_ptr = dst.data_ptr<float>();
            const float* s_ptr = src.data_ptr<float>();
            parallel_for(0, dst.size(), 32768, [=](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) d_ptr[i] += s_ptr[i];
            });
            return;
        }

        Dispatch_DType(dst.dtype, {
            scalar_type* d_ptr = dst.data_ptr<scalar_type>();
            const scalar_type* s_ptr = src.data_ptr<scalar_type>();
            for (int64_t i = 0; i < dst.size(); ++i) d_ptr[i] += s_ptr[i];
        });
    }

    void copy_from_vector(Tensor& tensor, const std::vector<float>& values) {
        if (values.size() < static_cast<size_t>(tensor.size())) {
            throw std::runtime_error("copy_from_vector: values vector smaller than tensor size");
        }
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
        if (dst.dtype != src.dtype)
            throw std::runtime_error("make_contiguous requires identical dtypes");
        if (dst.size() != src.size())
            throw std::runtime_error("make_contiguous requires identical sizes");
        if (src.size() == 0) return;

        if (src.dtype == DType::Float32) {
            auto grid = kern::make_grid(src.shape, {src.strides, make_strides(src.shape)});
            kern::host::copy(src.data_ptr<float>(), dst.data_ptr<float>(), grid);
            return;
        }

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
