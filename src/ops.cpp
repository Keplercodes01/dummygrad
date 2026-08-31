#include "ops.h"
#include <stdexcept>

// Forward declarations for the pure C++ CPU kernels
namespace cpu {
    void fill(Tensor& tensor, float value);
    void add_inplace(Tensor& dst, const Tensor& src);
    void copy_from_vector(Tensor& tensor, const std::vector<float>& values);
    void make_contiguous(Tensor& dst, const Tensor& src);
}

namespace ops {
    void fill(Tensor& tensor, float value) {
        switch (tensor.device) {
            case Device::CPU: cpu::fill(tensor, value); break;
            default: throw std::runtime_error("ops::fill: Device not supported");
        }
    }

    void add_inplace(Tensor& dst, const Tensor& src) {
        if (dst.device != src.device) throw std::runtime_error("ops::add_inplace: Device mismatch");
        switch (dst.device) {
            case Device::CPU: cpu::add_inplace(dst, src); break;
            default: throw std::runtime_error("ops::add_inplace: Device not supported");
        }
    }

    void copy_from_vector(Tensor& tensor, const std::vector<float>& values) {
        switch (tensor.device) {
            case Device::CPU: cpu::copy_from_vector(tensor, values); break;
            default: throw std::runtime_error("ops::copy_from_vector: Device not supported");
        }
    }

    void make_contiguous(Tensor& dst, const Tensor& src) {
        if (dst.device != src.device) throw std::runtime_error("ops::make_contiguous: Device mismatch");
        switch (dst.device) {
            case Device::CPU: cpu::make_contiguous(dst, src); break;
            default: throw std::runtime_error("ops::make_contiguous: Device not supported");
        }
    }
}
