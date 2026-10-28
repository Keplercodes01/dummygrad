#include "ops.h"
#include <stdexcept>

// After an in-place write: bump the storage's version (cached casts are now stale) and, for a TPU
// tensor, drop its TPU buffer. TPU tensors keep their authoritative copy in host memory; the buffer
// is a copy that host writes make stale.
static void written(Tensor& t) {
    if (!t.storage) return;
    t.storage->bump_version();
    if (t.device == Device::TPU) t.storage->tpu_handle.reset();
}

// Forward declarations for the pure C++ CPU kernels
namespace cpu {
    void fill(Tensor& tensor, float value);
    void add_inplace(Tensor& dst, const Tensor& src);
    void copy_from_vector(Tensor& tensor, const std::vector<float>& values);
    void make_contiguous(Tensor& dst, const Tensor& src);
}

#ifdef USE_CUDA
namespace cuda {
    void fill(Tensor& tensor, float value);
    void add_inplace(Tensor& dst, const Tensor& src);
    void copy_from_vector(Tensor& tensor, const std::vector<float>& values);
    void make_contiguous(Tensor& dst, const Tensor& src);
}
#endif

namespace ops {
    void fill(Tensor& tensor, float value) {
        switch (tensor.device) {
            case Device::CPU:
            case Device::MPS:
            case Device::TPU:
                cpu::fill(tensor, value);
                written(tensor);
                break;
#ifdef USE_CUDA
            case Device::CUDA: cuda::fill(tensor, value); written(tensor); break;
#endif
            default: throw std::runtime_error("ops::fill: Device not supported");
        }
    }

    void add_inplace(Tensor& dst, const Tensor& src) {
        if (dst.device != src.device) throw std::runtime_error("ops::add_inplace: Device mismatch");
        switch (dst.device) {
            case Device::CPU:
            case Device::MPS:
            case Device::TPU:
                cpu::add_inplace(dst, src);
                written(dst);
                break;
#ifdef USE_CUDA
            case Device::CUDA: cuda::add_inplace(dst, src); written(dst); break;
#endif
            default: throw std::runtime_error("ops::add_inplace: Device not supported");
        }
    }

    void copy_from_vector(Tensor& tensor, const std::vector<float>& values) {
        switch (tensor.device) {
            case Device::CPU:
            case Device::MPS:
            case Device::TPU:
                cpu::copy_from_vector(tensor, values);
                written(tensor);
                break;
#ifdef USE_CUDA
            case Device::CUDA: cuda::copy_from_vector(tensor, values); written(tensor); break;
#endif
            default: throw std::runtime_error("ops::copy_from_vector: Device not supported");
        }
    }

    void make_contiguous(Tensor& dst, const Tensor& src) {
        if (dst.device != src.device) throw std::runtime_error("ops::make_contiguous: Device mismatch");
        switch (dst.device) {
            case Device::CPU:
            case Device::MPS:
            case Device::TPU:
                cpu::make_contiguous(dst, src);
                written(dst);
                break;
#ifdef USE_CUDA
            case Device::CUDA: cuda::make_contiguous(dst, src); written(dst); break;
#endif
            default: throw std::runtime_error("ops::make_contiguous: Device not supported");
        }
    }
}
