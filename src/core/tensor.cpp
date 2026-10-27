#include "tensor.h"
#include "functional.h"
#include "ops.h"
#include "autograd.h"
#include "tpu/pjrt_client.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <unordered_map>
#include <atomic>
#include <queue>
#include <functional>
#include <stdexcept>

static std::atomic<uint64_t> g_to_device_count{0}, g_to_device_bytes{0};
static std::atomic<uint64_t> g_to_host_count{0}, g_to_host_bytes{0};

TransferStats transfer_stats() {
    TransferStats s;
    s.to_device_count = g_to_device_count;
    s.to_device_bytes = g_to_device_bytes;
    s.to_host_count = g_to_host_count;
    s.to_host_bytes = g_to_host_bytes;
    return s;
}

void reset_transfer_stats() {
    g_to_device_count = g_to_device_bytes = g_to_host_count = g_to_host_bytes = 0;
}

Tensor::Tensor(std::vector<int64_t> s, Device d, DType type, bool req_grad)
    : shape(std::move(s)), global_offset(0), device(d), dtype(type), requires_grad(req_grad) {  
    int64_t total = 1;  
    for (int64_t dim : shape) total *= dim;
    storage = std::make_shared<Storage>(total, d, type);
    strides = make_strides(shape);
}

Tensor::Tensor(std::shared_ptr<Storage> stor, std::vector<int64_t> s, std::vector<int64_t> str, int64_t offset, Device d, DType type, bool req_grad)
    : storage(std::move(stor)), shape(std::move(s)), strides(std::move(str)), global_offset(offset), device(d), dtype(type), requires_grad(req_grad) {}

int64_t Tensor::flat_idx(const std::vector<int64_t>& idx) const {
    int64_t pos = 0; 
    for (int64_t i = 0; i < (int64_t)idx.size(); i++) {
        int64_t coord = idx[i];
        if (coord < 0) coord += shape[i];
        if (coord < 0 || coord >= shape[i]) throw std::out_of_range("Tensor::flat_idx: index out of bounds");
        pos += coord * strides[i];
    }
    return pos;
}

bool Tensor::is_contiguous() const { return strides == make_strides(shape); }
int64_t Tensor::ndim() const { return (int64_t)shape.size(); }
int64_t Tensor::size() const {
    int64_t total = 1;
    for (int64_t dim : shape) total *= dim;
    return total;
}
void Tensor::zero_grad() { grad = nullptr; }

void Tensor::copy_from_vector(const std::vector<float>& values) {
    if ((int64_t)values.size() != size()) throw std::runtime_error("copy_from_vector: size mismatch");
    ops::copy_from_vector(*this, values);
}
void Tensor::fill_(float value) { ops::fill(*this, value); }

std::shared_ptr<Tensor> Tensor::_view(const std::vector<int64_t>& new_shape) {
    int64_t new_total = 1; 
    for (int64_t d : new_shape) new_total *= d;
    if (new_total != size()) throw std::runtime_error("view: element count mismatch..."); 
    if (!is_contiguous()) throw std::runtime_error("view: tensor is not contiguous, use reshape() or make_contiguous()");

    return std::make_shared<Tensor>(storage, new_shape, make_strides(new_shape), global_offset, device, dtype, requires_grad);
}

std::shared_ptr<Tensor> Tensor::_reshape(const std::vector<int64_t>& new_shape) {
    int64_t new_total = 1; 
    for (int64_t d : new_shape) new_total *= d;
    if (new_total != size()) throw std::runtime_error("reshape: element count mismatch..."); 

    if (is_contiguous()) {
        return _view(new_shape);
    } else {   
        auto copy = std::make_shared<Tensor>(shape, device, dtype, requires_grad);
        ops::make_contiguous(*copy, *this);
        return std::make_shared<Tensor>(copy->storage, new_shape, make_strides(new_shape), 0, device, dtype, requires_grad);
    }
}
std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int64_t>& new_shape) {
    try {
        return ::reshape(const_cast<Tensor*>(this)->shared_from_this(), new_shape);
    } catch (const std::bad_weak_ptr&) {
        return _reshape(new_shape);
    }
}

std::shared_ptr<Tensor> Tensor::_transpose(int64_t ax0, int64_t ax1) {
    if (ax0 < 0) ax0 += ndim();
    if (ax1 < 0) ax1 += ndim();
    if (ax0 < 0 || ax0 >= ndim() || ax1 < 0 || ax1 >= ndim()) throw std::runtime_error("transpose: axis out of range");

    auto new_shape = shape;
    auto new_strides = strides;
    std::swap(new_shape[ax0], new_shape[ax1]);
    std::swap(new_strides[ax0], new_strides[ax1]);

    return std::make_shared<Tensor>(storage, std::move(new_shape), std::move(new_strides), global_offset, device, dtype, requires_grad);
}
std::shared_ptr<Tensor> Tensor::transpose(int64_t ax0, int64_t ax1) {
    try {
        return ::transpose(const_cast<Tensor*>(this)->shared_from_this(), ax0, ax1);
    } catch (const std::bad_weak_ptr&) {
        return _transpose(ax0, ax1);
    }
}

void Tensor::_show_recursive(int64_t dim, int64_t pos, bool grad_mode) const {
    if (dim == ndim() - 1) {
        std::cout << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            int64_t p = pos + i * strides[dim];
            if (grad_mode) {
                float g_val = (grad) ? grad->data_ptr<float>()[p] : 0.0f;
                std::cout << g_val;
            } else {
                std::cout << data_ptr<float>()[p];
            }
            if (i < shape[dim] - 1) std::cout << ", ";
        }
        std::cout << "]";
    } else {
        std::cout << "[";
        for (int64_t i = 0; i < shape[dim]; ++i) {
            if (i > 0) {
                std::cout << ",\n";
                for (int64_t d = 0; d <= dim; ++d) std::cout << " ";
            }
            _show_recursive(dim + 1, pos + i * strides[dim], grad_mode);
        }
        std::cout << "]";
    }
}
void Tensor::_show_shape() const {
    std::cout << ", shape=(";
    for (int64_t i = 0; i < ndim(); ++i) {
        std::cout << shape[i];
        if (i < ndim() - 1) std::cout << ",";
    }
    std::cout << ")\n";
}
void Tensor::show() const {
    if (ndim() == 0) {
        std::cout << data_ptr<float>()[0];
        _show_shape();
        return;
    }
    _show_recursive(0, 0, false);
    _show_shape();
}

void Tensor::show_grad() const {
    if (ndim() == 0) {
        float g_val = (grad) ? grad->data_ptr<float>()[0] : 0.0f;
        std::cout << g_val;
        _show_shape();
        return;
    }
    _show_recursive(0, 0, true);
    _show_shape();
}

std::shared_ptr<Tensor> make_contiguous(const std::shared_ptr<Tensor>& a) {
    if (a->is_contiguous()) return a;
    auto out = std::make_shared<Tensor>(a->shape, a->device, a->dtype, a->requires_grad);
    ops::make_contiguous(*out, *a);
    return out;
}

void tensor_add_inplace(std::shared_ptr<Tensor>& dst, const std::shared_ptr<Tensor>& src) {
    if (!src) return;
    if (!dst) {
        dst = std::make_shared<Tensor>(src->shape, src->device, src->dtype, false);
        ops::make_contiguous(*dst, *src); // acts as deep copy
        return;
    }
    // Backends add element-wise over flat memory, so both operands must be dense.
    // Gradients routinely arrive as strided views (TransposeBackward), e.g. a tied
    // embedding weight that is also used transposed as the LM head.
    if (!dst->is_contiguous()) dst = make_contiguous(dst);
    if (!src->is_contiguous()) {
        ops::add_inplace(*dst, *make_contiguous(src));
        return;
    }
    ops::add_inplace(*dst, *src);
}

Edge get_grad_edge(const std::shared_ptr<Tensor>& t) {
    if (!t || !t->requires_grad || !grad_enabled()) return {nullptr, 0};
    if (t->grad_fn) return {t->grad_fn, 0};
    if (!t->grad_accumulator) {
        t->grad_accumulator = std::make_shared<AccumulateGrad>(t);
    }
    return {t->grad_accumulator, 0};
}

void Tensor::backward(bool retain_graph) {
    if (this->size() != 1) throw std::runtime_error("backward: tensor must be scalar (size 1)");
    if (this->requires_grad) {
        if (!this->grad) {
            this->grad = std::make_shared<Tensor>(this->shape, this->device, this->dtype, false);
        }
        this->grad->fill_(1.0f);
    }
    if (this->grad_fn) {
        AutogradEngine::get().execute(this->grad_fn, this->shape, this->device, this->dtype, retain_graph);
        if (!retain_graph) this->grad_fn = nullptr;
    }
}

std::shared_ptr<Tensor> Tensor::to(Device target_device, int device_id) {
    return to(target_device, this->dtype, device_id);
}

std::shared_ptr<Tensor> Tensor::to(DType target_dtype) {
    return to(this->device, target_dtype, 0);
}

std::shared_ptr<Tensor> Tensor::to(Device target_device, DType target_dtype, int device_id) {
    if (this->device == target_device && this->dtype == target_dtype) {
        return shared_from_this();
    }

    if (!this->is_contiguous()) {
        auto contiguous_self = make_contiguous(const_cast<Tensor*>(this)->shared_from_this());
        return contiguous_self->to(target_device, target_dtype, device_id);
    }

#ifdef USE_CUDA
    // Float dtype conversion on the GPU itself, no host round trip
    auto is_float = [](DType d) { return d == DType::Float32 || d == DType::Float16 || d == DType::BFloat16; };
    if (this->device == Device::CUDA && target_device == Device::CUDA && is_float(this->dtype) && is_float(target_dtype)) {
        auto out = std::make_shared<Tensor>(this->shape, Device::CUDA, target_dtype, this->requires_grad);
        cuda::convert(this->data_ptr<void>(), this->dtype, out->data_ptr<void>(), target_dtype, this->size());
        return out;
    }
#endif

    // Step 1: If dtype conversion is needed and tensor is on an accelerator (CUDA/TPU/MPS),
    // stage through CPU to perform the dtype conversion.
    if (this->dtype != target_dtype && this->device != Device::CPU) {
        auto host_tensor = this->to(Device::CPU, this->dtype, 0);
        auto converted_host = host_tensor->to(Device::CPU, target_dtype, 0);
        if (target_device == Device::CPU) {
            return converted_host;
        }
        return converted_host->to(target_device, target_dtype, device_id);
    }

    // Step 2: DType conversion on CPU (same device CPU -> CPU)
    if (this->dtype != target_dtype && this->device == Device::CPU && target_device == Device::CPU) {
        auto out = std::make_shared<Tensor>(this->shape, Device::CPU, target_dtype, this->requires_grad);
        int64_t n = this->size();
        const void* src = this->data_ptr<void>();
        void* dst = out->data_ptr<void>();

        if (this->dtype == DType::Float32) {
            const float* s = reinterpret_cast<const float*>(src);
            if (target_dtype == DType::Float16) {
                float16* d = reinterpret_cast<float16*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = float16(s[i]);
            } else if (target_dtype == DType::BFloat16) {
                bfloat16* d = reinterpret_cast<bfloat16*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = bfloat16(s[i]);
            } else if (target_dtype == DType::Int8) {
                int8_t* d = reinterpret_cast<int8_t*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(s[i]);
            }
        } else if (this->dtype == DType::Float16) {
            const float16* s = reinterpret_cast<const float16*>(src);
            if (target_dtype == DType::Float32) {
                float* d = reinterpret_cast<float*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
            } else if (target_dtype == DType::BFloat16) {
                bfloat16* d = reinterpret_cast<bfloat16*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = bfloat16(static_cast<float>(s[i]));
            } else if (target_dtype == DType::Int8) {
                int8_t* d = reinterpret_cast<int8_t*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(static_cast<float>(s[i]));
            }
        } else if (this->dtype == DType::BFloat16) {
            const bfloat16* s = reinterpret_cast<const bfloat16*>(src);
            if (target_dtype == DType::Float32) {
                float* d = reinterpret_cast<float*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
            } else if (target_dtype == DType::Float16) {
                float16* d = reinterpret_cast<float16*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = float16(static_cast<float>(s[i]));
            } else if (target_dtype == DType::Int8) {
                int8_t* d = reinterpret_cast<int8_t*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<int8_t>(static_cast<float>(s[i]));
            }
        } else if (this->dtype == DType::Int8) {
            const int8_t* s = reinterpret_cast<const int8_t*>(src);
            if (target_dtype == DType::Float32) {
                float* d = reinterpret_cast<float*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
            } else if (target_dtype == DType::Float16) {
                float16* d = reinterpret_cast<float16*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = float16(static_cast<float>(s[i]));
            } else if (target_dtype == DType::BFloat16) {
                bfloat16* d = reinterpret_cast<bfloat16*>(dst);
                for (int64_t i = 0; i < n; ++i) d[i] = bfloat16(static_cast<float>(s[i]));
            }
        }
        return out;
    }

    // Step 3: If dtype needs conversion and device is changing from CPU to accelerator:
    if (this->dtype != target_dtype && this->device == Device::CPU) {
        auto converted_host = this->to(Device::CPU, target_dtype, 0);
        return converted_host->to(target_device, target_dtype, device_id);
    }

    // Step 4: Device transfer with matching dtype (this->dtype == target_dtype)
    auto out = std::make_shared<Tensor>(this->shape, target_device, this->dtype, this->requires_grad);
    size_t copy_bytes = this->size() * dtype_size(this->dtype);
    if (this->device == Device::CPU && target_device != Device::CPU) {
        g_to_device_count++;
        g_to_device_bytes += copy_bytes;
    } else if (this->device != Device::CPU && target_device == Device::CPU) {
        g_to_host_count++;
        g_to_host_bytes += copy_bytes;
    }

    if (this->device == Device::CPU && target_device == Device::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(out->data_ptr<void>(), this->data_ptr<void>(), copy_bytes, cudaMemcpyHostToDevice);
#else
        throw std::runtime_error("Tensor::to: CUDA requested but dummygrad was built without CUDA support");
#endif
    } else if (this->device == Device::CUDA && target_device == Device::CPU) {
#ifdef USE_CUDA
        cudaMemcpy(out->data_ptr<void>(), this->data_ptr<void>(), copy_bytes, cudaMemcpyDeviceToHost);
#else
        throw std::runtime_error("Tensor::to: CUDA requested but dummygrad was built without CUDA support");
#endif
    } else if (this->device == Device::CPU && target_device == Device::TPU) {
        auto& mgr = PJRTTPUManager::get();
        if (!mgr.is_available()) {
            throw std::runtime_error("Tensor::to(TPU): " + mgr.error_message());
        }
        std::memcpy(out->storage->data, this->data_ptr<void>(), copy_bytes);
        out->storage->tpu_handle = mgr.create_buffer_from_host(
            this->data_ptr<void>(), this->shape, this->dtype, device_id
        );
    } else if (this->device == Device::TPU && target_device == Device::CPU) {
        if (this->storage->tpu_handle) {
            PJRTTPUManager::get().copy_to_host(*this->storage->tpu_handle, out->data_ptr<void>(), copy_bytes);
        } else {
            std::memcpy(out->data_ptr<void>(), this->data_ptr<void>(), copy_bytes);
        }
    } else if (this->device == Device::CPU && target_device == Device::MPS) {
        // Direct zero-copy staging on Apple Silicon Unified Memory Architecture
        std::memcpy(out->data_ptr<void>(), this->data_ptr<void>(), copy_bytes);
    } else if (this->device == Device::MPS && target_device == Device::CPU) {
        std::memcpy(out->data_ptr<void>(), this->data_ptr<void>(), copy_bytes);
    } else {
        // Automatically stage through host CPU memory
        return this->to(Device::CPU, this->dtype, 0)->to(target_device, target_dtype, device_id);
    }

    return out;
}
