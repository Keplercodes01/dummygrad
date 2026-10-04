#include "tensor.h"
#include "functional.h"
#include "ops.h"
#include "autograd.h"
#include "tpu/pjrt_client.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <unordered_map>
#include <queue>
#include <functional>
#include <stdexcept>

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
    ops::add_inplace(*dst, *src);
}

Edge get_grad_edge(const std::shared_ptr<Tensor>& t) {
    if (!t || !t->requires_grad) return {nullptr, 0};
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
    if (this->device == target_device) {
        return shared_from_this();
    }

    if (!this->is_contiguous()) {
        auto contiguous_self = make_contiguous(const_cast<Tensor*>(this)->shared_from_this());
        return contiguous_self->to(target_device, device_id);
    }

    auto out = std::make_shared<Tensor>(this->shape, target_device, this->dtype, this->requires_grad);
    size_t copy_bytes = this->size() * dtype_size(this->dtype);

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
        return this->to(Device::CPU)->to(target_device, device_id);
    }

    return out;
}
