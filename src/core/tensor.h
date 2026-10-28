#pragma once
#include "types.h"
#include "storage.h"
#include "node.h"
#include "grad_mode.h"
#include "utils.h"
#include <vector>
#include <memory>
#include <iostream>

#include <type_traits>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::shared_ptr<Storage> storage;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t global_offset = 0;
    Device device;
    DType dtype;

    bool requires_grad = true;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<Node> grad_fn;
    std::shared_ptr<Node> grad_accumulator;

    explicit Tensor(std::vector<int64_t> s, Device d = Device::CPU, DType type = DType::Float32, bool req_grad=true);

    explicit Tensor(std::vector<int64_t> s, bool req_grad) 
        : Tensor(s, Device::CPU, DType::Float32, req_grad) {}

    // Lightweight constructor for sharing existing storage without reallocating
    Tensor(std::shared_ptr<Storage> stor, std::vector<int64_t> s, std::vector<int64_t> str, int64_t offset, Device d, DType type, bool req_grad);

    template<typename T = float>
    T* data_ptr() {
        if (!storage || !storage->data) return nullptr;
        check_element_type<T>();
        char* base = static_cast<char*>(storage->data) + global_offset * dtype_size(dtype);
        if constexpr (std::is_void_v<T>) {
            return base;
        } else {
            return reinterpret_cast<T*>(base);
        }
    }

    template<typename T = float>
    const T* data_ptr() const {
        if (!storage || !storage->data) return nullptr;
        check_element_type<T>();
        const char* base = static_cast<const char*>(storage->data) + global_offset * dtype_size(dtype);
        if constexpr (std::is_void_v<T>) {
            return base;
        } else {
            return reinterpret_cast<const T*>(base);
        }
    }
    // Typed access must match the dtype (see has_dtype_v); raw byte access is always allowed
    template<typename T>
    void check_element_type() const {
        using U = std::remove_cv_t<T>;
        if constexpr (has_dtype_v<U>) {
            if (dtype != dtype_of<U>()) {
                throw std::runtime_error("Tensor::data_ptr: element type does not match the tensor's dtype");
            }
        }
    }

    template<typename T> T data_at(int64_t i) const {
        if (device != Device::CPU) {
            throw std::runtime_error("Tensor::data_at: cannot directly access device tensor on CPU. Call .cpu() first.");
        }
        return data_ptr<T>()[i];
    }
    template<typename T> T grad_at(int64_t i) const {
        if (!grad) return static_cast<T>(0);
        if (grad->device != Device::CPU) {
            throw std::runtime_error("Tensor::grad_at: cannot directly access device grad on CPU. Call .cpu() first.");
        }
        return grad->data_ptr<T>()[i];
    }

    std::shared_ptr<Tensor> to(Device target_device, int device_id = 0);
    std::shared_ptr<Tensor> to(Device target_device, DType target_dtype, int device_id = 0);
    std::shared_ptr<Tensor> to(DType target_dtype);
    std::shared_ptr<Tensor> cuda() { return to(Device::CUDA); }
    std::shared_ptr<Tensor> cpu() { return to(Device::CPU); }
    std::shared_ptr<Tensor> tpu(int device_id = 0) { return to(Device::TPU, device_id); }
    std::shared_ptr<Tensor> mps() { return to(Device::MPS); }

    int64_t flat_idx(const std::vector<int64_t>& idx) const;
    bool is_contiguous() const;
    int64_t ndim() const;
    int64_t size() const;
    void zero_grad();

    void copy_from_vector(const std::vector<float>& values);
    void fill_(float value);

    std::shared_ptr<Tensor> _view(const std::vector<int64_t>& new_shape);
    std::shared_ptr<Tensor> _reshape(const std::vector<int64_t>& new_shape);
    std::shared_ptr<Tensor> reshape(const std::vector<int64_t>& new_shape);

    std::shared_ptr<Tensor> _transpose(int64_t ax0, int64_t ax1);
    std::shared_ptr<Tensor> transpose(int64_t ax0, int64_t ax1);

    void _show_recursive(int64_t dim, int64_t pos, bool grad_mode) const;
    void _show_shape() const;
    void show() const;
    void show_grad() const;

    void backward(bool retain_graph = false);
};

// Host<->accelerator copies made through Tensor::to, counted process-wide. Every copy inside a
// training step is a CPU fallback or a sync point, so these counters show where an op lacks a
// device kernel. Reset before the region you want to measure.
struct TransferStats {
    uint64_t to_device_count = 0, to_device_bytes = 0;
    uint64_t to_host_count = 0, to_host_bytes = 0;
};
TransferStats transfer_stats();
void reset_transfer_stats();

std::shared_ptr<Tensor> make_contiguous(const std::shared_ptr<Tensor>& a);
// dst += src for gradient accumulation. A null dst becomes a dense copy of src, and a strided
// dst is replaced by a dense copy first, so never pass a view you expect to be written through.
void tensor_add_inplace(std::shared_ptr<Tensor>& dst, const std::shared_ptr<Tensor>& src);
// Whether a gradient can be taken over and written in place: g is the only reference to the tensor
// and to its storage, and it is a dense buffer covering exactly that storage. Producers hand grads
// over by move, so this holds for a fresh gradient and fails for one shared by several edges
// (AddBackward) or viewing another tensor's memory.
bool grad_is_exclusive(const std::shared_ptr<Tensor>& g);
Edge get_grad_edge(const std::shared_ptr<Tensor>& t);
