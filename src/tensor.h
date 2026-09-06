#pragma once
#include "types.h"
#include "storage.h"
#include "node.h"
#include "utils.h"
#include <vector>
#include <memory>
#include <iostream>

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


    template<typename T> T* data_ptr() { return static_cast<T*>(storage->data) + global_offset; }
    template<typename T> const T* data_ptr() const { return static_cast<const T*>(storage->data) + global_offset; }
    template<typename T> T data_at(int64_t i) const { return data_ptr<T>()[i]; }
    template<typename T> T grad_at(int64_t i) const { return grad ? grad->data_ptr<T>()[i] : static_cast<T>(0); }

    std::shared_ptr<Tensor> to(Device target_device);
    std::shared_ptr<Tensor> cuda() { return to(Device::CUDA); }
    std::shared_ptr<Tensor> cpu() { return to(Device::CPU); }

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

std::shared_ptr<Tensor> make_contiguous(const std::shared_ptr<Tensor>& a);
void tensor_add_inplace(std::shared_ptr<Tensor>& dst, const std::shared_ptr<Tensor>& src);
Edge get_grad_edge(const std::shared_ptr<Tensor>& t);
