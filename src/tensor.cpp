#include "tensor.h"
#include "functional.h"
#include "ops.h"
#include "autograd.h"
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

int64_t Tensor::flat_idx(const std::vector<int64_t>& idx) const {
    int64_t pos = 0; 
    for (int64_t i = 0; i < (int64_t)idx.size(); i++) 
        pos += idx[i] * strides[i];
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

    auto t = std::make_shared<Tensor>(new_shape, device, dtype, false);
    t->storage = storage;
    t->strides = make_strides(new_shape);
    t->global_offset = global_offset;
    return t;
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
        auto t = std::make_shared<Tensor>(new_shape, device, dtype, requires_grad);
        t->storage = copy->storage;
        t->strides = make_strides(new_shape);
        t->global_offset = 0;
        return t;
    }
}
std::shared_ptr<Tensor> Tensor::reshape(const std::vector<int64_t>& new_shape) { return _reshape(new_shape); }

std::shared_ptr<Tensor> Tensor::_transpose(int64_t ax0, int64_t ax1) {
    if (ax0 < 0 || ax0 >= ndim() || ax1 < 0 || ax1 >= ndim()) throw std::runtime_error("transpose: axis out of shape");
    auto t = std::make_shared<Tensor>(shape, device, dtype, requires_grad);
    t->storage = storage;
    t->shape = shape;
    t->strides = strides;
    t->global_offset = global_offset;
    std::swap(t->shape[ax0], t->shape[ax1]);
    std::swap(t->strides[ax0], t->strides[ax1]);
    return t;
}
std::shared_ptr<Tensor> Tensor::transpose(int64_t ax0, int64_t ax1) { return _transpose(ax0, ax1); }

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
void Tensor::show() const { _show_recursive(0, 0, false); _show_shape(); }
void Tensor::show_grad() const { _show_recursive(0, 0, true);  _show_shape(); }

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
    this->fill_(1.0f);
    if (this->grad_fn) {
        AutogradEngine::get().execute(this->grad_fn, this->shape, this->device, this->dtype, retain_graph);
        if (!retain_graph) this->grad_fn = nullptr;
    }
}
