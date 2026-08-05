// Dummygrad 
#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>
#include <cstring>

// forward declarations
struct Node;
class Tensor;

// edge connecting Nodes in the execution graph
struct Edge {
    std::shared_ptr<Node> function;
    size_t input_slot{0};
};

// abstract base class for autograd execution nodes 
struct Node : public std::enable_shared_from_this<Node> {
    std::vector<Edge> next_edges;

    virtual ~Node() = default;

    // takes incoming output gradient tensors, returns outgoing gradient tensors for next_edges
    virtual std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) = 0;

    // release saved tensors to break reference cycles
    virtual void release_variables() {}

    void add_next_edge(const std::shared_ptr<Node>& fn, size_t input_slot = 0) {
        next_edges.push_back({fn, input_slot});
    }
};

// Storage buffer (holds ONLY data array)
struct Storage {
    std::vector<float> data;

    explicit Storage(int size) : data(size, 0.0f) {}
};

// Stride creation helper
inline std::vector<int> make_strides(const std::vector<int>& shape) {
    int ndim = (int)shape.size();
    assert(ndim > 0 && "make_strides: shape cannot be empty");
    std::vector<int> st(ndim);
    st[ndim - 1] = 1; 
    for (int i = ndim - 2; i >= 0; i--) 
        st[i] = st[i + 1] * shape[i + 1];
    return st;
}

// Unravel flat index helper
inline std::vector<int> unravel(int flat, const std::vector<int>& shape) {
    int ndim = (int)shape.size();
    if (ndim == 0) return {};
    assert(flat >= 0 && "unravel: flat index cannot be negative");
    std::vector<int> idx(ndim);
    for (int d = ndim - 1; d >= 0; d--) {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    return idx;
}

// Tensor Class 
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::shared_ptr<Storage> storage;
    std::vector<int> shape;
    std::vector<int> strides;
    int offset = 0;

    bool requires_grad = true;
    
    // gradients are Tensor objects 
    std::shared_ptr<Tensor> grad;

    std::shared_ptr<Node> grad_fn;
    std::shared_ptr<Node> grad_accumulator;

    explicit Tensor(std::vector<int> s, bool req_grad = true)
        : shape(std::move(s)), offset(0), requires_grad(req_grad) {  
        int total = 1;  
        for (int dim : shape) total *= dim;
        storage = std::make_shared<Storage>(total);
        strides = make_strides(shape);
    }

    // direct raw pointer accessors
    float* data_ptr() { return storage->data.data() + offset; }
    const float* data_ptr() const { return storage->data.data() + offset; }

    float* grad_ptr() {
        if (!grad) {
            grad = std::make_shared<Tensor>(shape, false);
        }
        return grad->data_ptr();
    }

    // Element access helpers
    float data_at(int i) const { return data_ptr()[i]; }
    float grad_at(int i) const { return grad ? grad->data_ptr()[i] : 0.0f; }

    // get a flat index of where the desired element is in the data buffer using batch indices and strides   
    int flat_idx(const std::vector<int>& idx) const {
        int pos = 0; 
        for (int i = 0; i < (int)idx.size(); i++) 
            pos += idx[i] * strides[i];
        return pos;
    }

    // check contiguity
    bool is_contiguous() const {
        return strides == make_strides(shape);
    }

    // get the total no. of dimensions of the shape array 
    int ndim() const { return (int)shape.size(); }

    // get the size of the tensor
    int size() const {
        int total = 1;
        for (int dim : shape) total *= dim;
        return total;
    }

    // zero out gradients 
    void zero_grad() {
        grad = nullptr;
    }    

    // fill the values manually if anyone wants to 
    void fill(const std::vector<float>& values) {
        if ((int)values.size() != size()) 
            throw std::runtime_error("fill: size mismatch");

        if (is_contiguous()) {
            std::memcpy(data_ptr(), values.data(), size() * sizeof(float));
        } else {
            float* ptr = data_ptr(); 
            for (int i = 0; i < size(); i++) 
                ptr[flat_idx(unravel(i, shape))] = values[i];
        }
    }

    // Low-level zero-overhead view (requires contiguous tensor)
    std::shared_ptr<Tensor> _view(const std::vector<int>& new_shape) {
        int new_total = 1; 
        for (int d : new_shape) new_total *= d;
        if (new_total != size())
            throw std::runtime_error("view: element count mismatch..."); 
        if (!is_contiguous())
            throw std::runtime_error("view: tensor is not contiguous, use reshape() or make_contiguous()");

        auto t = std::make_shared<Tensor>(new_shape, requires_grad);
        t->storage = storage;
        t->strides = make_strides(new_shape);
        t->offset = offset;
        return t;
    }

    // Low-level general reshape (handles contiguous & non-contiguous tensors)
    std::shared_ptr<Tensor> _reshape(const std::vector<int>& new_shape) {
        int new_total = 1; 
        for (int d : new_shape) new_total *= d;
        if (new_total != size())
            throw std::runtime_error("reshape: element count mismatch..."); 

        if (is_contiguous()) {
            return _view(new_shape);
        } else {   
            auto copy = std::make_shared<Tensor>(shape, requires_grad);
            float* copy_ptr = copy->data_ptr();
            const float* src_data = storage->data.data();
            for (int i = 0; i < size(); i++)
                copy_ptr[i] = src_data[offset + flat_idx(unravel(i, shape))];

            auto t = std::make_shared<Tensor>(new_shape, requires_grad);
            t->storage = copy->storage;
            t->strides = make_strides(new_shape);
            t->offset = 0;
            return t;
        }
    }
    // Backward-compatibility alias
    std::shared_ptr<Tensor> reshape(const std::vector<int>& new_shape) {
        return _reshape(new_shape);
    }

    // Low-level transpose
    std::shared_ptr<Tensor> _transpose(int ax0, int ax1) {
        if (ax0 < 0 || ax0 >= ndim() || ax1 < 0 || ax1 >= ndim())
            throw std::runtime_error("transpose: axis out of shape");

        auto t = std::make_shared<Tensor>(shape, requires_grad);
        t->storage = storage;
        t->shape = shape;
        t->strides = strides;
        t->offset = offset;

        std::swap(t->shape[ax0], t->shape[ax1]);
        std::swap(t->strides[ax0], t->strides[ax1]);
        return t;
    }
    // Backward-compatibility alias
    std::shared_ptr<Tensor> transpose(int ax0, int ax1) {
        return _transpose(ax0, ax1);
    }

    // show methods 
    void _show_recursive(int dim, int pos, bool grad_mode) const {
        if (dim == ndim() - 1) {
            std::cout << "[";
            for (int i = 0; i < shape[dim]; ++i) {
                int p = pos + i * strides[dim];
                if (grad_mode) {
                    float g_val = (grad) ? grad->data_ptr()[p] : 0.0f;
                    std::cout << g_val;
                } else {
                    std::cout << storage->data[offset + p];
                }
                if (i < shape[dim] - 1) std::cout << ", ";
            }
            std::cout << "]";
        } else {
            std::cout << "[";
            for (int i = 0; i < shape[dim]; ++i) {
                if (i > 0) {
                    std::cout << ",\n";
                    for (int d = 0; d <= dim; ++d) std::cout << " ";
                }
                _show_recursive(dim + 1, pos + i * strides[dim], grad_mode);
            }
            std::cout << "]";
        }
    }

    void _show_shape() const {
        std::cout << ", shape=(";
        for (int i = 0; i < ndim(); ++i) {
            std::cout << shape[i];
            if (i < ndim() - 1) std::cout << ",";
        }
        std::cout << ")\n";
    }

    void show()      const { _show_recursive(0, 0, false); _show_shape(); }
    void show_grad() const { _show_recursive(0, 0, true);  _show_shape(); }

    // Autograd Engine Entrypoint
    void backward(bool retain_graph = false);
};

// Make contiguous helper
inline std::shared_ptr<Tensor> make_contiguous(const std::shared_ptr<Tensor>& a) {
    if (a->is_contiguous()) return a;
    auto out = std::make_shared<Tensor>(a->shape, a->requires_grad);
    float* out_ptr = out->data_ptr();
    const float* src_data = a->storage->data.data();

    for (int i = 0; i < a->size(); i++)
        out_ptr[i] = src_data[a->offset + a->flat_idx(unravel(i, a->shape))];

    return out;
}

//Autograd stuff 

// In-place Tensor Gradient Accumulation (Zero unnecessary heap allocations!)
inline void tensor_add_inplace(std::shared_ptr<Tensor>& dst, const std::shared_ptr<Tensor>& src) {
    if (!src) return;
    if (!dst) {
        // Clone src tensor into dst
        dst = std::make_shared<Tensor>(src->shape, false);
        std::memcpy(dst->data_ptr(), src->data_ptr(), src->size() * sizeof(float));
        return;
    }
    assert(dst->shape == src->shape);
    float* d_ptr = dst->data_ptr();
    const float* s_ptr = src->data_ptr();
    int n = dst->size();
    for (int i = 0; i < n; ++i) {
        d_ptr[i] += s_ptr[i];
    }
}

// Leaf Node: Accumulates gradients into leaf Tensor's .grad Tensor
struct AccumulateGrad : public Node {
    std::weak_ptr<Tensor> variable;

    explicit AccumulateGrad(std::shared_ptr<Tensor> var) : variable(var) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (auto var = variable.lock()) {
            if (!grads.empty() && grads[0]) {
                // In-place gradient accumulation!
                tensor_add_inplace(var->grad, grads[0]);
            }
        }
        return {};
    }
};

// Helper: Retrieve or create gradient edge for an input Tensor
inline Edge get_grad_edge(const std::shared_ptr<Tensor>& t) {
    if (!t || !t->requires_grad) return {nullptr, 0};
    if (t->grad_fn) return {t->grad_fn, 0};
    if (!t->grad_accumulator) {
        t->grad_accumulator = std::make_shared<AccumulateGrad>(t);
    }
    return Edge(t->grad_accumulator, 0);
}

// Implementation of Tensor::backward()
inline void Tensor::backward(bool retain_graph) {
    if (this->size() != 1) {
        throw std::runtime_error("backward: tensor must be scalar (size 1)");
    }

    this->grad_ptr()[0] = 1.0f;

    // 1. Node-based Autograd Graph Execution
    if (this->grad_fn) {
        std::unordered_map<Node*, int> in_degree;
        
        std::function<void(Node*)> compute_degrees = [&](Node* node) {
            if (!node) return;
            for (const auto& edge : node->next_edges) {
                if (edge.function) {
                    if (!in_degree.contains(edge.function.get())) {
                        in_degree[edge.function.get()] = 0;
                        compute_degrees(edge.function.get());
                    }
                    in_degree[edge.function.get()]++;
                }
            }
        };
        
        in_degree[this->grad_fn.get()] = 0;
        compute_degrees(this->grad_fn.get());

        std::queue<std::shared_ptr<Node>> ready;
        ready.push(this->grad_fn);

        std::unordered_map<Node*, std::vector<std::shared_ptr<Tensor>>> node_grads;
        auto seed_grad = std::make_shared<Tensor>(this->shape, false);
        seed_grad->data_ptr()[0] = 1.0f;
        node_grads[this->grad_fn.get()] = {seed_grad};

        while (!ready.empty()) {
            auto curr = ready.front();
            ready.pop();

            std::vector<std::shared_ptr<Tensor>> incoming = node_grads[curr.get()];
            std::vector<std::shared_ptr<Tensor>> outgoing = curr->apply(incoming);

            for (size_t i = 0; i < curr->next_edges.size(); ++i) {
                const auto& edge = curr->next_edges[i];
                if (!edge.function) continue;

                Node* next_node = edge.function.get();
                std::shared_ptr<Tensor> grad_to_pass = (i < outgoing.size()) ? outgoing[i] : nullptr;
                if (!grad_to_pass) continue;

                if (!node_grads.contains(next_node)) {
                    node_grads[next_node] = {grad_to_pass};
                } else {
                    // Accumulate incoming gradient Tensors in-place without heap re-allocations!
                    tensor_add_inplace(node_grads[next_node][0], grad_to_pass);
                }

                in_degree[next_node]--;
                if (in_degree[next_node] == 0) {
                    ready.push(edge.function);
                }
            }

            // Break reference cycles: clear next_edges and release saved variables if graph is not retained
            if (!retain_graph) {
                curr->next_edges.clear();
                curr->release_variables();
            }
        }

        if (!retain_graph) {
            this->grad_fn = nullptr;
        }
    }
}


