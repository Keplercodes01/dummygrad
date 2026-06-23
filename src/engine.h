//Dummygrad

#pragma once
#include<iostream>
#include<unordered_set>
#include<memory>
#include<set>
#include<cmath>
#include<random>
#include<iostream>
#include<vector>
#include<functional>
#include<algorithm>
#include<limits>

//flat buffer of data and grad 
struct Storage {
    std::vector<float> data;
    std::vector<float> grad;
    //float* cuda_data = nullptr;
    //float* cuda_grad = nullptr;
    //bool on_gpu = false;

    explicit Storage(int size) : data(size, 0.0f), grad(size, 0.0f) {}
};

//make_strides
inline std::vector<int> make_strides(const std::vector<int>& shape) {
    int ndim = shape.size();
    std::vector<int> st(ndim); 
    st[ndim-1] = 1;
    for(int i = ndim-2; i>=0; i--) 
        st[i] = st[i+1] * shape[i+1];
    return st;
}

//unravel
inline std::vector<int> unravel(int flat, const std::vector<int>& shape) {
    int ndim = (int)shape.size();
    std::vector<int> idx(ndim);
    for(int d = ndim-1; d>=0; d--) {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    return idx;
}

//tensor class 
class Tensor {
public:
    std::shared_ptr<Storage> storage;
    std::vector<int> shape;
    std::vector<int> strides;
    int offset = 0;

    std::function<void()> backward_fn;
    std::vector<std::shared_ptr<Tensor>> prev;

    Tensor(std::vector<int> s) : shape(std::move(s)), offset(0) {  

        //get total elements of tensor 
        int total = 1;  
        for (int dim : shape) 
            total *= dim;
        storage = std::make_shared<Storage>(total);
        strides = make_strides(shape);
    }

    float& data_at(int i) { return storage->data[offset + i]; }
    float& grad_at(int i) { return storage->grad[offset + i]; }

    //flat index 
    int flat_idx(const std::vector<int>& idx) const {
        int pos = 0; 
        for(int i=0; i<(int)idx.size(); i++) 
            pos += idx[i] * strides[i];
        return pos;
    }

    //is_contiguous
    bool is_contiguous() const {
        return strides == make_strides(shape);
    }

    //sizes
    int ndim() const { return (int)shape.size(); }

    int size() {
        int total = 1;
        for(int d : shape) total *= d;
        return total;
    }

    //zero_grad
    void zero_grad() {
        for(int i=0; i < size(); i++) {
            grad_at(i) = 0.0f; 
        }
    }    

    //fill the tensor manually   
    void fill(const std::vector<float>& values) {
        if ((int)values.size() != size()) {
            throw std::runtime_error("fill: size mismatch..."); 
        }
        for(int i=0; i < size(); i++) 
            data_at(i) = values[i];
    }

    //reshape
    std::shared_ptr<Tensor> reshape(std::vector<int> new_shape) {
        int new_total = 1; 
        for(int d : new_shape) new_total *= d;
        if(new_total != size())
            throw std::runtime_error("reshape: element count mismatch..."); 

        if(is_contiguous()) {
            //view - no copy
            auto t = std::make_shared<Tensor>(new_shape);
            t->storage = storage;
            t->strides = make_strides(new_shape);
            t->offset = offset;
            return t;
        }else {
            //not contiguous - must copy the data first 
            auto copy = std::make_shared<Tensor>(shape);
            for(int i = 0; i<size(); i++)
                copy->data_at(i) = storage->data[offset + flat_idx(unravel(i, shape))];

            //now that its contiguous return the view with the new shape
            auto t = std::make_shared<Tensor>(new_shape);
            t->storage = copy->storage;
            t->strides = make_strides(new_shape);
            t->offset = 0;
            return t;
        }
    }

    //transpose
    std::shared_ptr<Tensor> transpose(int ax0, int ax1) {
        if(ax0 < 0 || ax0 >= ndim() || ax1 < 0 || ax1 >= ndim())
            throw std::runtime_error("transpose: axis out of shape");

        auto t = std::make_shared<Tensor>(shape);
        t->storage = storage;
        t->shape = shape;
        t->strides = strides;
        t->offset = offset;

        std::swap(t->shape[ax0], t->shape[ax1]);
        std::swap(t->strides[ax0], t->strides[ax1]);
        return t;
    }

    //show methods
    void _show_recursive(int dim, int pos, bool grad_mode) const {
        if (dim == ndim() - 1) {
            std::cout << "[";
            for (int i = 0; i < shape[dim]; ++i) {
                int p = pos + i * strides[dim];
                std::cout << (grad_mode ? storage->grad[offset + p]
                                        : storage->data[offset + p]);
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

    //autograd 
    void backward(bool retain_graph = false) {

        if(this->size() != 1) {
            throw std::runtime_error("backward: tensor must be size 1");
        }

        this->grad_at(0) = 1.0f;

        std::vector<Tensor*> topo;
        std::unordered_set<Tensor*> visited;

        std::function<void(Tensor*)> build_topo = [&](Tensor* t) {
            if(visited.count(t)) return;
            visited.insert(t);

            for(auto& parent : t->prev) {
                build_topo(parent.get());
            }
            topo.push_back(t);
        };
        build_topo(this);

        for(int i = topo.size()-1; i>=0; i--) {
            if(topo[i]->backward_fn) {
                topo[i]->backward_fn();
            }
        }
        if(!retain_graph) {
            for(auto t : topo) {
                t->prev.clear();
                t->backward_fn = nullptr;
            }
        }
    }
};

//make_contiguous
inline std::shared_ptr<Tensor> make_contiguous(const std::shared_ptr<Tensor>& a) {
    if(a->is_contiguous()) return a;
    auto out = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i<a->size(); i++)
        out->data_at(i) = a->storage->data[a->offset + a->flat_idx(unravel(i, a->shape))];
    return out;
}
