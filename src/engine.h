//Dummygrad

#include<unordered_set>
#include<memory>
#include<set>
#include<cmath>
#include<random>
#include<iostream>
#include<vector>
#include<functional>
#include<algorithm>

struct Storage {
    std::vector<float> data;
    std::vector<float> grad;
    //float* cuda_data = nullptr;
    //float* cuda_grad = nullptr;
    //bool on_gpu = false;

    Storage(int size) : data(size, 0.0f), grad(size, 0.0f) {}
};

//Tensor 
class Tensor {
    public:
        std::shared_ptr<Storage> storage;
        std::vector<int> shape;
        std::vector<int> strides;
        std::function<void()> backward_fn;
        int offset = 0;
        std::vector<std::shared_ptr<Tensor>> prev;

        Tensor(std::vector<int> s) : shape(s), offset(0) {  

            //get total elements of tensor 
            int total = 1;  
            for (int dim : shape) {
                total *= dim;
            }

            //allocate and initialise
            storage = std::make_shared<Storage>(total);

            //strides
            strides.resize(shape.size());
            strides.back() = 1;
            for(int i = shape.size() - 2; i>=0; i--) {
                strides[i] = strides[i+1] * shape[i+1];
            }
        }

        float& data_at(int i) { return storage->data[offset + i]; }
        float& grad_at(int i) { return storage->grad[offset + i]; }

        int size() { return storage->data.size(); } 

        //zero_grad
        void zero_grad() {
            for(int i=0; i<size(); i++) {
                grad_at(i) = 0.0f; 
            }
        }    

        //fill the tensor manually   
        void fill(std::vector<float> values) {
            if (values.size() != size()) {
                throw std::runtime_error("size mismatch..."); 
            }
            storage->data = values;
        }

        //autograd 
        void backward() {

            if(this->size() > 1) {
                throw std::runtime_error("Grad can only be initialized to 1.0 for scalars (size 1). Sum or mean your tensor first, man!");
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

            for(int i=topo.size()-1; i>=0; i--) {
                if(topo[i]->backward_fn) {
                    topo[i]->backward_fn();
                }
            }
        }
};

inline std::mt19937 g_gen(std::random_device{}());

inline void manual_seed(unsigned int seed) {
    g_gen.seed(seed);
}

//random init
inline std::shared_ptr<Tensor> randn(std::vector<int> shape) {
    auto t = std::make_shared<Tensor>(shape);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for(int i=0; i<t->size(); i++) {
        t->data_at(i) = dis(g_gen);
    }
    return t;
}

//xavier init
inline std::shared_ptr<Tensor> xavier(std::vector<int> shape) {
    if(shape.size() != 2) throw std::runtime_error("xavier: only 2D tensors supported. cmon man.");

    auto t = std::make_shared<Tensor>(shape);
    float std = std::sqrt(1.0f / (float)shape[0]);
    std::normal_distribution<float> dis(0.0f, std);
    for(int i = 0; i < t->size(); i++) {
        t->data_at(i) = dis(g_gen);
    }
    return t;
}

//kaiming init
inline std::shared_ptr<Tensor> kaiming(std::vector<int> shape) {
    if(shape.size() != 2) throw std::runtime_error("kaiming: only 2D tensors supported. cmon man.");

    auto t = std::make_shared<Tensor>(shape);
    float std = std::sqrt(2.0f / (float)shape[0]);
    std::normal_distribution<float> dis(0.0f, std);
    for(int i = 0; i < t->size(); i++) {
        t->data_at(i) = dis(g_gen);
    }
    return t;
}

//view


//Operations

//add
inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {

    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in addition. Cmon man..");
    }

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) + b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i);
                b->grad_at(i) += self->grad_at(i);
            }
        }    
    };

    return out;
}
                         
//sub
inline std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {

    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in subtraction. Cmon man..");
    }

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) - b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i);
                b->grad_at(i) -= self->grad_at(i);
            }
        }
    };

    return out;
}

//mul
inline std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {

    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in multiplication. Cmon man..");
    }

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) * b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += b->data_at(i) * self->grad_at(i);
                b->grad_at(i) += a->data_at(i) * self->grad_at(i);
            }
        }
    };

    return out;
}

//divide
inline std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {

    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in division. Cmon man..");
    }

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = a->data_at(i) / b->data_at(i);
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out; 

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i) / b->data_at(i);
                b->grad_at(i) -= a->data_at(i) * self->grad_at(i) / (b->data_at(i) * b->data_at(i));
            }
        }
    };

    return out;
}

//transpose 
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a) {
    int n = a->shape.size();

    auto out = std::make_shared<Tensor>(a->shape);
    out->storage = a->storage;
    out->strides = a->strides;

    //swap the last two dims and strides
    std::swap(out->shape[n-2], out->shape[n-1]);
    std::swap(out->strides[n-2], out->strides[n-1]);

    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            int r = a->shape[n-2];
            int c = a->shape[n-1];
            int batch_size = 1;
            for(int i = 0; i<n-2; i++) { batch_size *= a->shape[i]; }

            for(int batch = 0; batch<batch_size; batch++) {
                for(int i=0; i<r; i++) {
                    for(int j=0; j<c; j++) {
                        a->grad_at(a->strides[n-2]*i + a->strides[n-1]*j + batch*r*c) 
                            += self->grad_at(self->strides[n-2]*j + self->strides[n-1]*i + batch*r*c);
                    }
                }
            }
        }
    };

    return out;
}

//the mighty matmul
inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {

    int n1 = a->shape.size();
    int n2 = b->shape.size();

    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    //check the batch dimensions match 
    for(int i = 0; i < n1-2; i++) {
        if(a->shape[i] != b->shape[i]) {
            throw std::runtime_error("Batch dimensions mismatch for matmul.. CMON MAN.");
        }
    }
    //check inner dimensions match
    if(c1 != r2) {
        throw std::runtime_error("Matmul dimensions mismatch.. CMON MAN.");  
    }

    //copy the batch dimensions from a 
    std::vector<int> out_shape;
    for(int i = 0; i<n1-2; i++) {
        out_shape.push_back(a->shape[i]);
    }
    out_shape.push_back(r1);
    out_shape.push_back(c2);

    auto out = std::make_shared<Tensor>(out_shape);
    int nout = out->shape.size();

    int batch_size = 1;
    for(int i = 0; i<n1-2; i++) { batch_size *= a->shape[i]; }

    for(int batch = 0; batch<batch_size; batch++) {
        for(int i = 0; i<r1; i++) {
            for(int j = 0; j<c2; j++) {
                float sum = 0.0f;
                for(int k = 0; k<c1; k++) {
                    sum += a->data_at(a->strides[n1-2]*i + a->strides[n1-1]*k + batch*r1*c1) 
                         * b->data_at(b->strides[n2-2]*k + b->strides[n2-1]*j + batch*r2*c2);                       
                }
                out->data_at(out->strides[nout-2]*i + out->strides[nout-1]*j + batch*r1*c2) = sum;
            }
        }
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out, r1, c1, r2, c2, batch_size, n1, n2, nout]() {
        if(auto self = weak_out.lock()) {
            for(int batch = 0; batch<batch_size; batch++) {
                //a.grad
                for(int i=0; i<r1; i++) {
                    for(int j=0; j<c1; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<c2; k++) {
                            sum += self->grad_at(self->strides[nout-2]*i + self->strides[nout-1]*k + batch*r1*c2) 
                                 * b->data_at(b->strides[n2-2]*j + b->strides[n2-1]*k + batch*r2*c2);
                        }
                        a->grad_at(a->strides[n1-2]*i + a->strides[n1-1]*j + batch*r1*c1) += sum;
                    }
                }
                //b.grad
                for(int i=0; i<r2; i++) {
                    for(int j=0; j<c2; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<r1; k++) {
                            sum += a->data_at(a->strides[n1-2]*k + a->strides[n1-1]*i + batch*r1*c1) 
                                 * self->grad_at(self->strides[nout-2]*k + self->strides[nout-1]*j + batch*r1*c2);
                        }
                        b->grad_at(b->strides[n2-2]*i + b->strides[n2-1]*j + batch*r2*c2) += sum;
                    }    
                }
            }
        }
    };

    return out;
}

//pow
inline std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, const int n) {
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::pow(a->data_at(i), n);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += n * std::pow(a->data_at(i), n-1) * self->grad_at(i);
            }
        }
    };

    return out;
}

//sqrt
inline std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::sqrt(a->data_at(i));
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += (0.5f / std::sqrt(a->data_at(i))) * self->grad_at(i);
            }
        }
    };

    return out;
}

//log
inline std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape); 

    float epsilon = 1e-8f;
    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = std::log(a->data_at(i) + epsilon);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, epsilon]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i) * (1.0f / (a->data_at(i) + epsilon));
            }
        }
    };

    return out;
}

//exp
inline std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data_at(i) = std::exp(a->data_at(i));
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(i) * self->data_at(i);
            }
        }
    };

    return out; 
}

//sum
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}); 

    float total = 0.0f;
    for(int i = 0; i<a->size(); i++) {
        total += a->data_at(i);
    }
    out->data_at(0) = total;
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += self->grad_at(0);
            }
        }
    };

    return out;
}

//mean
inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}); 

    float sum = 0.0f;
    for(int i=0; i<a->size(); i++) {
        sum += a->data_at(i);
    }
    out->data_at(0) = sum/(a->size());
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            float gradient = self->grad_at(0) / a->size();
            for(int i = 0; i<a->size(); i++) {
                a->grad_at(i) += gradient;
            }
        }
    };

    return out;
}

//softmax
inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a) {
    int n = a->shape.size();
    int last = a->shape[n-1];  // size of last dimension
    int outer = a->size() / last;  // product of all other dimensions

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i < outer; i++) {
        int offset = i * last;

        // find max for numerical stability
        float max_val = a->data_at(offset);
        for(int j = 1; j < last; j++) {
            max_val = std::max(max_val, a->data_at(offset + j));
        }

        // compute softmax
        float sum = 0.0f;
        for(int j = 0; j < last; j++) {
            sum += std::exp(a->data_at(offset + j) - max_val);
        }
        for(int j = 0; j < last; j++) {
            out->data_at(offset + j) = std::exp(a->data_at(offset + j) - max_val) / sum;
        }
    }

    out->prev.push_back(a);
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, last, outer]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i < outer; i++) {
                int offset = i * last;

                float max_val = a->data_at(offset);
                for(int j = 1; j < last; j++) {
                    max_val = std::max(max_val, a->data_at(offset + j));
                }
                float sum = 0.0f;
                for(int j = 0; j < last; j++) {
                    sum += std::exp(a->data_at(offset + j) - max_val);
                }
                for(int k = 0; k < last; k++) {
                    float y_k = std::exp(a->data_at(offset + k) - max_val);
                    a->grad_at(offset + k) += (y_k * (sum - y_k) / (sum * sum)) * self->grad_at(offset + k);
                }
            }
        }
    };

    return out;
}

//Activation functions

//relu
inline std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::max(0.0f, a->data_at(i)); 
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                if(a->data_at(i)>0.0f) {
                    a->grad_at(i) += self->grad_at(i);
                }
            }
        }
    };

    return out;
}

//tanh
inline std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data_at(i) = std::tanh(a->data_at(i));
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad_at(i) += (1.0 - (self->data_at(i) * self->data_at(i))) * self->grad_at(i);
            }
        }
    };

    return out;
}

//Loss functions

//CrossEntropyLoss
inline std::shared_ptr<Tensor> CrossEntropyLoss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    if(pred->shape != target->shape) {
        throw std::runtime_error("The shape of your prediction and target doesn't match man..");
    }

    float sum_loss = 0.0f;
    for(int i=0; i<pred->size(); i++) {
        sum_loss -= target->data_at(i) * std::log(pred->data_at(i));
    }

    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});

    float n = static_cast<float>(pred->size());

    out->data_at(0) = sum_loss / n; 
    out->prev.push_back(pred);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [pred, target, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<pred->size(); i++) {
                pred->grad_at(i) += -(target->data_at(i) / (pred->data_at(i) * n)) * self->grad_at(0); 
            }
        }
    };

    return out;
}

//Optimizers

//SGD
inline void SGD(const std::shared_ptr<Tensor>& param, const float& lr) {
    for(int i = 0; i<param->size(); i++) {
        param->data_at(i) -= lr * param->grad_at(i);
    }
}

//Adam the great 
class Adam {
    std::vector<float> m, v;
    int t = 0; 
    float lr, b1, b2, E;

    public:
        Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float E = 1e-8f)
            : lr(lr), b1(b1), b2(b2), E(E) {}

        void step(const std::shared_ptr<Tensor>& param) {
            if(m.empty()) {
                m.resize(param->size(), 0.0f);
                v.resize(param->size(), 0.0f);
            }
            t++;
            for(int i = 0; i<param->size(); i++) {
                m[i] = b1*m[i] + (1-b1)*param->grad_at(i);
                v[i] = b2*v[i] + (1-b2)*param->grad_at(i)*param->grad_at(i);

                float m_hat = m[i] / (1 - std::pow(b1, t)); 
                float v_hat = v[i] / (1 - std::pow(b2, t)); 

                //update
                param->data_at(i) -= lr * m_hat / (std::sqrt(v_hat) + E);
            }
        }
};


//Broadcasting 

//broadcast
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a, int axis, int n) {
    int r = a->shape[0];
    int c = a->shape[1];

    auto out = std::make_shared<Tensor>(
            axis == 0 ? std::vector<int>{n, c} : std::vector<int>{r, n}
    );

    if(axis == 0) {
        for(int i = 0; i<n; i++) {
            for(int j = 0; j<c; j++) {
                out->data_at(i*c + j) = a->data_at(j);
            }
        }
    }
    else {
        for(int i = 0; i<r; i++) {
            for(int j = 0; j<n; j++) {
                out->data_at(i*n + j) = a->data_at(i);
            }
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, r, c, n, axis]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int i = 0; i<n; i++) {
                    for(int j = 0; j<c; j++) {
                        a->grad_at(j) += self->grad_at(i*c + j);
                    }
                }
            }
            else {
                for(int i = 0; i<r; i++) {
                    for(int j = 0; j<n; j++) {
                        a->grad_at(i) += self->grad_at(i*n + j);
                    }
                }
            }
        }
    };

    return out;
}

//collapse
inline std::shared_ptr<Tensor> collapse(const std::shared_ptr<Tensor>& a, int axis) {
    int r = a->shape[0];
    int c = a->shape[1];

    auto out = std::make_shared<Tensor>(
            axis == 0 ? std::vector<int>{1, c} : std::vector<int>{r, 1}
    );

    if(axis == 0) {
        for(int i = 0; i<c; i++) {
            float total = 0.0f; 
            for(int j = 0; j<r; j++) {
                total += a->data_at(j*c + i);
            }
            out->data_at(i) = total;
        }
    }
    else {
        for(int i = 0; i<r; i++) {
            float total = 0.0f;
            for(int j = 0; j<c; j++) {
                total += a->data_at(i*c + j);
            }
            out->data_at(i) = total;
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, r, c, axis]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int i = 0; i<c; i++) {
                    for(int j = 0; j<r; j++) {
                        a->grad_at(j*c + i) += self->grad_at(i);
                    }
                }
            }
            else {
                for(int i = 0; i<r; i++) {
                    for(int j = 0; j<c; j++) {
                        a->grad_at(i*c + j) += self->grad_at(i);
                    }
                }
            }
        }
    };

    return out;
}




































