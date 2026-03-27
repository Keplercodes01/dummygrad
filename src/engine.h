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

//Tensor 
class Tensor {
    public:
        std::vector<float> data;
        std::vector<float> grad;
        std::vector<int> shape;
        std::vector<int> strides;
        std::function<void()> backward_fn;
        std::vector<std::shared_ptr<Tensor>> prev;

        Tensor(std::vector<int> s) : shape(s) {  

            //get total elements of tensor 
            int total = 1;  
            for (int dim : shape) {
                total *= dim;
            }

            //allocate and initialise
            data.resize(total, 0.0f);
            grad.resize(total, 0.0f);

            //compute strides
            strides.resize(shape.size());
            strides.back() = 1;
            for(int i = shape.size() - 2; i>=0; i--) {
                strides[i] = strides[i+1] * shape[i+1];
            }
        }

        int size() { return data.size(); } 

        //zero_grad
        void zero_grad() {
            for(int i=0; i<grad.size(); i++) {
                grad[i] = 0.0f; 
            }
        }    

        //fill the tensor manually   
        void fill(std::vector<float> values) {
            if (values.size() != data.size()) {
                throw std::runtime_error("size mismatch..."); 
            }
            data = values;
        }

        //autograd 
        void backward() {

            if(this->size() > 1) {
                throw std::runtime_error("Grad can only be initialized to 1.0 for scalars (size 1). Sum or mean your tensor first, man!");
            }

            this->grad[0] = 1.0f;

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

//randn        
inline std::mt19937 g_gen(std::random_device{}());

inline void manual_seed(unsigned int seed) {
    g_gen.seed(seed);
}

inline std::shared_ptr<Tensor> randn(std::vector<int> shape) {

    auto t = std::make_shared<Tensor>(shape);
    float k = std::sqrt(1.0f/(float)shape[0]);

    std::uniform_real_distribution<float> dis(-k, k);

    for(int i=0; i<t->size(); i++) {
        t->data[i] = dis(g_gen);
    }

    return t;
}

//Operations

//add
inline std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {

    if(a->shape != b->shape) {
        throw std::runtime_error("Shape mismatch in addition. Cmon man..");
    }

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += self->grad[i];
                b->grad[i] += self->grad[i];
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
        out->data[i] = a->data[i] - b->data[i];
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += self->grad[i];
                b->grad[i] -= self->grad[i];
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
        out->data[i] = a->data[i] * b->data[i];
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += b->data[i] * self->grad[i];
                b->grad[i] += a->data[i] * self->grad[i];
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
        out->data[i] = a->data[i] / b->data[i];
    }
    out->prev.push_back(a);
    out->prev.push_back(b);

    std::weak_ptr<Tensor> weak_out = out; 

    out->backward_fn = [a, b, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += self->grad[i] / b->data[i];
                b->grad[i] -= a->data[i] * self->grad[i] / (b->data[i] * b->data[i]);
            }
        }
    };

    return out;
}

//transpose 
inline std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& a) {
    int n = a->shape.size();

    auto out = std::make_shared<Tensor>(a->shape);
    out->data = a->data;
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
                        a->grad[a->strides[n-2]*i + a->strides[n-1]*j + batch*r*c] 
                            += self->grad[self->strides[n-2]*j + self->strides[n-1]*i + batch*r*c];
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
                    sum += a->data[a->strides[n1-2]*i + a->strides[n1-1]*k + batch*r1*c1] 
                         * b->data[b->strides[n2-2]*k + b->strides[n2-1]*j + batch*r2*c2];                       
                }
                out->data[out->strides[nout-2]*i + out->strides[nout-1]*j + batch*r1*c2] = sum;
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
                            sum += self->grad[self->strides[nout-2]*i + self->strides[nout-1]*k + batch*r1*c2] 
                                 * b->data[b->strides[n2-2]*j + b->strides[n2-1]*k + batch*r2*c2];
                        }
                        a->grad[a->strides[n1-2]*i + a->strides[n1-1]*j + batch*r1*c1] += sum;
                    }
                }
                //b.grad
                for(int i=0; i<r2; i++) {
                    for(int j=0; j<c2; j++) {
                        float sum = 0.0f;
                        for(int k=0; k<r1; k++) {
                            sum += a->data[a->strides[n1-2]*k + a->strides[n1-1]*i + batch*r1*c1] 
                                 * self->grad[self->strides[nout-2]*k + self->strides[nout-1]*j + batch*r1*c2];
                        }
                        b->grad[b->strides[n2-2]*i + b->strides[n2-1]*j + batch*r2*c2] += sum;
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
        out->data[i] = std::pow(a->data[i], n);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad[i] += n * std::pow(a->data[i], n-1) * self->grad[i];
            }
        }
    };

    return out;
}

//sqrt
inline std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<a->size(); i++) {
        out->data[i] = std::sqrt(a->data[i]);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<a->size(); i++) {
                a->grad[i] += (0.5f / std::sqrt(a->data[i])) * self->grad[i];
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
        out->data[i] = std::log(a->data[i] + epsilon);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, epsilon]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += self->grad[i] * (1.0f / (a->data[i] + epsilon));
            }
        }
    };
    return out;
}

//exp
inline std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i=0; i<out->size(); i++) {
        out->data[i] = std::exp(a->data[i]);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += self->grad[i] * self->data[i];
            }
        }
    };
    return out; 
}

//sum
inline std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a) {
    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1}); 

    float total = 0.0f;
    for(float val : a->data) {
        total += val;
    }
    out->data[0] = total;
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(float &g : a->grad) {
                g += self->grad[0];
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
        sum += a->data[i];
    }
    out->data[0] = sum/(a->size());
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            float gradient = self->grad[0] / a->size();
            for(float &g : a->grad) {
                g += gradient; 
            }
        }
    };

    return out;
}

//softmax
inline std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& a) {
    int r = a->shape[0];
    int c = a->shape[1];

    auto out = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i<r; i++) {
        float max_val = a->data[i*c];
        for(int m = 1; m<c; m++) {
            max_val = std::max(max_val, a->data[i*c + m]);
        }
        float sum = 0.0f;
        for(int j = 0; j<c; j++) {
            sum += std::exp(a->data[i*c + j] - max_val);
        }
        for(int k = 0; k<c; k++) {
            out->data[i*c + k] = std::exp(a->data[i*c + k] - max_val) / sum;
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, r, c]() {
        if(auto self = weak_out.lock()) {
            for(int i = 0; i<r; i++) {
                float max_val = a->data[i*c];
                for(int m = 1; m<c; m++) {
                    max_val = std::max(max_val, a->data[i*c + m]);
                }
                float sum = 0.0f;
                for(int j = 0; j<c; j++) {
                    sum += std::exp(a->data[i*c + j] - max_val);
                }
                for(int k = 0; k<c; k++) {
                    float y_k = std::exp(a->data[i*c + k] - max_val);
                    a->grad[i*c + k] += (y_k * (sum - y_k) / (sum*sum)) * self->grad[i*c + k];
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
        out->data[i] = std::max(0.0f, a->data[i]); 
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                if(a->data[i]>0.0f) {
                    a->grad[i] += self->grad[i];
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
        out->data[i] = std::tanh(a->data[i]);
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<a->size(); i++) {
                a->grad[i] += (1.0 - (self->data[i] * self->data[i])) * self->grad[i];
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
        sum_loss -= target->data[i] * std::log(pred->data[i]);
    }

    auto out = std::make_shared<Tensor>(std::vector<int>{1, 1});

    float n = static_cast<float>(pred->size());

    out->data[0] = sum_loss / n; 
    out->prev.push_back(pred);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [pred, target, weak_out, n]() {
        if(auto self = weak_out.lock()) {
            for(int i=0; i<pred->size(); i++) {
                pred->grad[i] += -(target->data[i] / (pred->data[i] * n)) * self->grad[0]; 
            }
        }
    };

    return out;
}

//Optimizers

//SGD
inline void SGD(const std::shared_ptr<Tensor>& param, const float& lr) {
    for(int i = 0; i<param->size(); i++) {
        param->data[i] -= lr * param->grad[i];
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
                m[i] = b1*m[i] + (1-b1)*param->grad[i];
                v[i] = b2*v[i] + (1-b2)*param->grad[i]*param->grad[i];

                float m_hat = m[i] / (1 - std::pow(b1, t)); 
                float v_hat = v[i] / (1 - std::pow(b2, t)); 

                //update
                param->data[i] -= lr * m_hat / (std::sqrt(v_hat) + E);
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
                out->data[i*c + j] = a->data[j];
            }
        }
    }
    else {
        for(int i = 0; i<r; i++) {
            for(int j = 0; j<n; j++) {
                out->data[i*n + j] = a->data[i];
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
                        a->grad[j] += self->grad[i*c + j];
                    }
                }
            }
            else {
                for(int i = 0; i<r; i++) {
                    for(int j = 0; j<n; j++) {
                        a->grad[i] += self->grad[i*n + j];
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
                total += a->data[j*c + i];
            }
            out->data[i] = total;
        }
    }
    else {
        for(int i = 0; i<r; i++) {
            float total = 0.0f;
            for(int j = 0; j<c; j++) {
                total += a->data[i*c + j];
            }
            out->data[i] = total;
        }
    }
    out->prev.push_back(a);

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_fn = [a, weak_out, r, c, axis]() {
        if(auto self = weak_out.lock()) {
            if(axis == 0) {
                for(int i = 0; i<c; i++) {
                    for(int j = 0; j<r; j++) {
                        a->grad[j*c + i] += self->grad[i];
                    }
                }
            }
            else {
                for(int i = 0; i<r; i++) {
                    for(int j = 0; j<c; j++) {
                        a->grad[i*c + j] += self->grad[i];
                    }
                }
            }
        }
    };

    return out;
}




































