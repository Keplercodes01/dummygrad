//Dummygrad

#pragma once
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

        int size() {
            if(shape.empty()) return 0;
            int total = 1;
            for(int d : shape) total *= d;
            return total;
        }

        //zero_grad
        void zero_grad() {
            int n = size();
            for(int i=0; i<n; i++) {
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

        //show tensor
        void show_recursive_data(int dim,  int offset) {
            if(dim == shape.size()-1) {
                //its the last dimension so print the actual values 
                std::cout<<"[";
                for(int i = 0; i<shape[dim]; i++) {
                    std::cout<< data_at(offset + i);
                    if(i < shape[dim]-1) std::cout<< ",";
                }
                std::cout<< "]"; 
            }else {
                std::cout<< "[";
                for(int i = 0; i<shape[dim]; i++) {
                    show_recursive_data(dim+1, offset + i*strides[dim]);  
                    if(i<shape[dim]-1) {
                        std::cout<< ",\n";
                        for(int d = 0; d<=dim; d++) {
                            std::cout<< " ";
                        }
                    }
                }
                std::cout<< "]"; 
            }
        }
        void show() {
            show_recursive_data(0, 0);
            std::cout<< ", shape=(";
            for(int i = 0; i<shape.size(); i++) {
                std::cout<< shape[i];
                if(i<shape.size()-1) std::cout<< ",";
            }
            std::cout<< ")" <<std::endl;
        }

        //show gradients
        void show_recursive_grad(int dim,  int offset) {
            if(dim == shape.size()-1) {
                //its the last dimension so print the actual values 
                std::cout<<"[";
                for(int i = 0; i<shape[dim]; i++) {
                    std::cout<< grad_at(offset + i);
                    if(i < shape[dim]-1) std::cout<< ",";
                }
                std::cout<< "]"; 
            }else {
                std::cout<< "[";
                for(int i = 0; i<shape[dim]; i++) {
                    show_recursive_grad(dim+1, offset + i*strides[dim]);  
                    if(i<shape[dim]-1) {
                        std::cout<< ",\n";
                        for(int d = 0; d<=dim; d++) {
                            std::cout<< " ";
                        }
                    }
                }
                std::cout<< "]"; 
            }
        }
        void show_grad() {
            show_recursive_grad(0, 0);
            std::cout<< ", shape=(";
            for(int i = 0; i<shape.size(); i++) {
                std::cout<< shape[i];
                if(i<shape.size()-1) std::cout<< ",";
            }
            std::cout<< ")" <<std::endl;
        }

        //autograd 
        void backward(bool retain_graph = false) {

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
            if(!retain_graph) {
                for(auto t : topo) {
                    t->prev.clear();
                    t->backward_fn = nullptr;
                }
            }
        }
};
