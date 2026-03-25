#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(dummygrad, m) {
    m.doc() = "Dummygrad - A tensor autograd engine in C++";

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>>())

        //repr
        .def("__repr__", [](Tensor& t) {
            std::function<std::string(std::vector<float>&, std::vector<int>&, int, int)> fmt;
            fmt = [&](std::vector<float>& data, std::vector<int>& shape, int offset, int indent) -> std::string {
                if(shape.size() == 1) {
                    std::string s = "[";
                    for(int i = 0; i < shape[0]; i++) {
                        s += std::to_string(data[offset + i]);
                        if(i < shape[0]-1) s += ", ";
                    }
                    return s + "]";
                }
                int stride = 1;
                for(int i = 1; i < shape.size(); i++) stride *= shape[i];
                std::vector<int> inner(shape.begin()+1, shape.end());
                std::string pad(indent + 1, ' ');
                std::string s = "[";
                for(int i = 0; i < shape[0]; i++) {
                    if(i > 0) s += pad;
                    s += fmt(data, inner, offset + i*stride, indent + 1);
                    if(i < shape[0]-1) s += ",\n";
                }
                return s + "]";
            };
            std::string result = "Tensor(";
            result += fmt(t.data, t.shape, 0, 7);  // 7 = length of "Tensor("
            result += ", shape=[";
            for(int i = 0; i < t.shape.size(); i++) {
                result += std::to_string(t.shape[i]);
                if(i < t.shape.size()-1) result += ", ";
            }
            return result + "])";
        })
        
        //show_grad
        .def("show_grad", [](Tensor& t) {
            std::function<std::string(std::vector<float>&, std::vector<int>&, int, int)> fmt;
            fmt = [&](std::vector<float>& data, std::vector<int>& shape, int offset, int indent) -> std::string {
                if(shape.size() == 1) {
                    std::string s = "[";
                    for(int i = 0; i < shape[0]; i++) {
                        s += std::to_string(data[offset + i]);
                        if(i < shape[0]-1) s += ", ";
                    }
                    return s + "]";
                }
                int stride = 1;
                for(int i = 1; i < shape.size(); i++) stride *= shape[i];
                std::vector<int> inner(shape.begin()+1, shape.end());
                std::string pad(indent + 1, ' ');
                std::string s = "[";
                for(int i = 0; i < shape[0]; i++) {
                    if(i > 0) s += pad;
                    s += fmt(data, inner, offset + i*stride, indent + 1);
                    if(i < shape[0]-1) s += ",\n";
                }
                return s + "]";
            };
            std::string result = "Grad(";
            result += fmt(t.grad, t.shape, 0, 5);  // 5 = length of "Grad("
            result += ", shape=[";
            for(int i = 0; i < t.shape.size(); i++) {
                result += std::to_string(t.shape[i]);
                if(i < t.shape.size()-1) result += ", ";
            }
            result += "])";
            py::print(result);
        })

        .def("__matmul__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return matmul(a, b); })
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        .def("get", &Tensor::get)
        .def("set", &Tensor::set)
        .def("size", &Tensor::size)
        .def("fill", &Tensor::fill)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("strides", &Tensor::strides);

    m.def("manual_seed", [](unsigned int seed) { manual_seed(seed); });
    m.def("randn", [](std::vector<int> shape) { return randn(shape); });
    m.def("add", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return add(a, b); });
    m.def("sub", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return sub(a, b); });
    m.def("mul", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return mul(a, b); });
    m.def("div", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return div(a, b); });
    m.def("transpose", [](const std::shared_ptr<Tensor>& a) { return transpose(a); });
    m.def("matmul", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) { return matmul(a, b); });
    m.def("pow", [](const std::shared_ptr<Tensor>& a, int n) { return pow(a, n); });
    m.def("sqrt", [](const std::shared_ptr<Tensor>& a) { return sqrt(a); });
    m.def("log", [](const std::shared_ptr<Tensor>& a) { return log(a); });
    m.def("exp", [](const std::shared_ptr<Tensor>& a) { return exp(a); });
    m.def("sum", [](const std::shared_ptr<Tensor>& a) { return sum(a); });
    m.def("mean", [](const std::shared_ptr<Tensor>& a) { return mean(a); });
    m.def("softmax", [](const std::shared_ptr<Tensor>& a) { return softmax(a); });
    m.def("relu", [](const std::shared_ptr<Tensor>& a) { return relu(a); });
    m.def("tanh", [](const std::shared_ptr<Tensor>& a) { return tanh(a); });
    m.def("broadcast", [](const std::shared_ptr<Tensor>& a, int axis, int n) { return broadcast(a, axis, n); });
    m.def("collapse", [](const std::shared_ptr<Tensor>& a, int axis) { return collapse(a, axis); });
    m.def("CrossEntropyLoss", [](const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) { return CrossEntropyLoss(pred, target); });
    m.def("SGD", [](const std::shared_ptr<Tensor>& param, float lr) { SGD(param, lr); });

    py::class_<Adam>(m, "Adam")
        .def(py::init<float, float, float, float>(),
             py::arg("lr") = 0.001f,
             py::arg("b1") = 0.9f,
             py::arg("b2") = 0.999f,
             py::arg("E") = 1e-8f)
        .def("step", [](Adam& self, const std::shared_ptr<Tensor>& param) { self.step(param); });
}
