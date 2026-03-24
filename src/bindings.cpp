#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(dummygrad, m) {
    m.doc() = "Dummygrad - A tensor autograd engine in C++";

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>>())
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
    m.def("get_row", [](const std::shared_ptr<Tensor>& a, int row) { return get_row(a, row); });
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
