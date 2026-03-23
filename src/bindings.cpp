#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(dummygrad, m) {
    m.doc() = "Dummygrad - A tensor autograd engine in C++";

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>>())        // ← added this
        .def("show", &Tensor::show)
        .def("show_grad", &Tensor::show_grad)
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

    m.def("manual_seed", &manual_seed);
    m.def("randn", &randn);
    m.def("get_row", &get_row);
    m.def("add", &add);
    m.def("sub", &sub);
    m.def("mul", &mul);
    m.def("div", &div);
    m.def("transpose", &transpose);
    m.def("matmul", &matmul);

    // ← lambdas for cmath name conflicts
    m.def("pow",  [](const std::shared_ptr<Tensor>& a, int n) { return pow(a, n); });
    m.def("sqrt", [](const std::shared_ptr<Tensor>& a) { return sqrt(a); });
    m.def("log",  [](const std::shared_ptr<Tensor>& a) { return log(a); });
    m.def("exp",  [](const std::shared_ptr<Tensor>& a) { return exp(a); });
    m.def("tanh", [](const std::shared_ptr<Tensor>& a) { return tanh(a); });

    m.def("sum", &sum);
    m.def("mean", &mean);
    m.def("softmax", &softmax);
    m.def("relu", &relu);
    m.def("broadcast", &broadcast);
    m.def("collapse", &collapse);
    m.def("CrossEntropyLoss", &CrossEntropyLoss);
    m.def("SGD", &SGD);

    py::class_<Adam>(m, "Adam")
        .def(py::init<float, float, float, float>(),
             py::arg("lr") = 0.001f,
             py::arg("b1") = 0.9f,
             py::arg("b2") = 0.999f,
             py::arg("E") = 1e-8f)
        .def("step", &Adam::step);
}
