from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "dummygrad",
        ["src/bindings.cpp"],
        extra_compile_args=["-O2", "-std=c++17"],
    ),
]

setup(
    name="dummygrad",
    ext_modules=ext_modules,
)
