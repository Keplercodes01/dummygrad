# dummygrad

A tensor autograd engine written from scratch in C++, with Python bindings via pybind11.

Supports n-dimensional tensors, automatic differentiation, and a full suite of ops — enough to train MLPs, RNNs, and Transformers.

---

## Features

- N-dimensional tensors with automatic stride computation
- Full autograd engine with topological sort backward pass
- Batched n-dimensional matmul with correct gradient computation
- N-dimensional softmax along the last axis
- Zero-copy `view` and `transpose` via shared Storage architecture
- Embedding lookup via `C[x]` with autograd support
- Operator overloading — `+`, `-`, `*`, `/`, `@`, scalar ops
- Weight initialization — `randn`, `xavier`, `kaiming`, `zeros`, `ones`
- Adam and SGD optimizers
- Common ops: `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `log`, `exp`, `sum`, `mean`, `transpose`, `matmul`
- Activations: `relu`, `tanh`, `softmax`
- Loss: `CrossEntropyLoss` — does not include softmax, call it explicitly
- Broadcasting and collapsing along axes (2D, row/column vectors)
- Python bindings via pybind11

---

## Installation

### Linux
```bash
git clone https://github.com/Keplercodes01/dummygrad.git
cd dummygrad
pip install pybind11
pip install .
```

### macOS
```bash
xcode-select --install
git clone https://github.com/Keplercodes01/dummygrad.git
cd dummygrad
pip install pybind11
pip install .
```

### Google Colab
```python
!git clone https://github.com/Keplercodes01/dummygrad.git
%cd dummygrad
!apt-get install -y build-essential
!pip install pybind11
!pip install .
```

### Windows
```bash
Stop using Windows.
```
---

## Usage

```python
import dummygrad as dummy

# create tensors
a = dummy.randn([3, 4])
b = dummy.randn([4, 2])

# forward pass
c = a @ b
d = dummy.relu(c)
loss = dummy.mean(d)

# backward pass
loss.backward()

# inspect gradients
a.show_grad()
```

### Operator overloading

```python
c = a + b
c = a - b
c = a * b
c = a / b
c = a @ b       # matmul
c = a * 0.1     # scalar mul
c = 0.1 * a     # scalar rmul
c = -a          # negation
```

### Weight initialization

```python
dummy.manual_seed(42)

W = dummy.xavier([fan_in, fan_out])   # for tanh
W = dummy.kaiming([fan_in, fan_out])  # for relu
W = dummy.randn([3, 4])               # standard normal
b = dummy.zeros([1, 64])
```

### Embedding lookup

```python
C = dummy.xavier([27, 10])   # embedding table
x = [[1, 2, 3], [4, 5, 6]]  # batch of indices
emb = C[x]                   # shape [2, 3, 10] — autograd supported
```

### View and transpose

```python
a = dummy.randn([2, 6])
b = a.view([3, 4])           # zero copy
c = dummy.transpose(a)       # zero copy
```

### Optimizers

```python
# SGD
dummy.SGD(param, lr=0.01)

# Adam
optimizer = dummy.Adam(lr=0.001, b1=0.9, b2=0.999, E=1e-8)
optimizer.step(param)
```

### Broadcasting

```python
v = dummy.randn([1, 3])
v_broadcast = dummy.broadcast(v, axis=0, n=4)    # [1,3] → [4,3]
v_collapsed = dummy.collapse(v_broadcast, axis=0) # sum along axis
```

### One hot encoding

```python
indices = dummy.tensor([2.0, 0.0, 1.0], [3])
targets = dummy.one_hot(indices, num_classes=5)  # shape [3, 5]
```

---

## Design Philosophy

No excessive abstraction. Every major step is explicit — if you don't know what softmax does before a cross entropy loss, you shouldn't be touching the engine. Sharp tools for sharp engineers.

The autograd engine uses a `Storage` architecture — tensors share underlying memory through `shared_ptr<Storage>`, enabling zero-copy `view` and `transpose`. Designed with CUDA support in mind.

---

## Project Structure

```
dummygrad/
├── setup.py
└── src/
    ├── engine.h        # Tensor class, Storage, autograd
    ├── ops.h           # matmul, transpose, add, sub, mul, div, pow, sqrt, log, exp, sum, mean
    ├── activations.h   # relu, tanh, softmax
    ├── loss.h          # CrossEntropyLoss
    ├── init.h          # randn, xavier, kaiming, zeros, ones, tensor, one_hot
    ├── optimizers.h    # SGD, Adam
    ├── scalar_ops.h    # scalar mul, add, sub, div, neg
    ├── broadcasting.h  # broadcast, collapse
    └── bindings.cpp    # pybind11 Python bindings
```

---

## Roadmap

- [ ] CUDA kernel support
- [ ] N-dimensional broadcasting
- [ ] CNN support via im2col
- [ ] Gradient checking utility

---

## Built by

[@Keplercodes01](https://github.com/Keplercodes01)
