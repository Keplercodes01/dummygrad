# dummygrad

A tensor autograd engine written from scratch in C++, with Python bindings via pybind11.

Supports n-dimensional tensors, automatic differentiation, and a full suite of ops — enough to train MLPs, RNNs, and Transformers.

---

## Features

- N-dimensional tensors with automatic stride computation
- Full autograd engine with topological sort backward pass
- Batched n-dimensional matmul with correct gradient computation
- Broadcasting and collapsing along axes
- Adam and SGD optimizers
- Common ops: add, sub, mul, div, pow, sqrt, log, exp, sum, mean, transpose, matmul
- Activations: relu, tanh, softmax
- Loss: CrossEntropyLoss
- Python bindings via pybind11 — use it like a Python library

---

## Installation

```bash
git clone https://github.com/Keplercodes01/dummygrad.git
cd dummygrad
pip install pybind11
pip install .
```

---

## Usage

```python
import dummygrad as dummy

# create tensors
a = dummy.randn([3, 4])
b = dummy.randn([4, 2])

# forward pass
c = dummy.matmul(a, b)
d = dummy.relu(c)
loss = dummy.mean(d)

# backward pass
loss.backward()

# inspect gradients
a.show_grad()
```

### Matrix multiply with @ operator

```python
c = a @ b
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
# broadcast a [1, 3] tensor to [4, 3]
v = dummy.randn([1, 3])
v_broadcast = dummy.broadcast(v, axis=0, n=4)

# collapse back along axis
v_collapsed = dummy.collapse(v_broadcast, axis=0)
```

### Manual seed

```python
dummy.manual_seed(42)
```

---

## Training a simple MLP

```python
import dummygrad as dummy

dummy.manual_seed(42)

# weights and biases
W1 = dummy.randn([784, 128])
b1 = dummy.randn([1, 128])
W2 = dummy.randn([128, 10])
b2 = dummy.randn([1, 10])

optimizer = dummy.Adam(lr=0.001)

for step in range(100):
    # forward
    x = dummy.randn([32, 784])  # batch of 32
    h = dummy.relu(dummy.add(dummy.matmul(x, W1), dummy.broadcast(b1, 0, 32)))
    logits = dummy.add(dummy.matmul(h, W2), dummy.broadcast(b2, 0, 32))
    probs = dummy.softmax(logits)

    # loss (you provide one-hot targets)
    loss = dummy.CrossEntropyLoss(probs, targets)
    loss.backward()

    # update
    for param in [W1, b1, W2, b2]:
        optimizer.step(param)
        param.zero_grad()
```

---

## Design Philosophy

No excessive abstraction. Every major step is explicit — if you don't know what softmax does before a cross entropy loss, you shouldn't be touching the engine. Sharp tools for sharp engineers.

The core autograd loop is ~835 lines of pure C++. No dependencies except the standard library.

---

## Project Structure

```
dummygrad/
├── setup.py
└── src/
    ├── engine.h        # entire engine — tensors, ops, autograd, optimizers
    └── bindings.cpp    # pybind11 Python bindings
```

---

## Built by

[@Keplercodes01](https://github.com/Keplercodes01)
