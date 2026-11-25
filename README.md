# dummygrad

[![CI](https://github.com/Keplercodes01/dummygrad/actions/workflows/ci.yml/badge.svg)](https://github.com/Keplercodes01/dummygrad/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keplercodes01/dummygrad/blob/main/notebooks/dummygrad_colab.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Keplercodes01/dummygrad/blob/main/notebooks/dummygrad_colab.ipynb)

A pure C++ deep learning engine engineered for absolute speed, zero abstraction bloat, and minimal codebase maintenance.

Got rid of Python and its bindings, GIL, multi-layer dispatcher overhead. It's a raw, frontier-grade execution directly on the metal.


```cpp
#include "dummy_core.h"

manual_seed(0);
Sequential net;
net.add<Linear>(2, 64).add<GELU>().add<Linear>(64, 1);
optim::AdamW opt(1e-2f);

auto x = normal({256, 2});
auto y = sum(square(x), 1);                                    // learn |x|^2

for (int step = 0; step < 2000; ++step) {
    net.zero_grad();
    auto loss = mse_loss(net.forward(x), y);
    loss->backward();
    opt.step(net.parameters());
}

// Gradients of gradients, like everything else, are ordinary ops
auto point = tensor({1, 2}, {1, 2}, Device::CPU, /*requires_grad=*/true);
auto g = autograd::grad({sum(net.forward(point))}, {point}, {}, true, /*create_graph=*/true)[0];
```

| | |
|---|---|
| **Autograd** | gradients of any order, Jacobians, Hessians, hooks, checkpointing, custom ops, anomaly detection |
| **Ops** | about 300: broadcasting, views, indexing, reductions, sorting, `einsum`, convolutions, pooling, losses |
| **Layers** | transformers (RoPE, GQA, SwiGLU, RMSNorm), mixture of experts, convolutions, RNN/LSTM/GRU, normalization, dropout |
| **Language models** | Llama, Mistral, Qwen2 and GPT-2 checkpoints, tokenizers, generation with a KV cache, beam search, LoRA |
| **Training** | 15 optimizers, 12 schedules, fp16/bf16 autocast, multi-GPU, resumable checkpoints, data loaders, EMA |
| **Numerics** | `linalg`, `fft`, `distributions`, random numbers that repeat on every device |
| **Devices** | NVIDIA GPUs (CUDA) and CPUs (x86, ARM); experimental Metal and TPU |
| **Reproducible** | gradients are summed in a fixed order, with no atomics: the same code and seed give the same bits |

**Contents:** [Getting started](#getting-started) · [Research toolkit](#research-toolkit) · [Training](#a-complete-training-run) · [Language models](#language-models-load-generate-finetune) · [Mixed precision](#mixed-precision) · [Your own ops](#writing-your-own-ops) · [Performance](#performance) · [Testing](#testing) · [Limitations](#status--known-limitations) · [Changelog](CHANGELOG.md)

---

## What's New

**v0.9.0: the research release.** Higher-order autograd (`autograd::grad` with `create_graph`), about 300 ops, convolutions of every kind, recurrent and transformer layers, 15 optimizers, `linalg`, `fft` and `distributions`, and a CPU backend 2.4 to 3.3 times faster. Upgrading from v0.8.0:
- `mean`, `var` and `std_dev` reduce the dimension you name. On tensors with more than 2 dimensions, 0 and 1 used to mean the last two.
- A second backward through a freed graph throws instead of silently losing gradients. Pass `retain_graph=true` the first time.

Also new: `a + b` and `x->index({Slice(1, 3), 0})`, KV caches and text generation, Hugging Face checkpoints and tokenizers, LoRA, mixture of experts, background data loading, a GPU allocator that splits blocks, and masked attention, sorting, scatter and FFT on the GPU.

**v0.8.0: the audit release.** Every finding of a full audit fixed, 12-21% faster in fp32 and 4-11% in fp16 on a T4, and multi-GPU training that works.

Earlier releases are in [CHANGELOG.md](CHANGELOG.md).

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│  Models    GPT, Llama family, Transformer, MARS, generation  │
├──────────────────────────────────────────────────────────────┤
│  Modules   layers, recurrent, attention, optimizers          │
├──────────────────────────────────────────────────────────────┤
│  Ops       free functions; each one a value and its gradient │
├──────────────────────────────────────────────────────────────┤
│  Autograd  any order, deterministic sums, hooks              │
├──────────────────────────────────────────────────────────────┤
│  Kernels   one source for CPU and GPU + cuBLAS, attention    │
├──────────────────────────────────────────────────────────────┤
│  Memory    caching allocators, versioned storage             │
└──────────────────────────────────────────────────────────────┘
```

- **One kernel, two backends**: general op kernels are functors in [`src/kernels`](src/kernels), compiled both as threaded CPU loops and as CUDA kernels, so the code the CPU tests cover is the code the GPU runs. The hot training kernels (matmul, attention, norms, cross entropy) are written for the GPU by hand.
- **Low per-op overhead**: ops are plain C++ functions with no Python or dispatcher layer, which is why small models step several times faster than in PyTorch (see [Performance](#performance)).
- **cuBLAS matmuls**: `Float32` (TF32 tensor cores only with `set_allow_tf32(true)`, off by default as in PyTorch), and `Float16` / `BFloat16` under `AutocastGuard`. Linear adds its bias in the cuBLASLt epilogue; the tied LM head (`matmul_nt`) reads the embedding transposed in place.
- **Attention**: one fused op for self, cross, masked and cached attention. cuBLAS GEMMs for sequences up to 2048 tokens (any head dimension), and for long unmasked self-attention FlashAttention-style tiled kernels, with memory linear in sequence length (head dimensions up to 128; everything else uses the GEMM path a chunk of heads at a time). Every path is deterministic.
- **Memory**: GPU memory is split and joined as needed, so batches of changing sizes reuse it; copies to the GPU go through page-locked memory and do not make the program wait.
- **Fused, deterministic kernels**: RMSNorm, LayerNorm, cross entropy, embedding, RoPE, GQA head repetition and AdamW; reductions sum in a fixed order (no atomics), so identical runs give bitwise-identical gradients.
- **Multi-GPU**: `MultiGPURunner` runs one replica per GPU in a single process and averages gradients GPU to GPU after each backward pass.
- **Apple Silicon**: MPS tensors live in unified memory; a few elementwise ops, masks, loss gradients and AdamW run as Metal kernels, everything else on the CPU over the same memory. Not covered by tests.
- **CPU**: threaded, vectorized kernels (AVX2, AVX-512 or NEON, whatever the machine has) around OpenBLAS, Accelerate or MKL, or a built-in matrix multiply without one. dummygrad's own kernels give the same results at any thread count.

---

## Getting Started

### Try it in your browser (no install)

The fastest way to see dummygrad run is on a free GPU in Colab or Kaggle. The same notebook works on both.

**Colab:**
1. Click the **Open in Colab** badge at the top of this page.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**.

**Kaggle** (if Colab is unavailable to you):
1. Click the **Open in Kaggle** badge at the top of this page.
2. In the **Settings** panel, set **Accelerator → GPU T4 x2** and turn **Internet** on (Kaggle asks for phone verification once).
3. Select **Run All**.

The notebook builds dummygrad with CUDA, runs the test suite, the quickstart example, and a transformer training demo — about 5 minutes end to end.

### 1. Install the prerequisites

| Platform | Backends | Status |
|---|---|---|
| Linux (Ubuntu / Debian) | CPU, CUDA | CPU tested in CI on every change; CUDA tested on Colab (T4, CUDA 13) |
| macOS (Apple Silicon) | CPU, Metal | CPU tested in CI on every change; Metal kernels not covered by tests |
| Windows | CPU, CUDA via WSL2 | Use the Linux steps inside WSL2 |

You need a C++17 compiler (GCC ≥ 9 or Clang ≥ 11), CMake ≥ 3.18 and git. CUDA and BLAS are optional.

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y build-essential cmake git libopenblas-dev
cmake --version          # needs 3.18+; on older distros: pip install --upgrade cmake
```

**macOS**
```bash
xcode-select --install   # Apple's compiler toolchain
brew install cmake git
```

**Windows:** install [WSL2 with Ubuntu](https://learn.microsoft.com/windows/wsl/install), then follow the Ubuntu steps inside it. For GPU support, use NVIDIA's [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) guide.

**NVIDIA GPU (optional):** install the NVIDIA driver and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.0 or newer), then check both are visible:
```bash
nvidia-smi               # shows your GPU
nvcc --version           # if not found: export PATH=/usr/local/cuda/bin:$PATH
```

### 2. Build

```bash
git clone https://github.com/Keplercodes01/dummygrad.git
cd dummygrad
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake prints whether it found CUDA (`Building with GPU support`) or is building CPU-only. Either is fine — dummygrad picks the backend at configure time.

| CMake option | Default | Effect |
|---|---|---|
| `ENABLE_CUDA` | `ON` | Build the CUDA backend when `nvcc` is found; set `OFF` to force CPU-only |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (CMake ≥ 3.24) | GPU generations to compile for, e.g. `75` for T4, `80` for A100, `89` for RTX 40xx |
| `DUMMYGRAD_USE_BLAS` | `ON` | Use a BLAS library for CPU matrix products when one is found |
| `DUMMYGRAD_BUILD_TESTS` | `ON`* | Build the test suites: `test_engine`, `test_research`, `test_models` |
| `DUMMYGRAD_BUILD_EXAMPLES` | `ON`* | Build the examples: `quickstart`, `custom_op`, `research_tour`, `train_lm`, `generate`, `train_colab` |

\* Only when dummygrad is the top-level project; off when used inside your own project.

### 3. Check that it works

```bash
./build/test_engine      # gradient checks, CUDA-vs-CPU checks, engine tests, training
./build/test_research    # first and second derivatives, values, layers, optimizers
./build/test_models      # indexing, generation, checkpoints, tokenizers, data loading, memory
./build/quickstart       # a tour of the API, trains a small network on XOR
```

The suites should end with `N passed, 0 failed` and take a few seconds; see [Testing](#testing). `quickstart` should show the loss falling towards zero. You can also run everything through CTest: `cd build && ctest --output-on-failure`.

### 4. Your first program

[`examples/quickstart.cpp`](examples/quickstart.cpp) is a complete, runnable program. The essentials:

```cpp
#include "dummy_core.h"

// Tensors are std::shared_ptr<Tensor>. requires_grad=true makes autograd track them.
auto x = ones({2, 3}, Device::CPU, DType::Float32, /*requires_grad=*/true);

// Ops are free functions that build the graph as they run.
auto y = sum(x * 3.0f + 1.0f);
y->backward();            // x->grad now holds dy/dx
x->show_grad();           // prints 3s

// Layers own their parameters; move them to a device with .to().
Linear hidden(2, 8), output(8, 1);
hidden.to(Device::CUDA);  // or Device::CPU
output.to(Device::CUDA);
std::vector<std::shared_ptr<Tensor>> params = {hidden.W, hidden.b, output.W, output.b};
Adam optimizer(0.05f);

for (int step = 0; step < 500; ++step) {
    optimizer.zero_grad(params);
    auto loss = mse(output.forward(tanh(hidden.forward(X))), Y);
    loss->backward();
    optimizer.step(params);
}
```

A few conventions worth knowing:
- Inputs and parameters live on one device; copy data over with `tensor->to(Device::CUDA)` and read results back with `tensor->item()` (one element) or `tensor->to_vector()`.
- Integer data such as token ids and class labels is stored in `Float32` tensors.
- Collect a layer's `parameters()` *after* calling `.to()` — moving replaces the tensors.

- `+ - * /` work between tensors (with broadcasting) and numbers; matrix products are `matmul(a, b)`. `a += b` makes `a` a new tensor. `==` and `<` compare pointers: use `eq(a, b)`, `lt(a, b)`.
- Index as in Python: `x->index({0, Slice(1, 3), Ellipsis, -1})`, `x->index({rows, cols})` with index tensors, `x->index({Mask(gt(x, 0.0f))})`; `x->index_put({...}, value)` returns a changed copy. Both are differentiable. `std::cout << x` prints.
- Reductions over a dimension keep it as size 1 unless you pass `keepdim=false`: `sum(x, 1, false)`. `sum(x)` is a scalar.

[`examples/research_tour.cpp`](examples/research_tour.cpp) goes further: second derivatives, linear algebra, a variational autoencoder, a convolutional network.

### 5. Use dummygrad in your own project

dummygrad is a CMake library. The simplest way to depend on it is `FetchContent`, which downloads and builds it with your project:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(dummygrad
    GIT_REPOSITORY https://github.com/Keplercodes01/dummygrad.git
    GIT_TAG        v0.9.0)
FetchContent_MakeAvailable(dummygrad)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE dummygrad::dummygrad)
```

Then `#include "dummy_core.h"` in `main.cpp` and build with the usual `cmake -S . -B build && cmake --build build`. CUDA is enabled automatically when `nvcc` is found, and the `ENABLE_CUDA` / `CMAKE_CUDA_ARCHITECTURES` options above work from your project too.

Prefer vendoring? Add the repository as a git submodule and replace the `FetchContent` lines with `add_subdirectory(external/dummygrad)`.

### Building models with Module

Every layer and model is a `Module`. Declare a module's parameters and children once in `visit()`, and parameter collection, device moves, train/eval mode, and checkpointing come with it:

```cpp
struct MyBlock : Module {
    Linear proj{64, 64};
    RMSNorm norm{64};
    std::shared_ptr<Tensor> scale = ones({1, 64}, Device::CPU, DType::Float32, true);

    void visit(ModuleVisitor& v) override {
        v.module("proj", proj);           // children: their parameters are named "proj.weight", ...
        v.module("norm", norm);
        v.parameter("scale", scale);      // your own trainable tensors
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        return mul(proj.forward(norm.forward(x)), scale);
    }
};

MyBlock block;
Adam optimizer(1e-3f);
optimizer.step(block.parameters());       // all parameters, recursively, each listed once
block.to(Device::CUDA);                   // parameters and buffers
block.eval();                             // train/eval mode, propagated to children
block.num_parameters();
block.save_safetensors("block.safetensors");
block.load_safetensors("block.safetensors");   // strict: mismatched names or shapes throw
```

- For a list of children, use indexed names: `v.module("blocks." + std::to_string(i), *blocks[i])`.
- Non-trainable state that should move with the module goes in `v.buffer(name, tensor)`; pass `/*persistent=*/false` for state you recompute (like RoPE tables) so it isn't saved.
- `load_safetensors(path, /*strict=*/false)` loads the matching subset of a checkpoint, e.g. to initialize part of a new model. Shape mismatches still throw.
- `requires_grad_(false)` freezes a module's parameters; `.to()` keeps them frozen.

### A complete training run

[`examples/train_lm.cpp`](examples/train_lm.cpp) is a template for real experiments: a character-level language model with train/validation data, AdamW with warmup and cosine decay, gradient clipping, metric logging, periodic evaluation, sampling, and resumable checkpoints. Run `./build/train_lm my_text.txt` on any text file, and add `--resume` to continue after an interruption. The core of it:

```cpp
auto parts = TokenDataset::from_bytes("my_text.txt").split(0.9);   // or from_file("train.bin") for uint16 ids
TokenDataset train = parts.first, val = parts.second;

ModernTransformer model(256, seq_len, 64, 4, 2, 2);
model.to(device);
auto params = model.parameters();
Adam optimizer(3e-3f, 0.9f, 0.95f, 1e-8f, /*weight_decay=*/0.1f);
optimizer.set_weight_decay(no_weight_decay_params(model), 0.0f);  // no decay for biases and norm gains
CosineAnnealingLR schedule(3e-3f, steps, /*warmup_steps=*/20, /*min_lr=*/3e-4f);
MetricsLogger logger("metrics.csv");                    // step,metric,value
std::mt19937 rng(0);

for (int64_t step = 0; step < steps; ++step) {
    optimizer.lr = schedule.get_lr();
    Batch b = train.sample(batch_size, seq_len, rng, device);   // targets = inputs shifted by one
    optimizer.zero_grad(params);
    auto loss = fused_cross_entropy(model.forward(b.inputs), b.targets);
    loss->backward();
    float grad_norm = clip_grad_norm_(params, 1.0f);
    optimizer.step(params);
    schedule.step();
    logger.log(step, {{"train_loss", loss->cpu()->data_ptr<float>()[0]}, {"grad_norm", grad_norm}});

    if ((step + 1) % 500 == 0)                         // model, optimizer state and step in one file
        save_checkpoint("ckpt.safetensors", model, &optimizer, TrainingState{step + 1, {{"val_loss", val_loss}}});
}

// Later, or after a crash:
TrainingState state = load_checkpoint("ckpt.safetensors", model, &optimizer);   // continue from state.step
```

- **Checkpoints** hold the model, the optimizer's moment estimates and step counts, and any scalars you add (`TrainingState::scalars`), and are written atomically, so a crash mid-save keeps the previous file intact. Resuming reproduces an uninterrupted run exactly: gradients are summed in a fixed order, so the same code with the same seed gives bitwise-identical results on CPU.
- **`DataLoader(X, Y, batch_size, shuffle, seed)`** batches in-memory tensors along their first dimension, or the samples of your own `Dataset` (files to read, images to augment); call `set_epoch(e)` for a deterministic per-epoch shuffle, `prefetch(workers)` to make batches on background threads, and iterate with `for (auto [x, y] : loader)`.
- **`TokenDataset`** memory-maps token streams (`from_file` for raw uint16/int32 ids, `from_bytes` for byte-level text), samples random windows with `sample()`, and walks the whole stream for evaluation with `windows()`. `TokenLoader` draws windows ahead of time; batch i depends only on the seed and i, so resumed runs see the same data.
- **Optimizers:** `Adam` (AdamW, decoupled weight decay) and `SGD` (momentum, weight decay); the rest are in `optim::` (see [Research toolkit](#research-toolkit)) and work the same way. Their state lives on each parameter's device and follows the model if you move it.
- **`MetricsLogger`** writes long-format CSV, so any metric can be logged at any step. Load it with `pandas.read_csv(path).pivot(index="step", columns="metric", values="value")`.

### Mixed precision

On CUDA, `AutocastGuard` runs every matmul, and the matrix products inside attention, on fp16 (or bf16) tensor cores. Inputs are rounded to half precision, products accumulate in fp32, and outputs are fp32. The attention softmax, parameters, gradients, optimizer state and every other op stay fp32, so nothing else in your code changes. Each matmul's backward reuses the half-precision copies its forward made, so the conversion is paid once. fp16 needs loss scaling so that small gradients don't round to zero, and `GradScaler` handles it:

```cpp
GradScaler scaler;                                       // skip it for bf16
for (int64_t step = 0; step < steps; ++step) {
    optimizer.zero_grad(params);
    std::shared_ptr<Tensor> loss;
    {
        AutocastGuard autocast(DType::Float16);          // matmuls in here run on tensor cores
        loss = fused_cross_entropy(model.forward(b.inputs), b.targets);
    }
    scaler.scale(loss)->backward();                      // backward reuses each matmul's forward precision
    scaler.unscale(params);                              // true gradients, e.g. for clipping
    clip_grad_norm_(params, 1.0f);
    scaler.step(optimizer, params);                      // skipped if a gradient overflowed
    scaler.update();
}
```

The scaler follows PyTorch's `GradScaler`: it starts at 65536, halves on overflow and doubles after 2000 clean steps. `scaler.skipped_steps()` counts skipped steps, and `scaler.save_to(state)` / `scaler.load_from(state)` carry it through `TrainingState` checkpoints. Use `dg_bench --fp16` to measure the speedup on your GPU.

### Language models: load, generate, finetune

[`examples/generate.cpp`](examples/generate.cpp) is a complete program: `./build/generate path/to/Llama-3.2-1B "The capital of France is"`. The folder holds a model as downloaded from Hugging Face: `config.json`, `tokenizer.json` and `model.safetensors` or its shards.

```cpp
auto model = load_llama("Llama-3.2-1B");                   // Llama, Mistral, Qwen2; load_gpt2 for GPT-2
auto tok = Tokenizer::from_file("Llama-3.2-1B/tokenizer.json");
model.to(Device::CUDA);

GenerateOptions o;                                         // also top_k, min_p, repetition_penalty, eos, on_tokens
o.max_new_tokens = 100;
o.temperature = 0.8f;                                      // 0: greedy
o.top_p = 0.95f;
std::cout << tok.decode(generate(model, tok.encode_tensor("Once upon a time"), o));
auto best = beam_search(model, tok.encode_tensor("Once upon a time"), {});

// Finetune with low-rank adapters: everything else is frozen
add_lora(model, /*rank=*/8, /*alpha=*/16.0f, {"W_q", "W_v"});
auto params = trainable_parameters(model);
// ... train params as usual ...
save_lora(model, "adapters.safetensors");
merge_lora(model);                                         // into the weights
save_llama(model, "my-model");                             // readable by transformers

auto mine = Tokenizer::train(text, 4096, {"<|end|>"});     // byte-level BPE
MixtureOfExperts moe(512, 1024, /*experts=*/8, /*top_k=*/2);   // add moe.aux_loss to your loss
```

- Loading is tested against plain-loop implementations of the Hugging Face forward passes on small random models, in both directions for the Llama family. No full-size checkpoint is part of the tests.
- Models load in Float32 (a 1B model takes about 5 GB). Prompts in a batch must have the same length.
- Tokenizers split words exactly for ASCII and for Latin, Greek, Cyrillic, Hebrew, Arabic, Korean, Japanese and Chinese letters, approximately for other scripts, and do not apply Unicode normalization. Decoding is always exact.

### Research toolkit

[`examples/research_tour.cpp`](examples/research_tour.cpp) is a runnable tour. Tensors are `TensorPtr` (`std::shared_ptr<Tensor>`), ops are free functions with PyTorch's names and, with few exceptions, its semantics. Differentiable ops support second derivatives.

```cpp
// Higher-order gradients: differentiate through a gradient
auto dx = autograd::grad({score}, {x}, {}, /*retain_graph=*/true, /*create_graph=*/true)[0];
auto penalty = mean(square(sub_scalar(norm(dx, 2.0f, 1), 1.0f)));
penalty->backward();
auto H = autograd::hessian(f, x);                          // also jacobian, hvp, vjp, jvp

// Recompute a block in backward instead of storing its activations
auto y = autograd::checkpoint([&](const TensorPtr& t) { return block.forward(t); }, x);

// A new op from a value and a vector-Jacobian product; second derivatives come for free
TensorPtr swishy(const TensorPtr& x) {
    return autograd::custom_function("swishy", [](const TensorList& in) { return mul(in[0], sigmoid(in[0])); }, {x}, {/*save input*/ 0},
        [](const OpContext& c, const TensorPtr& g) -> TensorList {
            auto v = c.input(0); auto s = sigmoid(v);
            return {mul(g, mul(s, add_scalar(mul(v, rsub_scalar(1.0f, s)), 1.0f)))};
        });
}

auto scores = einsum("bqd,bkd->bqk", {q, k});
auto [u, s, vt] = linalg::svd(a);                          // linalg: qr, eigh, cholesky, solve, ...
auto spectrum = fft::rfft(signal);                         // complex tensors have a trailing dimension of 2
distributions::Normal posterior(mu, sigma);
auto z = posterior.rsample();                              // gradients flow to mu and sigma

auto net = std::make_shared<Sequential>();
net->add<Conv2d>(3, 64, 3, 1, 1).add<BatchNorm2d>(64).add<ReLU>().add<MaxPool2d>(2).add<Flatten>().add<Linear>(64 * 16 * 16, 10);
LSTM lstm(128, 256, {/*layers=*/2, /*dropout=*/0.1f, /*bidirectional=*/true});
optim::Muon muon(0.02f);                                   // optim: SGD, AdamW, NAdam, RAdam, Lion, LBFGS, ...
optim::OneCycleLR schedule(0.02f, total_steps);
```

| Area | What is there |
|---|---|
| Autograd | `grad`, `create_graph`, `jacobian`, `hessian`, `hvp`, `vjp`, `jvp`, hooks, `retain_grad`, `checkpoint`, `custom_function`, anomaly detection, in-place-modification checks |
| Ops | elementwise, broadcasting, views, indexing and scatter, reductions, sorting, `einsum`, convolutions, pooling, resampling, losses including CTC |
| Layers | containers, convolutions, normalizations, dropout, recurrent, attention, transformers, pooling, embeddings |
| Optimizers | SGD, Adam(W), AMSGrad, NAdam, RAdam, Adamax, Adagrad, Adadelta, RMSprop, Rprop, LBFGS, Lion, Muon, Adafactor; 12 schedules; EMA, SWA |
| Numerics | `linalg`, `fft`, `distributions`, reproducible counter-based random numbers |

On a GPU, `linalg` factorizations and a few searches (`unique`, `searchsorted`, `bincount`, `histc`, `nonzero`) run on the host and copy their data there and back; nearly everything else runs on the device. Run `./build/test_research` to check all of it on your machine.

### Writing your own ops

`make_op` and `autograd::custom_function` ([Research toolkit](#research-toolkit)) are the short way. For full control over the kernels, a new op is a forward function plus a backward `Node`. [`examples/custom_op.cpp`](examples/custom_op.cpp) is a complete, compiled example (squared ReLU). The shape of it:

```cpp
struct SquaredReluBackward : public Node {
    std::shared_ptr<Tensor> x;                              // what backward needs
    explicit SquaredReluBackward(std::shared_ptr<Tensor> x) : x(std::move(x)) {}
    void release_variables() override { x.reset(); }        // free saved tensors after backward
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        // grads[0] is dL/dy (always dense); return dL/dx, one entry per input
    }
};

std::shared_ptr<Tensor> squared_relu(const std::shared_ptr<Tensor>& x_in) {
    DG_PROFILE_OP("squared_relu");             // optional: an entry in the forward profile
    auto x = contiguous(x_in);                 // kernels that index raw memory need dense inputs
    auto out = /* compute y = relu(x)^2 */;
    out->requires_grad = needs_grad(x);        // false inside NoGradGuard
    if (needs_grad(x)) {
        auto grad_fn = std::make_shared<SquaredReluBackward>(x);
        grad_fn->add_next_edge(get_grad_edge(x).function);   // one edge per input, same order as apply()
        out->grad_fn = grad_fn;
    }
    return out;
}
```

Then check the backward pass against finite differences before trusting it in a model:

```cpp
std::string err = dummygrad::gradcheck(
    [](const std::vector<std::shared_ptr<Tensor>>& in) { return squared_relu(in[0]); },
    {randn({4, 5})});
if (!err.empty()) std::cerr << err << "\n";   // e.g. "input 0 element 3: analytic 0.12 vs numeric 0.48"
```

`gradcheck` works for any function of tensors — a single op, a layer, or a whole block. Use CPU inputs (GPU matmuls under TF32 or autocast are too imprecise for finite differences) and keep inputs away from kinks such as `relu` at 0.

### Evaluation and debugging

```cpp
// No graph is recorded inside the scope: less memory and time for evaluation and inference
{
    NoGradGuard no_grad;
    auto val_logits = model.forward(val_inputs);
}

// Stop at the first NaN/Inf gradient and name the op that produced it
{
    DetectAnomalyGuard detect;
    loss->backward();   // throws e.g. "detect_anomaly: RdivScalarBackward produced Inf in the gradient for its input 0"
}
```

Anomaly detection scans every gradient (and copies GPU gradients to the host), so enable it only while hunting a problem.

To see where a step's time goes, profile a few steps. Forward ops are listed by function name (`linear`, `matmul`, `rms_norm`, `autocast_cast` for the fp16 conversions) and backward nodes by node name (`LinearBackward`, `MatmulBackward`):

```cpp
reset_profile();
set_profiling(true);
for (int i = 0; i < 3; ++i) { optimizer.zero_grad(params); loss_fn(model.forward(x), y)->backward(); }
set_profiling(false);
for (const auto& e : profile_report(ProfilePhase::Forward))   // or ProfilePhase::Backward
    std::printf("%8.2f ms  %4lld calls  %s\n", e.total_ms / 3, (long long)e.calls / 3, e.name.c_str());
```

The profiler synchronizes the GPU at every op so each time is exact, which makes profiled steps slower than normal ones. A forward op's time is its own work only: ops it calls have their own entries. Ops you write show up once their forward starts with `DG_PROFILE_OP("my_op");`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA compiler not found. Building CPU-only.` | `nvcc` isn't on your `PATH`. Run `export PATH=/usr/local/cuda/bin:$PATH`, or pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. Delete `build/` and configure again. |
| `CMake 3.18 or higher is required` | `pip install --upgrade cmake`, then open a new shell. |
| `nvcc fatal : Unsupported gpu architecture` | Your CMake or CUDA is too old for `native`. Pass your GPU's compute capability, e.g. `-DCMAKE_CUDA_ARCHITECTURES=75`. |
| `Illegal instruction` when running a binary on another machine | Builds use `-march=native`. Rebuild on the machine that runs the program. |
| `BLAS found` is missing from the CMake output | Optional. Install `libopenblas-dev` (Linux) for faster CPU matmuls. |
| A `test_engine` check fails | Please [open an issue](https://github.com/Keplercodes01/dummygrad/issues) with the full output and your GPU / OS. |

---

## Example: Training a Transformer (GPT) in C++

```cpp
#include "dummy_core.h"
#include <iostream>

int main() {
    // Hyperparameters
    int vocab_size  = 50257;
    int max_seq_len = 1024;
    int d_model     = 768;
    int n_heads     = 12;
    int n_layers    = 12;

    int B = 4;  // batch size
    int T = 64; // sequence length

    // 1. Initialize GPT model and AdamW optimizer
    GPT model(vocab_size, max_seq_len, d_model, n_heads, n_layers);
#ifdef USE_CUDA
    model.cuda();
    Device device = Device::CUDA;
#else
    model.cpu();
    Device device = Device::CPU;
#endif

    auto params = model.parameters();
    Adam optimizer(1e-4f);

    // 2. Training loop
    for (int step = 1; step <= 100; ++step) {
        optimizer.zero_grad(params);

        // Generate synthetic training batch
        auto in_cpu = std::make_shared<Tensor>(std::vector<int64_t>{B, T}, Device::CPU, DType::Float32, false);
        auto tgt_cpu = std::make_shared<Tensor>(std::vector<int64_t>{B, T}, Device::CPU, DType::Float32, false);
        for (int i = 0; i < B * T; ++i) {
            in_cpu->data_ptr<float>()[i]  = static_cast<float>((step + i) % vocab_size);
            tgt_cpu->data_ptr<float>()[i] = static_cast<float>((step + i + 1) % vocab_size);
        }

        auto inputs  = (device == Device::CUDA) ? in_cpu->to(Device::CUDA) : in_cpu;
        auto targets = (device == Device::CUDA) ? tgt_cpu->to(Device::CUDA) : tgt_cpu;

        // Forward pass -> Fused cross entropy loss -> Backward pass -> Step
        auto logits = model.forward(inputs);
        auto loss = fused_cross_entropy(logits, targets);

        loss->backward();
        optimizer.step(params);

        if (step % 10 == 0) {
            float loss_val = loss->cpu()->data_ptr<float>()[0];
            std::cout << "Step " << step << " | Loss: " << loss_val << std::endl;
        }
    }

    return 0;
}
```

---

## Example: Multi-GPU Data-Parallel Training

```cpp
#include "dummy_core.h"

int main() {
    int num_gpus = 2;  // e.g. Kaggle's T4 x2

    // One replica per GPU in a single process. The factory runs with GPU `rank` current; replica 0's
    // weights are then copied to the others.
    MultiGPURunner<ModernTransformer> runner(num_gpus, [](int rank) {
        ModernTransformer model(32000, 2048, 512, 8, 4, 2);
        model.cuda();
        return model;
    });
    std::vector<Adam> opts(num_gpus, Adam(3e-4f));

    for (int step = 0; step < 1000; ++step) {
        // Forward and backward on every GPU at once, each on its own shard of the batch; then the
        // gradients are averaged across GPUs (GPU to GPU, in a fixed order)
        runner.parallel_step([&](int rank, ModernTransformer& model) {
            auto inputs  = zeros({2, 32}, Device::CUDA);  // load this rank's shard here
            auto targets = ones({2, 32}, Device::CUDA);
            model.zero_grad();
            fused_cross_entropy(model.forward(inputs), targets)->backward();
        });
        // Identical gradients and identical optimizer steps keep the replicas identical
        runner.for_each_replica([&](int rank, ModernTransformer& model) { opts[rank].step(model.parameters()); });
    }
    return 0;
}
```

Gradients are averaged, as in PyTorch DDP (`runner.average_gradients = false` sums them). An exception on any GPU is rethrown by `parallel_step`. `all_reduce_gradients(replica_parameters)` does the reduction on its own for your own training loops.

---

## TPU (experimental)

An experimental TPU backend loads the OpenXLA PJRT runtime (`libtpu.so`) at run time. Sigmoid, SiLU, LeakyReLU, MSE and L1 run as compiled XLA programs on the TPU (each distinct program compiles once and is cached); every other op runs on the host, which holds TPU tensors' authoritative copy. `MultiTPURunner` runs one replica per chip and averages gradients on the host after each step. It is a prototype: nothing in CI or the notebook runs it.

```cpp
MultiTPURunner<GPT> tpu_runner([](size_t rank) {
    return GPT(vocab_size, max_seq_len, d_model, n_heads, n_layers);
});
tpu_runner.parallel_step([&](size_t rank, GPT& model) {
    auto loss = fused_cross_entropy(model.forward(batch_inputs[rank]), batch_targets[rank]);
    loss->backward();
});  // gradients averaged across chips
```

---

## SafeTensors

Model weights, optimizer state and datasets are read and written in the `.safetensors` format of Hugging Face and PyTorch, in C++ alone. Files are memory-mapped, so `io::SafeTensorsFile` can read single tensors out of a checkpoint larger than memory. F32, F16, BF16 and I8 tensors keep their type; F64, integer and boolean ones become Float32.

```cpp
io::save_safetensors("model.safetensors", {{"gate.weight", w_gate}, {"norm.gamma", gamma}});
auto tensors = io::load_safetensors("model.safetensors", Device::CPU);   // all of them
auto one = io::SafeTensorsFile("model.safetensors").read("norm.gamma");  // or one
```

---

## Modern LLM Research Primitives (LLaMA 3, Mistral, DeepSeek Stack)

`dummygrad` comes built-in with the complete modern LLM architectural stack:

- **RMSNorm**: Root Mean Square Layer Normalization without mean subtraction ($\frac{x}{\text{RMS}(x)} \odot \gamma$) with analytical autograd backward pass.
- **RoPE (Rotary Position Embeddings)**: Continuous coordinate rotation for Query and Key tensors (`precompute_freqs_cis`, `apply_rotary_emb`) with orthogonal $R^T$ backward pass.
- **SwiGLU**: Gated feed-forward network ($\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$) fully differentiable across all backends.
- **Gradient Clipping**: In-place global $L_2$ norm clipping (`clip_grad_norm_`) preventing exploding gradients in deep transformer training.
- **Cosine Annealing with Warmup**: `CosineAnnealingLR` scheduler for optimal convergence.

```cpp
#include "dummy_core.h"

// 1. RMSNorm
RMSNorm norm(768);
auto x_norm = norm.forward(x);

// 2. Rotary Position Embeddings (RoPE)
auto [cos_t, sin_t] = precompute_freqs_cis(64, 4096);
auto q_rot = apply_rotary_emb(q, cos_t, sin_t);
auto k_rot = apply_rotary_emb(k, cos_t, sin_t);

// 3. SwiGLU Gated Feed-Forward Block
SwiGLU ffn(768);
auto ffn_out = ffn.forward(x_norm);

// 4. Gradient Clipping & LR Scheduler
loss->backward();
float total_norm = clip_grad_norm_(model.parameters(), 1.0f);

// 5. Full Turnkey ModernTransformer (LLaMA 3 / Mistral Architecture)
//    RMSNorm + RoPE + Grouped Query Attention (GQA) + SwiGLU + Tied Output Projection
ModernTransformer modern_model(
    /*vocab_size=*/32000,
    /*max_seq_len=*/4096,
    /*d_model=*/768,
    /*n_heads=*/12,
    /*n_layers=*/12,
    /*n_kv_heads=*/4 // GQA: 12 Query heads share 4 KV heads
);
auto modern_logits = modern_model.forward(input_ids);
auto loss = fused_cross_entropy(modern_logits, targets);
```

---

## Modern Attention Masking Techniques

`dummygrad` provides a comprehensive suite of hardware-accelerated modern masking primitives:

- **Sliding Window Attention (SWA)**: Used in Mistral 7B and Gemma 2 (`masking::sliding_window_mask`). Bounds memory and attention span to a local window $W$.
- **Prefix-LM Masking**: Used in T5, PaLM, and Fill-in-the-Middle (`masking::prefix_causal_mask`). Enables bidirectional prefix conditioning with causal generation.
- **ALiBi (Attention with Linear Biases)**: Eliminates positional embeddings with geometric head slopes $m_h = 2^{-8h/H}$ (`masking::alibi_bias`).
- **Arbitrary Key-Padding Mask**: Automatically broadcastable additive masks (`masking::create_padding_mask`) for variable-length batched training.
- **Document-Packing Block-Diagonal Mask**: Prevents attention contamination between multiplexed packed sequences (`masking::document_causal_mask`).

```cpp
#include "dummy_core.h"

// 1. Sliding Window Attention (e.g. 512 token local window for Mistral)
auto swa_scores = masking::sliding_window_mask(scores, 512);

// 2. Prefix-Causal Mask (first 256 tokens bidirectional prompt, rest causal)
auto prefix_scores = masking::prefix_causal_mask(scores, 256);

// 3. ALiBi Positional Bias
auto alibi = masking::alibi_bias(12, 1024); // [1, 12, 1024, 1024]
auto logits = gpt.forward(input_ids, /*pos_ids=*/nullptr, /*mask=*/alibi); // GPT takes an additive mask

// 4. Sequence Packing / Sample Multiplexing (LLaMA 3 block-diagonal mask)
auto doc_mask = masking::document_causal_mask(doc_ids);
auto out = gpt.forward(input_ids, /*pos_ids=*/nullptr, /*mask=*/doc_mask);
```

---

## Performance

Training-step time (forward, backward, AdamW) for the same ModernTransformer architecture (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) in dummygrad v0.9.0 and PyTorch 2.11 (eager, and `torch.compile`d) on a Kaggle T4. Median ms per step over at least 50 steps; each table is one session of `bench/compare.py`, which runs one program after the other. PyTorch runs at its best: fused AdamW, `F.rms_norm`, `scaled_dot_product_attention`.

**fp16 mixed precision** (`AutocastGuard` + `GradScaler`, against `torch.autocast` + `GradScaler`):

| config | params | v0.8.0 | v0.9.0 | PyTorch eager | PyTorch compile | peak MB: v0.9.0 / eager / compile |
|---|---|---|---|---|---|---|
| tiny (d=128, 2 layers, batch 8x64) | 0.4M | **1.6 ms** | **1.7 ms** | 9.7 ms | 5.5 ms | 18 / 33 / 30 |
| small (d=256, 4 layers, batch 16x128) | 3.9M | 14.2 ms | 11.4 ms | 18.6 ms | **9.8 ms** | 226 / 260 / 170 |
| medium (d=512, 6 layers, batch 8x256) | 23.1M | 49.1 ms | 40.3 ms | 43.4 ms | **30.2 ms** | 748 / 820 / 627 |
| large (d=768, 12 layers, batch 8x512) | 100.2M | 297 ms | 246 ms | 238 ms | **198 ms** | 4147 / 4592 / 3152 |

**fp32** (TF32 enabled on both sides; the T4 has none):

| config | v0.8.0 | v0.9.0 | PyTorch eager | PyTorch compile |
|---|---|---|---|---|
| tiny | **1.6 ms** | **1.6 ms** | 7.3 ms | 4.6 ms |
| small | 23.6 ms | 24.0 ms | 23.1 ms | **20.2 ms** |
| medium | 114 ms | 121 ms | 107 ms | **101 ms** |
| large | 887 ms | 852 ms | 854 ms | **808 ms** |

A T4 slows down as it heats up: in fp32 the same medium step took 102 ms at 54 °C and 130 ms at 81 °C, which is more than the differences between columns. With the two versions alternating in another fp32 session, v0.9.0 averaged 24.1 ms against v0.8.0's 24.7 (small), 116.3 against 116.4 (medium) and 925 against 942 (large), and PyTorch eager was level on small and 2 to 4% faster on medium.

dummygrad's advantage is per-op overhead (no Python, no dispatcher), which dominates small models. From small up, fp16 is 21 to 24% faster than in v0.8.0; eager PyTorch is 3% ahead on large, and `torch.compile` is ahead everywhere but tiny. `./build/dg_bench_ops` times single operations: generation, attention with masks, sorting, scatter, FFT, convolutions.

**CPU**, on two 4-thread CI machines with OpenBLAS, v0.8.0 and v0.9.0's CPU backend in the same job:

| config | v0.8.0 | v0.9.0 | v0.8.0 | v0.9.0 |
|---|---|---|---|---|
| tiny | 51.0 ms | **18.9 ms** | 39.5 ms | **12.0 ms** |
| small | 1086 ms | **460 ms** | 791 ms | **306 ms** |
 Reproduce with the [notebook](notebooks/dummygrad_colab.ipynb), or run `./build/dg_bench` and `python bench/bench_torch.py` on your own GPU; `--profile` breaks forward and backward time down by op.

---

## Testing

```bash
./build/test_research            # or with part of a name: ./build/test_research grad2/conv
./build/test_models
./build/test_engine
```

- **Gradients:** about 240 ops against finite differences, and nearly all of their second derivatives.
- **GPU against CPU**, when there is one: the same ops, optimizers, reductions, special values (NaN, infinity), and whole models trained for a few steps.
- **Values:** against definitions and closed forms (CTC against every alignment, convolutions against their sum, factorizations against their product, samplers against their moments, ...).
- **Models:** cached against uncached logits, sampling distributions, beam search against every continuation, checkpoints against plain-loop forward passes, tokenizers, adapters, experts, data loading, and the memory pool under random load.
- **Engine:** determinism, graph lifetime, exceptions, resuming runs bit for bit, two-GPU training.

On a Kaggle T4 x2 that is about 1280 checks; without a GPU, about 870. CI runs the CPU ones on Linux (with and without a BLAS) and macOS for every change and compiles the CUDA code; GPU runs happen on Kaggle.

---

## Status & Known Limitations

dummygrad is pre-1.0. The CPU and CUDA paths are covered by the test suites, in fp32 and, for training, fp16.

- **Mixed precision covers matrix products.** Under `AutocastGuard`, CUDA matmuls and attention (on its cuBLAS path, up to 2048 tokens) run on fp16/bf16 tensor cores. So do the activations between them in the built-in blocks: gated and activated feed-forward outputs, RoPE, queries, keys and values. Normalization, the residual stream, the attention softmax, and attention beyond 2048 tokens stay fp32, as do parameters, gradients of parameters, and optimizer state. bf16 needs an Ampere or newer GPU.
- **Attention** runs as cuBLAS GEMMs for sequences up to 2048 tokens (any head dimension) and, without masks, as tiled kernels beyond that for head dimensions up to 128 (other sizes zero-padded to 16, 32, 64 or 128). Larger head dimensions, masks and cross-attention past 2048 tokens use the GEMM path a chunk of heads at a time, whose memory is quadratic in the sequence length. Attention dropout, and masks that need gradients, run as separate ops.
- **Data types:** ops compute in fp32. There are no float64, integer, boolean or complex tensors: indices and masks are floats (exact up to 2^24), and complex numbers are pairs. `vmap`, sparse tensors, `torch.compile`-style fusion, and multi-process or sharded training are not implemented.
- **Multi-GPU** runs in one process and averages gradients GPU to GPU, one parameter at a time (no bucketing or overlap with backward yet). Its test needs two GPUs (Kaggle's T4 x2).
- **On the CPU even for GPU tensors:** `linalg` factorizations, `unique`, `searchsorted`, `bincount`, `histc`, `nonzero` and masked selection copy their data to the host and back. So does `fft` when CMake does not find cuFFT.
- **Generation** takes prompts of one length per batch and draws tokens on the host. There is no quantized (int8, 4-bit) inference.
- **Metal and TPU** paths are not covered by the test suite (macOS CI runs the CPU backend; there is no TPU in CI).

Bug reports with a failing test case are the fastest way to get something fixed.

---

## Built by

[@Keplercodes01](https://github.com/Keplercodes01)

