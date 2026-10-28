# dummygrad

[![CI](https://github.com/Keplercodes01/dummygrad/actions/workflows/ci.yml/badge.svg)](https://github.com/Keplercodes01/dummygrad/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keplercodes01/dummygrad/blob/main/notebooks/dummygrad_colab.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Keplercodes01/dummygrad/blob/main/notebooks/dummygrad_colab.ipynb)

A pure C++ deep learning engine engineered for absolute speed, zero abstraction bloat, and minimal codebase maintenance.

No Python. No bindings. No GIL. No multi-layer dispatcher overhead. Just raw, frontier-grade execution directly on the metal.

> *"I honestly made it for my selfish reasons. I train my neural nets in C++ with my engine. Read the source code—it's fast and it's fun."*

---

## What's New

Full details for every release are in [CHANGELOG.md](CHANGELOG.md).

### v0.7.0: PyTorch-level speed at 23M parameters, 6x faster at 0.4M
- **fp16 on a T4: 50.5 vs PyTorch's 50.7 ms at 23M parameters, 14.4 vs 21.9 ms at 3.9M, 1.7 vs 10.8 ms at 0.4M.** See [Performance](#performance).
- **Fused Linear:** the bias is added inside the matrix multiply instead of by a separate pass over the output, and one backward node returns all three gradients.
- **Backward runs on the calling thread for GPU training,** instead of hopping between worker threads, which made the two smaller models 2.1x and 20% faster. Gradients are bit-identical either way.
- **Forward profiling:** `dg_bench --profile` (or `set_profiling(true)`) now times forward ops too, including the fp16 conversions, so you can see where a whole step goes.

### v0.6.1: faster fp16 backward
- **fp16 medium step 57.2 → 53.8 ms on a T4** (PyTorch fp16: 51.8 ms). Each matmul's backward reuses the half-precision copies its forward made instead of converting its inputs again, which takes `MatmulBackward` from 16.7 to 13.9 ms. Saved activations are half-size copies too.
- **Frozen weights cost no gradient work:** when one matmul input doesn't require grad (a frozen base model under LoRA, a fixed input), backward skips that GEMM and doesn't keep the tensor it would have needed.

### v0.6.0: mixed precision
- **fp16 training nearly halves step time on larger models.** `AutocastGuard` runs matmuls and attention on tensor cores, and `GradScaler` handles loss scaling. On a T4, the 23M-parameter step goes from 107.6 to 57.2 ms. PyTorch fp16 takes 51.4 ms; dummygrad is faster at 0.4M and 3.9M. See [Mixed precision](#mixed-precision).
- **Everything else stays fp32**: parameters, gradients, optimizer state, softmax and normalization. Your training code only adds the guard and the scaler.
- **Safer dtypes:** reading an fp16 tensor as `float` now throws instead of silently computing garbage, and CPU fp16 conversion rounds correctly.
- **Runs on Kaggle too:** the notebook works on Kaggle's free T4s as well as Colab.

### v0.5.1
- **Faster than PyTorch at every size we benchmark.** On a T4: 2.9 vs 8.6 ms (0.4M params), 23.9 vs 27.6 ms (3.9M), 103.6 vs 107.7 ms (23M).
- **Attention on cuBLAS** for sequences up to 2048 tokens, any head dimension: backward went from 17 to 6.6 ms on the 23M model. Longer sequences keep the memory-efficient tiled kernels.
- **GPT is bitwise reproducible on the GPU too:** LayerNorm backward no longer uses atomics.
- **Fixed:** LayerNorm on CUDA could silently replace its parameters, so the optimizer stopped updating them.

### v0.5.0: the speed release
- **Faster than PyTorch on small models, at parity on larger ones.** On a T4, a ModernTransformer training step takes 3.5 ms vs PyTorch's 9.2 ms (0.4M params), 23.6 vs 27.1 ms (3.9M), and 112.6 vs 108.6 ms (23M). See [Performance](#performance).
- **Everything stays on the GPU.** New kernels for broadcast and scalar ops, RMSNorm, cross entropy, embedding lookup, GQA and gradient clipping removed 1.5 GB of host-device copies per step.
- **Tiled attention:** FlashAttention-style forward and an atomic-free FlashAttention-2 backward.
- **Reproducible on the GPU:** the new kernels use fixed-order sums instead of atomics, so ModernTransformer training gives the same results run to run.
- **Profiler:** `dg_bench --profile` (or `set_profiling(true)`) shows where forward and backward time goes, per op.
- **Fixed:** broadcasting now follows NumPy rules; `[1, H, S, S]` operands such as ALiBi biases were applied to the wrong heads with batch size above 1.

### v0.4.0: the workflow release
- **Build models from modules.** A `Module` base class gives every layer and model `parameters()`, `to()`, train/eval, freezing, and strict checkpoint loading from one `visit()` declaration.
- **Train like a real project.** Resumable checkpoints (model, optimizer state, step, scalars), `DataLoader` and `TokenDataset`, a CSV metrics logger, AdamW and SGD, and a complete [training template](examples/train_lm.cpp).
- **Write and trust your own ops.** `dummygrad::gradcheck` verifies custom backward passes, `NoGradGuard` runs evaluation without graphs, and `DetectAnomalyGuard` names the op that produced a NaN.
- **Reproducible:** gradients are summed in a fixed order, so the same code and seed give bitwise-identical results, and resumed runs match uninterrupted ones exactly.
- **Fixed:** `GPT` was not causal (it attended to future tokens), and SGD never updated CUDA parameters.

### v0.3.0: the trust release
- **No more silently wrong results on views.** Ops applied to transposed or sliced tensors (`relu(transpose(x))`, ops on `slice()` results) now compute on the right elements, and gradients through slices along any dimension are correct.
- **Every op and layer is gradient-checked**, including with strided inputs, and a dedicated test checks that strided and dense inputs give identical results.
- **CI on every change:** the test suite runs on Linux and macOS for every pull request.

### v0.2.1
- A full [Getting Started](#getting-started) guide: per-OS install, a runnable [quickstart](examples/quickstart.cpp), using dummygrad from your own CMake project, troubleshooting.

### v0.2.0: the correctness release
- **Correct CUDA training.** Strided copies on the GPU now honor strides (transposed tensors, including the tied LM head, were wrong), softmax and LayerNorm reduce across the whole block for rows wider than 32, and flash-attention backward always produces correct weight gradients.
- **Correct attention everywhere.** Grouped-query attention no longer reads scrambled K/V heads, and strided gradients flowing through RoPE and tied embeddings are handled on every backend.
- **A safer autograd engine.** No more gradient corruption in residual connections, and no use-after-free races in the multithreaded backward pass — including when a backward node throws.
- **Multi-GPU fixes.** Backward runs on each replica's own GPU, and the CUDA caching allocator keeps a separate pool per device.
- **CUDA RoPE kernels**, FP16/BF16-safe strided copies, the `test_engine` suite and a one-click [Colab notebook](notebooks/dummygrad_colab.ipynb).

> **Upgrading from 0.1:** checkpoints still load, but models using CUDA, grouped-query attention or tied embeddings were trained with incorrect activations or gradients — retrain rather than fine-tune.

---

## Architecture & Performance Highlights

```
┌──────────────────────────────────────────────────────────┐
│  Models: GPT / ModernTransformer (GQA, RoPE, SwiGLU) / CNNs │
├──────────────────────────────────────────────────────────┤
│  Autograd: deterministic gradient sums, calling-thread GPU │
├──────────────────────────────────────────────────────────┤
│  Attention: cuBLAS GEMMs ≤ 2048 tokens, tiled kernels past │
├──────────────────────────────────────────────────────────┤
│  Compute: cuBLAS / cuBLASLt (FP32, opt-in TF32, FP16, BF16)│
├──────────────────────────────────────────────────────────┤
│  Multi-GPU: one process, GPU-to-GPU gradient averaging     │
├──────────────────────────────────────────────────────────┤
│  Memory: CUDA caching allocator                            │
└──────────────────────────────────────────────────────────┘
```

- **Low per-op overhead**: ops are plain C++ functions with no Python or dispatcher layer, which is why small models step several times faster than in PyTorch (see [Performance](#performance)).
- **cuBLAS matmuls**: `Float32` (TF32 tensor cores only with `set_allow_tf32(true)`, off by default as in PyTorch), and `Float16` / `BFloat16` under `AutocastGuard`. Linear adds its bias in the cuBLASLt epilogue; the tied LM head (`matmul_nt`) reads the embedding transposed in place.
- **Attention**: cuBLAS GEMMs for sequences up to 2048 tokens (any head dimension), and FlashAttention-style tiled kernels beyond that, with memory linear in sequence length (head dimensions up to 128; larger ones use the GEMM path a chunk of heads at a time). Every path is deterministic.
- **Fused, deterministic kernels**: RMSNorm, LayerNorm, cross entropy, embedding, RoPE, GQA head repetition and AdamW; reductions sum in a fixed order (no atomics), so identical runs give bitwise-identical gradients.
- **Multi-GPU**: `MultiGPURunner` runs one replica per GPU in a single process and averages gradients GPU to GPU after each backward pass.
- **Apple Silicon**: MPS tensors live in unified memory; a few elementwise ops, masks, loss gradients and AdamW run as Metal kernels, everything else on the CPU over the same memory. Not covered by tests.
- **CPU SIMD Fallback**: Vectorized kernels for AVX2/FMA on x86_64 and NEON on ARM with optional OpenBLAS/oneMKL integration.

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
| `DUMMYGRAD_BUILD_TESTS` | `ON`* | Build the `test_engine` test suite |
| `DUMMYGRAD_BUILD_EXAMPLES` | `ON`* | Build the examples: `quickstart`, `custom_op`, `train_lm`, `train_colab` |

\* Only when dummygrad is the top-level project; off when used inside your own project.

### 3. Check that it works

```bash
./build/test_engine      # gradient checks, CUDA-vs-CPU checks, engine tests, training
./build/quickstart       # a tour of the API, trains a small network on XOR
```

`test_engine` should end with `N passed, 0 failed`; it compares every op's gradients against finite differences, checks the CUDA backend against the CPU when a GPU is present, and trains a small transformer end to end. `quickstart` should show the loss falling towards zero. You can also run everything through CTest: `cd build && ctest --output-on-failure`.

### 4. Your first program

[`examples/quickstart.cpp`](examples/quickstart.cpp) is a complete, runnable program. The essentials:

```cpp
#include "dummy_core.h"

// Tensors are std::shared_ptr<Tensor>. requires_grad=true makes autograd track them.
auto x = ones({2, 3}, Device::CPU, DType::Float32, /*requires_grad=*/true);

// Ops are free functions that build the graph as they run.
// sum() reduces one axis and keeps it, so two sums give a [1, 1] scalar.
auto y = sum(sum(mul_scalar(x, 3.0f), -1), -2);
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
- Inputs and parameters live on one device; copy data over with `tensor->to(Device::CUDA)` and read results back with `tensor->cpu()->data_ptr<float>()`.
- Integer data such as token ids and class labels is stored in `Float32` tensors.
- Collect a layer's `parameters()` *after* calling `.to()` — moving replaces the tensors.

The bigger building blocks — `ModernTransformer`, `GPT`, `MultiHeadAttention`, `RMSNorm`, `SwiGLU`, RoPE — are shown in the sections below.

### 5. Use dummygrad in your own project

dummygrad is a CMake library. The simplest way to depend on it is `FetchContent`, which downloads and builds it with your project:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(dummygrad
    GIT_REPOSITORY https://github.com/Keplercodes01/dummygrad.git
    GIT_TAG        v0.7.0)
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
- **`DataLoader(X, Y, batch_size, shuffle, seed)`** batches in-memory tensors along their first dimension; call `set_epoch(e)` for a deterministic per-epoch shuffle and iterate with `for (auto [x, y] : loader)`.
- **`TokenDataset`** reads token streams (`from_file` for raw uint16/int32 ids, `from_bytes` for byte-level text), samples random windows with `sample()`, and walks the whole stream for evaluation with `windows()`.
- **Optimizers:** `Adam` (AdamW, decoupled weight decay) and `SGD` (momentum, weight decay). Their state lives on each parameter's device and follows the model if you move it.
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

### Writing your own ops

A new op is a forward function plus a backward `Node`. [`examples/custom_op.cpp`](examples/custom_op.cpp) is a complete, compiled example (squared ReLU). The shape of it:

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

## Zero-Copy SafeTensors & Model Serialization

`dummygrad` features native, pure C++ **SafeTensors** and binary model checkpointing with **zero Python dependencies** and zero CPU copies via memory-mapped I/O (`mmap`):

- **Hugging Face Interoperability**: Read and write standard `.safetensors` files directly loadable in PyTorch / Hugging Face Transformers.
- **Zero-Copy Deserialization**: Uses `mmap` to map tensors directly from disk into virtual address space in sub-microsecond latency.
- **Pure C++ Checkpointing**: Save and restore complete state dicts (`std::unordered_map<std::string, std::shared_ptr<Tensor>>`) with single-line calls.

```cpp
#include "dummy_core.h"

// 1. Save model weights to Hugging Face SafeTensors format
std::unordered_map<std::string, std::shared_ptr<Tensor>> weights = {
    {"transformer.w_gate.weight", w_gate},
    {"transformer.norm.gamma", rms_gamma}
};
io::save_safetensors("model.safetensors", weights);

// 2. Instantaneous zero-copy load from SafeTensors file
auto loaded = io::load_safetensors("model.safetensors", Device::CPU);
std::cout << "Loaded " << loaded.size() << " tensors with zero memory copies.\n";
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

Training-step time (forward, backward, AdamW) for the same ModernTransformer architecture (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) in dummygrad and eager PyTorch, float32, on a Colab T4 (measured with v0.5.1; v0.7.0's fused Linear and single-threaded GPU backward apply to fp32 too, so current numbers should be lower):

| config | params | dummygrad | PyTorch 2.11 | |
|---|---|---|---|---|
| tiny (d=128, 2 layers, batch 8x64) | 0.4M | **2.9 ms** | 8.6 ms | 3.0x faster |
| small (d=256, 4 layers, batch 16x128) | 3.9M | **23.9 ms** | 27.6 ms | 13% faster |
| medium (d=512, 6 layers, batch 8x256) | 23.1M | **103.6 ms** | 107.7 ms | 4% faster |

dummygrad's advantage is per-op overhead (no Python, no dispatcher), which dominates small models; at larger sizes both are bound by cuBLAS matrix multiplies (`MatmulBackward` is 47 of medium's 104 ms). Reproduce it with the [Colab notebook](notebooks/dummygrad_colab.ipynb), or run `./build/dg_bench` and `python bench/bench_torch.py` on your own GPU. `./build/dg_bench --profile` breaks forward and backward time down by op.

With fp16 mixed precision (`dg_bench --fp16`: `AutocastGuard` + `GradScaler`, against `torch.autocast` + `GradScaler`), v0.7.0 on a Kaggle T4:

| config | dummygrad fp16 | PyTorch 2.11 fp16 | |
|---|---|---|---|
| tiny | **1.7 ms** | 10.8 ms | 6.4x faster |
| small | **14.4 ms** | 21.9 ms | 34% faster |
| medium | **50.5 ms** | **50.7 ms** | even |

At 23M parameters both are bound by the GPU: cuBLAS matrix multiplies are nearly half of dummygrad's step. dummygrad still converts every matmul input from fp32 (3.3 ms of medium's 18 ms forward pass, per `--profile`), while PyTorch's autocast keeps more of its activations in fp16; that is the next thing to remove.

---

## Status & Known Limitations

dummygrad is pre-1.0: the core training path (CPU and CUDA, FP32) is covered by `test_engine`; other paths are less exercised.

- **Mixed precision covers matrix products.** Under `AutocastGuard`, CUDA matmuls and attention (on its cuBLAS path, up to 2048 tokens) run on fp16/bf16 tensor cores. Normalization and elementwise kernels, the softmax, and attention beyond 2048 tokens stay fp32. Tensors are always stored in fp32; non-fp32 tensors are only for storage and conversion. bf16 needs an Ampere or newer GPU.
- **Attention** runs as cuBLAS GEMMs for sequences up to 2048 tokens (any head dimension) and as tiled kernels beyond that for head dimensions up to 128 (other sizes zero-padded to 16, 32, 64 or 128). Larger head dimensions past 2048 tokens use the GEMM path a chunk of heads at a time, whose memory is quadratic in the sequence length.
- **Multi-GPU** runs in one process and averages gradients GPU to GPU, one parameter at a time (no bucketing or overlap with backward yet). Its test needs two GPUs (Kaggle's T4 x2).
- **Metal and TPU** paths are not covered by the test suite (macOS CI runs the CPU backend; there is no TPU in CI).

Bug reports with a failing `test_engine` case are the fastest way to get something fixed.

---

## Built by

[@Keplercodes01](https://github.com/Keplercodes01)

