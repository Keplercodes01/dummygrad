# Changelog

All notable changes to dummygrad are documented here. Versions follow [Semantic Versioning](https://semver.org/); while the major version is 0, minor releases may change behavior.

## [0.2.1] - 2026-10-24

Documentation and packaging: everything needed to install dummygrad, try it, and build on it.

### Added
- **Getting Started guide** in the README: per-platform prerequisites (Ubuntu, macOS, Windows via WSL2, CUDA), build and verification steps, a walkthrough of a first program, and a troubleshooting table.
- **`examples/quickstart.cpp`**: a runnable tour of tensors, autograd and training (a two-layer network on XOR). Built by default and run by CTest as `example_quickstart`, so the README's code is compiled on every build.
- **Use dummygrad from your own project** with `FetchContent` or `add_subdirectory`, linking the new `dummygrad::dummygrad` target.
- The Colab notebook runs the quickstart and builds a separate project that depends on dummygrad via `FetchContent`.

### Changed
- When dummygrad is included in another CMake project, tests and examples are no longer built (`DUMMYGRAD_BUILD_TESTS` and the new `DUMMYGRAD_BUILD_EXAMPLES` default to `ON` only for top-level builds).

## [0.2.0] - 2026-10-24

The correctness release. Training the transformer models in 0.1 computed wrong results in several places — most severely on CUDA, where transposed tensors, wide softmax/LayerNorm rows and attention gradients were all affected. This release fixes those paths, hardens the multithreaded autograd engine and multi-GPU support, and adds a test suite that runs on a free Colab GPU.

### Fixed — CUDA correctness
- **Strided copies ignored strides.** `make_contiguous` on CUDA copied memory linearly, so every transposed tensor that was densified on the GPU — including the tied LM head in `ModernTransformer` and attention inputs — held the wrong values. It is now a strided GPU kernel, and it handles FP16/BF16/Int8 (it previously read past the end of half-precision buffers).
- **Softmax and LayerNorm on rows wider than 32.** The kernels launch up to 256 threads but reduced only within the first warp, so normalization used a partial sum. They now reduce across the whole block.
- **Flash-attention backward** read its incoming gradient (a transposed view) as dense memory, and produced no weight gradients when the layer input did not require grad.
- **Tied embedding gradients.** The LM head's gradient arrives as a strided view and was accumulated element-wise in the wrong order.
- `matmul` rejects partial batch broadcasting (e.g. `[B,H,…] @ [B,1,…]`) instead of reading out of bounds.
- Flash attention supports head dimensions up to 512 with register-sized accumulators; larger sizes raise an error instead of leaving the output unwritten.

### Fixed — models & ops (all backends)
- **Grouped-query attention read scrambled K/V.** The KV-head repeat copied transposed views as if they were contiguous, mixing sequence positions and heads. Affects any model with `n_kv_heads < n_heads`.
- **Strided gradients in backward nodes.** Several backward kernels (e.g. RoPE on CPU) read transposed incoming gradients as dense memory; on CPU, accumulating a strided gradient (tied embeddings) threw. The engine now densifies incoming gradients once, and accumulation accepts strided operands.
- Broadcast ops (`cast_n_add/sub/mul/div`) share one batch mapping (`broadcast_batch_index`) between forward and backward, including grouped (GQA-style) broadcasting.
- `conv2d` supports separate stride/padding per axis; `conv1d` now passes its arguments correctly.
- `cross_entropy` averages over samples instead of over every element.
- LayerNorm/RMSNorm produce gradients when only their parameters require grad.
- SafeTensors header parsing tolerates whitespace and keeps scalar (shape `[]`) entries.

### Fixed — autograd engine
- **Gradient aliasing in residual networks.** `AddBackward`, `AddScalarBackward` and views hand the same gradient tensor to several edges; the engine accumulated into it in place, sometimes while another worker thread was still reading it. Gradients are now copied before in-place accumulation.
- **Use-after-free races.** `backward()` could return while worker tasks still referenced its stack frame, at normal completion and after an exception. It now waits for every in-flight task; exceptions propagate and the engine stays usable afterwards.
- `Edge::input_slot` is documented and used consistently; dependency counting no longer recurses on deep graphs.

### Fixed — multi-GPU
- **Backward ran on GPU 0.** Autograd worker threads did not inherit the caller's CUDA device, so backward for replicas on GPU 1+ launched kernels on GPU 0. Workers now use the caller's device.
- **Allocator blocks could cross GPUs.** The CUDA caching allocator had one pool for all devices; it is now per-device.
- Gradients are aggregated after every replica's backward completes, then broadcast back.
- cuBLAS handles are per device and per thread.

### Added
- CUDA RoPE kernels (forward and backward).
- `test_engine` test suite (`tests/test_engine.cpp`, registered with CTest): finite-difference gradient checks for every op and layer, CUDA-vs-CPU parity checks, autograd engine stress tests, and an end-to-end transformer that must give every parameter a gradient and reduce its loss.
- Colab notebook (`notebooks/dummygrad_colab.ipynb`): build, test and train on a free GPU with no local setup.
- CMake: project version (`DUMMYGRAD_VERSION`), `DUMMYGRAD_BUILD_TESTS` option, `CMAKE_CUDA_ARCHITECTURES` defaulting to the attached GPU, Threads linkage.
- `CHANGELOG.md`, and "Status & Known Limitations" in the README.

### Changed
- CUDA libraries are linked `PUBLIC`, so programs including dummygrad headers find `cuda_runtime.h` without extra flags.
- README examples for multi-GPU training and attention masking corrected.
- The prebuilt `src/test` binary is no longer tracked.

### Upgrading
Checkpoints from 0.1 still load. Models using CUDA, grouped-query attention or tied embeddings were trained with incorrect activations or gradients, so retraining is recommended over fine-tuning.
