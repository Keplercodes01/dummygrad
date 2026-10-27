# Changelog

All notable changes to dummygrad are documented here. Versions follow [Semantic Versioning](https://semver.org/); while the major version is 0, minor releases may change behavior.

## [Unreleased]

### Added
- **Forward profiling.** `set_profiling(true)` now times forward ops as well as backward nodes. Forward entries are named after the op (`matmul`, `rms_norm`, `add`, `attention`, `autocast_cast` for fp16/bf16 conversions) and count self time: an op that calls other ops counts only its own work. Ops a backward node runs internally stay part of that node's time. `dg_bench --profile` prints a forward table next to the backward one, each with its total.
- `DG_PROFILE_OP("name")` gives your own ops a forward entry (`examples/custom_op.cpp` shows it).
- `ProfileEntry::phase` and `profile_report(ProfilePhase)` select one phase.
- Tests: forward and backward entries land in their own phases, nested ops record self time, and ops inside a backward node get no forward entry.

### Changed
- Profiling lives in `core/profiler.h` / `profiler.cpp` (still included by `autograd.h`). `profile_report()` with no argument returns both phases. Before, it returned backward entries only.

## [0.6.1] - 2026-10-27

Faster fp16 backward for matmuls. On a T4 (Kaggle), the fp16 medium step goes from 57.2 to 53.8 ms (PyTorch 2.11 fp16: 51.8 ms), and `MatmulBackward` from 16.7 to 13.9 ms. small goes from 22.0 to 21.3 ms.

### Changed
- **CUDA matmul backward reuses the forward's operands.** The backward node keeps the forward GEMM's inputs (contiguous, already rounded to fp16/bf16 under autocast), so backward no longer re-casts or re-densifies A and B; only dC is rounded, once for both products. Under fp16 autocast a saved activation is a half-size copy, and its fp32 original can be freed as soon as nothing else holds it. In fp32, a transposed operand (the tied LM head) is kept as the dense copy the forward already made, instead of being copied again in backward.
- **Frozen matmul operands skip their gradient.** When only one input of a CUDA matmul requires grad (a fixed input, a frozen weight in LoRA-style fine-tuning), backward runs only that input's GEMM and keeps only the operand it needs. Previously both gradients were computed and the unused one discarded.
- `cuda::matmul` can return its GEMM operands (`saved_a` / `saved_b`), and `cuda::matmul_backward` takes the input shapes plus optional gradient outputs, so either GEMM can be skipped. CUDA matmuls record a dedicated backward node; it still reports as `MatmulBackward` in profiles.
- Backward through a CUDA matmul whose saved operands were already released now throws (pass `retain_graph=true` to run backward twice) instead of silently producing no gradients.

### Added
- Tests: fp16/bf16 matmul parity for a transposed weight; a frozen operand yields only the other gradient, bit-identical to the fully trainable product (fp32 and fp16).

## [0.6.0] - 2026-10-27

Mixed-precision training: matmuls and attention on fp16/bf16 tensor cores. On a T4 (Kaggle), fp16 cuts the medium ModernTransformer step (23M parameters) from 107.6 to 57.2 ms; `MatmulBackward` goes from 48.3 to 16.7 ms and attention backward from 6.9 to 4.5 ms.

| config | fp32 | fp16 | PyTorch 2.11 fp16 |
|---|---|---|---|
| tiny (0.4M params) | 4.1 ms | 4.7 ms | 12.4 ms |
| small (3.9M) | 28.2 ms | 22.0 ms | 23.8 ms |
| medium (23.1M) | 107.6 ms | 57.2 ms | 51.4 ms |

fp16 does not pay off at tiny sizes, where converting operands and the scaler's extra pass cost more than tensor cores save. PyTorch's autocast also runs elementwise and normalization ops in fp16, which accounts for most of its lead at 23M.

### Added
- **`AutocastGuard`** (`core/autocast.h`): while one is alive, CUDA matmuls round their fp32 inputs to fp16 or bf16 and run on tensor cores, accumulating and writing fp32. Everything else stays fp32: parameters, gradients, optimizer state, and the normalization and elementwise kernels. Each op's backward uses the precision its forward ran with, so `backward()` can run outside the guard. Thread-local, like `NoGradGuard`.
- **Attention under autocast** (cuBLAS path, sequences up to 2048 tokens): Q, K, V and dO are rounded once per call, and the softmax probabilities P and score gradients dS are written in half precision for the GEMMs that read them. Scores, the softmax, lse and every GEMM output stay fp32, and the chunk size accounts for the extra half-precision buffers. The tiled kernels for longer sequences stay fp32.
- **`GradScaler`** (`nn/grad_scaler.h`): dynamic loss scaling for fp16, with PyTorch's semantics (`scale`, `unscale`, `step`, `update`). Gradients are unscaled and checked for inf/NaN on the GPU in one pass with a single host sync. Its state round-trips through `TrainingState` checkpoints (`save_to` / `load_from`).
- `dg_bench --fp16` / `--bf16` and `bench_torch.py --fp16` / `--bf16`; the notebook runs the fp16 comparison.
- The notebook runs on Kaggle as well as Colab (working directory picked per platform, nvcc located when it is not on PATH); the README has an Open in Kaggle badge.
- Tests: fp16 rounding edge cases and an exact round trip of all 65536 fp16 values; fp16 (and bf16 on Ampere+) matmul forward and gradients against fp32 for plain, batched, shared-weight and broadcast products; attention output, lse and gradients under autocast against fp32, causal and not, and the tiled kernels unchanged by autocast; GradScaler skip, backoff, growth and checkpointing; the tiny transformer trains under fp16 autocast.

### Changed
- The cuBLAS GEMM call is one shared helper (`gemm_strided_batched` in `cuda_common.h`) for matmul and attention, in fp32 or half precision.
- **`cuda::matmul` takes and returns Float32 only**, with the GEMM precision as a separate `compute` argument. The old fp16/bf16 paths stored their outputs and gradients in half precision, which no other CUDA kernel could read. `matmul_backward` no longer zero-fills its outputs before GEMMs that overwrite them.
- **Typed data access is checked:** `data_ptr<float>()` (and `float16`, `bfloat16`, `int8_t`) throws if the tensor's dtype differs, instead of reinterpreting the bytes. CUDA kernels take `float*`, so passing an fp16 tensor to any of them used to compute garbage silently. `data_ptr<void>()` and other raw types are unchecked.

### Fixed
- **fp32 to fp16 conversion on the CPU rounds to nearest even.** It used to truncate, biasing every value toward zero, and turned finite values beyond 65504 into NaN instead of infinity.

## [0.5.1] - 2026-10-25

Backward-pass tuning. On a T4, the medium ModernTransformer step (23M parameters) went from 112.6 to 103.6 ms and is now 4% faster than eager PyTorch (107.7 ms); attention backward went from 17 to 6.6 ms and AccumulateGrad from about 4 to 1.1 ms per step.

### Changed
- **Attention runs on cuBLAS for sequences up to 2048 tokens** (`cuda_attention.cu`). The [seq, seq] score matrices are materialized a chunk of heads at a time (at most 256 MB of scratch) and every product is a strided-batched GEMM: forward is S = QK^T, softmax, PV, and backward recomputes S and dO V^T and runs dV = P^T dO, dK = dS^T Q, dQ = dS K. The tiled kernels ran attention backward at about 8% of T4 fp32 peak; at these lengths cuBLAS is several times faster. Longer sequences keep the tiled kernels, whose memory is linear in the sequence length. `cuda::set_attention_gemm_max_seq` moves the cutoff (0 forces the tiled kernels). Any head dimension now gets the fast, deterministic path at these lengths, not just 16/32/64/128.
- **LayerNorm backward on CUDA is deterministic.** gamma/beta gradients were accumulated with one atomic per element per row; they are now fixed-order partial sums like the other kernels, so GPT-style models are bitwise reproducible on the GPU too. LayerNorm moved to `cuda_core_ops.cu` and was rewritten like RMSNorm.
- **LayerNorm saves less for backward.** The CUDA forward allocated an `x - mean` buffer the size of the input that it never filled or read, and kept it alive until backward; backward now recomputes x̂ from x and the per-row mean and inverse std on every device. The redundant zero-fills of the output gradients are gone too.
- **AccumulateGrad adopts a parameter's first gradient instead of copying it** when nothing else references the gradient buffer, saving one device copy per parameter per step. The engine now hands gradients to the next node by move so that the consumer holds the only reference.

### Fixed
- **LayerNorm on CUDA no longer replaces its parameters.** With gamma/beta on another device, the CUDA forward reassigned them to new GPU tensors, so an optimizer holding the originals stopped updating the layer. A device mismatch now raises an error, like RMSNorm.

### Added
- Tests: GPT and ModernTransformer gradients are bitwise identical across repeated runs on CUDA; the attention parity tests also run on the tiled and untiled kernels; adopted gradients never alias between parameters.

## [0.5.0] - 2026-10-24

The speed release. A training step no longer moves data between GPU and CPU, attention runs on tiled kernels, and the new kernels reduce in a fixed order (no atomics), so ModernTransformer training is reproducible on the GPU. On a T4, a ModernTransformer training step is 2.6x faster than eager PyTorch at 0.4M parameters, 15% faster at 3.9M, and within 4% at 23M (it was 11x slower at the start of this release).

| config | 0.4.0 | 0.5.0 | PyTorch 2.11 (eager) |
|---|---|---|---|
| tiny (0.4M params) | 24.1 ms | 3.5 ms | 9.2 ms |
| small (3.9M) | 318.9 ms | 23.6 ms | 27.1 ms |
| medium (23.1M) | 1190.2 ms | 112.6 ms | 108.6 ms |

### Fixed
- Library code no longer prints to stdout when saving or loading SafeTensors files.

### Added
- **Benchmark** (`bench/bench.cpp`, target `dg_bench`): full ModernTransformer training steps at three sizes, timed as forward / backward / optimizer, with host-device traffic per step. `bench/bench_torch.py` runs the same architecture in PyTorch; the Colab notebook runs both.
- `transfer_stats()` / `reset_transfer_stats()`: counts host-device copies made through `Tensor::to`, which exposes ops that fall back to the CPU.
- CI compiles the CUDA backend (in NVIDIA's CUDA container) on every change, and smoke-runs the benchmark on CPU.
- **GPU kernels for the ops that fell back to the CPU** (`cuda_core_ops.cu`): broadcast add/sub/mul/div forward and backward, scalar ops, RMSNorm, fused cross entropy, embedding lookup, GQA head repetition, and the gradient-norm reduction. A training step no longer copies activations or gradients between GPU and CPU. All reductions are deterministic (fixed-order partial sums, no atomics), so GPU results are reproducible too.

- **Tiled attention kernels** (`cuda_attention.cu`): FlashAttention-style forward (K/V tiles in shared memory, online softmax, saved log-sum-exp) and a FlashAttention-2 backward split into a dK/dV pass and a dQ pass, for head dims 16, 32, 64 and 128. Each row is owned by a group of head_dim / 16 threads holding register slices combined with warp shuffles, which keeps shared memory small and occupancy high. The previous kernel recomputed every dot product in every thread and accumulated dK/dV with one atomic per element per query-key pair; the new backward has no atomics and is deterministic. Other head dims keep the untiled kernels.
- **Backward profiler:** `set_profiling(true)` times every backward node (serialized and synchronized for exact numbers) and `profile_report()` lists time and calls per op; `dg_bench --profile` prints the most expensive backward ops.
- Matmul with a shared weight (e.g. every `Linear` on `[B, T, D]` input) runs as one GEMM over `[B*T, D]` in forward and backward instead of one GEMM per batch element.

### Changed
- **Broadcasting follows NumPy rules** (shapes aligned from the right; each dimension 1 or equal). The previous rule mapped a smaller batch onto contiguous groups of a larger one, which applied `[1, H, S, S]` operands such as ALiBi biases to the wrong heads whenever the batch size was above 1. Incompatible shapes now raise an error instead of reading out of bounds.
- The four broadcast ops and six scalar ops are each one implementation (`broadcast_binary`, `scalar_op`) instead of separate copies; `add`, `mul`, `cast_n_*`, `*_scalar`, `neg` are thin wrappers.
- `Embedding` looks up rows directly instead of multiplying a one-hot matrix by the table.
- `fused_cross_entropy` saves per-row log-sum-exp instead of the full probability matrix and recomputes the softmax in backward; on the GPU the loss stays on the device.
- `RMSNorm` runs natively on each device instead of building a CPU copy of the layer on every call.
- `Node::name()` gives backward functions readable names in error messages (e.g. `ScalarOpBackward(rdiv)`).

## [0.4.0] - 2026-10-24

The workflow release: everything around the model that day-to-day research needs. Write and verify custom ops, build models from modules, train with resumable checkpoints, data loaders and metric logs, and get bitwise-reproducible gradients.

### Added
- **`dummygrad::gradcheck`** (`testing.h`, included by `dummy_core.h`): checks the gradients of any function of tensors — a custom op, a layer, a whole block — against finite differences and reports the first mismatch. `test_engine` now runs its CPU gradient checks through it.
- **`NoGradGuard`** / `set_grad_enabled`: no autograd graph is recorded inside the scope, for evaluation and inference. Outputs don't require grad and hold no references to their inputs. Thread-local, like PyTorch's `no_grad`.
- **`DetectAnomalyGuard`** / `set_detect_anomaly`: backward throws at the first NaN/Inf gradient and names the backward function that produced it.
- **`examples/custom_op.cpp`** and a README guide to writing your own ops: forward, backward `Node`, graph edges, and verification with `gradcheck`. Built and run by CTest.
- `needs_grad(t)`: whether an op should record a graph edge for `t` (false inside `NoGradGuard`).
- **`Module` base class** (`nn/module.h`): a module declares its parameters, buffers and children in `visit()`, and gets `parameters()`, `named_parameters()`, `state_dict()`, `to()`, `train()`/`eval()`, `requires_grad_()`, `zero_grad()`, `num_parameters()`, `save_safetensors()` and `load_safetensors()`. Every built-in layer and model is now a `Module`; their hand-written copies of these methods (about 380 lines) are gone. Checkpoint tensor names are unchanged.

- **Resumable checkpoints** (`training.h`): `save_checkpoint` / `load_checkpoint` store the model, optimizer state, step and arbitrary scalars in one SafeTensors file, written atomically. Resuming reproduces an uninterrupted run exactly.
- **Deterministic gradients:** gradients that meet at a node are summed in a fixed order (node creation order) instead of thread arrival order, so identical runs give bitwise-identical gradients on CPU. The summing also moved out of the engine's global lock.
- **Data loading** (`data.h`): `DataLoader` for in-memory tensors with deterministic per-epoch shuffling, and `TokenDataset` for language-model token streams (raw uint16/int32 id files, byte-level text, random and sequential windows, train/validation split).
- **`MetricsLogger`:** long-format CSV metric logging with optional console echo and append-on-resume.
- **`SGD` optimizer class** with momentum and weight decay; `Adam` and `SGD` expose `state_dict` / `load_state_dict`. `AdamW` is an alias of `Adam`.
- **`examples/train_lm.cpp`:** a complete training-run template (character-level LM with validation, schedule, clipping, logging, checkpoints, `--resume`, sampling), run by CTest.

### Fixed
- **`SGD` on CUDA never updated parameters:** it copied the parameter to the CPU without its gradient, so the update was skipped. The free function is replaced by the `SGD` class, which works on every device.
- `clip_grad_norm_` accumulated the squared norm in single precision, losing accuracy on large models; it now uses double precision and works on any device. Its unused `norm_type` parameter is removed.
- **`GPT` was not causal.** `TransformerBlock` passed `causal` into `MultiHeadAttention`'s `n_kv_heads` parameter, so every GPT layer attended to future tokens and used a single shared K/V head. GPT is now causal with full multi-head attention; GPT checkpoints from earlier versions have the wrong `W_k`/`W_v` shapes and will not load (and were trained on a model that could see the future).
- `MARS::rollout` built a full autograd graph at every step and cleared `requires_grad` on the caller's seed tensor; it now runs under `NoGradGuard` and leaves its input untouched.

### Changed
- **Optimizer state is stored as tensors on each parameter's device** instead of CPU vectors plus hand-managed CUDA buffers. State is tied to the parameter's identity, so a new tensor that reuses a freed parameter's address no longer inherits stale moments, and state follows parameters across `to()`. `Adam`'s fields are now `beta1`, `beta2`, `eps` (previously `b1`, `b2`, `E`).
- **Loading checkpoints is strict by default.** Missing, unexpected and wrong-shape tensors raise an error listing them, instead of being silently skipped (a renamed layer previously meant training from random initialization unnoticed). Pass `strict=false` to load a subset; shape mismatches always raise.
- Loading copies values into the existing parameter tensors instead of swapping their storage, so tensor identity (and optimizer state attached to it) survives a load, and shapes are checked.
- `to()` preserves `requires_grad`, so frozen parameters stay frozen; previously every move re-enabled gradients. Tied tensors stay tied across moves.

## [0.3.0] - 2026-10-24

The trust release: results no longer go silently wrong for strided tensors, every op is covered by gradient checks, and CI tests every change on Linux and macOS.

### Fixed
- **Ops give correct results for strided inputs.** Forward kernels indexed raw memory, so an op applied to a view from `transpose()` or `slice()` — e.g. `relu(transpose(x))` — silently computed on the wrong elements. Every op that indexes raw memory now densifies its inputs first (no cost for already-dense tensors).
- **Slicing along any dimension except the first** produced zero gradients (an error in 0.2.0). `SliceBackward` now writes the gradient window directly.
- `one_hot` reads strided index tensors correctly.

### Added
- `contiguous(x)`: a differentiable dense copy of a strided view; returns `x` itself when already contiguous.
- **Continuous integration** (GitHub Actions): builds and runs the test suite on Linux and macOS for every pull request and push to `main`, and builds a downstream project via `FetchContent`.
- **Test coverage**: gradient checks for `concat`, `slice`, `broadcast`, `var`, `std_dev`, the attention masks, `cross_entropy`, `l1_loss`, `scale_n_shift`, `conv1d`, `conv2d`, `max_pool2d` and `Embedding`; strided-input variants of every elementwise, reduction, normalization, loss and binary op; and a check that strided and dense inputs give identical forward results (91 CPU checks in total).

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
