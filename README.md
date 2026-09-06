# dummygrad

A pure C++ deep learning engine engineered for absolute speed, zero abstraction bloat, and minimal codebase maintenance.

No Python. No bindings. No GIL. No multi-layer dispatcher overhead. Just raw, frontier-grade execution directly on the metal.

> *"I honestly made it for my selfish reasons. I train my neural nets in C++ with my engine. Read the source code—it's fast and it's fun."*

---

## Architecture & Performance Highlights

```
┌────────────────────────────────────────────────────────┐
│  High-Level Models: GPT / Transformers / Linear / CNNs  │
├────────────────────────────────────────────────────────┤
│  Multi-GPU Engine: Native P2P NVLink DMA & Reduction   │
├────────────────────────────────────────────────────────┤
│  Memory: Static Arena (0 ns) + CUDACachingAllocator    │
├────────────────────────────────────────────────────────┤
│  Attention: Tiled FlashAttention (Fwd + Bwd, O(N) Mem) │
├────────────────────────────────────────────────────────┤
│  Compute: cuBLAS Tensor Cores (TF32 / FP16 / BF16)     │
├────────────────────────────────────────────────────────┤
│  Fused Kernels: LayerNorm, AdamW, GELU, Bias-Epilogue  │
├────────────────────────────────────────────────────────┤
│  Execution: Hardware CUDA Graphs (0 CPU Latency)       │
└────────────────────────────────────────────────────────┘
```

- **Zero-Overhead Dispatch**: Pure C++ inline execution drops op launch overhead from PyTorch's $\sim 3\text{ }\mu\text{s}$ down to sub-$100\text{ ns}$.
- **Hardware Tensor Cores (cuBLAS)**: Full hardware saturation across `Float32` (with TF32 Tensor Cores), `Float16`, and `BFloat16` using row-major transpose identities that eliminate memory transposition copies.
- **Tiled FlashAttention**: Custom CUDA forward and backward attention kernels using online softmax. Eliminates the $O(N^2)$ memory bottleneck, executing in strictly $O(N)$ memory.
- **Single-Pass Fused Kernels**:
  - **LayerNorm**: Warp-shuffle parallel reductions for row mean and variance; fused affine scaling and backward passes.
  - **AdamW**: Updates parameters, gradients, momentum $m$, and variance $v$ in a single memory pass.
  - **Activations & Epilogues**: Fast vectorized GELU, ReLU, and fused bias addition kernels.
- **Hardware CUDA Graphs**: Record entire forward + backward + optimizer steps into an executable GPU graph (`CUDAGraph`) to eliminate 100% of CPU dispatch latency across training iterations.
- **Zero-Allocation Static Arena**: `CUDAScratchpadArena` with `ArenaScope` RAII helper gives $0\text{ ns}$ allocation overhead for intermediate activation buffers, eliminating memory fragmentation and OOM spikes during 100k+ step training runs.
- **Native Multi-GPU Scaling**: Single-process Peer-to-Peer direct NVLink/PCIe DMA reduction (`DeviceMesh` + `MultiGPURunner`). No Python `torchrun`, no sockets, no process spawning.
- **CPU SIMD Fallback**: Vectorized kernels for AVX2/FMA on x86_64 and NEON on ARM with optional OpenBLAS/oneMKL integration.

---

## Quickstart & Build

### Prerequisites
- C++17 compatible compiler (`g++` $\ge 9$, `clang++` $\ge 11$)
- CMake $\ge 3.18$
- NVIDIA CUDA Toolkit $\ge 11.0$ (for GPU execution)
- Optional: OpenBLAS / oneMKL / BLAS (for CPU acceleration)

### Build
```bash
git clone https://github.com/Keplercodes01/dummygrad.git
cd dummygrad
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

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

    // 1. Initialize GPT model on CUDA with FP16/BF16 Tensor Cores
    GPT model(vocab_size, max_seq_len, d_model, n_heads, n_layers);
    Adam optimizer(1e-4f);

    // 2. Pre-allocate 2GB GPU scratchpad arena for 0-allocation steps
    CUDAScratchpadArena::get().init(2ULL * 1024 * 1024 * 1024);

    // 3. Training step loop
    for (int step = 0; step < 1000; ++step) {
        {
            ArenaScope scope; // Activates scratchpad arena; auto-resets in 0 ns at scope exit

            auto logits = model.forward(input_ids, pos_ids);
            auto loss = cross_entropy(logits, targets);

            loss->backward(); // Executes FlashAttention backward & fused LayerNorm backward

            for (auto& param : model.parameters()) {
                optimizer.step(param); // Fused AdamW GPU kernel
            }
        }

        std::cout << "Step " << step << " complete." << std::endl;
    }

    return 0;
}
```

---

## Example: Multi-GPU Parallel Training (NVLink P2P)

```cpp
#include "dummy_core.h"

int num_gpus = 4;

// Spin up model replicas across all 4 GPUs in a single C++ process
MultiGPURunner<GPT> runner(num_gpus, [](int rank) {
    return GPT(vocab_size, max_seq_len, d_model, n_heads, n_layers);
});

// Execute parallel steps with direct NVLink DMA reduction
runner.parallel_step([](int rank, GPT& model) {
    auto logits = model.forward(batch_inputs[rank], batch_pos[rank]);
    auto loss = cross_entropy(logits, batch_targets[rank]);
    loss->backward();
    // Gradients are automatically synchronized across all GPUs via direct NVLink DMA!
});
```

---

## Built by

[@Keplercodes01](https://github.com/Keplercodes01)

