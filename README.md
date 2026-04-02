# cuda-inference

A from-scratch MLP inference engine in C++17/CUDA. No frameworks. Weights trained in PyTorch, everything else built by hand — memory management, GPU kernels, weight loading, the lot.

**96.90% accuracy on MNIST** · **11× speedup over naive matmul** · RTX 3060 12GB

---

## Architecture

```
test_images.npy  ──►  Tensor<float>  ──►  Network::forward_gpu()
test_labels.npy  ──►  std::vector<int>

Network::forward_gpu()
  │
  ├── Linear fc1: [N, 784] × [784, 128]ᵀ + bias  →  [N, 128]
  ├── ReLU (CUDA kernel)
  ├── Linear fc2: [N, 128] × [128,  64]ᵀ + bias  →  [N,  64]
  ├── ReLU (CUDA kernel)
  ├── Linear fc3: [N,  64] × [ 64,  10]ᵀ + bias  →  [N,  10]
  └── Softmax (CUDA kernel)
        │
        └──► argmax  ──►  predicted class
```

Each `Linear` layer uses cuBLAS `cublasSgemm` for the matmul. Weights are Xavier-initialized and loaded from `.npy` files exported by the PyTorch training script.

---

## Performance — Matmul Kernels

Benchmarked on a single `[1000, 784] × [784, 128]` matmul, RTX 3060 12GB, CUDA 12.x.

| Kernel              | Time     | Throughput  | Speedup vs Naive |
|---------------------|----------|-------------|------------------|
| Naive               | ~1.80 ms | 2.3 GB/s    | 1×               |
| Tiled (shared mem)  | ~0.58 ms | 7.1 GB/s    | 3×               |
| cuBLAS              | ~0.16 ms | 25.7 GB/s   | **11×**          |

**Naive** — one thread per output element, reads directly from global memory. Bandwidth-bound immediately due to redundant DRAM loads.

**Tiled** — threads cooperatively load 16×16 tiles into shared memory before computing. Reduces global memory traffic by the tile dimension, cuts latency proportionally.

**cuBLAS** — vendor BLAS using Tensor Cores and optimised register tiling. Serves as the production kernel; naive and tiled are retained for comparison.

---

## Accuracy

| Implementation    | Correct / 1000 | Accuracy |
|-------------------|----------------|----------|
| PyTorch baseline  | —              | 97.34%   |
| C++ CPU           | 969 / 1000     | 96.90%   |
| C++ GPU (cuBLAS)  | 969 / 1000     | 96.90%   |

The 0.44% gap vs PyTorch is floating-point accumulation order — different matmul traversal order produces slightly different rounding. Not a bug; same weights, same model.

---

## Project Structure

```
cuda-inference/
├── include/
│   ├── tensor.cuh       — RAII Tensor<T>: CPU/GPU, move semantics, deleted copy
│   ├── linear.cuh       — Linear layer, Xavier init, to_gpu()
│   ├── activations.cuh  — ReLU, Softmax (CPU)
│   ├── network.cuh      — MLP: fc1(784→128), fc2(128→64), fc3(64→10)
│   ├── kernels.cuh      — naive matmul, tiled matmul, cuBLAS, ReLU, Softmax kernels
│   └── npy.cuh          — .npy file parser
└── src/
    ├── tensor.cu
    ├── linear.cu
    ├── activations.cu
    ├── network.cu        — forward_cpu, forward_gpu, load_weights
    ├── kernels.cu        — all CUDA kernels
    ├── npy.cu            — .npy parser implementation
    └── main.cu
```

---

## Build & Run

**Requirements:** CUDA 12.x, cuBLAS, C++17, `make`

```bash
# Train and export weights (one-time)
cd weights/
python export_weights.ipynb          # trains MLP on MNIST, saves *.npy weight files

# Build and run inference
make
./inference
```

Expected output:
```
Test images: [1000, 784]
Test labels: [1000]  (first 5: 7 2 1 0 4 )
Weights loaded from weights

CPU inference:
  Correct: 969/1000
  Accuracy: 96.90%

GPU inference:
  Correct: 969/1000
  Accuracy: 96.90%

=== Summary ===
PyTorch baseline: 97.34%
CPU inference:    96.90%
GPU inference:    96.90%
CPU==GPU: yes
```

---

## The Hard Bug

The engine produced **8.80% accuracy** — near-random — despite weights and images loading correctly. A layer-by-layer trace confirmed the forward pass math was pixel-perfect against PyTorch all the way through fc1, fc2, fc3, and softmax. Sample 0 predicted class 7 with 99.98% confidence. The correct answer was also 7.

The bug was in label loading. PyTorch exports labels as `int64`. The `.npy` parser loaded them into a `float*`. An `int64` value of `7` has bytes `07 00 00 00 00 00 00 00`; reinterpreted as `float32` that's `~9.8e-45`, which casts to `int` as `0`. Every label was wrong — the predictions were fine, the ground truth was garbage.

Fix: read the `.npy` header directly, skip to the raw data, and `fread` into `int64_t` before casting to `int`.

---

## Key Implementation Notes

- **`Tensor<T>`** owns its memory via RAII. Copy constructor is deleted — tensors are move-only to prevent accidental deep copies across the CPU/GPU boundary.
- **`to_gpu()` / `to_cpu()`** use `cudaMalloc` + `cudaMemcpy`; the CPU pointer is freed on `to_gpu()` and vice versa. No double-ownership.
- **Xavier init** in `Linear`: weights drawn from `U(-√(6/(in+out)), √(6/(in+out)))`. Keeps activations in a sane range through a deep stack.
- **Tiled matmul** uses `__syncthreads()` barriers on both tile loads to prevent read-after-write and write-after-read hazards in shared memory.
- **Softmax kernel** applies the max-subtraction stability trick before `expf` to prevent overflow on large logits.