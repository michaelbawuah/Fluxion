# Fluxion

**A deep learning systems engine built from first principles.**

Fluxion is an educational/research deep learning framework that implements tensors, reverse-mode automatic differentiation, neural-network layers, optimizers, transformer components, and a small GPT model without relying on PyTorch for its core execution engine. PyTorch is used as a numerical reference for correctness validation.

The project is also a systems-performance study: profile the framework, identify bottlenecks, optimize them, validate correctness after each change, and measure whether the optimization actually improves end-to-end performance.

## Architecture

```text
Tensor
  ↓
Reverse-mode autograd
  ↓
Neural-network modules + losses + optimizers
  ↓
Attention + Transformer blocks
  ↓
GPT
  ↓
PyTorch numerical validation
  ↓
Native C++ / Apple Accelerate backend
  ↓
Profiling + controlled benchmarking
  ↓
CUDA / GPU backend (next)
```

## Current status

- **70 automated tests passing**
- Reverse-mode autograd with broadcasting, reductions, indexing, reshape, transpose, and batched matrix multiplication
- Neural-network modules including Linear, ReLU, Sigmoid, Softmax, LayerNorm, and Embedding
- SGD and Adam optimizers
- Causal multi-head self-attention, Transformer blocks, and GPT
- Character-level autoregressive training and generation
- PyTorch reference validation for forward computations **and gradients**
- Native C++ Linear forward/backward path using **pybind11 + Apple Accelerate**
- CPU profiling, controlled A/B optimization experiments, and multi-workload benchmarking

## Correctness: Fluxion vs PyTorch

Fluxion is validated against PyTorch using identical inputs and copied parameters. The current validation suite checks both forward outputs and reverse-mode gradients through increasingly complex components.

| Component | Max output abs. error | Max gradient abs. error | Result |
|---|---:|---:|:---:|
| Linear | `0.000e+00` | `0.000e+00` | PASS |
| LayerNorm | `4.441e-16` | `1.332e-15` | PASS |
| Causal Attention | `1.110e-16` | `4.441e-16` | PASS |
| TransformerBlock | `4.441e-16` | `2.665e-15` | PASS |
| GPT | `1.305e-15` | `4.974e-14` | PASS |

The small absolute discrepancies are consistent with floating-point evaluation-order differences. Relative error can appear larger for gradients whose reference values are extremely close to zero, so the validation reports both absolute and relative error.

Run the validation:

```bash
PYTHONPATH=. python validation/validate_pytorch.py
```

## Performance

These are **measured results for the current CPU implementation on Apple Silicon/macOS using float64**. They are workload- and machine-specific, not universal performance claims.

### Autograd gradient accumulation

Profiling identified repeated zero-filled gradient-buffer allocation as a source of overhead. Fluxion changed first-write gradient accumulation from a `zeros_like + add` strategy to **copy-on-first-write**, followed by in-place accumulation.

Controlled 7-trial A/B benchmark:

| Metric | Result |
|---|---:|
| Median speedup | **1.125x** |
| Mean speedup | **1.126x** |
| Std. deviation | `0.017` |
| Trial range | `1.100x – 1.161x` |

A `1.125x` speedup corresponds to about **11.1% lower elapsed time** for this benchmarked workload.

```bash
PYTHONPATH=. python benchmarks/benchmark_autograd_ab.py
```

### Native C++ Linear in GPT

Fluxion includes a fused native Linear path implemented in C++ and exposed through pybind11. On macOS it uses Apple Accelerate for matrix operations.

After adding a regression test to guarantee that the regular and native Linear paths are actually distinct, a 7-trial GPT training-step benchmark measured:

| Metric | Result |
|---|---:|
| Median speedup | **1.050x** |
| Mean speedup | **1.051x** |
| Std. deviation | `0.009` |
| Trial range | `1.041x – 1.067x` |

Tested configuration: batch size 8, sequence length 16, embedding dimension 16, 2 Transformer layers, 20 warmup steps, and 200 measured steps per trial.

```bash
PYTHONPATH=. python benchmarks/benchmark_gpt_native.py
```

### CPU training-step matrix

The benchmark matrix compares regular Fluxion, Fluxion with NativeLinear, and PyTorch CPU over multiple workloads.

| Workload | Tokens | Fluxion | Native C++ | PyTorch CPU | Native / Fluxion | PyTorch / Native |
|---|---:|---:|---:|---:|---:|---:|
| Small | 64 | 1.204 ms | 1.137 ms | 0.854 ms | **1.059x** | **1.332x** |
| Medium | 256 | 3.173 ms | 2.931 ms | 1.667 ms | **1.083x** | **1.758x** |
| Large | 512 | 9.423 ms | 10.106 ms | 5.182 ms | **0.932x** | **1.950x** |

The native Linear path improves the small and medium workloads, but it becomes slower than regular Fluxion in the large workload. This negative result is intentionally reported: accelerating one operator does not remove framework-level costs such as autograd bookkeeping, other tensor operations, attention/normalization work, memory movement, dispatch overhead, and backend/kernel boundaries. PyTorch's advantage also grows with workload size, motivating the next profiling and GPU stages rather than cherry-picking only favorable CPU cases.

```bash
PYTHONPATH=. python benchmarks/benchmark_cpu_matrix.py
```

## Benchmark methodology

Performance work follows a simple rule:

> **baseline → profile → optimize → validate correctness → re-benchmark**

Benchmarks use warmup iterations and repeated measured trials. Correctness checks are kept separate from timing, and profiler output is used for hotspot discovery rather than treated as authoritative wall-clock speedup evidence.

One benchmark-integrity bug was caught during development: the regular `Linear` path had accidentally been routed through the native implementation, making an earlier regular-vs-native comparison invalid. The baseline was restored, a regression test was added to guarantee distinct execution paths, and the benchmark was rerun. Only the repaired results are reported above.

## Repository layout

```text
src/fluxion/
├── tensor.py              # Tensor object and backward traversal
├── autograd.py
├── ops.py                 # Differentiable tensor operations
├── nn/
│   ├── module.py
│   ├── layers.py
│   └── losses.py
├── optim/
│   ├── sgd.py
│   └── adam.py
└── transformer/
    ├── attention.py
    ├── layers.py
    └── gpt.py

native/
└── fluxion_native.cpp     # Native C++ Linear backend

tests/                     # Unit and reference-correctness tests
validation/                # Fluxion ↔ PyTorch numerical validation
benchmarks/                # Profiling and controlled performance experiments
examples/                  # Training / generation examples
```

## Quick start

Create an environment and install Fluxion in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,reference]"
```

Run the complete test suite:

```bash
PYTHONPATH=. python -m pytest -q
```

Current checkpoint:

```text
70 passed
```

## Native backend

The current native backend is macOS-specific because it uses Apple Accelerate. The extension is compiled locally with Apple Clang, pybind11, and Accelerate.

Example build command:

```bash
clang++ \
  -O3 \
  -Wall \
  -shared \
  -std=c++17 \
  -undefined dynamic_lookup \
  $(python -m pybind11 --includes) \
  native/fluxion_native.cpp \
  -framework Accelerate \
  -o fluxion_native$(python3-config --extension-suffix)
```

A future portability step will separate the backend interface from the platform implementation so Linux can use an appropriate BLAS backend before GPU work.

## Roadmap

**Completed:** tensor engine, reverse-mode autograd, neural-network modules, optimizers, attention/Transformer/GPT, PyTorch numerical validation, profiling, gradient-accumulation optimization, native C++ Linear acceleration, and CPU benchmark matrix.

**Next:** portable Linux native backend and reproducible machine metadata, followed by NVIDIA/CUDA kernels, GPU correctness validation, profiling, and CPU/GPU comparisons against PyTorch.

Longer-term work may include improved dtype/device abstractions, additional fused kernels, better native build tooling, and more extensive benchmark workloads.

## Project philosophy

Fluxion is not intended to claim that a small from-scratch framework outperforms PyTorch. The goal is to understand and demonstrate the systems underneath modern deep learning frameworks:

- how reverse-mode autograd constructs and traverses computation graphs,
- how tensor shapes and broadcasting affect gradient propagation,
- where Python/framework overhead appears,
- when native kernels help and when they do not,
- how to validate a custom implementation against a trusted reference,
- and how profiling evidence should drive optimization decisions.

The project deliberately preserves negative results and benchmark methodology because understanding **why an optimization fails to improve end-to-end performance** is part of systems engineering.

## Portable CPU backend (Milestone B)

The native extension now has a platform boundary instead of hard-coding Apple Accelerate throughout the implementation:

```text
Fluxion Python
      |
      v
fluxion_native (pybind11 interface)
      |
      +-- macOS -> Apple Accelerate / CBLAS
      |
      +-- Linux -> CBLAS (OpenBLAS or system BLAS)
      |
      +-- NVIDIA CUDA backend -> next stage
```

Build the CPU extension with the platform-aware build helper:

```bash
python -m pip install pybind11
python native/build_native.py
```

On macOS the build links Apple Accelerate. On Linux it prefers OpenBLAS when available and otherwise uses the system BLAS library. The compiled extension exposes backend metadata so benchmark logs can identify the implementation being measured:

```python
from fluxion.native import backend_name, build_info

print(backend_name())
print(build_info())
```

For Ubuntu/AWS Linux development, install a compiler and OpenBLAS headers before building, for example:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libopenblas-dev python3-dev
python -m pip install pybind11
python native/build_native.py
PYTHONPATH=. python -m pytest -q
```

This portability layer is intentionally completed before CUDA work so CPU and GPU backends can share a stable Python-facing boundary while retaining platform-specific implementations underneath.
