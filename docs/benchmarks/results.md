# README measurements · 2026-10-04

The README figures are generated from a fresh run of Fluxion's own training
loop. The raw values are retained so readers can inspect the measurements,
rebuild the figures, or run the experiment on another machine.

Source: [`8febc2c1e69a3a86777f4246eb66cebe453ff6cb`](https://github.com/michaelbawuah/Fluxion/tree/8febc2c1e69a3a86777f4246eb66cebe453ff6cb).
SHA-256 hashes of the measured source files are recorded in
[`results.json`](2026-10-04/results.json). The documentation changes do not alter
that engine implementation.

## What was measured

**Learning:** the network in [`train_tiny_network.py`](../../examples/train_tiny_network.py)
is reproduced with seed 0, float64, `Linear(1, 8) → ReLU → Linear(8, 1)`,
mean squared error, and 1,000 SGD updates at learning rate 0.01. The five training
inputs are 1–5 and their targets are twice the input. Loss is recorded before
each update and once after the final update.

MSE falls from **0.2563583788 to 7.8073562 × 10⁻¹³**. The final predictions at
unseen inputs 6, 7, 8, and 10 differ from 12, 14, 16, and 20 by less than
3.57 × 10⁻⁶. This is an end-to-end sanity check of gradients and parameter
updates on a synthetic regression problem, not evidence of language-model
quality or real-world generalization.

**Scaling:** a complete GPT training step includes gradient reset, forward,
cross-entropy, backward, and the Adam update. The configuration is batch 8,
vocabulary 32, embedding width 16, four attention heads, feed-forward width 32,
two transformer blocks, float64, seed 0, and Adam learning rate 0.001. Inputs
and targets are fixed random token arrays for each sequence length.

Each point uses five independently initialized trials. Each trial has 10 warmup
steps and 40 individually timed steps. The plotted value is the **median of the
five trial mean step times**; shading shows their interquartile range. Backend
order alternates between trials. Model creation, input generation, explicit
pre-trial collection, and warmup are excluded; normal Python garbage collection
remains enabled during the timed steps. Every timed sample is retained.

| Sequence length | NumPy path, ms/step | Native Linear path, ms/step | NumPy tokens/s |
| ---: | ---: | ---: | ---: |
| 8 | 2.379 | 2.938 | 26,902 |
| 16 | 3.128 | 3.949 | 40,921 |
| 32 | 5.173 | 7.251 | 49,483 |
| 64 | 10.762 | 14.882 | 47,574 |
| 128 | 29.531 | 37.825 | 34,676 |

The native path replaces Linear projections with the actual C++ extension.
Attention, normalization, embeddings, and the surrounding autograd graph remain
on NumPy/Python. On **this host**, native Linear is slower: NumPy uses optimized
OpenBLAS 0.3.30, while the extension links the system Netlib BLAS 3.12.0 library.
These results include the effect of different BLAS implementations and do not
isolate Python-versus-C++ overhead. They do not establish a general native
speedup or slowdown. No PyTorch timing comparison is made in these figures.

## Machine and verification

The run used a shared virtualized Linux x86-64 host reporting an AMD EPYC 9V74
CPU, nine visible logical CPUs, and an eight-CPU cgroup quota. Python was
3.12.14, NumPy 2.3.5, pybind11 3.1.0, and the compiler GCC 13.3.0. BLAS/OpenMP
thread variables were set to one before NumPy was imported; threadpool metadata
confirms one OpenBLAS thread. Netlib BLAS is the sequential system library.
Timing variation from the shared host is visible in the retained trials.

The unmodified native build script produced a Linux `cblas` float64 extension.
The complete test suite reported **72 passed and 2 skipped**; the two skips are
CUDA backend tests. PyTorch 2.14.1+cpu was used as a numerical reference.

| Reference comparison | Maximum absolute output error | Maximum absolute gradient error |
| --- | ---: | ---: |
| Linear | 4.441 × 10⁻¹⁶ | 2.220 × 10⁻¹⁶ |
| LayerNorm | 4.441 × 10⁻¹⁶ | 8.882 × 10⁻¹⁶ |
| Causal attention | 2.220 × 10⁻¹⁶ | 4.441 × 10⁻¹⁶ |
| Transformer block | 7.216 × 10⁻¹⁶ | 2.665 × 10⁻¹⁵ |
| GPT | 2.887 × 10⁻¹⁵ | 4.263 × 10⁻¹⁴ |

These are the five fixed comparisons in
[`validate_pytorch.py`](../../validation/validate_pytorch.py), which reported
`PASS` for each. Its summary uses maximum absolute error and prints a result;
the test suite separately contains assertion-based PyTorch comparisons.
Relative error can be misleading when a reference gradient is near zero.
The recorded checks apply to these configurations, not every possible input.
CUDA was not built or exercised, and the figures contain no GPU measurements.

## Reproduce or inspect

Create a Python 3.11+ environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,reference]' pybind11 matplotlib threadpoolctl
```

The optional native build needs a C++17 compiler and platform BLAS development
headers/libraries. On Ubuntu/Debian, install `g++` and `libopenblas-dev` (or
`libblas-dev`); on macOS, install the Xcode command-line tools for Accelerate.
Python development headers must also be available for your interpreter.

```bash
# Optional for ordinary NumPy execution; required for the full current suite.
python native/build_native.py

# PyTorch is also required by the current tests' reference imports.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.validate_pytorch

# Keep the committed reference run intact; write a separate measurement set.
PYTHONPATH=src:. python docs/benchmarks/run_readme_benchmarks.py \
  --output docs/benchmarks/local-run

# Rebuild the README figures from the committed data, or supply --data.
python docs/media/render_charts.py
python docs/media/render_charts.py --data docs/benchmarks/local-run/results.json
```

If the native extension is absent, the measurement script records only the
NumPy path. It never silently substitutes NumPy for a native result. On macOS,
the native backend is labeled `native_accelerate`. A different BLAS library,
NumPy/Python version, CPU, or shared-host load can materially change timings.

| Retained file | Contents |
| --- | --- |
| [`results.json`](2026-10-04/results.json) | Configuration, environment, source hashes, loss/predictions, all timing samples and summaries |
| [`training.csv`](2026-10-04/training.csv) | Loss after 0–1,000 parameter updates |
| [`predictions.csv`](2026-10-04/predictions.csv) | Final training and unseen-input predictions with targets |
| [`scaling.csv`](2026-10-04/scaling.csv) | Per-backend timing medians, quartiles, and throughput |
| [`scaling_samples.csv`](2026-10-04/scaling_samples.csv) | Every timed step, including its backend, sequence length, and trial |
| [`verification.json`](2026-10-04/verification.json) | Test/reference results and native build metadata |
| [`run_readme_benchmarks.py`](run_readme_benchmarks.py) | Deterministic measurement procedure |
| [`render_charts.py`](../media/render_charts.py) | PNG/SVG figure builder |
