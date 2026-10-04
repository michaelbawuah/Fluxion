# Fluxion visuals and measurements

The [project README](../README.md) connects the tensor engine, its gradients, and complete training workloads. These figures show each part in more detail.

| Figure | What it shows | Files |
| --- | --- | --- |
| Architecture | Tensor operations, reverse-mode autograd, model training, and the NumPy/native/CUDA execution boundaries | [PNG](media/architecture.png) · [SVG](media/architecture.svg) |
| Autograd | The three gradient contributions to a shared input in `y = x * x + x` | [PNG](media/autograd.png) · [SVG](media/autograd.svg) |
| Training progress | Recorded loss and final predictions from the seeded tiny-network example | [PNG](media/training-progress.png) · [SVG](media/training-progress.svg) |
| GPT sequence scaling | Full CPU training-step time and throughput, with repeated-trial variation | [PNG](media/sequence-scaling.png) · [SVG](media/sequence-scaling.svg) |

[Measurement procedure and environment](benchmarks/results.md) · [Recorded JSON/CSV data](benchmarks/2026-10-04/) · [Earlier macOS results](benchmarks/historical-results.md)

The editable SVGs and [diagram](media/render_diagrams.py) / [chart](media/render_charts.py) renderers are retained alongside the figures. The current charts use the recorded Linux CPU run; CUDA measurements are not included.
