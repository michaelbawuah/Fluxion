# Historical macOS CPU checkpoint

These results were reported in the README at source commit [`8febc2c1e69a3a86777f4246eb66cebe453ff6cb`](https://github.com/michaelbawuah/Fluxion/tree/8febc2c1e69a3a86777f4246eb66cebe453ff6cb). They describe an earlier Apple Silicon/macOS float64 checkpoint and are retained as project history. Original per-trial files, full machine configuration, dependency versions, and measurement timestamps were not committed with that report, so these tables cannot support a precise comparison with the newly recorded Linux measurements.

## Gradient accumulation A/B experiment

The experiment compares zero-allocation-plus-add with copy-on-first-write gradient accumulation. The original seven-trial report gave:

| Metric | Reported value |
|---|---:|
| Median speedup | 1.125× |
| Mean speedup | 1.126× |
| Standard deviation | 0.017 |
| Trial range | 1.100×–1.161× |

A speedup of 1.125× corresponds to about 11.1% lower elapsed time for that workload. The reproduction entry point is [`benchmark_autograd_ab.py`](../../benchmarks/benchmark_autograd_ab.py).

## Native Linear in a GPT training step

The original repaired seven-trial experiment used batch size 8, sequence length 16, embedding dimension 16, two Transformer layers, 20 warmup steps, and 200 timed steps per trial:

| Metric | Reported value |
|---|---:|
| Median speedup | 1.050× |
| Mean speedup | 1.051× |
| Standard deviation | 0.009 |
| Trial range | 1.041×–1.067× |

The report documented an earlier baseline-integrity issue: regular `Linear` had accidentally used the native path, making the first comparison invalid. The baseline was restored and a regression test added before the figures above were reported. The reproduction entry point is [`benchmark_gpt_native.py`](../../benchmarks/benchmark_gpt_native.py).

## CPU training-step workload matrix

| Workload | Tokens per step | Regular Fluxion | NativeLinear | PyTorch CPU | Native speedup, regular/native | PyTorch speedup, native/PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| Small | 64 | 1.204 ms | 1.137 ms | 0.854 ms | 1.059× | 1.332× |
| Medium | 256 | 3.173 ms | 2.931 ms | 1.667 ms | 1.083× | 1.758× |
| Large | 512 | 9.423 ms | 10.106 ms | 5.182 ms | 0.932× | 1.950× |

NativeLinear helped the two smaller workloads and slowed the largest. The benchmark times the complete model step: native Linear does not replace attention, normalization, graph traversal, optimizer work, or every memory movement. That negative result is part of the experiment, rather than a reason to omit the workload.

The reproduction entry point is [`benchmark_cpu_matrix.py`](../../benchmarks/benchmark_cpu_matrix.py). Exact timings depend on hardware, BLAS, thread settings, Python and library versions, tensor shapes, and timing protocol. No GPU measurement was provided by this checkpoint.
