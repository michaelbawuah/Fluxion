from __future__ import annotations

import statistics
import time

import numpy as np

from fluxion.nn.layers import Linear, NativeLinear
from fluxion.tensor import Tensor


BATCH_SIZE = 256
INPUT_DIM = 512
OUTPUT_DIM = 512

WARMUP_STEPS = 20
MEASURED_STEPS = 200
TRIALS = 7


def measure(function) -> dict[str, float]:
    for _ in range(WARMUP_STEPS):
        function()

    timings_ms: list[float] = []

    for _ in range(MEASURED_STEPS):
        start_time = time.perf_counter()

        function()

        end_time = time.perf_counter()

        timings_ms.append(
            (end_time - start_time) * 1000.0
        )

    return {
        "median": statistics.median(timings_ms),
        "mean": statistics.mean(timings_ms),
        "std": statistics.stdev(timings_ms),
        "min": min(timings_ms),
    }


def summarize_trials(
    values: list[float],
) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def print_trial_result(
    trial_number: int,
    regular_forward: float,
    native_forward: float,
    regular_backward: float,
    native_backward: float,
) -> None:
    forward_speedup = (
        regular_forward
        / native_forward
    )

    backward_speedup = (
        regular_backward
        / native_backward
    )

    print(
        f"Trial {trial_number:>2}: "
        f"forward {forward_speedup:>6.3f}x | "
        f"forward+backward {backward_speedup:>6.3f}x"
    )


def main() -> None:
    np.random.seed(0)

    x_data = np.random.randn(
        BATCH_SIZE,
        INPUT_DIM,
    )

    weight_data = np.random.randn(
        INPUT_DIM,
        OUTPUT_DIM,
    )

    bias_data = np.random.randn(
        OUTPUT_DIM,
    )

    regular = Linear(
        INPUT_DIM,
        OUTPUT_DIM,
    )

    native = NativeLinear(
        INPUT_DIM,
        OUTPUT_DIM,
    )

    regular.weight.data = weight_data.copy()
    regular.bias.data = bias_data.copy()

    native.weight.data = weight_data.copy()
    native.bias.data = bias_data.copy()

    x_regular = Tensor(
        x_data.copy(),
        requires_grad=True,
    )

    x_native = Tensor(
        x_data.copy(),
        requires_grad=True,
    )

    # Correctness check before benchmarking.
    regular_output = regular(
        x_regular
    )

    native_output = native(
        x_native
    )

    if not np.allclose(
        regular_output.data,
        native_output.data,
    ):
        raise RuntimeError(
            "NativeLinear forward does not match Linear."
        )

    regular.zero_grad()
    native.zero_grad()

    x_regular.grad = None
    x_native.grad = None

    regular_output.sum().backward()
    native_output.sum().backward()

    if not np.allclose(
        x_regular.grad,
        x_native.grad,
    ):
        raise RuntimeError(
            "NativeLinear input gradient does not match Linear."
        )

    if not np.allclose(
        regular.weight.grad,
        native.weight.grad,
    ):
        raise RuntimeError(
            "NativeLinear weight gradient does not match Linear."
        )

    if not np.allclose(
        regular.bias.grad,
        native.bias.grad,
    ):
        raise RuntimeError(
            "NativeLinear bias gradient does not match Linear."
        )

    def regular_forward() -> None:
        regular(
            x_regular
        )

    def native_forward() -> None:
        native(
            x_native
        )

    def regular_forward_backward() -> None:
        regular.zero_grad()
        x_regular.grad = None

        output = regular(
            x_regular
        )

        loss = output.sum()

        loss.backward()

    def native_forward_backward() -> None:
        native.zero_grad()
        x_native.grad = None

        output = native(
            x_native
        )

        loss = output.sum()

        loss.backward()

    forward_speedups: list[float] = []
    backward_speedups: list[float] = []

    print(
        "Fluxion Linear Native Fusion Benchmark"
    )
    print(
        "======================================"
    )
    print(
        f"Batch size:      {BATCH_SIZE}"
    )
    print(
        f"Input dim:       {INPUT_DIM}"
    )
    print(
        f"Output dim:      {OUTPUT_DIM}"
    )
    print(
        f"Warmup steps:    {WARMUP_STEPS}"
    )
    print(
        f"Measured steps:  {MEASURED_STEPS}"
    )
    print(
        f"Independent runs:{TRIALS}"
    )
    print()

    for trial in range(
        1,
        TRIALS + 1,
    ):
        regular_forward_result = measure(
            regular_forward
        )

        native_forward_result = measure(
            native_forward
        )

        regular_backward_result = measure(
            regular_forward_backward
        )

        native_backward_result = measure(
            native_forward_backward
        )

        forward_speedup = (
            regular_forward_result["median"]
            / native_forward_result["median"]
        )

        backward_speedup = (
            regular_backward_result["median"]
            / native_backward_result["median"]
        )

        forward_speedups.append(
            forward_speedup
        )

        backward_speedups.append(
            backward_speedup
        )

        print_trial_result(
            trial,
            regular_forward_result["median"],
            native_forward_result["median"],
            regular_backward_result["median"],
            native_backward_result["median"],
        )

    forward_summary = summarize_trials(
        forward_speedups
    )

    backward_summary = summarize_trials(
        backward_speedups
    )

    print()
    print(
        "Aggregate speedups"
    )
    print(
        "------------------"
    )

    print(
        "Forward:"
    )
    print(
        f"  median: {forward_summary['median']:.3f}x"
    )
    print(
        f"  mean:   {forward_summary['mean']:.3f}x"
    )
    print(
        f"  std:    {forward_summary['std']:.3f}"
    )
    print(
        f"  range:  "
        f"{forward_summary['min']:.3f}x"
        f" - "
        f"{forward_summary['max']:.3f}x"
    )

    print()

    print(
        "Forward + backward:"
    )
    print(
        f"  median: {backward_summary['median']:.3f}x"
    )
    print(
        f"  mean:   {backward_summary['mean']:.3f}x"
    )
    print(
        f"  std:    {backward_summary['std']:.3f}"
    )
    print(
        f"  range:  "
        f"{backward_summary['min']:.3f}x"
        f" - "
        f"{backward_summary['max']:.3f}x"
    )


if __name__ == "__main__":
    main()