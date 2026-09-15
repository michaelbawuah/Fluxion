from __future__ import annotations

import statistics
import time

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT


VOCAB_SIZE = 32
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
EMBED_DIM = 16
NUM_HEADS = 4
HIDDEN_DIM = 32
NUM_LAYERS = 2

WARMUP_STEPS = 20
MEASURED_STEPS = 200
TRIALS = 7


def measure(function) -> float:
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

    return statistics.median(
        timings_ms
    )


def copy_parameters(
    source: GPT,
    target: GPT,
) -> None:
    source_parameters = source.parameters()
    target_parameters = target.parameters()

    if len(source_parameters) != len(target_parameters):
        raise RuntimeError(
            "Regular and native GPT parameter counts differ."
        )

    for source_parameter, target_parameter in zip(
        source_parameters,
        target_parameters,
    ):
        target_parameter.data = (
            source_parameter.data.copy()
        )


def main() -> None:
    np.random.seed(0)

    regular_model = GPT(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_native_linear=False,
    )

    native_model = GPT(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_native_linear=True,
    )

    copy_parameters(
        regular_model,
        native_model,
    )

    loss_function = CrossEntropyLoss()

    regular_optimizer = Adam(
        regular_model.parameters(),
        lr=0.001,
    )

    native_optimizer = Adam(
        native_model.parameters(),
        lr=0.001,
    )

    token_ids = np.random.randint(
        0,
        VOCAB_SIZE,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
    )

    targets = np.random.randint(
        0,
        VOCAB_SIZE,
        size=(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
    )

    # Correctness check before benchmarking.
    regular_logits = regular_model(
        token_ids
    )

    native_logits = native_model(
        token_ids
    )

    if not np.allclose(
        regular_logits.data,
        native_logits.data,
    ):
        max_error = np.max(
            np.abs(
                regular_logits.data
                - native_logits.data
            )
        )

        raise RuntimeError(
            "Native GPT forward does not match "
            f"regular GPT. Max error: {max_error}"
        )

    def regular_training_step() -> None:
        regular_optimizer.zero_grad()

        logits = regular_model(
            token_ids
        )

        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()

        regular_optimizer.step()

    def native_training_step() -> None:
        native_optimizer.zero_grad()

        logits = native_model(
            token_ids
        )

        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()

        native_optimizer.step()

    speedups: list[float] = []

    print(
        "Fluxion GPT Native Linear Benchmark"
    )
    print(
        "==================================="
    )
    print(
        f"Batch size:      {BATCH_SIZE}"
    )
    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )
    print(
        f"Embed dim:       {EMBED_DIM}"
    )
    print(
        f"Layers:          {NUM_LAYERS}"
    )
    print(
        f"Warmup steps:    {WARMUP_STEPS}"
    )
    print(
        f"Measured steps:  {MEASURED_STEPS}"
    )
    print(
        f"Trials:          {TRIALS}"
    )
    print()

    for trial in range(
        1,
        TRIALS + 1,
    ):
        regular_time = measure(
            regular_training_step
        )

        native_time = measure(
            native_training_step
        )

        speedup = (
            regular_time
            / native_time
        )

        speedups.append(
            speedup
        )

        print(
            f"Trial {trial:>2}: "
            f"regular {regular_time:>7.3f} ms | "
            f"native {native_time:>7.3f} ms | "
            f"speedup {speedup:>6.3f}x"
        )

    print()
    print(
        "Aggregate"
    )
    print(
        "---------"
    )
    print(
        f"Median speedup: "
        f"{statistics.median(speedups):.3f}x"
    )
    print(
        f"Mean speedup:   "
        f"{statistics.mean(speedups):.3f}x"
    )
    print(
        f"Std:            "
        f"{statistics.stdev(speedups):.3f}"
    )
    print(
        f"Range:          "
        f"{min(speedups):.3f}x - "
        f"{max(speedups):.3f}x"
    )


if __name__ == "__main__":
    main()