from __future__ import annotations

import time

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT


VOCAB_SIZE = 32
BATCH_SIZE = 8
EMBED_DIM = 16
NUM_HEADS = 4
HIDDEN_DIM = 32
NUM_LAYERS = 2

SEQUENCE_LENGTHS = [
    8,
    16,
    32,
    64,
    128,
]

WARMUP_STEPS = 3
MEASURED_STEPS = 20


def benchmark_sequence_length(
    sequence_length: int,
) -> float:
    """
    Measure the average time for one complete GPT training step.

    A training step contains:

        forward pass
        loss computation
        backward pass
        optimizer update
    """

    np.random.seed(0)

    model = GPT(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=sequence_length,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )

    loss_function = CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    token_ids = np.random.randint(
        0,
        VOCAB_SIZE,
        size=(
            BATCH_SIZE,
            sequence_length,
        ),
    )

    targets = np.random.randint(
        0,
        VOCAB_SIZE,
        size=(
            BATCH_SIZE,
            sequence_length,
        ),
    )

    def training_step() -> None:
        optimizer.zero_grad()

        logits = model(
            token_ids
        )

        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

    # Warm up the Python / NumPy execution path.
    for _ in range(WARMUP_STEPS):
        training_step()

    start_time = time.perf_counter()

    for _ in range(MEASURED_STEPS):
        training_step()

    end_time = time.perf_counter()

    average_seconds = (
        end_time - start_time
    ) / MEASURED_STEPS

    return average_seconds * 1000.0


def main() -> None:
    print(
        "Fluxion GPT Sequence-Length Scaling Benchmark"
    )
    print(
        "--------------------------------------------"
    )

    print(
        f"{'Sequence':>10} | "
        f"{'Step Time':>12} | "
        f"{'Slowdown':>10} | "
        f"{'Tokens/sec':>12}"
    )

    print(
        "-" * 55
    )

    baseline_time = None

    for sequence_length in SEQUENCE_LENGTHS:
        step_time_ms = benchmark_sequence_length(
            sequence_length
        )

        if baseline_time is None:
            baseline_time = step_time_ms

        slowdown = (
            step_time_ms
            / baseline_time
        )

        tokens_per_step = (
            BATCH_SIZE
            * sequence_length
        )

        tokens_per_second = (
            tokens_per_step
            / (
                step_time_ms
                / 1000.0
            )
        )

        print(
            f"{sequence_length:>10} | "
            f"{step_time_ms:>9.3f} ms | "
            f"{slowdown:>8.2f}x | "
            f"{tokens_per_second:>12.0f}"
        )


if __name__ == "__main__":
    main()