from __future__ import annotations

import statistics
import time
from collections.abc import Callable

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


def measure(
    function: Callable[[], None],
) -> dict[str, float]:
    """
    Measure one operation many times and summarize the timing distribution.
    """

    for _ in range(WARMUP_STEPS):
        function()

    timings_ms: list[float] = []

    for _ in range(MEASURED_STEPS):
        start_time = time.perf_counter()

        function()

        end_time = time.perf_counter()

        elapsed_ms = (
            end_time - start_time
        ) * 1000.0

        timings_ms.append(
            elapsed_ms
        )

    return {
        "mean": statistics.mean(
            timings_ms
        ),
        "median": statistics.median(
            timings_ms
        ),
        "std": statistics.stdev(
            timings_ms
        ),
        "min": min(
            timings_ms
        ),
    }


def print_result(
    name: str,
    result: dict[str, float],
) -> None:
    print(
        f"{name:<20} | "
        f"median {result['median']:>8.3f} ms | "
        f"mean {result['mean']:>8.3f} ms | "
        f"std {result['std']:>8.3f} ms | "
        f"min {result['min']:>8.3f} ms"
    )


def main() -> None:
    np.random.seed(0)

    model = GPT(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
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

    def forward_pass() -> None:
        model(
            token_ids
        )

    def forward_backward() -> None:
        optimizer.zero_grad()

        logits = model(
            token_ids
        )

        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()

    def full_training_step() -> None:
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

    print(
        "Fluxion GPT Benchmark"
    )
    print(
        "====================="
    )
    print(
        f"Warmup steps:   {WARMUP_STEPS}"
    )
    print(
        f"Measured steps: {MEASURED_STEPS}"
    )
    print()

    forward_result = measure(
        forward_pass
    )

    backward_result = measure(
        forward_backward
    )

    training_result = measure(
        full_training_step
    )

    print_result(
        "Forward",
        forward_result,
    )

    print_result(
        "Forward + backward",
        backward_result,
    )

    print_result(
        "Training step",
        training_result,
    )


if __name__ == "__main__":
    main()