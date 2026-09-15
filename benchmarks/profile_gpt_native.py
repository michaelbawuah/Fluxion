from __future__ import annotations

import cProfile
import pstats

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT


VOCAB_SIZE = 64
SEQUENCE_LENGTH = 32
BATCH_SIZE = 16
EMBED_DIM = 32
NUM_HEADS = 4
HIDDEN_DIM = 64
NUM_LAYERS = 2

WARMUP_STEPS = 3
PROFILE_STEPS = 100


def main() -> None:
    np.random.seed(0)

    model = GPT(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_native_linear=True,
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

    for _ in range(WARMUP_STEPS):
        training_step()

    profiler = cProfile.Profile()

    profiler.enable()

    for _ in range(PROFILE_STEPS):
        training_step()

    profiler.disable()

    print(
        "Fluxion GPT Native Profile"
    )
    print(
        "=========================="
    )

    stats = pstats.Stats(
        profiler
    )

    stats.sort_stats(
        "cumulative"
    )

    stats.print_stats(
        30
    )


if __name__ == "__main__":
    main()