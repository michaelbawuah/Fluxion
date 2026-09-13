from __future__ import annotations

import cProfile
import pstats

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT


np.random.seed(42)

vocab_size = 64
sequence_length = 32
batch_size = 16

model = GPT(
    vocab_size=vocab_size,
    max_sequence_length=sequence_length,
    embed_dim=32,
    num_heads=4,
    hidden_dim=64,
    num_layers=2,
)

criterion = CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=0.001,
)

inputs = np.random.randint(
    0,
    vocab_size,
    size=(
        batch_size,
        sequence_length,
    ),
    dtype=np.int64,
)

targets = np.random.randint(
    0,
    vocab_size,
    size=(
        batch_size,
        sequence_length,
    ),
    dtype=np.int64,
)


def training_step() -> None:
    optimizer.zero_grad()

    logits = model(
        inputs
    )

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    optimizer.step()


# Warm-up
for _ in range(3):
    training_step()


profiler = cProfile.Profile()

profiler.enable()

for _ in range(100):
    training_step()

profiler.disable()


stats = pstats.Stats(
    profiler
)

stats.sort_stats(
    "cumulative"
)

print()
print("Fluxion GPT Profile")
print("-------------------")

stats.print_stats(25)