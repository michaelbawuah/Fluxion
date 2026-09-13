from __future__ import annotations

import time

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT


def benchmark(
    name: str,
    function,
    iterations: int,
) -> None:
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        function()

        end = time.perf_counter()

        times.append(
            end - start
        )

    average = sum(times) / len(times)

    print(
        f"{name:<20} "
        f"{average * 1000:.3f} ms"
    )


np.random.seed(42)

vocab_size = 32
sequence_length = 16
batch_size = 8

model = GPT(
    vocab_size=vocab_size,
    max_sequence_length=sequence_length,
    embed_dim=16,
    num_heads=4,
    hidden_dim=32,
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


def forward_pass():
    model(inputs)


def backward_pass():
    optimizer.zero_grad()

    logits = model(inputs)

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()


def training_step():
    optimizer.zero_grad()

    logits = model(inputs)

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    optimizer.step()


print()
print("Fluxion GPT Baseline Benchmark")
print("------------------------------")

# Warm-up runs
for _ in range(3):
    training_step()

benchmark(
    "Forward pass:",
    forward_pass,
    iterations=20,
)

benchmark(
    "Forward + backward:",
    backward_pass,
    iterations=20,
)

benchmark(
    "Full training step:",
    training_step,
    iterations=20,
)

print()