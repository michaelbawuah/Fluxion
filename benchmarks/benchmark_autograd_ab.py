from __future__ import annotations

import statistics
import time

import numpy as np

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.tensor import Tensor
from fluxion.transformer.gpt import GPT

VOCAB_SIZE = 32
BATCH = 8
SEQ_LEN = 16
EMBED = 16
HEADS = 4
HIDDEN = 32
LAYERS = 2
WARMUP = 20
MEASURED = 200
TRIALS = 7

CURRENT_ACCUMULATE = Tensor._accumulate_grad


def legacy_accumulate(self: Tensor, gradient: np.ndarray) -> None:
    """Historical first-write strategy: allocate zeros, then add."""
    if not self.requires_grad:
        return
    gradient = np.asarray(gradient, dtype=self.data.dtype)
    if gradient.shape != self.data.shape:
        raise ValueError("gradient shape mismatch")
    if self.grad is None:
        self.grad = np.zeros_like(self.data)
    self.grad += gradient


def make_workload():
    np.random.seed(123)
    model = GPT(VOCAB_SIZE, SEQ_LEN, EMBED, HEADS, HIDDEN, LAYERS)
    loss_fn = CrossEntropyLoss()
    tokens = np.random.randint(0, VOCAB_SIZE, size=(BATCH, SEQ_LEN))
    targets = np.random.randint(0, VOCAB_SIZE, size=(BATCH, SEQ_LEN))
    return model, loss_fn, tokens, targets


def run_step(model, loss_fn, tokens, targets):
    model.zero_grad()
    logits = model(tokens)
    loss_fn(logits, targets).backward()


def measure(accumulator) -> float:
    Tensor._accumulate_grad = accumulator
    model, loss_fn, tokens, targets = make_workload()
    for _ in range(WARMUP):
        run_step(model, loss_fn, tokens, targets)
    start = time.perf_counter()
    for _ in range(MEASURED):
        run_step(model, loss_fn, tokens, targets)
    return (time.perf_counter() - start) * 1000.0 / MEASURED


def main():
    legacy_times, current_times, speedups = [], [], []
    print("Fluxion Autograd Gradient-Accumulation A/B")
    print("Legacy=zeros_like+add vs Current=copy-on-first-write")
    print("=" * 68)
    try:
        for trial in range(TRIALS):
            if trial % 2 == 0:
                legacy = measure(legacy_accumulate)
                current = measure(CURRENT_ACCUMULATE)
            else:
                current = measure(CURRENT_ACCUMULATE)
                legacy = measure(legacy_accumulate)
            speedup = legacy / current
            legacy_times.append(legacy)
            current_times.append(current)
            speedups.append(speedup)
            print(f"Trial {trial + 1}: legacy {legacy:.3f} ms | current {current:.3f} ms | speedup {speedup:.3f}x")
    finally:
        Tensor._accumulate_grad = CURRENT_ACCUMULATE
    print("\nAggregate")
    print("---------")
    print(f"Median speedup: {statistics.median(speedups):.3f}x")
    print(f"Mean speedup:   {statistics.mean(speedups):.3f}x")
    print(f"Std:            {statistics.pstdev(speedups):.3f}")
    print(f"Range:          {min(speedups):.3f}x - {max(speedups):.3f}x")


if __name__ == "__main__":
    main()
