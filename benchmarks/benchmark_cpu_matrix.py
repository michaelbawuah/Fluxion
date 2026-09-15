from __future__ import annotations

import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxion.nn.losses import CrossEntropyLoss
from fluxion.optim.adam import Adam
from fluxion.transformer.gpt import GPT

WARMUP = 10
MEASURED = 50
TRIALS = 5
VOCAB = 64
WORKLOADS = [
    ("small", 4, 16, 16, 4, 32, 2),
    ("medium", 8, 32, 32, 4, 64, 2),
    ("large", 8, 64, 32, 4, 64, 3),
]


class TorchBlock(nn.Module):
    def __init__(self, d, h, hidden):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, hidden)
        self.ff2 = nn.Linear(hidden, d)
        self.h = h
        self.hd = d // h

    def forward(self, x):
        z = self.n1(x)
        b, s, d = z.shape
        q = self.q(z).reshape(b, s, self.h, self.hd).transpose(1, 2)
        k = self.k(z).reshape(b, s, self.h, self.hd).transpose(1, 2)
        v = self.v(z).reshape(b, s, self.h, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        a = a.transpose(1, 2).reshape(b, s, d)
        x = x + self.o(a)
        z = self.n2(x)
        return x + self.ff2(torch.relu(self.ff1(z)))


class TorchGPT(nn.Module):
    def __init__(self, vocab, seq, d, h, hidden, layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([TorchBlock(d, h, hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, ids):
        s = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(s, device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


def time_fluxion(batch, seq, d, heads, hidden, layers, native):
    np.random.seed(7)
    model = GPT(VOCAB, seq, d, heads, hidden, layers, use_native_linear=native)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    ids = np.random.randint(0, VOCAB, (batch, seq))
    targets = np.random.randint(0, VOCAB, (batch, seq))
    def step():
        optimizer.zero_grad()
        loss = loss_fn(model(ids), targets)
        loss.backward()
        optimizer.step()
    for _ in range(WARMUP): step()
    start = time.perf_counter()
    for _ in range(MEASURED): step()
    return (time.perf_counter() - start) * 1000 / MEASURED


def time_torch(batch, seq, d, heads, hidden, layers):
    torch.manual_seed(7)
    model = TorchGPT(VOCAB, seq, d, heads, hidden, layers).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.randint(0, VOCAB, (batch, seq))
    targets = torch.randint(0, VOCAB, (batch, seq))
    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        loss.backward()
        optimizer.step()
    for _ in range(WARMUP): step()
    start = time.perf_counter()
    for _ in range(MEASURED): step()
    return (time.perf_counter() - start) * 1000 / MEASURED


def median_trials(fn):
    return statistics.median(fn() for _ in range(TRIALS))


def main():
    torch.set_num_threads(1)
    print("Fluxion CPU Training-Step Benchmark Matrix (float64, median of 5 trials)")
    print("=" * 94)
    print(f"{'workload':<9} {'tokens':>7} {'Fluxion':>12} {'Native':>12} {'PyTorch':>12} {'N/Python':>10} {'Torch/Native':>13}")
    for name, batch, seq, d, heads, hidden, layers in WORKLOADS:
        args = (batch, seq, d, heads, hidden, layers)
        regular = median_trials(lambda: time_fluxion(*args, native=False))
        native = median_trials(lambda: time_fluxion(*args, native=True))
        pytorch = median_trials(lambda: time_torch(*args))
        tokens = batch * seq
        print(f"{name:<9} {tokens:>7} {regular:>9.3f} ms {native:>9.3f} ms {pytorch:>9.3f} ms {regular/native:>9.3f}x {native/pytorch:>12.3f}x")


if __name__ == "__main__":
    main()
