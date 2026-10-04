"""Reproduce Fluxion's README CPU figures and retain all timing samples.

Run from the repository root after installing Fluxion. The optional native
extension must be built to include its results; NumPy is always measured.
This deliberately small synthetic workload tests the engine, not model quality.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

# Set these before importing NumPy so the results do not depend on ambient
# OpenBLAS/MKL/OpenMP thread defaults. This script runs CPU measurements only.
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
for name in THREAD_ENV:
    os.environ[name] = "1"

import numpy as np

from fluxion.native import build_info, is_available
from fluxion.nn.layers import Linear, ReLU
from fluxion.nn.losses import CrossEntropyLoss, MSELoss
from fluxion.nn.module import Module
from fluxion.optim.adam import Adam
from fluxion.optim.sgd import SGD
from fluxion.tensor import Tensor
from fluxion.transformer.gpt import GPT


ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCE_SHA = "8febc2c1e69a3a86777f4246eb66cebe453ff6cb"
SEED = 0
SCALING_CONFIG = {
    "vocab_size": 32,
    "batch_size": 8,
    "embed_dim": 16,
    "num_heads": 4,
    "hidden_dim": 32,
    "num_layers": 2,
    "sequence_lengths": [8, 16, 32, 64, 128],
    "warmup_steps_per_trial": 10,
    "measured_steps_per_trial": 40,
    "trials": 5,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "seed": SEED,
    "dtype": "float64",
}


class TinyNetwork(Module):
    def __init__(self) -> None:
        self.layer1 = Linear(1, 8)
        self.relu = ReLU()
        self.layer2 = Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(self.relu(self.layer1(x)))


def tiny_network() -> dict:
    """Use the same network, seed and training data as train_tiny_network.py."""
    np.random.seed(SEED)
    x = Tensor(np.arange(1, 6, dtype=np.float64).reshape(-1, 1))
    target = Tensor(2 * x.data)
    model = TinyNetwork()
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    losses = []
    for step in range(1001):
        prediction = model(x)
        loss = loss_fn(prediction, target)
        losses.append(float(loss.data.item()))
        if step < 1000:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    unseen = Tensor(np.array([6.0, 7.0, 8.0, 10.0]).reshape(-1, 1))
    return {
        "config": {
            "architecture": [1, 8, 1], "activation": "ReLU",
            "seed": SEED, "updates": 1000, "optimizer": "SGD",
            "learning_rate": 0.01, "loss": "mean squared error",
            "dtype": "float64",
        },
        "steps": list(range(1001)),
        "loss": losses,
        "train": {
            "x": x.data.ravel().tolist(),
            "prediction": model(x).data.ravel().tolist(),
            "target": target.data.ravel().tolist(),
        },
        "unseen": {
            "x": unseen.data.ravel().tolist(),
            "prediction": model(unseen).data.ravel().tolist(),
            "target": (2 * unseen.data).ravel().tolist(),
        },
        "interpretation": "Synthetic y=2x regression; unseen x values are a sanity check, not a real-world generalization benchmark.",
    }


def scaling_trial(sequence_length: int, native: bool) -> list[float]:
    np.random.seed(SEED)
    c = SCALING_CONFIG
    model = GPT(
        c["vocab_size"], sequence_length, c["embed_dim"], c["num_heads"],
        c["hidden_dim"], c["num_layers"], use_native_linear=native,
    )
    optimizer = Adam(model.parameters(), lr=c["learning_rate"])
    loss_fn = CrossEntropyLoss()
    shape = (c["batch_size"], sequence_length)
    ids = np.random.randint(0, c["vocab_size"], size=shape)
    targets = np.random.randint(0, c["vocab_size"], size=shape)

    def step() -> None:
        optimizer.zero_grad()
        loss = loss_fn(model(ids), targets)
        loss.backward()
        optimizer.step()

    gc.collect()
    for _ in range(c["warmup_steps_per_trial"]):
        step()
    timings = []
    for _ in range(c["measured_steps_per_trial"]):
        start = time.perf_counter_ns()
        step()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return timings


def scaling() -> dict:
    native_name = f"native_{build_info()['backend']}" if is_available() else None
    backends = ["numpy"] + ([native_name] if native_name is not None else [])
    samples = {(name, length): [] for name in backends
               for length in SCALING_CONFIG["sequence_lengths"]}
    # Rotate backend order per trial to avoid always giving one backend the
    # first measurements. Each trial rebuilds the model from the same seed.
    for trial in range(SCALING_CONFIG["trials"]):
        order = backends if trial % 2 == 0 else list(reversed(backends))
        for length in SCALING_CONFIG["sequence_lengths"]:
            for backend in order:
                timings = scaling_trial(length, backend != "numpy")
                samples[backend, length].append(timings)
                print(f"trial={trial + 1} backend={backend} sequence={length} mean={statistics.mean(timings):.3f} ms", flush=True)
    rows = []
    for backend in backends:
        for length in SCALING_CONFIG["sequence_lengths"]:
            trial_samples = samples[backend, length]
            means = [statistics.mean(values) for values in trial_samples]
            median = statistics.median(means)
            rows.append({
                "backend": backend, "sequence_length": length,
                "batch_size": SCALING_CONFIG["batch_size"],
                "trial_ms": means, "median_ms": median,
                "q25_ms": float(np.quantile(means, 0.25)),
                "q75_ms": float(np.quantile(means, 0.75)),
                "min_trial_ms": min(means), "max_trial_ms": max(means),
                "tokens_per_second": SCALING_CONFIG["batch_size"] * length / (median / 1000),
                "trial_step_ms": trial_samples,
            })
    return {
        "config": SCALING_CONFIG,
        "rows": rows,
        "methodology": {
            "timed_scope": "zero_grad, forward, cross-entropy, backward, Adam update",
            "summary": "median of five trial mean step times; q25/q75 over those five means",
            "excluded": "model initialization, input generation, explicit pre-trial gc.collect, warmup",
            "gc": "Python garbage collection remains enabled during timed steps",
            "backend_scope": "native replaces Linear projections only; attention and the remaining graph stay on NumPy/Python",
            "caveats": [
                "Synthetic random tokens and targets; timings assess execution cost, not useful language-model training.",
                "Shared virtualized Linux CPU host; timings are local observations, not hardware-independent performance claims.",
                "NumPy and the native extension may link different BLAS implementations; backend ratios include that difference.",
                "No CUDA or GPU results are collected.",
            ],
        },
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_manifest() -> dict[str, str]:
    paths = [ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py")),
             ROOT / "native" / "fluxion_native.cpp", ROOT / "native" / "build_native.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def environment() -> dict:
    model_name = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
    try:
        from threadpoolctl import threadpool_info
        pools = [{k: value for k, value in pool.items() if k != "filepath"}
                 for pool in threadpool_info()]
    except ImportError:
        pools = None
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": model_name, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": quota.read_text().strip() if quota.exists() else None,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": package_version("torch"),
        "pybind11_version": package_version("pybind11"),
        "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
        "thread_pools": pools,
        "native_available": is_available(),
        "native_build_info": build_info() if is_available() else None,
    }


def write_csv(output: Path, result: dict) -> None:
    with (output / "training.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["updates", "mean_squared_error"])
        writer.writerows(zip(result["training"]["steps"], result["training"]["loss"]))
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "x", "prediction", "target"])
        for split in ("train", "unseen"):
            data = result["training"][split]
            writer.writerows((split, x, p, y) for x, p, y in
                             zip(data["x"], data["prediction"], data["target"]))
    with (output / "scaling.csv").open("w", newline="") as handle:
        columns = ["backend", "sequence_length", "batch_size", "median_ms", "q25_ms", "q75_ms", "tokens_per_second"]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["scaling"]["rows"])
    with (output / "scaling_samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "sequence_length", "trial", "step", "milliseconds"])
        for row in result["scaling"]["rows"]:
            for trial, values in enumerate(row["trial_step_ms"], 1):
                writer.writerows((row["backend"], row["sequence_length"], trial, step, ms)
                                 for step, ms in enumerate(values, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks" / datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    before = source_manifest()
    result = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": BASE_SOURCE_SHA,
        "source_sha_note": "Pinned base commit; verify source_sha256 against your checkout when reproducing on another revision.",
        "source_sha256": before,
        "environment": environment(), "training": tiny_network(), "scaling": scaling(),
    }
    assert before == source_manifest(), "Source changed during measurement"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(args.output, result)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
