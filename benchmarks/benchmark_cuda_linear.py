from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from fluxion.cuda import build_info
from fluxion.ops import cuda_linear, native_linear
from fluxion.tensor import Tensor


def measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x=Tensor(rng.normal(size=(args.batch,args.in_features)))
    w=Tensor(rng.normal(size=(args.in_features,args.out_features)))
    b=Tensor(rng.normal(size=(args.out_features,)))

    cuda_times=[]
    native_times=[]
    for _ in range(args.trials):
        native_times.append(measure(lambda: native_linear(x,w,b),args.warmup,args.iterations))
        cuda_times.append(measure(lambda: cuda_linear(x,w,b),args.warmup,args.iterations))

    print("CUDA build:", build_info())
    print(f"shape: ({args.batch}, {args.in_features}) @ ({args.in_features}, {args.out_features})")
    print(f"native CPU median: {statistics.median(native_times):.3f} ms")
    print(f"CUDA end-to-end median: {statistics.median(cuda_times):.3f} ms")
    print(f"CPU/CUDA speedup: {statistics.median(native_times)/statistics.median(cuda_times):.3f}x")
    print("NOTE: CUDA timing includes host↔device copies and allocation on every call.")


if __name__ == "__main__":
    main()
