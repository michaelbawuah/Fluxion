"""Build Fluxion's optional CUDA extension on an NVIDIA CUDA system."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig


def run(command: list[str]) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, check=True)


def pybind11_includes() -> list[str]:
    output = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"], text=True
    ).strip()
    return shlex.split(output)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "native" / "cuda" / "fluxion_cuda.cu"
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = root / f"fluxion_cuda{suffix}"
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        raise SystemExit("nvcc was not found. Install the NVIDIA CUDA Toolkit and ensure nvcc is on PATH.")
    command = [
        nvcc, "-O3", "--shared", "-std=c++17", "-Xcompiler", "-fPIC",
        *pybind11_includes(), str(source), "-o", str(output),
    ]
    run(command)
    print(f"Built: {output}")
    check = (
        "import fluxion_cuda; "
        "print('CUDA available:', fluxion_cuda.is_available()); "
        "print('Device:', fluxion_cuda.device_name()); "
        "print('Build info:', fluxion_cuda.build_info())"
    )
    run([sys.executable, "-c", check])


if __name__ == "__main__":
    main()
