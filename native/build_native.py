"""Build Fluxion's native CPU extension on macOS or Linux."""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig


def run(command: list[str]) -> None:
    print("+", shlex.join(command))
    subprocess.run(command, check=True)


def pybind11_includes() -> list[str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--includes"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "pybind11 is required. Install it with: python -m pip install pybind11"
        ) from exc
    return shlex.split(output)


def linux_blas_flags() -> tuple[list[str], str]:
    # Prefer OpenBLAS when installed, but a system CBLAS/BLAS implementation is valid.
    if ctypes.util.find_library("openblas"):
        return ["-lopenblas"], "openblas"
    if ctypes.util.find_library("blas"):
        return ["-lblas"], "blas"
    raise SystemExit(
        "No Linux BLAS library found. Install OpenBLAS development packages "
        "(for Ubuntu: sudo apt-get install libopenblas-dev)."
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "native" / "fluxion_native.cpp"
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = root / f"fluxion_native{suffix}"

    system = platform.system()
    compiler = os.environ.get("CXX", "clang++" if system == "Darwin" else "c++")

    command = [
        compiler,
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        *pybind11_includes(),
        str(source),
    ]

    if system == "Darwin":
        command.extend(["-undefined", "dynamic_lookup", "-framework", "Accelerate"])
        backend = "accelerate"
    elif system == "Linux":
        blas_flags, backend = linux_blas_flags()
        command.extend(["-fPIC", *blas_flags])
    else:
        raise SystemExit(f"Unsupported platform for native build: {system}")

    command.extend(["-o", str(output)])
    run(command)

    print(f"Built: {output}")
    print(f"Platform: {system}")
    print(f"CPU backend: {backend}")

    # Verify that this interpreter can import the extension we just built.
    check = (
        "import fluxion_native; "
        "print('Loaded backend:', fluxion_native.backend_name()); "
        "print('Build info:', fluxion_native.build_info())"
    )
    run([sys.executable, "-c", check])


if __name__ == "__main__":
    main()
