"""Runtime helpers for Fluxion's optional CUDA extension."""

from __future__ import annotations

import importlib


def _extension():
    try:
        return importlib.import_module("fluxion_cuda")
    except ImportError as exc:
        raise RuntimeError(
            "Fluxion CUDA extension is not available. "
            "Build it on an NVIDIA CUDA system with: python native/cuda/build_cuda.py"
        ) from exc


def is_available() -> bool:
    try:
        module = importlib.import_module("fluxion_cuda")
    except ImportError:
        return False
    try:
        return bool(module.is_available())
    except RuntimeError:
        return False


def device_name() -> str:
    return str(_extension().device_name())


def build_info() -> dict:
    return dict(_extension().build_info())
