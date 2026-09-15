"""Runtime helpers for Fluxion's optional native CPU backend."""

from __future__ import annotations

from typing import Any


def _extension() -> Any:
    try:
        import fluxion_native
    except ImportError as exc:
        raise RuntimeError(
            "Fluxion native extension is not available. "
            "Build it with: python native/build_native.py"
        ) from exc
    return fluxion_native


def is_available() -> bool:
    """Return whether the native extension can be imported."""
    try:
        import fluxion_native  # noqa: F401
    except ImportError:
        return False
    return True


def backend_name() -> str:
    """Return the CPU backend compiled into the native extension."""
    extension = _extension()
    if hasattr(extension, "backend_name"):
        return str(extension.backend_name())
    return "legacy"


def build_info() -> dict[str, str]:
    """Return portable metadata describing the compiled native backend."""
    extension = _extension()
    if hasattr(extension, "build_info"):
        return dict(extension.build_info())
    return {"backend": "legacy", "platform": "unknown", "dtype": "float64"}
