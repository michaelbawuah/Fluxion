from __future__ import annotations

from typing import Any

from fluxion.tensor import Tensor


class Module:
    """Base class for all neural-network modules in Fluxion."""

    def parameters(self) -> list[Tensor]:
        """Return all trainable parameters owned by this module."""

        params: list[Tensor] = []

        for value in self.__dict__.values():
            params.extend(self._collect_parameters(value))

        return params

    def _collect_parameters(self, value: Any) -> list[Tensor]:
        """Recursively find trainable tensors inside nested objects."""

        if isinstance(value, Tensor):
            if value.requires_grad:
                return [value]

            return []

        if isinstance(value, Module):
            return value.parameters()

        if isinstance(value, (list, tuple)):
            params: list[Tensor] = []

            for item in value:
                params.extend(self._collect_parameters(item))

            return params

        if isinstance(value, dict):
            params: list[Tensor] = []

            for item in value.values():
                params.extend(self._collect_parameters(item))

            return params

        return []

    def zero_grad(self) -> None:
        """Reset gradients of all trainable parameters."""

        for parameter in self.parameters():
            parameter.grad = None

    def forward(self, *args, **kwargs):
        """Compute the forward pass."""

        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Allow modules to be called like functions."""

        return self.forward(*args, **kwargs)


class Sequential(Module):
    """Apply a sequence of modules one after another."""

    def __init__(self, *modules: Module) -> None:
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        """Pass the input through every module in order."""

        for module in self.modules:
            x = module(x)

        return x