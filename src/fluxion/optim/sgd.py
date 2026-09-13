from __future__ import annotations

from fluxion.tensor import Tensor


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
    ) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Update all parameters using their gradients."""

        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            parameter.data -= self.lr * parameter.grad

    def zero_grad(self) -> None:
        """Reset gradients for all parameters."""

        for parameter in self.parameters:
            parameter.grad = None
            