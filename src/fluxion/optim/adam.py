from __future__ import annotations

import numpy as np

from fluxion.tensor import Tensor


class Adam:
    """Adam optimizer."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1).")

        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.parameters = list(parameters)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self.m = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

        self.v = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

    def step(self) -> None:
        """Update all parameters using the Adam algorithm."""

        self.t += 1

        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue

            grad = parameter.grad

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1) * grad
            )

            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2) * (grad ** 2)
            )

            m_hat = self.m[i] / (
                1.0 - self.beta1 ** self.t
            )

            v_hat = self.v[i] / (
                1.0 - self.beta2 ** self.t
            )

            parameter.data -= (
                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )

    def zero_grad(self) -> None:
        """Reset gradients for all parameters."""

        for parameter in self.parameters:
            parameter.grad = None