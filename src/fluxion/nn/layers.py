from __future__ import annotations

import numpy as np

from fluxion.nn.module import Module
from fluxion.tensor import Tensor


class Linear(Module):
    """A fully connected neural-network layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be positive.")

        if out_features <= 0:
            raise ValueError("out_features must be positive.")

        scale = 1.0 / np.sqrt(in_features)

        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True,
        )

        self.bias = (
            Tensor(
                np.zeros(out_features),
                requires_grad=True,
            )
            if bias
            else None
        )
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                "Linear expects an input with at least 2 dimensions."
            )

        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                "Input feature dimension does not match Linear."
            )

        output = x @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return output


class ReLU(Module):
    """Rectified Linear Unit activation."""

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(
            np.maximum(x.data, 0.0),
            requires_grad=x.requires_grad,
        )

        out._prev = (x,)
        out._op = "relu"

        def _backward() -> None:
            if out.grad is None:
                return

            if x.requires_grad:
                grad_x = (
                    (x.data > 0.0)
                    * out.grad
                )

                x._accumulate_grad(grad_x)

        out._backward = _backward

        return out

class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (-x).exp())

class Softmax(Module):
    """Numerically stable softmax activation."""

    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        shifted = x - x.max(
            axis=self.axis,
            keepdims=True,
        )

        exponentials = shifted.exp()

        denominator = exponentials.sum(
            axis=self.axis,
            keepdims=True,
        )

        return exponentials / denominator

class LayerNorm(Module):
    """Layer normalization over the final tensor dimension."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ) -> None:
        if normalized_shape <= 0:
            raise ValueError(
                "normalized_shape must be positive."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be positive."
            )

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = Tensor(
            np.ones(normalized_shape),
            requires_grad=True,
        )

        self.bias = Tensor(
            np.zeros(normalized_shape),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                "The final input dimension must match "
                "normalized_shape."
            )

        mean = x.mean(
            axis=-1,
            keepdims=True,
        )

        centered = x - mean

        variance = (
            centered ** 2
        ).mean(
            axis=-1,
            keepdims=True,
        )

        normalized = centered / (
            variance + self.eps
        ).sqrt()

        return (
            normalized * self.weight
            + self.bias
        )

class Embedding(Module):
    """Learnable lookup table for token embeddings."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        if num_embeddings <= 0:
            raise ValueError(
                "num_embeddings must be positive."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        scale = 1.0 / np.sqrt(embedding_dim)

        self.weight = Tensor(
            np.random.randn(
                num_embeddings,
                embedding_dim,
            ) * scale,
            requires_grad=True,
        )

    def forward(
        self,
        indices: np.ndarray,
    ) -> Tensor:
        indices = np.asarray(indices)

        if not np.issubdtype(
            indices.dtype,
            np.integer,
        ):
            raise ValueError(
                "Embedding indices must be integers."
            )

        if np.any(indices < 0) or np.any(
            indices >= self.num_embeddings
        ):
            raise ValueError(
                "Embedding index is out of range."
            )

        return self.weight[
            indices
        ]

class NativeLinear(Module):
    """
    Linear layer backed by Fluxion's fused native C++ operator.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be positive."
            )

        scale = 1.0 / np.sqrt(input_dim)

        self.weight = Tensor(
            np.random.randn(
                input_dim,
                output_dim,
            ) * scale,
            requires_grad=True,
        )

        self.bias = Tensor(
            np.zeros(
                output_dim,
            ),
            requires_grad=True,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        from fluxion.ops import native_linear

        if x.ndim < 2:
            raise ValueError(
                "NativeLinear expects input with at least 2 dimensions."
            )

        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                "Input feature dimension does not match NativeLinear."
            )

        # The C++ kernel accepts a 2D matrix.
        if x.ndim == 2:
            return native_linear(
                x,
                self.weight,
                self.bias,
            )

        # Transformer inputs have shape:
        #
        #     (batch, sequence, features)
        #
        # Flatten every leading dimension into one matrix dimension,
        # call the 2D native kernel, then restore the original shape.
        original_shape = x.shape

        flattened = x.reshape(
            -1,
            original_shape[-1],
        )

        output = native_linear(
            flattened,
            self.weight,
            self.bias,
        )

        return output.reshape(
            *original_shape[:-1],
            self.weight.shape[1],
        )