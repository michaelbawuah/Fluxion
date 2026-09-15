from __future__ import annotations

import numpy as np

from fluxion.nn.layers import (
    Embedding,
    LayerNorm,
    Linear,
    NativeLinear,
    ReLU,
)
from fluxion.nn.module import Module, Sequential
from fluxion.tensor import Tensor
from fluxion.transformer.attention import CausalSelfAttention


class PositionalEmbedding(Module):
    """Learned positional embeddings for Transformer inputs."""

    def __init__(
        self,
        max_sequence_length: int,
        embed_dim: int,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive."
            )

        if embed_dim <= 0:
            raise ValueError(
                "embed_dim must be positive."
            )

        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim

        self.embedding = Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embed_dim,
        )

    def forward(
        self,
        sequence_length: int,
    ) -> Tensor:
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "sequence_length exceeds max_sequence_length."
            )

        positions = np.arange(
            sequence_length,
            dtype=np.int64,
        )

        return self.embedding(
            positions
        )


class TransformerBlock(Module):
    """A GPT-style Transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        *,
        use_native_linear: bool = False,
    ) -> None:
        if embed_dim <= 0:
            raise ValueError(
                "embed_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        self.norm1 = LayerNorm(
            embed_dim
        )

        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_native_linear=use_native_linear,
        )

        self.norm2 = LayerNorm(
            embed_dim
        )

        linear_cls = (
            NativeLinear
            if use_native_linear
            else Linear
        )

        self.feed_forward = Sequential(
            linear_cls(
                embed_dim,
                hidden_dim,
            ),
            ReLU(),
            linear_cls(
                hidden_dim,
                embed_dim,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TransformerBlock expects input with shape "
                "(batch, sequence, embed_dim)."
            )

        # Pre-norm residual block: normalize before each sublayer, then add the
        # sublayer output back to the residual stream. This keeps the main
        # information path explicit and mirrors the GPT-style block structure.
        normalized = self.norm1(x)

        attended = self.attention(
            normalized
        )

        x = x + attended

        # Feed-forward sublayer
        normalized = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized
        )

        x = x + feed_forward_output

        return x