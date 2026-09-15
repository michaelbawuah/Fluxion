from __future__ import annotations

import math

import numpy as np

from fluxion.nn.layers import Linear, NativeLinear, Softmax
from fluxion.nn.module import Module
from fluxion.tensor import Tensor


class ScaledDotProductAttention(Module):
    """Scaled dot-product attention."""

    def __init__(
        self,
        *,
        causal: bool = False,
    ) -> None:
        self.causal = causal
        self.softmax = Softmax(axis=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Compute:

        softmax(Q K^T / sqrt(d_k)) V
        """

        if query.ndim < 2:
            raise ValueError(
                "query must have at least 2 dimensions."
            )

        if key.ndim < 2:
            raise ValueError(
                "key must have at least 2 dimensions."
            )

        if value.ndim < 2:
            raise ValueError(
                "value must have at least 2 dimensions."
            )

        if query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "query and key must have the same feature dimension."
            )

        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "key and value must have the same sequence length."
            )

        d_k = query.shape[-1]

        scores = (
            query
            @ key.transpose(-1, -2)
        )

        scores = scores / math.sqrt(d_k)

        if self.causal:
            query_length = query.shape[-2]
            key_length = key.shape[-2]

            if query_length != key_length:
                raise ValueError(
                    "Causal attention currently requires equal "
                    "query and key sequence lengths."
                )

            causal_mask = np.triu(
                np.full(
                    (
                        query_length,
                        key_length,
                    ),
                    -np.inf,
                ),
                k=1,
            )

            scores = scores + Tensor(
                causal_mask
            )

        attention_weights = self.softmax(
            scores
        )

        return attention_weights @ value


class MultiHeadAttention(Module):
    """Multi-head attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        causal: bool = False,
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

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (
            embed_dim // num_heads
        )

        linear_cls = (
            NativeLinear
            if use_native_linear
            else Linear
        )

        self.query_projection = linear_cls(
            embed_dim,
            embed_dim,
        )

        self.key_projection = linear_cls(
            embed_dim,
            embed_dim,
        )

        self.value_projection = linear_cls(
            embed_dim,
            embed_dim,
        )

        self.output_projection = linear_cls(
            embed_dim,
            embed_dim,
        )

        self.attention = ScaledDotProductAttention(
            causal=causal,
        )

    def _split_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.permute(
            0,
            2,
            1,
            3,
        )

    def _combine_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        batch_size, _, sequence_length, _ = x.shape

        x = x.permute(
            0,
            2,
            1,
            3,
        )

        return x.reshape(
            batch_size,
            sequence_length,
            self.embed_dim,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        if query.ndim != 3:
            raise ValueError(
                "query must have shape "
                "(batch, sequence, embed_dim)."
            )

        if key.ndim != 3:
            raise ValueError(
                "key must have shape "
                "(batch, sequence, embed_dim)."
            )

        if value.ndim != 3:
            raise ValueError(
                "value must have shape "
                "(batch, sequence, embed_dim)."
            )

        if query.shape[-1] != self.embed_dim:
            raise ValueError(
                "query feature dimension does not match embed_dim."
            )

        if key.shape[-1] != self.embed_dim:
            raise ValueError(
                "key feature dimension does not match embed_dim."
            )

        if value.shape[-1] != self.embed_dim:
            raise ValueError(
                "value feature dimension does not match embed_dim."
            )

        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attended = self.attention(
            q,
            k,
            v,
        )

        attended = self._combine_heads(
            attended
        )

        return self.output_projection(
            attended
        )


class SelfAttention(Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        use_native_linear: bool = False,
    ) -> None:
        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            causal=False,
            use_native_linear=use_native_linear,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.attention(
            x,
            x,
            x,
        )


class CausalSelfAttention(Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        use_native_linear: bool = False,
    ) -> None:
        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            causal=True,
            use_native_linear=use_native_linear,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.attention(
            x,
            x,
            x,
        )