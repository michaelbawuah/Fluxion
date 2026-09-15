from __future__ import annotations

import numpy as np

from fluxion.nn.layers import (
    Embedding,
    LayerNorm,
    Linear,
    NativeLinear,
)
from fluxion.nn.module import Module
from fluxion.tensor import Tensor
from fluxion.transformer.layers import (
    PositionalEmbedding,
    TransformerBlock,
)


class GPT(Module):
    """A small GPT-style causal language model."""

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        *,
        use_native_linear: bool = False,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embed_dim=embed_dim,
        )

        self.blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                use_native_linear=use_native_linear,
            )
            for _ in range(num_layers)
        ]

        self.final_norm = LayerNorm(
            embed_dim
        )

        linear_cls = (
            NativeLinear
            if use_native_linear
            else Linear
        )

        self.output_projection = linear_cls(
            embed_dim,
            vocab_size,
        )

    def forward(
        self,
        token_ids: np.ndarray,
    ) -> Tensor:
        token_ids = np.asarray(
            token_ids
        )

        if token_ids.ndim != 2:
            raise ValueError(
                "GPT expects token_ids with shape "
                "(batch, sequence)."
            )

        if not np.issubdtype(
            token_ids.dtype,
            np.integer,
        ):
            raise ValueError(
                "token_ids must contain integers."
            )

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "sequence length exceeds max_sequence_length."
            )

        if np.any(token_ids < 0) or np.any(
            token_ids >= self.vocab_size
        ):
            raise ValueError(
                "token ID is out of vocabulary range."
            )

        token_embeddings = self.token_embedding(
            token_ids
        )

        position_embeddings = self.position_embedding(
            sequence_length
        )

        # Token embeddings encode *what* each token is; positional embeddings
        # encode *where* it occurs. Their sum is the Transformer input stream.
        x = (
            token_embeddings
            + position_embeddings
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Produce one unnormalized vocabulary score (logit) per token position.
        # Cross-entropy turns these logits into the next-token training signal.
        logits = self.output_projection(
            x
        )

        return logits