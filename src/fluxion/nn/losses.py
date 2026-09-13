from __future__ import annotations

import numpy as np

from fluxion.nn.module import Module
from fluxion.tensor import Tensor


class MSELoss(Module):
    """Mean Squared Error loss."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        error = prediction - target
        squared_error = error ** 2
        return squared_error.mean()


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for classification and language modeling.

    Supported shapes:

        Classification:
            logits:  (batch_size, num_classes)
            targets: (batch_size,)

        Language modeling:
            logits:  (batch_size, sequence_length, num_classes)
            targets: (batch_size, sequence_length)
    """

    def forward(
        self,
        logits: Tensor,
        targets: np.ndarray,
    ) -> Tensor:
        targets = np.asarray(targets)

        if logits.ndim not in (2, 3):
            raise ValueError(
                "CrossEntropyLoss expects logits with shape "
                "(batch, classes) or "
                "(batch, sequence, classes)."
            )

        if logits.ndim == 2:
            batch_size, num_classes = logits.shape

            if targets.shape != (batch_size,):
                raise ValueError(
                    "For 2D logits, targets must have shape "
                    "(batch_size,)."
                )

            flat_logits = logits
            flat_targets = targets

        else:
            batch_size, sequence_length, num_classes = logits.shape

            if targets.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "For 3D logits, targets must have shape "
                    "(batch_size, sequence_length)."
                )

            flat_logits = logits.reshape(
                batch_size * sequence_length,
                num_classes,
            )

            flat_targets = targets.reshape(
                batch_size * sequence_length
            )

        if not np.issubdtype(
            flat_targets.dtype,
            np.integer,
        ):
            raise ValueError(
                "targets must contain integer class indices."
            )

        if np.any(flat_targets < 0) or np.any(
            flat_targets >= num_classes
        ):
            raise ValueError(
                "target class index is out of range."
            )

        max_logits = flat_logits.max(
            axis=1,
            keepdims=True,
        )

        shifted_logits = (
            flat_logits - max_logits
        )

        log_sum_exp = (
            shifted_logits.exp()
            .sum(
                axis=1,
                keepdims=True,
            )
            .log()
            + max_logits
        )

        prediction_count = flat_logits.shape[0]

        prediction_indices = np.arange(
            prediction_count
        )

        correct_class_logits = flat_logits[
            prediction_indices,
            flat_targets,
        ]

        correct_class_logits = correct_class_logits.reshape(
            prediction_count,
            1,
        )

        losses = (
            log_sum_exp
            - correct_class_logits
        )

        return losses.mean()