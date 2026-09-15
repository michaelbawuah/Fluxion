from __future__ import annotations

import numpy as np

from fluxion.tensor import Tensor


def _sum_to_shape(
    gradient: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    """
    Reduce a broadcasted gradient back to the shape
    of the original tensor.
    """

    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(
                axis=axis,
                keepdims=True,
            )

    return gradient


def add(a: Tensor, b: Tensor) -> Tensor:
    """Add two tensors with NumPy-style broadcasting."""

    data = a.data + b.data

    out = Tensor(
        data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    if out.requires_grad:
        out._prev = (a, b)
        out._op = "add"

        def _backward() -> None:
            if out.grad is None:
                return

            if a.requires_grad:
                a._accumulate_grad(
                    _sum_to_shape(
                        out.grad,
                        a.shape,
                    )
                )

            if b.requires_grad:
                b._accumulate_grad(
                    _sum_to_shape(
                        out.grad,
                        b.shape,
                    )
                )

        out._backward = _backward

    return out


def multiply(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise multiplication."""

    out = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
    )

    out._prev = (a, b)
    out._op = "multiply"

    def _backward() -> None:
        if out.grad is None:
            return

        if a.requires_grad:
            grad_a = b.data * out.grad

            a._accumulate_grad(
                _sum_to_shape(
                    grad_a,
                    a.shape,
                )
            )

        if b.requires_grad:
            grad_b = a.data * out.grad

            b._accumulate_grad(
                _sum_to_shape(
                    grad_b,
                    b.shape,
                )
            )

    out._backward = _backward

    return out


def negate(a: Tensor) -> Tensor:
    """Elementwise negation."""

    out = Tensor(
        -a.data,
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "negate"

    def _backward() -> None:
        if out.grad is None:
            return

        if a.requires_grad:
            a._accumulate_grad(-out.grad)

    out._backward = _backward

    return out


def subtract(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise subtraction."""

    return add(
        a,
        negate(b),
    )


def divide(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise division."""

    return multiply(
        a,
        power(b, -1),
    )


def power(a: Tensor, exponent: float) -> Tensor:
    """Raise every element of a tensor to a scalar power."""

    out = Tensor(
        a.data ** exponent,
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = f"power({exponent})"

    def _backward() -> None:
        if out.grad is None:
            return

        if a.requires_grad:
            grad_a = (
                exponent
                * (a.data ** (exponent - 1))
                * out.grad
            )

            a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def sum_tensor(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum tensor elements along the requested axis."""

    out = Tensor(
        np.array(
            a.data.sum(
                axis=axis,
                keepdims=keepdims,
            )
        ),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "sum"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        grad = out.grad

        if axis is not None:
            axes = (
                (axis,)
                if isinstance(axis, int)
                else axis
            )

            axes = tuple(
                ax % a.ndim
                for ax in axes
            )

            if not keepdims:
                for ax in sorted(axes):
                    grad = np.expand_dims(
                        grad,
                        axis=ax,
                    )

        grad_a = np.broadcast_to(
            grad,
            a.shape,
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def mean_tensor(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the mean along the requested axis."""

    out = Tensor(
        np.array(
            a.data.mean(
                axis=axis,
                keepdims=keepdims,
            )
        ),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "mean"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        grad = out.grad

        if axis is None:
            divisor = a.size

        else:
            axes = (
                (axis,)
                if isinstance(axis, int)
                else axis
            )

            axes = tuple(
                ax % a.ndim
                for ax in axes
            )

            divisor = int(
                np.prod(
                    [
                        a.shape[ax]
                        for ax in axes
                    ]
                )
            )

            if not keepdims:
                for ax in sorted(axes):
                    grad = np.expand_dims(
                        grad,
                        axis=ax,
                    )

        grad_a = (
            np.broadcast_to(
                grad,
                a.shape,
            )
            / divisor
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication with support for batched tensors.

    The final two dimensions are treated as matrix dimensions.
    Earlier dimensions are treated as batch dimensions and
    may broadcast according to NumPy matmul rules.
    """

    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(
            "matmul requires tensors with at least 2 dimensions."
        )

    out = Tensor(
        np.matmul(
            a.data,
            b.data,
        ),
        requires_grad=a.requires_grad or b.requires_grad,
    )

    out._prev = (a, b)
    out._op = "matmul"

    def _backward() -> None:
        if out.grad is None:
            return

        if a.requires_grad:
            grad_a = np.matmul(
                out.grad,
                np.swapaxes(
                    b.data,
                    -1,
                    -2,
                ),
            )

            a._accumulate_grad(
                _sum_to_shape(
                    grad_a,
                    a.shape,
                )
            )

        if b.requires_grad:
            grad_b = np.matmul(
                np.swapaxes(
                    a.data,
                    -1,
                    -2,
                ),
                out.grad,
            )

            b._accumulate_grad(
                _sum_to_shape(
                    grad_b,
                    b.shape,
                )
            )

    out._backward = _backward

    return out


def exp(a: Tensor) -> Tensor:
    """Elementwise exponential."""

    out = Tensor(
        np.exp(a.data),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "exp"

    def _backward() -> None:
        if out.grad is None:
            return

        if a.requires_grad:
            a._accumulate_grad(
                out.data * out.grad
            )

    out._backward = _backward

    return out


def log(a: Tensor) -> Tensor:
    """Elementwise natural logarithm."""

    out = Tensor(
        np.log(a.data),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "log"

    def _backward() -> None:
        if out.grad is None:
            return

        if a.requires_grad:
            a._accumulate_grad(
                out.grad / a.data
            )

    out._backward = _backward

    return out


def max_tensor(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the maximum along the requested axis."""

    out = Tensor(
        np.array(
            a.data.max(
                axis=axis,
                keepdims=keepdims,
            )
        ),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "max"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        if axis is None:
            max_value = a.data.max()

            mask = a.data == max_value
            number_of_maxima = mask.sum()

            grad_a = (
                mask
                * out.grad
                / number_of_maxima
            )

            a._accumulate_grad(grad_a)
            return

        axes = (
            (axis,)
            if isinstance(axis, int)
            else axis
        )

        axes = tuple(
            ax % a.ndim
            for ax in axes
        )

        max_values = a.data.max(
            axis=axes,
            keepdims=True,
        )

        mask = a.data == max_values

        number_of_maxima = mask.sum(
            axis=axes,
            keepdims=True,
        )

        grad = out.grad

        if not keepdims:
            for ax in sorted(axes):
                grad = np.expand_dims(
                    grad,
                    axis=ax,
                )

        grad_a = (
            mask
            * grad
            / number_of_maxima
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def getitem(a: Tensor, index) -> Tensor:
    """Index into a tensor while preserving autograd."""

    out = Tensor(
        a.data[index],
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "getitem"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        grad_a = np.zeros_like(a.data)

        np.add.at(
            grad_a,
            index,
            out.grad,
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def reshape(
    a: Tensor,
    shape: tuple[int, ...],
) -> Tensor:
    """Reshape a tensor while preserving autograd."""

    out = Tensor(
        a.data.reshape(shape),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "reshape"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        a._accumulate_grad(
            out.grad.reshape(a.shape)
        )

    out._backward = _backward

    return out


def sqrt(a: Tensor) -> Tensor:
    """Elementwise square root."""

    out = Tensor(
        np.sqrt(a.data),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "sqrt"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        grad_a = (
            out.grad
            / (2.0 * out.data)
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def transpose(
    a: Tensor,
    dim0: int,
    dim1: int,
) -> Tensor:
    """Swap two tensor dimensions while preserving autograd."""

    out = Tensor(
        np.swapaxes(
            a.data,
            dim0,
            dim1,
        ),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "transpose"

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        grad_a = np.swapaxes(
            out.grad,
            dim0,
            dim1,
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def permute(
    a: Tensor,
    dims: tuple[int, ...],
) -> Tensor:
    """Reorder tensor dimensions while preserving autograd."""

    if len(dims) != a.ndim:
        raise ValueError(
            "permute must specify every tensor dimension."
        )

    normalized_dims = tuple(
        dim % a.ndim
        for dim in dims
    )

    if len(set(normalized_dims)) != a.ndim:
        raise ValueError(
            "permute dimensions must be unique."
        )

    out = Tensor(
        np.transpose(
            a.data,
            axes=normalized_dims,
        ),
        requires_grad=a.requires_grad,
    )

    out._prev = (a,)
    out._op = "permute"

    inverse_dims = tuple(
        np.argsort(normalized_dims)
    )

    def _backward() -> None:
        if out.grad is None:
            return

        if not a.requires_grad:
            return

        grad_a = np.transpose(
            out.grad,
            axes=inverse_dims,
        )

        a._accumulate_grad(grad_a)

    out._backward = _backward

    return out


def native_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Tensor:
    """
    Fused Linear operation backed by the native C++ extension.

    Computes:

        output = x @ weight + bias

    Backward computes:

        dX = grad_output @ weight.T
        dW = x.T @ grad_output
        db = sum(grad_output, axis=0)
    """

    try:
        import fluxion_native
    except ImportError as exc:
        raise RuntimeError(
            "Fluxion native extension is not available. "
            "Build it with: python native/build_native.py"
        ) from exc

    if x.ndim != 2:
        raise ValueError(
            "native_linear expects x to be 2D."
        )

    if weight.ndim != 2:
        raise ValueError(
            "native_linear expects weight to be 2D."
        )

    if bias.ndim != 1:
        raise ValueError(
            "native_linear expects bias to be 1D."
        )

    if x.shape[1] != weight.shape[0]:
        raise ValueError(
            "x and weight dimensions are incompatible."
        )

    if bias.shape[0] != weight.shape[1]:
        raise ValueError(
            "bias size must match output dimension."
        )

    output_data = fluxion_native.linear_forward(
        x.data,
        weight.data,
        bias.data,
    )

    out = Tensor(
        output_data,
        requires_grad=(
            x.requires_grad
            or weight.requires_grad
            or bias.requires_grad
        ),
    )

    out._prev = (
        x,
        weight,
        bias,
    )
    out._op = "native_linear"

    def _backward() -> None:
        if out.grad is None:
            return

        grad_x, grad_weight, grad_bias = (
            fluxion_native.linear_backward(
                out.grad,
                x.data,
                weight.data,
            )
        )

        if x.requires_grad:
            x._accumulate_grad(grad_x)

        if weight.requires_grad:
            weight._accumulate_grad(
                grad_weight
            )

        if bias.requires_grad:
            bias._accumulate_grad(
                grad_bias
            )

    out._backward = _backward

    return out

def cuda_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Fused Linear operation backed by Fluxion's optional CUDA extension."""
    try:
        import fluxion_cuda
    except ImportError as exc:
        raise RuntimeError(
            "Fluxion CUDA extension is not available. "
            "Build it with: python native/cuda/build_cuda.py"
        ) from exc

    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("cuda_linear expects 2D x, 2D weight, and 1D bias.")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("x and weight dimensions are incompatible.")
    if bias.shape[0] != weight.shape[1]:
        raise ValueError("bias size must match output dimension.")

    output_data = fluxion_cuda.linear_forward(x.data, weight.data, bias.data)
    out = Tensor(
        output_data,
        requires_grad=x.requires_grad or weight.requires_grad or bias.requires_grad,
    )
    out._prev = (x, weight, bias)
    out._op = "cuda_linear"

    def _backward() -> None:
        if out.grad is None:
            return
        grad_x, grad_weight, grad_bias = fluxion_cuda.linear_backward(
            out.grad, x.data, weight.data
        )
        if x.requires_grad:
            x._accumulate_grad(grad_x)
        if weight.requires_grad:
            weight._accumulate_grad(grad_weight)
        if bias.requires_grad:
            bias._accumulate_grad(grad_bias)

    out._backward = _backward
    return out
