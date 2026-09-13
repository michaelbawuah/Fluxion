from __future__ import annotations

from typing import Any

import numpy as np

from fluxion.autograd import topological_sort


class Tensor:
    """Core multidimensional array object used by Fluxion."""

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        dtype: np.dtype | type | None = None,
    ) -> None:
        if isinstance(data, Tensor):
            array = data.data.copy()
        else:
            array = np.asarray(data, dtype=dtype)

        if array.dtype.kind in {"i", "u"}:
            array = array.astype(np.float64)

        self.data = array
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None

        self._prev: tuple["Tensor", ...] = ()
        self._op: str = ""
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array."""
        return self.data

    def backward(self, gradient: Any | None = None) -> None:
        """
        Compute gradients for every tensor that contributed to this tensor.

        For scalar outputs, the initial gradient defaults to 1.
        For non-scalar outputs, an explicit gradient must be supplied.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on a tensor that does not require gradients."
            )

        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "Gradient must be provided for non-scalar tensors."
                )

            initial_grad = np.ones_like(self.data)
        else:
            initial_grad = np.asarray(gradient, dtype=self.data.dtype)

            if initial_grad.shape != self.data.shape:
                raise ValueError(
                    f"Gradient shape {initial_grad.shape} does not match "
                    f"tensor shape {self.data.shape}."
                )

        self.grad = initial_grad

        graph = topological_sort(self)

        for tensor in reversed(graph):
            tensor._backward()

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: Any) -> "Tensor":
        from fluxion.ops import add

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return add(self, other)

    def __mul__(self, other: Any) -> "Tensor":
        from fluxion.ops import multiply

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return multiply(self, other)

    def __neg__(self) -> "Tensor":
        from fluxion.ops import negate

        return negate(self)

    def __sub__(self, other: Any) -> "Tensor":
        from fluxion.ops import subtract

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return subtract(self, other)

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __rsub__(self, other: Any) -> "Tensor":
        if not isinstance(other, Tensor):
           other = Tensor(other)

        return other - self

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __rtruediv__(self, other: Any) -> "Tensor":
        if not isinstance(other, Tensor):
           other = Tensor(other)

        return other / self

    def __truediv__(self, other: Any) -> "Tensor":
        from fluxion.ops import divide

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return divide(self, other)

    def __pow__(self, exponent: float | int) -> "Tensor":
        from fluxion.ops import power

        return power(self, exponent)

    def __matmul__(self, other: Any) -> "Tensor":
        from fluxion.ops import matmul

        if not isinstance(other, Tensor):
            other = Tensor(other)

        return matmul(self, other)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        from fluxion.ops import sum_tensor

        return sum_tensor(
        self,
        axis=axis,
        keepdims=keepdims,
    )

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        from fluxion.ops import mean_tensor

        return mean_tensor(
        self,
        axis=axis,
        keepdims=keepdims,
    )

    def max(
    self,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ) -> "Tensor":
     from fluxion.ops import max_tensor

     return max_tensor(
        self,
        axis=axis,
        keepdims=keepdims,
    )

    def exp(self) -> "Tensor":
        from fluxion.ops import exp

        return exp(self)

    def log(self) -> "Tensor":
       from fluxion.ops import log
       return log(self)

    def sqrt(self) -> "Tensor":
        from fluxion.ops import sqrt
        return sqrt(self)

    def __repr__(self) -> str:
        grad_suffix = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{grad_suffix})"

    def __getitem__(self, index) -> "Tensor":
        from fluxion.ops import getitem

        return getitem(self, index)

    def reshape(self, *shape: int) -> "Tensor":
        from fluxion.ops import reshape

        return reshape(self, shape)

    def transpose(
    self,
    dim0: int,
    dim1: int,
) -> "Tensor":
     from fluxion.ops import transpose

     return transpose(
         self,
         dim0,
         dim1,
    )

    def permute(
    self,
    *dims: int,
) -> "Tensor":
     from fluxion.ops import permute

     return permute(
        self,
        dims,
    )