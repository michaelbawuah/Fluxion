from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxion.tensor import Tensor


def topological_sort(root: "Tensor") -> list["Tensor"]:
    """
    Return tensors in topological order from graph leaves to the root.

    This ordering allows backward propagation to process the
    computation graph correctly.
    """
    visited: set[int] = set()
    order: list["Tensor"] = []

    def visit(tensor: "Tensor") -> None:
        tensor_id = id(tensor)

        if tensor_id in visited:
            return

        visited.add(tensor_id)

        for parent in tensor._prev:
            visit(parent)

        order.append(tensor)

    visit(root)

    return order

def test_fluxion_matches_pytorch_for_extended_operations():
    # Fluxion
    x = Tensor([4.0], requires_grad=True)
    w = Tensor([2.0], requires_grad=True)

    y = ((x ** 2) - (x * 2.0)) / w
    y.backward()

    # PyTorch
    torch_x = torch.tensor([4.0], requires_grad=True)
    torch_w = torch.tensor([2.0], requires_grad=True)

    torch_y = ((torch_x ** 2) - (torch_x * 2.0)) / torch_w
    torch_y.backward()

    np.testing.assert_allclose(y.data, torch_y.detach().numpy())
    np.testing.assert_allclose(x.grad, torch_x.grad.numpy())
    np.testing.assert_allclose(w.grad, torch_w.grad.numpy())