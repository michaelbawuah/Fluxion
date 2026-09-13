import numpy as np
import torch

from fluxion.optim.adam import Adam
from fluxion.optim.sgd import SGD
from fluxion.tensor import Tensor


def test_sgd_step_updates_parameter():
    parameter = Tensor([5.0], requires_grad=True)
    parameter.grad = np.array([2.0])

    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(
        parameter.data,
        np.array([4.8]),
    )


def test_sgd_zero_grad():
    parameter = Tensor([5.0], requires_grad=True)
    parameter.grad = np.array([2.0])

    optimizer = SGD([parameter], lr=0.1)
    optimizer.zero_grad()

    assert parameter.grad is None


def test_adam_matches_pytorch_first_step():
    parameter = Tensor(
        [1.0, -2.0, 3.0],
        requires_grad=True,
    )

    parameter.grad = np.array(
        [0.1, -0.2, 0.3],
        dtype=np.float64,
    )

    optimizer = Adam(
        [parameter],
        lr=0.001,
    )

    optimizer.step()

    torch_parameter = torch.tensor(
        [1.0, -2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_parameter.grad = torch.tensor(
        [0.1, -0.2, 0.3],
        dtype=torch.float64,
    )

    torch_optimizer = torch.optim.Adam(
        [torch_parameter],
        lr=0.001,
    )

    torch_optimizer.step()

    np.testing.assert_allclose(
        parameter.data,
        torch_parameter.detach().numpy(),
    )


def test_adam_matches_pytorch_multiple_steps():
    parameter = Tensor(
        [1.0, -2.0, 3.0],
        requires_grad=True,
    )

    optimizer = Adam(
        [parameter],
        lr=0.001,
    )

    torch_parameter = torch.tensor(
        [1.0, -2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_optimizer = torch.optim.Adam(
        [torch_parameter],
        lr=0.001,
    )

    gradients = [
        np.array([0.1, -0.2, 0.3]),
        np.array([0.2, -0.1, 0.4]),
        np.array([-0.1, 0.3, 0.2]),
    ]

    for grad in gradients:
        parameter.grad = grad.copy()

        torch_parameter.grad = torch.tensor(
            grad,
            dtype=torch.float64,
        )

        optimizer.step()
        torch_optimizer.step()

    np.testing.assert_allclose(
        parameter.data,
        torch_parameter.detach().numpy(),
    )


def test_adam_zero_grad():
    parameter = Tensor([5.0], requires_grad=True)
    parameter.grad = np.array([2.0])

    optimizer = Adam([parameter])
    optimizer.zero_grad()

    assert parameter.grad is None