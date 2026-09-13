import numpy as np
import torch

from fluxion.tensor import Tensor


def test_addition_backward():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)

    out = a + b
    out.backward()

    np.testing.assert_array_equal(a.grad, np.array([1.0]))
    np.testing.assert_array_equal(b.grad, np.array([1.0]))


def test_addition_chain_rule():
    x = Tensor([2.0], requires_grad=True)

    y = x + x
    z = y + x

    z.backward()

    np.testing.assert_array_equal(x.grad, np.array([3.0]))


def test_backward_requires_scalar_or_explicit_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)

    try:
        x.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError for non-scalar backward().")


def test_non_scalar_backward_with_explicit_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x + x

    y.backward(np.array([2.0, 4.0]))

    np.testing.assert_array_equal(x.grad, np.array([4.0, 8.0]))


def test_fluxion_matches_pytorch_for_multiply_add_expression():
    # Fluxion
    x = Tensor([2.0], requires_grad=True)
    w = Tensor([3.0], requires_grad=True)

    y = x * w + x
    y.backward()

    # PyTorch
    torch_x = torch.tensor([2.0], requires_grad=True)
    torch_w = torch.tensor([3.0], requires_grad=True)

    torch_y = torch_x * torch_w + torch_x
    torch_y.backward()

    # Compare forward output
    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    # Compare gradients
    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

    np.testing.assert_allclose(
        w.grad,
        torch_w.grad.numpy(),
    )


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


def test_sum_matches_pytorch():
    # Fluxion
    x = Tensor([2.0, 4.0, 6.0], requires_grad=True)

    y = x.sum()
    y.backward()

    # PyTorch
    torch_x = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)

    torch_y = torch_x.sum()
    torch_y.backward()

    np.testing.assert_allclose(y.data, torch_y.detach().numpy())
    np.testing.assert_allclose(x.grad, torch_x.grad.numpy())


def test_mean_matches_pytorch():
    # Fluxion
    x = Tensor([2.0, 4.0, 6.0], requires_grad=True)

    y = x.mean()
    y.backward()

    # PyTorch
    torch_x = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)

    torch_y = torch_x.mean()
    torch_y.backward()

    np.testing.assert_allclose(y.data, torch_y.detach().numpy())
    np.testing.assert_allclose(x.grad, torch_x.grad.numpy())

def test_matmul_matches_pytorch():
    # Fluxion
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    w = Tensor(
        [[2.0, 0.0], [1.0, 3.0]],
        requires_grad=True,
    )

    y = x @ w
    loss = y.sum()
    loss.backward()

    # PyTorch
    torch_x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        requires_grad=True,
    )
    torch_w = torch.tensor(
        [[2.0, 0.0], [1.0, 3.0]],
        requires_grad=True,
    )

    torch_y = torch_x @ torch_w
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

    np.testing.assert_allclose(
        w.grad,
        torch_w.grad.numpy(),
    )   

def test_broadcast_add_matches_pytorch():
    # Fluxion
    x = Tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    b = Tensor(
        [10.0, 20.0],
        requires_grad=True,
    )

    y = x + b
    loss = y.sum()
    loss.backward()

    # PyTorch
    torch_x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    torch_b = torch.tensor(
        [10.0, 20.0],
        requires_grad=True,
    )

    torch_y = torch_x + torch_b
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

    np.testing.assert_allclose(
        b.grad,
        torch_b.grad.numpy(),
    )

def test_exp_matches_pytorch():
    x = Tensor(
        [-1.0, 0.0, 1.0, 2.0],
        requires_grad=True,
    )

    y = x.exp()
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0],
        requires_grad=True,
    )

    torch_y = torch.exp(torch_x)
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )
def test_reverse_operators_match_pytorch():
    x = Tensor(
        [1.0, 2.0, 4.0],
        requires_grad=True,
    )

    y = 1 + x
    z = 3 * y
    q = 10 - z
    out = 2 / q

    loss = out.sum()
    loss.backward()

    torch_x = torch.tensor(
        [1.0, 2.0, 4.0],
        requires_grad=True,
    )

    torch_y = 1 + torch_x
    torch_z = 3 * torch_y
    torch_q = 10 - torch_z
    torch_out = 2 / torch_q

    torch_loss = torch_out.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        out.data,
        torch_out.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )
def test_log_matches_pytorch():
    x = Tensor(
        [0.5, 1.0, 2.0, 4.0],
        requires_grad=True,
    )

    y = x.log()
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [0.5, 1.0, 2.0, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.log(torch_x)
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )
def test_sum_axis_matches_pytorch():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    y = x.sum(axis=1)
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x.sum(dim=1)
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_mean_axis_keepdims_matches_pytorch():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    y = x.mean(
        axis=1,
        keepdims=True,
    )

    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x.mean(
        dim=1,
        keepdim=True,
    )

    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_sum_tuple_axes_matches_pytorch():
    x = Tensor(
        np.arange(
            24.0,
        ).reshape(2, 3, 4),
        requires_grad=True,
    )

    y = x.sum(axis=(0, 2))
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        np.arange(
            24.0,
        ).reshape(2, 3, 4),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x.sum(
        dim=(0, 2),
    )

    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

def test_max_matches_pytorch():
    x = Tensor(
        [1.0, 4.0, 2.0, 3.0],
        requires_grad=True,
    )

    y = x.max()
    y.backward()

    torch_x = torch.tensor(
        [1.0, 4.0, 2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.amax(torch_x)
    torch_y.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_max_axis_matches_pytorch():
    x = Tensor(
        [
            [1.0, 5.0, 2.0],
            [4.0, 3.0, 6.0],
        ],
        requires_grad=True,
    )

    y = x.max(
        axis=1,
        keepdims=True,
    )

    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [1.0, 5.0, 2.0],
            [4.0, 3.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.amax(
        torch_x,
        dim=1,
        keepdim=True,
    )

    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_max_tied_values_matches_pytorch():
    x = Tensor(
        [2.0, 5.0, 5.0],
        requires_grad=True,
    )

    y = x.max()
    y.backward()

    torch_x = torch.tensor(
        [2.0, 5.0, 5.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.amax(torch_x)
    torch_y.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

def test_getitem_single_index_matches_pytorch():
    x = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    y = x[1]
    y.backward()

    torch_x = torch.tensor(
        [10.0, 20.0, 30.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x[1]
    torch_y.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_getitem_slice_matches_pytorch():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    y = x[:, 1]
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x[:, 1]
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )


def test_getitem_repeated_indices_matches_pytorch():
    x = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    indices = np.array([0, 0, 2])

    y = x[indices]
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_indices = torch.tensor(
        [0, 0, 2],
        dtype=torch.long,
    )

    torch_y = torch_x[torch_indices]
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

def test_reshape_matches_pytorch():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    y = x.reshape(3, 2)

    weights = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    loss = (y * weights).sum()
    loss.backward()

    torch_x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x.reshape(3, 2)

    torch_weights = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float64,
    )

    torch_loss = (
        torch_y * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

def test_sqrt_matches_pytorch():
    x = Tensor(
        [0.25, 1.0, 4.0, 9.0],
        requires_grad=True,
    )

    y = x.sqrt()
    loss = y.sum()
    loss.backward()

    torch_x = torch.tensor(
        [0.25, 1.0, 4.0, 9.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch.sqrt(torch_x)
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )

def test_matmul_2d_still_matches_pytorch():
    a = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        requires_grad=True,
    )

    y = a @ b
    loss = y.sum()
    loss.backward()

    torch_a = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_b = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_a @ torch_b
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        a.grad,
        torch_a.grad.numpy(),
    )

    np.testing.assert_allclose(
        b.grad,
        torch_b.grad.numpy(),
    )


def test_batched_matmul_matches_pytorch():
    a = Tensor(
        np.arange(24.0).reshape(2, 3, 4),
        requires_grad=True,
    )

    b = Tensor(
        np.arange(40.0).reshape(2, 4, 5),
        requires_grad=True,
    )

    y = a @ b
    loss = y.sum()
    loss.backward()

    torch_a = torch.tensor(
        np.arange(24.0).reshape(2, 3, 4),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_b = torch.tensor(
        np.arange(40.0).reshape(2, 4, 5),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_a @ torch_b
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        a.grad,
        torch_a.grad.numpy(),
    )

    np.testing.assert_allclose(
        b.grad,
        torch_b.grad.numpy(),
    )


def test_broadcasted_batched_matmul_matches_pytorch():
    a = Tensor(
        np.arange(24.0).reshape(2, 3, 4),
        requires_grad=True,
    )

    b = Tensor(
        np.arange(20.0).reshape(4, 5),
        requires_grad=True,
    )

    y = a @ b
    loss = y.sum()
    loss.backward()

    torch_a = torch.tensor(
        np.arange(24.0).reshape(2, 3, 4),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_b = torch.tensor(
        np.arange(20.0).reshape(4, 5),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_a @ torch_b
    torch_loss = torch_y.sum()
    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        a.grad,
        torch_a.grad.numpy(),
    )

    np.testing.assert_allclose(
        b.grad,
        torch_b.grad.numpy(),
    )

def test_transpose_matches_pytorch():
    x = Tensor(
        np.arange(24.0).reshape(2, 3, 4),
        requires_grad=True,
    )

    y = x.transpose(-1, -2)

    weights = Tensor(
        np.arange(24.0).reshape(2, 4, 3),
    )

    loss = (y * weights).sum()
    loss.backward()

    torch_x = torch.tensor(
        np.arange(24.0).reshape(2, 3, 4),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_y = torch_x.transpose(-1, -2)

    torch_weights = torch.tensor(
        np.arange(24.0).reshape(2, 4, 3),
        dtype=torch.float64,
    )

    torch_loss = (
        torch_y * torch_weights
    ).sum()

    torch_loss.backward()

    np.testing.assert_allclose(
        y.data,
        torch_y.detach().numpy(),
    )

    np.testing.assert_allclose(
        x.grad,
        torch_x.grad.numpy(),
    )