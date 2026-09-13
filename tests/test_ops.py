import numpy as np

from fluxion.ops import add, multiply
from fluxion.tensor import Tensor


def test_add_forward():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])

    out = add(a, b)

    np.testing.assert_array_equal(out.data, np.array([4.0, 6.0]))


def test_add_tracks_graph():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)

    out = add(a, b)

    assert out.requires_grad is True
    assert out._prev == (a, b)
    assert out._op == "add"


def test_multiply_forward():
    a = Tensor([2.0])
    b = Tensor([3.0])

    out = multiply(a, b)

    np.testing.assert_array_equal(out.data, np.array([6.0]))


def test_multiply_tracks_graph():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)

    out = multiply(a, b)

    assert out.requires_grad is True
    assert out._prev == (a, b)
    assert out._op == "multiply"


def test_multiply_backward():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)

    out = a * b
    out.backward()

    np.testing.assert_array_equal(a.grad, np.array([3.0]))
    np.testing.assert_array_equal(b.grad, np.array([2.0]))


def test_multiply_add_chain_rule():
    x = Tensor([2.0], requires_grad=True)
    w = Tensor([3.0], requires_grad=True)

    y = x * w + x
    y.backward()

    np.testing.assert_array_equal(y.data, np.array([8.0]))
    np.testing.assert_array_equal(x.grad, np.array([4.0]))
    np.testing.assert_array_equal(w.grad, np.array([2.0]))