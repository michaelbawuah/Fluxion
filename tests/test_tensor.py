import numpy as np

from fluxion.tensor import Tensor


def test_tensor_from_list():
    x = Tensor([1.0, 2.0, 3.0])

    np.testing.assert_array_equal(x.data, np.array([1.0, 2.0, 3.0]))
    assert x.shape == (3,)
    assert x.ndim == 1
    assert x.size == 3
    assert x.requires_grad is False
    assert x.grad is None


def test_tensor_multidimensional_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert x.shape == (2, 2)
    assert x.ndim == 2
    assert x.size == 4


def test_integer_input_is_promoted_to_float():
    x = Tensor([1, 2, 3])

    assert np.issubdtype(x.dtype, np.floating)


def test_requires_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)

    assert x.requires_grad is True
    assert x.grad is None


def test_tensor_copy_does_not_share_data():
    original = Tensor([1.0, 2.0])
    copied = Tensor(original)

    copied.data[0] = 99.0

    assert original.data[0] == 1.0


def test_numpy_returns_underlying_array():
    x = Tensor([1.0, 2.0])

    assert x.numpy() is x.data