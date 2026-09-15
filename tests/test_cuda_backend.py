import numpy as np
import pytest

from fluxion.cuda import build_info, device_name, is_available
from fluxion.ops import cuda_linear
from fluxion.tensor import Tensor

pytestmark = pytest.mark.skipif(not is_available(), reason="CUDA backend is not available")


def test_cuda_backend_reports_metadata():
    info = build_info()
    assert info["backend"] == "cuda"
    assert info["dtype"] == "float64"
    assert info["device"] == device_name()
    assert info["cuda_runtime_version"] > 0


def test_cuda_linear_forward_and_backward_match_numpy():
    rng = np.random.default_rng(7)
    x_data = rng.normal(size=(5, 7))
    w_data = rng.normal(size=(7, 4))
    b_data = rng.normal(size=(4,))
    grad = rng.normal(size=(5, 4))

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    y = cuda_linear(x, w, b)
    np.testing.assert_allclose(y.data, x_data @ w_data + b_data, rtol=1e-10, atol=1e-10)

    y.backward(grad)
    np.testing.assert_allclose(x.grad, grad @ w_data.T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(w.grad, x_data.T @ grad, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b.grad, grad.sum(axis=0), rtol=1e-10, atol=1e-10)
    assert y._op == "cuda_linear"
