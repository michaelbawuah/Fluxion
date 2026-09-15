import platform

import numpy as np
import pytest

from fluxion.native import backend_name, build_info, is_available


pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Fluxion native extension is not built",
)


def test_native_backend_reports_build_metadata():
    name = backend_name()
    info = build_info()

    assert name in {"accelerate", "cblas"}
    assert info["backend"] == name
    assert info["dtype"] == "float64"

    if platform.system() == "Darwin":
        assert name == "accelerate"
        assert info["platform"] == "macos"
    elif platform.system() == "Linux":
        assert name == "cblas"
        assert info["platform"] == "linux"


def test_native_extension_linear_kernel_smoke_test():
    import fluxion_native

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    weight = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    bias = np.array([1.0, -1.0], dtype=np.float64)

    output = fluxion_native.linear_forward(x, weight, bias)
    expected = x @ weight + bias
    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
