// Keep one pybind11 API while selecting the platform BLAS implementation at
// compile time: Apple Accelerate on macOS, CBLAS/OpenBLAS on Linux.
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <tuple>

namespace py = pybind11;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::string backend_name() {
#ifdef __APPLE__
    return "accelerate";
#else
    return "cblas";
#endif
}

py::dict build_info() {
    py::dict info;
    info["backend"] = backend_name();
#ifdef __APPLE__
    info["platform"] = "macos";
#else
    info["platform"] = "linux";
#endif
    info["dtype"] = "float64";
    return info;
}

py::array_t<double> linear_forward(Array x, Array weight, Array bias) {
    auto x_info = x.request();
    auto weight_info = weight.request();
    auto bias_info = bias.request();

    if (x_info.ndim != 2 || weight_info.ndim != 2 || bias_info.ndim != 1) {
        throw std::runtime_error("linear_forward expects 2D x, 2D weight, and 1D bias");
    }

    const py::ssize_t batch_size = x_info.shape[0];
    const py::ssize_t input_dim = x_info.shape[1];
    const py::ssize_t output_dim = weight_info.shape[1];

    if (weight_info.shape[0] != input_dim) {
        throw std::runtime_error("x and weight dimensions are incompatible");
    }
    if (bias_info.shape[0] != output_dim) {
        throw std::runtime_error("bias dimension must match output dimension");
    }

    py::array_t<double> output({batch_size, output_dim});
    auto output_info = output.request();

    const double* x_ptr = static_cast<const double*>(x_info.ptr);
    const double* weight_ptr = static_cast<const double*>(weight_info.ptr);
    const double* bias_ptr = static_cast<const double*>(bias_info.ptr);
    double* output_ptr = static_cast<double*>(output_info.ptr);

    // Delegate the O(batch * input_dim * output_dim) matrix product to the
    // platform's optimized BLAS rather than reimplementing GEMM in C++.
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        batch_size, output_dim, input_dim,
        1.0, x_ptr, input_dim, weight_ptr, output_dim,
        0.0, output_ptr, output_dim
    );

    for (py::ssize_t i = 0; i < batch_size; ++i) {
        for (py::ssize_t j = 0; j < output_dim; ++j) {
            output_ptr[i * output_dim + j] += bias_ptr[j];
        }
    }
    return output;
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
linear_backward(Array grad_output, Array x, Array weight) {
    auto grad_info = grad_output.request();
    auto x_info = x.request();
    auto weight_info = weight.request();

    if (grad_info.ndim != 2 || x_info.ndim != 2 || weight_info.ndim != 2) {
        throw std::runtime_error("linear_backward expects 2D arrays");
    }

    const py::ssize_t batch_size = x_info.shape[0];
    const py::ssize_t input_dim = x_info.shape[1];
    const py::ssize_t output_dim = weight_info.shape[1];

    if (weight_info.shape[0] != input_dim ||
        grad_info.shape[0] != batch_size ||
        grad_info.shape[1] != output_dim) {
        throw std::runtime_error("linear_backward dimensions are incompatible");
    }

    py::array_t<double> grad_x({batch_size, input_dim});
    py::array_t<double> grad_weight({input_dim, output_dim});
    py::array_t<double> grad_bias(output_dim);

    auto grad_x_info = grad_x.request();
    auto grad_weight_info = grad_weight.request();
    auto grad_bias_info = grad_bias.request();

    const double* grad_ptr = static_cast<const double*>(grad_info.ptr);
    const double* x_ptr = static_cast<const double*>(x_info.ptr);
    const double* weight_ptr = static_cast<const double*>(weight_info.ptr);
    double* grad_x_ptr = static_cast<double*>(grad_x_info.ptr);
    double* grad_weight_ptr = static_cast<double*>(grad_weight_info.ptr);
    double* grad_bias_ptr = static_cast<double*>(grad_bias_info.ptr);

    // dX = G W^T
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        batch_size, input_dim, output_dim,
        1.0, grad_ptr, output_dim, weight_ptr, output_dim,
        0.0, grad_x_ptr, input_dim
    );

    // dW = X^T G
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        input_dim, output_dim, batch_size,
        1.0, x_ptr, input_dim, grad_ptr, output_dim,
        0.0, grad_weight_ptr, output_dim
    );

    // db = sum of G over the batch dimension.
    for (py::ssize_t j = 0; j < output_dim; ++j) {
        double sum = 0.0;
        for (py::ssize_t i = 0; i < batch_size; ++i) {
            sum += grad_ptr[i * output_dim + j];
        }
        grad_bias_ptr[j] = sum;
    }

    return {grad_x, grad_weight, grad_bias};
}

PYBIND11_MODULE(fluxion_native, m) {
    m.doc() = "Portable native C++ kernels for Fluxion";
    m.def("backend_name", &backend_name, "Name of the compiled CPU backend");
    m.def("build_info", &build_info, "Native backend build metadata");
    m.def("linear_forward", &linear_forward, "Fused Linear forward: XW + b");
    m.def("linear_backward", &linear_backward, "Fused Linear backward: dX, dW, db");
}
