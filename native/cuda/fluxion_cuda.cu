#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <tuple>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); \
} while (0)

// Baseline kernel mapping: one CUDA thread owns one output element. This is
// intentionally simple and easy to validate before introducing shared-memory
// tiling or a device-resident Tensor abstraction.
__global__ void linear_forward_kernel(const double* x, const double* w, const double* b,
                                      double* y, int batch, int in_dim, int out_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_dim;
    if (index >= total) return;
    int row = index / out_dim;
    int col = index % out_dim;
    double value = b[col];
    for (int k = 0; k < in_dim; ++k) value += x[row * in_dim + k] * w[k * out_dim + col];
    y[index] = value;
}

__global__ void grad_x_kernel(const double* g, const double* w, double* dx,
                              int batch, int in_dim, int out_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_dim;
    if (index >= total) return;
    int row = index / in_dim;
    int col = index % in_dim;
    double value = 0.0;
    for (int j = 0; j < out_dim; ++j) value += g[row * out_dim + j] * w[col * out_dim + j];
    dx[index] = value;
}

__global__ void grad_weight_kernel(const double* g, const double* x, double* dw,
                                   int batch, int in_dim, int out_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_dim * out_dim;
    if (index >= total) return;
    int row = index / out_dim;
    int col = index % out_dim;
    double value = 0.0;
    for (int i = 0; i < batch; ++i) value += x[i * in_dim + row] * g[i * out_dim + col];
    dw[index] = value;
}

__global__ void grad_bias_kernel(const double* g, double* db, int batch, int out_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_dim) return;
    double value = 0.0;
    for (int i = 0; i < batch; ++i) value += g[i * out_dim + col];
    db[col] = value;
}

bool is_available() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    return error == cudaSuccess && count > 0;
}

std::string device_name() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("No CUDA-capable GPU is available");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.name;
}

py::dict build_info() {
    py::dict info;
    info["backend"] = "cuda";
    info["dtype"] = "float64";
    info["device"] = device_name();
    info["cuda_runtime_version"] = CUDART_VERSION;
    return info;
}

py::array_t<double> linear_forward(Array x, Array weight, Array bias) {
    // NumPy owns host memory, so this baseline allocates device buffers and
    // performs host->device->host transfers on every call. Benchmarks therefore
    // report end-to-end CUDA cost rather than kernel-only execution time.
    auto xi=x.request(), wi=weight.request(), bi=bias.request();
    if (xi.ndim != 2 || wi.ndim != 2 || bi.ndim != 1) throw std::runtime_error("linear_forward expects 2D x, 2D weight, and 1D bias");
    int batch=(int)xi.shape[0], in_dim=(int)xi.shape[1], out_dim=(int)wi.shape[1];
    if (wi.shape[0] != xi.shape[1] || bi.shape[0] != wi.shape[1]) throw std::runtime_error("linear_forward dimensions are incompatible");
    py::array_t<double> output({batch,out_dim});
    double *dx=nullptr,*dw=nullptr,*db=nullptr,*dy=nullptr;
    size_t sx=(size_t)batch*in_dim*sizeof(double), sw=(size_t)in_dim*out_dim*sizeof(double), sb=(size_t)out_dim*sizeof(double), sy=(size_t)batch*out_dim*sizeof(double);
    try {
        CUDA_CHECK(cudaMalloc(&dx,sx)); CUDA_CHECK(cudaMalloc(&dw,sw)); CUDA_CHECK(cudaMalloc(&db,sb)); CUDA_CHECK(cudaMalloc(&dy,sy));
        CUDA_CHECK(cudaMemcpy(dx,xi.ptr,sx,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dw,wi.ptr,sw,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db,bi.ptr,sb,cudaMemcpyHostToDevice));
        int threads=256, blocks=(batch*out_dim+threads-1)/threads;
        linear_forward_kernel<<<blocks,threads>>>(dx,dw,db,dy,batch,in_dim,out_dim);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(output.request().ptr,dy,sy,cudaMemcpyDeviceToHost));
    } catch (...) { cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dy); throw; }
    cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(dy); return output;
}

std::tuple<py::array_t<double>,py::array_t<double>,py::array_t<double>> linear_backward(Array grad, Array x, Array weight) {
    // Backward mirrors the analytical Linear derivatives with three independent
    // kernels: dX = G W^T, dW = X^T G, and db = sum_rows(G).
    auto gi=grad.request(), xi=x.request(), wi=weight.request();
    if (gi.ndim != 2 || xi.ndim != 2 || wi.ndim != 2) throw std::runtime_error("linear_backward expects 2D arrays");
    int batch=(int)xi.shape[0], in_dim=(int)xi.shape[1], out_dim=(int)wi.shape[1];
    if (wi.shape[0] != xi.shape[1] || gi.shape[0] != xi.shape[0] || gi.shape[1] != wi.shape[1]) throw std::runtime_error("linear_backward dimensions are incompatible");
    py::array_t<double> gx({batch,in_dim}), gw({in_dim,out_dim}), gb(out_dim);
    double *dg=nullptr,*dx=nullptr,*dw=nullptr,*dgx=nullptr,*dgw=nullptr,*dgb=nullptr;
    size_t sg=(size_t)batch*out_dim*sizeof(double), sx=(size_t)batch*in_dim*sizeof(double), sw=(size_t)in_dim*out_dim*sizeof(double), sb=(size_t)out_dim*sizeof(double);
    try {
        CUDA_CHECK(cudaMalloc(&dg,sg)); CUDA_CHECK(cudaMalloc(&dx,sx)); CUDA_CHECK(cudaMalloc(&dw,sw)); CUDA_CHECK(cudaMalloc(&dgx,sx)); CUDA_CHECK(cudaMalloc(&dgw,sw)); CUDA_CHECK(cudaMalloc(&dgb,sb));
        CUDA_CHECK(cudaMemcpy(dg,gi.ptr,sg,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dx,xi.ptr,sx,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dw,wi.ptr,sw,cudaMemcpyHostToDevice));
        int threads=256;
        grad_x_kernel<<<(batch*in_dim+threads-1)/threads,threads>>>(dg,dw,dgx,batch,in_dim,out_dim);
        grad_weight_kernel<<<(in_dim*out_dim+threads-1)/threads,threads>>>(dg,dx,dgw,batch,in_dim,out_dim);
        grad_bias_kernel<<<(out_dim+threads-1)/threads,threads>>>(dg,dgb,batch,out_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(gx.request().ptr,dgx,sx,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(gw.request().ptr,dgw,sw,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(gb.request().ptr,dgb,sb,cudaMemcpyDeviceToHost));
    } catch (...) { cudaFree(dg);cudaFree(dx);cudaFree(dw);cudaFree(dgx);cudaFree(dgw);cudaFree(dgb);throw; }
    cudaFree(dg);cudaFree(dx);cudaFree(dw);cudaFree(dgx);cudaFree(dgw);cudaFree(dgb); return {gx,gw,gb};
}

PYBIND11_MODULE(fluxion_cuda, m) {
    m.doc()="Experimental CUDA kernels for Fluxion";
    m.def("is_available",&is_available);
    m.def("device_name",&device_name);
    m.def("build_info",&build_info);
    m.def("linear_forward",&linear_forward);
    m.def("linear_backward",&linear_backward);
}
