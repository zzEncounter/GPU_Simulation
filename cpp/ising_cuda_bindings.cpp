#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ising_cuda_backend.hpp"

namespace py = pybind11;

namespace {

using FlatArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

struct ParamsView {
    const double *ptr;
    std::size_t size;
};

auto view_params(const FlatArray &params) -> ParamsView {
    const auto buffer = params.request();
    return {static_cast<const double *>(buffer.ptr),
            static_cast<std::size_t>(buffer.size)};
}

void validate_params_shape(std::size_t num_qubits, std::size_t num_layers,
                           FlatArray params) {
    const auto buffer = params.request();
    const auto expected = static_cast<py::ssize_t>(num_qubits * num_layers * 2);
    if (buffer.ndim != 1) {
        throw std::invalid_argument("params must be a flat 1D float64 array.");
    }
    if (buffer.size != expected) {
        throw std::invalid_argument("params does not match num_qubits/layers.");
    }
}

auto make_energy_grad_dict(const standalone_backend::EnergyGradResult &result)
    -> py::dict {
    py::array_t<double> grad(result.gradient.size());
    std::memcpy(grad.mutable_data(), result.gradient.data(),
                result.gradient.size() * sizeof(double));
    py::dict out;
    out["energy"] = py::float_(result.energy);
    out["gradient"] = std::move(grad);
    if (!result.stage_timings_ms.empty()) {
        py::dict timings;
        for (const auto &[name, value_ms] : result.stage_timings_ms) {
            timings[py::str(name)] = py::float_(value_ms);
        }
        out["timings_ms"] = std::move(timings);
    }
    return out;
}

auto make_cuda_graph_benchmark_dict(
    const standalone_backend::CudaGraphBenchmarkResult &result) -> py::dict {
    py::dict out;
    out["normal_forward_ms"] = py::float_(result.normal_forward_ms);
    out["graph_forward_ms"] = py::float_(result.graph_forward_ms);
    out["speedup"] = py::float_(
        result.graph_forward_ms > 0.0
            ? result.normal_forward_ms / result.graph_forward_ms
            : 0.0);
    return out;
}

} // namespace

PYBIND11_MODULE(_cuda_backend, m) {
    m.doc() = "Standalone CUDA backend for the ring Ising adjoint workflow.";

    py::class_<standalone_backend::RingIsingCudaBackend>(m, "RingIsingCudaBackend")
        .def(py::init<std::size_t, std::size_t, double, const std::string &,
                      std::size_t, bool>(),
             py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
             py::arg("gradient_strategy") = "structured_adjoint",
             py::arg("structured_rotation_chunk_width") = 8,
             py::arg("double_buffer") = false)
        .def(
            "energy_and_grad",
            [](standalone_backend::RingIsingCudaBackend &self,
               FlatArray params,
               bool compute_gradient,
               bool profile) {
                const auto view = view_params(params);
                standalone_backend::EnergyGradResult result;
                {
                    py::gil_scoped_release release;
                    result = self.energy_and_grad(view.ptr, view.size,
                                                  compute_gradient, profile);
                }
                return make_energy_grad_dict(result);
            },
            py::arg("params"),
            py::arg("compute_gradient") = true,
            py::arg("profile") = false)
        .def(
            "benchmark_structured_forward_graph",
            [](standalone_backend::RingIsingCudaBackend &self,
               FlatArray params,
               std::size_t repeats) {
                const auto view = view_params(params);
                standalone_backend::CudaGraphBenchmarkResult result;
                {
                    py::gil_scoped_release release;
                    result = self.benchmark_structured_forward_graph(
                        view.ptr, view.size, repeats);
                }
                return make_cuda_graph_benchmark_dict(result);
            },
            py::arg("params"),
            py::arg("repeats") = 100)
        ;

    m.def(
        "energy_and_grad",
        [](std::size_t num_qubits, std::size_t num_layers, double field,
           const std::string &gradient_strategy,
           FlatArray params,
           bool compute_gradient,
           bool profile,
           std::size_t structured_rotation_chunk_width) {
            validate_params_shape(num_qubits, num_layers, params);
            const auto view = view_params(params);
            standalone_backend::EnergyGradResult result;
            {
                py::gil_scoped_release release;
                result = standalone_backend::energy_and_grad(
                    num_qubits, num_layers, field, gradient_strategy, view.ptr,
                    view.size, compute_gradient, profile,
                    structured_rotation_chunk_width);
            }
            return make_energy_grad_dict(result);
        },
        py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
        py::arg("gradient_strategy") = "structured_adjoint",
        py::arg("params"),
        py::arg("compute_gradient") = true,
        py::arg("profile") = false,
        py::arg("structured_rotation_chunk_width") = 8);

}
