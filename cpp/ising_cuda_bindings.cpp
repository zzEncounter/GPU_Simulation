#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ising_cuda_backend.hpp"

namespace py = pybind11;

namespace {

void validate_params_shape(std::size_t num_qubits, std::size_t num_layers,
                           py::array_t<double, py::array::c_style |
                                                   py::array::forcecast>
                               params) {
    const auto buffer = params.request();
    const auto expected = static_cast<py::ssize_t>(num_qubits * num_layers * 2);
    if (buffer.ndim != 1) {
        throw std::invalid_argument("params must be a flat 1D float64 array.");
    }
    if (buffer.size != expected) {
        throw std::invalid_argument("params does not match num_qubits/layers.");
    }
}

} // namespace

PYBIND11_MODULE(_cuda_backend, m) {
    m.doc() = "Standalone CUDA backend for the ring Ising adjoint workflow.";

    py::class_<standalone_backend::RingIsingCudaBackend>(m, "RingIsingCudaBackend")
        .def(py::init<std::size_t, std::size_t, double, const std::string &, bool, std::size_t>(),
             py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
             py::arg("gradient_strategy") = "checkpoint",
             py::arg("fuse_ring_cnot_layer") = true,
             py::arg("checkpoint_interval_ops") = 0)
        .def(
            "forward_energy",
            [](standalone_backend::RingIsingCudaBackend &self,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   params) {
                const auto buffer = params.request();
                const auto *params_ptr =
                    static_cast<const double *>(buffer.ptr);
                const auto params_size =
                    static_cast<std::size_t>(buffer.size);
                py::gil_scoped_release release;
                return self.forward_energy(params_ptr, params_size);
            },
            py::arg("params"))
        .def(
            "energy_and_grad",
            [](standalone_backend::RingIsingCudaBackend &self,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   params) {
                const auto buffer = params.request();
                const auto *params_ptr =
                    static_cast<const double *>(buffer.ptr);
                const auto params_size =
                    static_cast<std::size_t>(buffer.size);
                standalone_backend::EnergyGradResult result;
                {
                    py::gil_scoped_release release;
                    result = self.energy_and_grad(params_ptr, params_size);
                }

                py::array_t<double> grad(result.gradient.size());
                std::memcpy(grad.mutable_data(), result.gradient.data(),
                            result.gradient.size() * sizeof(double));
                return py::make_tuple(result.energy, grad);
            },
            py::arg("params"));

    m.def(
        "forward_energy",
        [](std::size_t num_qubits, std::size_t num_layers, double field,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               params) {
            validate_params_shape(num_qubits, num_layers, params);
            const auto buffer = params.request();
            const auto *params_ptr = static_cast<const double *>(buffer.ptr);
            const auto params_size = static_cast<std::size_t>(buffer.size);
            py::gil_scoped_release release;
            return standalone_backend::forward_energy(num_qubits, num_layers,
                                                      field, params_ptr,
                                                      params_size, true);
        },
        py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
        py::arg("params"));

    m.def(
        "energy_and_grad",
        [](std::size_t num_qubits, std::size_t num_layers, double field,
           const std::string &gradient_strategy,
           bool fuse_ring_cnot_layer,
           std::size_t checkpoint_interval_ops,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               params) {
            validate_params_shape(num_qubits, num_layers, params);
            const auto buffer = params.request();
            const auto *params_ptr = static_cast<const double *>(buffer.ptr);
            const auto params_size = static_cast<std::size_t>(buffer.size);
            standalone_backend::EnergyGradResult result;
            {
                py::gil_scoped_release release;
                result = standalone_backend::energy_and_grad(
                    num_qubits, num_layers, field, gradient_strategy,
                    fuse_ring_cnot_layer, params_ptr, params_size,
                    checkpoint_interval_ops);
            }

            py::array_t<double> grad(result.gradient.size());
            std::memcpy(grad.mutable_data(), result.gradient.data(),
                        result.gradient.size() * sizeof(double));
            return py::make_tuple(result.energy, grad);
        },
        py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
        py::arg("gradient_strategy") = "checkpoint",
        py::arg("fuse_ring_cnot_layer") = true,
        py::arg("checkpoint_interval_ops") = 0,
        py::arg("params"));
}
