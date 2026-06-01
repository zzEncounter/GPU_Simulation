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

auto make_state_ri_array(const std::vector<double> &flat_ri,
                         std::size_t num_states,
                         std::size_t state_size) -> py::array_t<double> {
    py::array_t<double> array(
        {static_cast<py::ssize_t>(num_states),
         static_cast<py::ssize_t>(state_size),
         static_cast<py::ssize_t>(2)});
    std::memcpy(array.mutable_data(), flat_ri.data(),
                flat_ri.size() * sizeof(double));
    return array;
}

auto make_energy_grad_dict(const standalone_backend::EnergyGradResult &result)
    -> py::dict {
    py::array_t<double> grad(result.gradient.size());
    std::memcpy(grad.mutable_data(), result.gradient.data(),
                result.gradient.size() * sizeof(double));
    py::dict out;
    out["energy"] = py::float_(result.energy);
    out["gradient"] = std::move(grad);
    out["forward_ms"] = py::float_(result.forward_ms);
    out["back_ms"] = py::float_(result.back_ms);
    out["gradient_ms"] = py::float_(result.gradient_ms);
    out["total_ms"] = py::float_(result.total_ms);
    return out;
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
            "energy_and_grad",
            [](standalone_backend::RingIsingCudaBackend &self,
               FlatArray params,
               bool measure_timings,
               bool compute_gradient) {
                const auto view = view_params(params);
                standalone_backend::EnergyGradResult result;
                {
                    py::gil_scoped_release release;
                    result =
                        self.energy_and_grad(view.ptr, view.size, measure_timings,
                                             compute_gradient);
                }
                return make_energy_grad_dict(result);
            },
            py::arg("params"),
            py::arg("measure_timings") = true,
            py::arg("compute_gradient") = true)
        .def(
            "dense_scan_experiment",
            [](standalone_backend::RingIsingCudaBackend &self,
               FlatArray params) {
                const auto view = view_params(params);
                standalone_backend::DenseScanExperimentResult result;
                {
                    py::gil_scoped_release release;
                    result = self.dense_scan_experiment(view.ptr, view.size);
                }

                py::array_t<double> grad(result.gradient.size());
                std::memcpy(grad.mutable_data(), result.gradient.data(),
                            result.gradient.size() * sizeof(double));

                py::dict out;
                out["energy"] = py::float_(result.energy);
                out["gradient"] = std::move(grad);
                out["forward_states_ri"] = make_state_ri_array(
                    result.forward_states_ri, result.num_forward_states,
                    result.state_size);
                out["backward_states_ri"] = make_state_ri_array(
                    result.backward_states_ri, result.num_backward_states,
                    result.state_size);
                out["cpu_reference_ms"] = py::float_(result.cpu_reference_ms);
                out["gpu_scan_ms"] = py::float_(result.gpu_scan_ms);
                out["sequential_statevector_ms"] =
                    py::float_(result.sequential_statevector_ms);
                return out;
            },
            py::arg("params"));

    m.def(
        "energy_and_grad",
        [](std::size_t num_qubits, std::size_t num_layers, double field,
           const std::string &gradient_strategy,
           bool fuse_ring_cnot_layer,
           std::size_t checkpoint_interval_ops,
           FlatArray params,
           bool measure_timings,
           bool compute_gradient) {
            validate_params_shape(num_qubits, num_layers, params);
            const auto view = view_params(params);
            standalone_backend::EnergyGradResult result;
            {
                py::gil_scoped_release release;
                result = standalone_backend::energy_and_grad(
                    num_qubits, num_layers, field, gradient_strategy,
                    fuse_ring_cnot_layer, view.ptr, view.size,
                    checkpoint_interval_ops, measure_timings,
                    compute_gradient);
            }
            return make_energy_grad_dict(result);
        },
        py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
        py::arg("gradient_strategy") = "checkpoint",
        py::arg("fuse_ring_cnot_layer") = true,
        py::arg("checkpoint_interval_ops") = 0,
        py::arg("params"),
        py::arg("measure_timings") = true,
        py::arg("compute_gradient") = true);

    m.def(
        "dense_scan_experiment",
        [](std::size_t num_qubits, std::size_t num_layers, double field,
           bool fuse_ring_cnot_layer,
           FlatArray params) {
            validate_params_shape(num_qubits, num_layers, params);
            const auto view = view_params(params);
            standalone_backend::DenseScanExperimentResult result;
            {
                py::gil_scoped_release release;
                result = standalone_backend::dense_scan_experiment(
                    num_qubits, num_layers, field, fuse_ring_cnot_layer,
                    view.ptr, view.size);
            }

            py::array_t<double> grad(result.gradient.size());
            std::memcpy(grad.mutable_data(), result.gradient.data(),
                        result.gradient.size() * sizeof(double));

            py::dict out;
            out["energy"] = py::float_(result.energy);
            out["gradient"] = std::move(grad);
            out["forward_states_ri"] = make_state_ri_array(
                result.forward_states_ri, result.num_forward_states,
                result.state_size);
            out["backward_states_ri"] = make_state_ri_array(
                result.backward_states_ri, result.num_backward_states,
                result.state_size);
            out["cpu_reference_ms"] = py::float_(result.cpu_reference_ms);
            out["gpu_scan_ms"] = py::float_(result.gpu_scan_ms);
            out["sequential_statevector_ms"] =
                py::float_(result.sequential_statevector_ms);
            return out;
        },
        py::arg("num_qubits"), py::arg("num_layers"), py::arg("field"),
        py::arg("fuse_ring_cnot_layer") = true, py::arg("params"));
}
