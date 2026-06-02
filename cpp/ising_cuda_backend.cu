#include "ising_cuda_backend.hpp"
#include "ising_cuda_backend_internal.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace standalone_backend {
namespace {

using detail::Complex;
template <typename T> using DeviceBuffer = detail::DeviceBuffer<T>;

auto elapsed_ms(std::chrono::steady_clock::time_point start,
                std::chrono::steady_clock::time_point end) -> double {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

enum class OpKind { RY, RZ, CNOT, FusedRYRZ, RingCNOTLayer };
enum class GradientStrategy {
    SaveParamStates,
    Checkpoint,
    DenseScan
};

struct OpDesc {
    OpKind kind;
    std::size_t wire0;
    std::size_t wire1;
    double theta0;
    double theta1;
    std::size_t param_index0;
    std::size_t param_index1;
    bool is_parametric;
};

auto build_ring_ops(std::size_t num_qubits, std::size_t num_layers,
                    const double *params,
                    bool fuse_ring_cnot_layer) -> std::vector<OpDesc> {
    std::vector<OpDesc> ops;
    ops.reserve(
        num_layers * (num_qubits + (fuse_ring_cnot_layer ? 1 : num_qubits)));

    for (std::size_t layer = 0; layer < num_layers; layer++) {
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            const auto base = (layer * num_qubits + wire) * 2;
            const double theta0 = params != nullptr ? params[base] : 0.0;
            const double theta1 = params != nullptr ? params[base + 1] : 0.0;
            ops.push_back({OpKind::FusedRYRZ, wire, 0, theta0,
                           theta1, base, base + 1, true});
        }
        if (fuse_ring_cnot_layer) {
            ops.push_back(
                {OpKind::RingCNOTLayer, 0, 0, 0.0, 0.0, 0, 0, false});
        } else {
            for (std::size_t wire = 0; wire < num_qubits; wire++) {
                ops.push_back({OpKind::CNOT, wire, (wire + 1) % num_qubits, 0.0,
                               0.0, 0, 0, false});
            }
        }
    }
    return ops;
}

void copy_device_buffer(Complex *dst, const Complex *src, std::size_t size);

void apply_op_inplace(Complex *state, std::size_t size, const OpDesc &op,
                      bool inverse, Complex *scratch, std::size_t num_qubits);

void apply_ops_range_inplace(Complex *state, std::size_t size,
                             const std::vector<OpDesc> &ops,
                             std::size_t begin, std::size_t end,
                             Complex *scratch, std::size_t num_qubits) {
    for (std::size_t op_index = begin; op_index < end; op_index++) {
        apply_op_inplace(state, size, ops[op_index], false, scratch,
                         num_qubits);
    }
}

void apply_op_inplace(Complex *state, std::size_t size, const OpDesc &op,
                      bool inverse, Complex *scratch,
                      std::size_t num_qubits) {
    switch (op.kind) {
    case OpKind::RY:
        detail::launch_apply_ry(state, size, op.wire0,
                                inverse ? -op.theta0 : op.theta0);
        break;
    case OpKind::RZ:
        detail::launch_apply_rz(state, size, op.wire0,
                                inverse ? -op.theta0 : op.theta0);
        break;
    case OpKind::CNOT:
        detail::launch_apply_cnot(state, size, op.wire0, op.wire1);
        break;
    case OpKind::FusedRYRZ:
        if (inverse) {
            detail::launch_apply_rz(state, size, op.wire0, -op.theta1);
            detail::launch_apply_ry(state, size, op.wire0, -op.theta0);
        } else {
            detail::launch_apply_ryrz(state, size, op.wire0, op.theta0,
                                      op.theta1);
        }
        break;
    case OpKind::RingCNOTLayer:
        if (scratch == nullptr) {
            throw std::runtime_error(
                "RingCNOTLayer requires a scratch buffer for fused execution.");
        }
        detail::launch_apply_ring_cnot_layer(scratch, state, size, num_qubits,
                                             inverse);
        copy_device_buffer(state, scratch, size);
        break;
    }
}

void apply_param_derivative(Complex *out, const Complex *state,
                            std::size_t size, const OpDesc &op) {
    switch (op.kind) {
    case OpKind::RY:
        detail::launch_apply_dry(out, state, size, op.wire0, op.theta0);
        break;
    case OpKind::RZ:
        detail::launch_apply_drz(out, state, size, op.wire0, op.theta0);
        break;
    case OpKind::FusedRYRZ:
        throw std::runtime_error(
            "FusedRYRZ should use fused_ryrz_gradients instead of "
            "apply_param_derivative.");
    case OpKind::CNOT:
        throw std::runtime_error("CNOT is not parametric.");
    case OpKind::RingCNOTLayer:
        throw std::runtime_error("RingCNOTLayer is not parametric.");
    }
}

void copy_device_buffer(Complex *dst, const Complex *src, std::size_t size) {
    detail::check_cuda(cudaMemcpyAsync(dst, src, sizeof(Complex) * size,
                                       cudaMemcpyDeviceToDevice, 0),
                       "cudaMemcpyAsyncDeviceToDevice");
}

auto validate_and_get_state_size(std::size_t num_qubits,
                                 std::size_t num_layers,
                                 std::size_t num_params) -> std::size_t {
    const auto expected_params = num_qubits * num_layers * 2;
    if (num_params != expected_params) {
        throw std::invalid_argument("num_params does not match the circuit.");
    }
    if (num_qubits >= 63) {
        throw std::invalid_argument(
            "This prototype only supports fewer than 63 qubits.");
    }
    return std::size_t{1} << num_qubits;
}

void validate_num_params(std::size_t expected_params, std::size_t num_params) {
    if (num_params != expected_params) {
        throw std::invalid_argument("num_params does not match the circuit.");
    }
}

auto state_slot(Complex *base, std::size_t state_size, std::size_t index)
    -> Complex * {
    return base + index * state_size;
}

auto parse_gradient_strategy(const std::string &strategy)
    -> GradientStrategy {
    if (strategy == "save_param_states") {
        return GradientStrategy::SaveParamStates;
    }
    if (strategy == "checkpoint") {
        return GradientStrategy::Checkpoint;
    }
    if (strategy == "dense_scan") {
        return GradientStrategy::DenseScan;
    }
    if (strategy == "bruteforce_parallel_q6") {
        return GradientStrategy::DenseScan;
    }
    throw std::invalid_argument(
        "Unknown gradient strategy. Expected one of: "
        "save_param_states, checkpoint, dense_scan.");
}

struct CublasHandle {
    cublasHandle_t handle{nullptr};

    CublasHandle() {
        detail::check_cublas(cublasCreate(&handle), "cublasCreate");
        detail::check_cublas(
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
            "cublasSetPointerMode");
    }

    CublasHandle(const CublasHandle &) = delete;
    auto operator=(const CublasHandle &) -> CublasHandle & = delete;

    CublasHandle(CublasHandle &&other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

auto next_power_of_two(std::size_t value) -> std::size_t {
    if (value == 0) {
        return 1;
    }
    std::size_t power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

auto to_int(std::size_t value, const char *context) -> int {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(context) + " does not fit in int.");
    }
    return static_cast<int>(value);
}

#include "ising_cuda_dense_helpers.inc"

} // namespace

struct RingIsingCudaBackend::Impl {
    std::size_t num_qubits;
    std::size_t num_layers;
    double field;
    std::size_t expected_params;
    std::size_t state_size;
    std::size_t num_ops;
    std::size_t checkpoint_interval_ops;
    std::size_t num_chunks;
    GradientStrategy strategy;
    bool fuse_ring_cnot_layer;
    DeviceBuffer<Complex> current;
    DeviceBuffer<Complex> lambda;
    DeviceBuffer<Complex> deriv;
    DeviceBuffer<Complex> scratch;
    DeviceBuffer<Complex> save_param_states;
    DeviceBuffer<Complex> checkpoints;
    DeviceBuffer<Complex> local_states;
    std::unique_ptr<CublasHandle> dense_cublas;
    DeviceBuffer<Complex> dense_gate_mats;
    DeviceBuffer<Complex> dense_dgate_mats;
    DeviceBuffer<Complex> dense_hamiltonian;
    DeviceBuffer<int> dense_param_gate_indices;
    DeviceBuffer<Complex> dense_prefix_scan;
    DeviceBuffer<Complex> dense_tmp_level;
    DeviceBuffer<Complex> dense_left_tmp;
    DeviceBuffer<int> dense_left_indices;
    DeviceBuffer<int> dense_right_indices;
    DeviceBuffer<const Complex *> dense_ptr_a;
    DeviceBuffer<const Complex *> dense_ptr_b;
    DeviceBuffer<Complex *> dense_ptr_c;
    DeviceBuffer<Complex> dense_ones_vector;
    DeviceBuffer<Complex> dense_psi_before;
    DeviceBuffer<Complex> dense_psi_after;
    DeviceBuffer<Complex> dense_forward_states;
    DeviceBuffer<Complex> dense_lambda_k;
    DeviceBuffer<Complex> dense_lambda_by_op;
    DeviceBuffer<Complex> dense_suffix_scan;
    DeviceBuffer<Complex> dense_lambda_for_params;
    DeviceBuffer<Complex> dense_psi_before_for_params;
    DeviceBuffer<Complex> dense_deriv_states;
    DeviceBuffer<double> dense_gradients;
    DeviceBuffer<int> dense_reverse_index;
    DeviceBuffer<Complex> dense_reversed_tmp;
    DeviceBuffer<Complex> dense_suffix_for_ops;
    DenseGateStorage dense_storage_cache;
    std::vector<std::size_t> dense_param_op_indices;
    std::vector<Complex> dense_hamiltonian_host;
    std::vector<Complex> dense_psi0_host;
    std::vector<int> dense_reverse_index_host;
    bool dense_static_device_uploaded{false};
    bool dense_reverse_indices_uploaded{false};

    Impl(std::size_t num_qubits_, std::size_t num_layers_, double field_,
         const std::string &gradient_strategy_,
         bool fuse_ring_cnot_layer_,
         std::size_t checkpoint_interval_ops_)
        : num_qubits(num_qubits_), num_layers(num_layers_), field(field_),
          expected_params(num_qubits_ * num_layers_ * 2),
          state_size(validate_and_get_state_size(num_qubits_, num_layers_,
                                                 num_qubits_ * num_layers_ * 2)),
          num_ops(fuse_ring_cnot_layer_
                      ? num_layers_ * (num_qubits_ + 1)
                      : 2 * num_qubits_ * num_layers_),
          checkpoint_interval_ops(checkpoint_interval_ops_), num_chunks(0),
          strategy(parse_gradient_strategy(gradient_strategy_)),
          fuse_ring_cnot_layer(fuse_ring_cnot_layer_),
          current(state_size), lambda(state_size), deriv(state_size),
          scratch(state_size) {
        if (strategy == GradientStrategy::Checkpoint &&
            (checkpoint_interval_ops_ == 0 || checkpoint_interval_ops_ >= num_ops)) {
            strategy = GradientStrategy::SaveParamStates;
        }

        if (strategy == GradientStrategy::DenseScan ||
            strategy == GradientStrategy::SaveParamStates) {
            save_param_states.allocate((expected_params / 2) * state_size);
        } else {
            num_chunks = (num_ops + checkpoint_interval_ops - 1) /
                         checkpoint_interval_ops;
            checkpoints.allocate((num_chunks + 1) * state_size);
            local_states.allocate((checkpoint_interval_ops + 1) * state_size);
        }
    }
};

auto run_dense_scan_experiment(RingIsingCudaBackend::Impl &impl,
                               const double *params, std::size_t num_params)
    -> DenseScanExperimentResult;
auto run_dense_scan_energy_and_grad_fast(RingIsingCudaBackend::Impl &impl,
                                         const double *params,
                                         std::size_t num_params,
                                         bool measure_timings)
    -> EnergyGradResult;


#include "ising_cuda_statevector_modes.inc"

#include "ising_cuda_dense_modes.inc"

RingIsingCudaBackend::RingIsingCudaBackend(std::size_t num_qubits,
                                           std::size_t num_layers,
                                           double field,
                                           const std::string &gradient_strategy,
                                           bool fuse_ring_cnot_layer,
                                           std::size_t checkpoint_interval_ops)
    : impl_(std::make_unique<Impl>(num_qubits, num_layers, field,
                                   gradient_strategy,
                                   fuse_ring_cnot_layer,
                                   checkpoint_interval_ops)) {}

RingIsingCudaBackend::~RingIsingCudaBackend() = default;

RingIsingCudaBackend::RingIsingCudaBackend(RingIsingCudaBackend &&) noexcept =
    default;

auto RingIsingCudaBackend::operator=(RingIsingCudaBackend &&) noexcept
    -> RingIsingCudaBackend & = default;

auto RingIsingCudaBackend::energy_and_grad(const double *params,
                                           std::size_t num_params,
                                           bool measure_timings,
                                           bool compute_gradient)
    -> EnergyGradResult {
    return run_energy_and_grad_checkpointed(*impl_, params, num_params,
                                            measure_timings, compute_gradient);
}

auto RingIsingCudaBackend::dense_scan_experiment(const double *params,
                                                 std::size_t num_params)
    -> DenseScanExperimentResult {
    return run_dense_scan_experiment(*impl_, params, num_params);
}

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 bool fuse_ring_cnot_layer,
                                 const double *params, std::size_t num_params,
                                 std::size_t checkpoint_interval_ops,
                                 bool measure_timings,
                                 bool compute_gradient) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 gradient_strategy,
                                 fuse_ring_cnot_layer,
                                 checkpoint_interval_ops);
    return backend.energy_and_grad(params, num_params, measure_timings,
                                   compute_gradient);
}

DenseScanExperimentResult dense_scan_experiment(
    std::size_t num_qubits, std::size_t num_layers, double field,
    bool fuse_ring_cnot_layer, const double *params, std::size_t num_params) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 "dense_scan",
                                 fuse_ring_cnot_layer, 0);
    return backend.dense_scan_experiment(params, num_params);
}

} // namespace standalone_backend
