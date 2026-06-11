#include "ising_cuda_backend.hpp"
#include "ising_cuda_backend_internal.cuh"

#include <algorithm>
#include <array>
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

enum class GradientStrategy {
    InverseWalk,
    SaveParamStates,
    Checkpoint,
    DenseScan,
    BlockFusedAdjoint,
    IntrablockParallel
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

auto build_pennylane_gate_level_ops(std::size_t num_qubits,
                                    std::size_t num_layers,
                                    const double *params)
    -> std::vector<OpDesc> {
    std::vector<OpDesc> ops;
    ops.reserve(3 * num_layers * num_qubits);

    for (std::size_t layer = 0; layer < num_layers; layer++) {
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            const auto base = (layer * num_qubits + wire) * 2;
            const double theta0 = params != nullptr ? params[base] : 0.0;
            const double theta1 = params != nullptr ? params[base + 1] : 0.0;
            ops.push_back(
                {OpKind::RY, wire, 0, theta0, 0.0, base, 0, true});
            ops.push_back(
                {OpKind::RZ, wire, 0, theta1, 0.0, base + 1, 0, true});
        }
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            ops.push_back(
                {OpKind::CNOT, wire, (wire + 1) % num_qubits, 0.0, 0.0, 0, 0,
                 false});
        }
    }
    return ops;
}

auto build_ops_for_strategy(std::size_t num_qubits, std::size_t num_layers,
                            const double *params, GradientStrategy strategy,
                            bool fuse_ring_cnot_layer)
    -> std::vector<OpDesc> {
    if (strategy == GradientStrategy::InverseWalk ||
        strategy == GradientStrategy::SaveParamStates) {
        return build_pennylane_gate_level_ops(num_qubits, num_layers, params);
    }
    return build_ring_ops(num_qubits, num_layers, params, fuse_ring_cnot_layer);
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
        detail::launch_apply_ryrz(state, size, op.wire0, op.theta0,
                                  op.theta1, inverse);
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
    if (strategy == "inverse_walk") {
        return GradientStrategy::InverseWalk;
    }
    if (strategy == "checkpoint") {
        return GradientStrategy::Checkpoint;
    }
    if (strategy == "dense_scan") {
        return GradientStrategy::DenseScan;
    }
    if (strategy == "block_fused_adjoint") {
        return GradientStrategy::BlockFusedAdjoint;
    }
    if (strategy == "intrablock_parallel") {
        return GradientStrategy::IntrablockParallel;
    }
    if (strategy == "bruteforce_parallel_q6") {
        return GradientStrategy::DenseScan;
    }
    throw std::invalid_argument(
        "Unknown gradient strategy. Expected one of: "
        "inverse_walk, save_param_states, checkpoint, dense_scan, "
        "block_fused_adjoint, intrablock_parallel.");
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
    std::size_t intrablock_block_size;
    std::size_t intrablock_num_blocks;
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
    DeviceBuffer<Complex> block_fused_lambda_states;
    DeviceBuffer<std::size_t> block_fused_wires;
    DeviceBuffer<std::size_t> block_fused_param_indices;
    DeviceBuffer<double> block_fused_thetas;
    DeviceBuffer<double> block_fused_phis;
    DeviceBuffer<double> block_fused_gradients;
    std::unique_ptr<CublasHandle> dense_cublas;
    DeviceBuffer<Complex> dense_gate_mats;
    DeviceBuffer<Complex> dense_dgate_mats;
    DeviceBuffer<Complex> dense_hamiltonian;
    DeviceBuffer<int> dense_param_gate_indices;
    DeviceBuffer<Complex> dense_prefix_scan;
    DeviceBuffer<Complex> dense_ones_vector;
    DeviceBuffer<Complex> dense_psi_before;
    DeviceBuffer<Complex> dense_psi_after;
    DeviceBuffer<Complex> dense_forward_states;
    DeviceBuffer<Complex> dense_total_matrix;
    DeviceBuffer<Complex> dense_lambda_k;
    DeviceBuffer<Complex> dense_eta_vector;
    DeviceBuffer<Complex> dense_eta_before;
    DeviceBuffer<Complex> dense_suffix_scan;
    DeviceBuffer<double> dense_gradients;
    DenseGateStorage dense_storage_cache;
    std::vector<std::size_t> dense_param_op_indices;
    std::vector<Complex> dense_hamiltonian_host;
    std::vector<Complex> dense_psi0_host;
    bool dense_static_device_uploaded{false};
    DeviceBuffer<OpDesc> intrablock_ops;
    DeviceBuffer<Complex> intrablock_boundary_states;
    DeviceBuffer<Complex> intrablock_lambda_boundaries;
    DeviceBuffer<Complex> intrablock_forward_states;
    DeviceBuffer<double> intrablock_gradients;
    DeviceBuffer<double> gate_level_gradients;

    Impl(std::size_t num_qubits_, std::size_t num_layers_, double field_,
         const std::string &gradient_strategy_,
         bool fuse_ring_cnot_layer_,
         std::size_t checkpoint_interval_ops_,
         std::size_t intrablock_block_size_)
        : num_qubits(num_qubits_), num_layers(num_layers_), field(field_),
          expected_params(num_qubits_ * num_layers_ * 2),
          state_size(validate_and_get_state_size(num_qubits_, num_layers_,
                                                 num_qubits_ * num_layers_ * 2)),
          num_ops(fuse_ring_cnot_layer_
                      ? num_layers_ * (num_qubits_ + 1)
                      : 2 * num_qubits_ * num_layers_),
          checkpoint_interval_ops(checkpoint_interval_ops_),
          intrablock_block_size(intrablock_block_size_ == 0
                                    ? std::size_t{64}
                                    : intrablock_block_size_),
          intrablock_num_blocks(0), num_chunks(0),
          strategy(parse_gradient_strategy(gradient_strategy_)),
          fuse_ring_cnot_layer(fuse_ring_cnot_layer_),
          current(state_size), lambda(state_size), deriv(state_size),
          scratch(state_size) {
        if ((strategy == GradientStrategy::Checkpoint ||
             strategy == GradientStrategy::BlockFusedAdjoint) &&
            (checkpoint_interval_ops_ == 0 || checkpoint_interval_ops_ >= num_ops)) {
            strategy = GradientStrategy::SaveParamStates;
        }

        if (strategy == GradientStrategy::SaveParamStates) {
            save_param_states.allocate(expected_params * state_size);
            gate_level_gradients.allocate(expected_params);
        } else if (strategy == GradientStrategy::InverseWalk) {
            gate_level_gradients.allocate(expected_params);
        } else if (strategy == GradientStrategy::Checkpoint ||
                   strategy == GradientStrategy::BlockFusedAdjoint) {
            num_chunks = (num_ops + checkpoint_interval_ops - 1) /
                         checkpoint_interval_ops;
            checkpoints.allocate((num_chunks + 1) * state_size);
            local_states.allocate((checkpoint_interval_ops + 1) * state_size);
            if (strategy == GradientStrategy::BlockFusedAdjoint) {
                block_fused_lambda_states.allocate(checkpoint_interval_ops *
                                                   state_size);
                block_fused_wires.allocate(checkpoint_interval_ops);
                block_fused_param_indices.allocate(2 * checkpoint_interval_ops);
                block_fused_thetas.allocate(checkpoint_interval_ops);
                block_fused_phis.allocate(checkpoint_interval_ops);
                block_fused_gradients.allocate(expected_params);
            }
        } else if (strategy == GradientStrategy::IntrablockParallel) {
            intrablock_num_blocks =
                (num_ops + intrablock_block_size - 1) / intrablock_block_size;
        }
    }
};
auto run_dense_scan_energy_and_grad_fast(RingIsingCudaBackend::Impl &impl,
                                         const double *params,
                                         std::size_t num_params)
    -> EnergyGradResult;
auto run_intrablock_parallel_energy_and_grad(RingIsingCudaBackend::Impl &impl,
                                             const double *params,
                                             std::size_t num_params,
                                             bool compute_gradient)
    -> EnergyGradResult;


#include "ising_cuda_statevector_modes.inc"

#include "ising_cuda_block_fused_adjoint_modes.inc"

#include "ising_cuda_dense_modes.inc"

#include "ising_cuda_intrablock_modes.inc"

RingIsingCudaBackend::RingIsingCudaBackend(std::size_t num_qubits,
                                           std::size_t num_layers,
                                           double field,
                                           const std::string &gradient_strategy,
                                           bool fuse_ring_cnot_layer,
                                           std::size_t checkpoint_interval_ops,
                                           std::size_t intrablock_block_size)
    : impl_(std::make_unique<Impl>(num_qubits, num_layers, field,
                                   gradient_strategy,
                                   fuse_ring_cnot_layer,
                                   checkpoint_interval_ops,
                                   intrablock_block_size)) {}

RingIsingCudaBackend::~RingIsingCudaBackend() = default;

RingIsingCudaBackend::RingIsingCudaBackend(RingIsingCudaBackend &&) noexcept =
    default;

auto RingIsingCudaBackend::operator=(RingIsingCudaBackend &&) noexcept
    -> RingIsingCudaBackend & = default;

auto RingIsingCudaBackend::energy_and_grad(const double *params,
                                           std::size_t num_params,
                                           bool compute_gradient)
    -> EnergyGradResult {
    if (impl_->strategy == GradientStrategy::BlockFusedAdjoint) {
        return run_energy_and_grad_block_fused_adjoint(*impl_, params, num_params,
                                                       compute_gradient);
    }
    return run_energy_and_grad_checkpointed(*impl_, params, num_params,
                                            compute_gradient);
}

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 bool fuse_ring_cnot_layer,
                                 const double *params, std::size_t num_params,
                                 std::size_t checkpoint_interval_ops,
                                 std::size_t intrablock_block_size,
                                 bool compute_gradient) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 gradient_strategy,
                                 fuse_ring_cnot_layer,
                                 checkpoint_interval_ops,
                                 intrablock_block_size);
    return backend.energy_and_grad(params, num_params, compute_gradient);
}

} // namespace standalone_backend
