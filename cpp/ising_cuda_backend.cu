#include "ising_cuda_backend.hpp"
#include "ising_cuda_backend_internal.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace standalone_backend {
namespace {

using detail::Complex;
template <typename T> using DeviceBuffer = detail::DeviceBuffer<T>;

enum class OpKind { RY, RZ, CNOT, FusedRYRZ, RingCNOTLayer };
enum class GradientStrategy { SaveParamStates, Checkpoint };

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
            ops.push_back({OpKind::FusedRYRZ, wire, 0, params[base],
                           params[base + 1], base, base + 1, true});
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
    throw std::invalid_argument(
        "Unknown gradient strategy. Expected one of: "
        "save_param_states, checkpoint.");
}

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

        if (strategy == GradientStrategy::SaveParamStates) {
            save_param_states.allocate((expected_params / 2) * state_size);
        } else {
            num_chunks = (num_ops + checkpoint_interval_ops - 1) /
                         checkpoint_interval_ops;
            checkpoints.allocate((num_chunks + 1) * state_size);
            local_states.allocate((checkpoint_interval_ops + 1) * state_size);
        }
    }
};

double run_forward_energy_only(std::size_t num_qubits, std::size_t num_layers,
                               double field, const double *params,
                               std::size_t num_params,
                               bool fuse_ring_cnot_layer) {
    const auto expected_params = num_qubits * num_layers * 2;
    validate_num_params(expected_params, num_params);
    const auto state_size =
        validate_and_get_state_size(num_qubits, num_layers, expected_params);
    const auto ops = build_ring_ops(num_qubits, num_layers, params,
                                    fuse_ring_cnot_layer);

    DeviceBuffer<Complex> current(state_size);
    DeviceBuffer<Complex> h_psi(state_size);
    DeviceBuffer<Complex> scratch(state_size);

    detail::launch_init_zero_state(current.get(), state_size);
    for (const auto &op : ops) {
        apply_op_inplace(current.get(), state_size, op, false, scratch.get(),
                         num_qubits);
    }

    detail::launch_apply_hamiltonian(h_psi.get(), current.get(), state_size,
                                     num_qubits, field);
    const Complex energy_complex =
        detail::complex_inner_product(current.get(), h_psi.get(), state_size);
    detail::check_cuda(cudaDeviceSynchronize(), "run_forward_energy_only completion");
    return energy_complex.real();
}

auto run_forward_energy_only(RingIsingCudaBackend::Impl &impl,
                             const double *params,
                             std::size_t num_params) -> double {
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);

    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (const auto &op : ops) {
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }
    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);
    const Complex energy_complex =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size);
    detail::check_cuda(cudaDeviceSynchronize(), "run_forward_energy_only completion");
    return energy_complex.real();
}

auto run_energy_and_grad_save_param_states(
    RingIsingCudaBackend::Impl &impl, const double *params,
    std::size_t num_params) -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);

    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (std::size_t op_index = 0; op_index < ops.size(); op_index++) {
        const auto &op = ops[op_index];
        if (op.is_parametric) {
            copy_device_buffer(
                state_slot(impl.save_param_states.get(), impl.state_size,
                           op.param_index0 / 2),
                impl.current.get(), impl.state_size);
        }
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }

    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);
    const Complex energy_complex =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size);

    EnergyGradResult result;
    result.energy = energy_complex.real();
    result.gradient.assign(impl.expected_params, 0.0);

    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        if (op.is_parametric) {
            const Complex *state_before =
                state_slot(impl.save_param_states.get(), impl.state_size,
                           op.param_index0 / 2);
            if (op.kind == OpKind::FusedRYRZ) {
                const auto [grad_theta, grad_phi] = detail::fused_ryrz_gradients(
                    impl.lambda.get(), state_before, impl.state_size, op.wire0,
                    op.theta0, op.theta1);
                result.gradient[op.param_index0] = grad_theta;
                result.gradient[op.param_index1] = grad_phi;
            } else {
                apply_param_derivative(impl.deriv.get(), state_before,
                                       impl.state_size, op);
                const Complex grad_complex =
                    detail::complex_inner_product(impl.lambda.get(), impl.deriv.get(),
                                                  impl.state_size);
                result.gradient[op.param_index0] = 2.0 * grad_complex.real();
            }
        }
        if (reverse_index != 0) {
            apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                             impl.scratch.get(), impl.num_qubits);
        }
    }

    detail::check_cuda(cudaDeviceSynchronize(),
                       "run_energy_and_grad_save_param_states completion");
    return result;
}

auto run_energy_and_grad_checkpointed(RingIsingCudaBackend::Impl &impl,
                                      const double *params,
                                      std::size_t num_params)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);
    const auto num_ops = ops.size();

    if (impl.strategy == GradientStrategy::SaveParamStates) {
        return run_energy_and_grad_save_param_states(impl, params, num_params);
    }

    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    copy_device_buffer(state_slot(impl.checkpoints.get(), impl.state_size, 0),
                       impl.current.get(), impl.state_size);

    for (std::size_t chunk_idx = 0; chunk_idx < impl.num_chunks; chunk_idx++) {
        const auto chunk_start = chunk_idx * impl.checkpoint_interval_ops;
        const auto chunk_end =
            std::min(chunk_start + impl.checkpoint_interval_ops, num_ops);
        apply_ops_range_inplace(impl.current.get(), impl.state_size, ops,
                                chunk_start, chunk_end, impl.scratch.get(),
                                impl.num_qubits);
        copy_device_buffer(
            state_slot(impl.checkpoints.get(), impl.state_size, chunk_idx + 1),
            impl.current.get(), impl.state_size);
    }

    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);
    const Complex energy_complex =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size);

    EnergyGradResult result;
    result.energy = energy_complex.real();
    result.gradient.assign(impl.expected_params, 0.0);

    for (std::size_t chunk_idx = impl.num_chunks; chunk_idx-- > 0;) {
        const auto chunk_start = chunk_idx * impl.checkpoint_interval_ops;
        const auto chunk_end =
            std::min(chunk_start + impl.checkpoint_interval_ops, num_ops);
        const auto local_count = chunk_end - chunk_start;

        copy_device_buffer(impl.current.get(),
                           state_slot(impl.checkpoints.get(), impl.state_size,
                                      chunk_idx),
                           impl.state_size);
        copy_device_buffer(state_slot(impl.local_states.get(), impl.state_size, 0),
                           impl.current.get(), impl.state_size);
        for (std::size_t local_idx = 0; local_idx < local_count; local_idx++) {
            apply_op_inplace(impl.current.get(), impl.state_size,
                             ops[chunk_start + local_idx], false,
                             impl.scratch.get(), impl.num_qubits);
            copy_device_buffer(
                state_slot(impl.local_states.get(), impl.state_size,
                           local_idx + 1),
                impl.current.get(), impl.state_size);
        }

        for (std::size_t op_idx = chunk_end; op_idx-- > chunk_start;) {
            const auto &op = ops[op_idx];
            const auto local_before = op_idx - chunk_start;
            if (op.is_parametric) {
                const Complex *state_before =
                    state_slot(impl.local_states.get(), impl.state_size,
                               local_before);
                if (op.kind == OpKind::FusedRYRZ) {
                    const auto [grad_theta, grad_phi] = detail::fused_ryrz_gradients(
                        impl.lambda.get(), state_before, impl.state_size,
                        op.wire0, op.theta0, op.theta1);
                    result.gradient[op.param_index0] = grad_theta;
                    result.gradient[op.param_index1] = grad_phi;
                } else {
                    apply_param_derivative(impl.deriv.get(), state_before,
                                           impl.state_size, op);
                    const Complex grad_complex =
                        detail::complex_inner_product(impl.lambda.get(),
                                                      impl.deriv.get(),
                                                      impl.state_size);
                    result.gradient[op.param_index0] = 2.0 * grad_complex.real();
                }
            }
            if (op_idx != 0) {
                apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                                 impl.scratch.get(), impl.num_qubits);
            }
        }
    }

    detail::check_cuda(cudaDeviceSynchronize(),
                       "run_energy_and_grad_checkpointed completion");
    return result;
}

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

auto RingIsingCudaBackend::forward_energy(const double *params,
                                          std::size_t num_params) -> double {
    return run_forward_energy_only(*impl_, params, num_params);
}

auto RingIsingCudaBackend::energy_and_grad(const double *params,
                                           std::size_t num_params)
    -> EnergyGradResult {
    return run_energy_and_grad_checkpointed(*impl_, params, num_params);
}

double forward_energy(std::size_t num_qubits, std::size_t num_layers,
                      double field, const double *params,
                      std::size_t num_params,
                      bool fuse_ring_cnot_layer) {
    return run_forward_energy_only(num_qubits, num_layers, field, params,
                                   num_params, fuse_ring_cnot_layer);
}

EnergyGradResult energy_and_grad(std::size_t num_qubits,
                                 std::size_t num_layers, double field,
                                 const std::string &gradient_strategy,
                                 bool fuse_ring_cnot_layer,
                                 const double *params, std::size_t num_params,
                                 std::size_t checkpoint_interval_ops) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 gradient_strategy,
                                 fuse_ring_cnot_layer,
                                 checkpoint_interval_ops);
    return backend.energy_and_grad(params, num_params);
}

} // namespace standalone_backend
