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
    BruteForceParallelQ6
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
    if (strategy == "bruteforce_parallel_q6") {
        return GradientStrategy::BruteForceParallelQ6;
    }
    throw std::invalid_argument(
        "Unknown gradient strategy. Expected one of: "
        "save_param_states, checkpoint, bruteforce_parallel_q6.");
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

void fill_zero_matrix(Complex *matrix, std::size_t dim) {
    std::fill(matrix, matrix + dim * dim, Complex(0.0, 0.0));
}

void fill_single_qubit_embedded_matrix(Complex *out, std::size_t dim,
                                       std::size_t wire,
                                       const std::array<Complex, 4> &local) {
    fill_zero_matrix(out, dim);
    const auto mask = std::size_t{1} << wire;
    for (std::size_t col = 0; col < dim; col++) {
        if ((col & mask) == 0U) {
            const auto row0 = col;
            const auto row1 = col | mask;
            out[row0 + col * dim] = local[0]; // m00
            out[row1 + col * dim] = local[1]; // m10
        } else {
            const auto row0 = col ^ mask;
            const auto row1 = col;
            out[row0 + col * dim] = local[2]; // m01
            out[row1 + col * dim] = local[3]; // m11
        }
    }
}

void fill_cnot_matrix(Complex *out, std::size_t dim, std::size_t control,
                      std::size_t target) {
    fill_zero_matrix(out, dim);
    const auto control_mask = std::size_t{1} << control;
    const auto target_mask = std::size_t{1} << target;
    for (std::size_t col = 0; col < dim; col++) {
        auto row = col;
        if ((row & control_mask) != 0U) {
            row ^= target_mask;
        }
        out[row + col * dim] = Complex(1.0, 0.0);
    }
}

auto apply_ring_cnot_layer_basis(std::size_t index, std::size_t num_qubits,
                                 bool inverse) -> std::size_t {
    auto transformed = index;
    if (!inverse) {
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            if (((transformed >> wire) & std::size_t{1}) != 0U) {
                transformed ^= (std::size_t{1} << ((wire + 1) % num_qubits));
            }
        }
    } else {
        for (std::size_t wire = num_qubits; wire-- > 0;) {
            if (((transformed >> wire) & std::size_t{1}) != 0U) {
                transformed ^= (std::size_t{1} << ((wire + 1) % num_qubits));
            }
        }
    }
    return transformed;
}

void fill_ring_cnot_layer_matrix(Complex *out, std::size_t dim,
                                 std::size_t num_qubits) {
    fill_zero_matrix(out, dim);
    for (std::size_t col = 0; col < dim; col++) {
        const auto row = apply_ring_cnot_layer_basis(col, num_qubits, false);
        out[row + col * dim] = Complex(1.0, 0.0);
    }
}

void fill_fused_ryrz_matrices(Complex *u, Complex *dtheta, Complex *dphi,
                              std::size_t dim, std::size_t wire, double theta,
                              double phi) {
    const double c = std::cos(theta * 0.5);
    const double s = std::sin(theta * 0.5);
    const double half_phi = phi * 0.5;
    const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
    const Complex phase1(std::cos(half_phi), std::sin(half_phi));
    const std::array<Complex, 4> local_u = {
        phase0 * Complex(c, 0.0),
        phase1 * Complex(s, 0.0),
        phase0 * Complex(-s, 0.0),
        phase1 * Complex(c, 0.0),
    };
    fill_single_qubit_embedded_matrix(u, dim, wire, local_u);

    if (dtheta != nullptr) {
        const std::array<Complex, 4> local_theta = {
            phase0 * Complex(-0.5 * s, 0.0),
            phase1 * Complex(0.5 * c, 0.0),
            phase0 * Complex(-0.5 * c, 0.0),
            phase1 * Complex(-0.5 * s, 0.0),
        };
        fill_single_qubit_embedded_matrix(dtheta, dim, wire, local_theta);
    }

    if (dphi != nullptr) {
        const Complex pref0(0.0, -0.5);
        const Complex pref1(0.0, 0.5);
        const Complex b00 = phase0 * Complex(c, 0.0);
        const Complex b10 = phase1 * Complex(s, 0.0);
        const Complex b01 = phase0 * Complex(-s, 0.0);
        const Complex b11 = phase1 * Complex(c, 0.0);
        const std::array<Complex, 4> local_phi = {
            pref0 * b00,
            pref1 * b10,
            pref0 * b01,
            pref1 * b11,
        };
        fill_single_qubit_embedded_matrix(dphi, dim, wire, local_phi);
    }
}

void fill_ring_ising_hamiltonian_matrix(Complex *out, std::size_t dim,
                                        std::size_t num_qubits, double field) {
    fill_zero_matrix(out, dim);
    for (std::size_t col = 0; col < dim; col++) {
        double diag_coeff = 0.0;
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            const auto next_wire = (wire + 1) % num_qubits;
            const double zi = ((col >> wire) & std::size_t{1}) != 0U ? -1.0 : 1.0;
            const double zj =
                ((col >> next_wire) & std::size_t{1}) != 0U ? -1.0 : 1.0;
            diag_coeff += -(zi * zj);
        }
        out[col + col * dim] += Complex(diag_coeff, 0.0);
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            const auto row = col ^ (std::size_t{1} << wire);
            out[row + col * dim] += Complex(-field, 0.0);
        }
    }
}

void fill_dense_gate_matrix_for_op(const OpDesc &op, std::size_t num_qubits,
                                   std::size_t dim, Complex *u,
                                   Complex *dtheta, Complex *dphi) {
    switch (op.kind) {
    case OpKind::FusedRYRZ:
        fill_fused_ryrz_matrices(u, dtheta, dphi, dim, op.wire0, op.theta0,
                                 op.theta1);
        break;
    case OpKind::CNOT:
        fill_cnot_matrix(u, dim, op.wire0, op.wire1);
        if (dtheta != nullptr) {
            fill_zero_matrix(dtheta, dim);
        }
        if (dphi != nullptr) {
            fill_zero_matrix(dphi, dim);
        }
        break;
    case OpKind::RingCNOTLayer:
        fill_ring_cnot_layer_matrix(u, dim, num_qubits);
        if (dtheta != nullptr) {
            fill_zero_matrix(dtheta, dim);
        }
        if (dphi != nullptr) {
            fill_zero_matrix(dphi, dim);
        }
        break;
    case OpKind::RY:
    case OpKind::RZ:
        throw std::runtime_error(
            "Dense q<=6 experiment expects fused RYRZ plus entanglers.");
    }
}

struct DenseGateStorage {
    std::vector<Complex> gate_mats;
    std::vector<Complex> dgate_mats;
    std::vector<int> param_gate_indices;
    std::vector<OpDesc> ops;
};

struct DenseScanGpuRunResult {
    double energy{0.0};
    std::vector<double> gradient;
    std::vector<Complex> forward_states;
    std::vector<Complex> backward_states;
    std::size_t num_ops{0};
    std::size_t state_size{0};
    double gpu_scan_ms{0.0};
    double forward_ms{0.0};
    double back_ms{0.0};
    double gradient_ms{0.0};
    double total_ms{0.0};
};

auto build_dense_gate_storage(std::size_t num_qubits, std::size_t num_layers,
                              double field, const double *params,
                              bool fuse_ring_cnot_layer)
    -> DenseGateStorage {
    (void)field;
    const auto ops = build_ring_ops(num_qubits, num_layers, params,
                                    fuse_ring_cnot_layer);
    const auto dim = std::size_t{1} << num_qubits;
    const auto mat_elements = dim * dim;
    const auto num_params = num_qubits * num_layers * 2;

    DenseGateStorage storage;
    storage.ops = ops;
    storage.gate_mats.assign(ops.size() * mat_elements, Complex(0.0, 0.0));
    storage.dgate_mats.assign(num_params * mat_elements, Complex(0.0, 0.0));
    storage.param_gate_indices.assign(num_params, -1);

    for (std::size_t op_index = 0; op_index < ops.size(); op_index++) {
        const auto &op = ops[op_index];
        auto *u = storage.gate_mats.data() + op_index * mat_elements;
        Complex *dtheta = nullptr;
        Complex *dphi = nullptr;
        if (op.is_parametric) {
            dtheta = storage.dgate_mats.data() + op.param_index0 * mat_elements;
            dphi = storage.dgate_mats.data() + op.param_index1 * mat_elements;
            storage.param_gate_indices[op.param_index0] = to_int(op_index, "op_index");
            storage.param_gate_indices[op.param_index1] = to_int(op_index, "op_index");
        }
        fill_dense_gate_matrix_for_op(op, num_qubits, dim, u, dtheta, dphi);
    }
    return storage;
}

void host_matvec(const Complex *matrix, const Complex *vector, Complex *out,
                 std::size_t dim) {
    for (std::size_t row = 0; row < dim; row++) {
        Complex acc(0.0, 0.0);
        for (std::size_t col = 0; col < dim; col++) {
            acc += matrix[row + col * dim] * vector[col];
        }
        out[row] = acc;
    }
}

void host_adjoint(const Complex *in, Complex *out, std::size_t dim) {
    for (std::size_t row = 0; row < dim; row++) {
        for (std::size_t col = 0; col < dim; col++) {
            out[row + col * dim] = thrust::conj(in[col + row * dim]);
        }
    }
}

auto host_inner_product(const Complex *lhs, const Complex *rhs, std::size_t dim)
    -> Complex {
    Complex acc(0.0, 0.0);
    for (std::size_t idx = 0; idx < dim; idx++) {
        acc += thrust::conj(lhs[idx]) * rhs[idx];
    }
    return acc;
}

void run_cpu_dense_reference(const DenseGateStorage &storage,
                            std::size_t num_qubits, double field,
                            std::vector<double> *gradient_out,
                            std::vector<Complex> *forward_out,
                            std::vector<Complex> *backward_out,
                            double *energy_out) {
    const auto dim = std::size_t{1} << num_qubits;
    const auto mat_elements = dim * dim;
    const auto num_ops = storage.ops.size();
    const auto num_params = storage.param_gate_indices.size();
    std::vector<Complex> forward((num_ops + 1) * dim, Complex(0.0, 0.0));
    std::vector<Complex> backward(num_ops * dim, Complex(0.0, 0.0));
    std::vector<double> gradient(num_params, 0.0);

    forward[0] = Complex(1.0, 0.0);
    std::vector<Complex> tmp(dim, Complex(0.0, 0.0));
    for (std::size_t op_index = 0; op_index < num_ops; op_index++) {
        const auto *u = storage.gate_mats.data() + op_index * mat_elements;
        const auto *in = forward.data() + op_index * dim;
        auto *out = forward.data() + (op_index + 1) * dim;
        host_matvec(u, in, out, dim);
    }

    std::vector<Complex> hamiltonian(mat_elements, Complex(0.0, 0.0));
    fill_ring_ising_hamiltonian_matrix(hamiltonian.data(), dim, num_qubits, field);
    std::vector<Complex> lambda_k(dim, Complex(0.0, 0.0));
    host_matvec(hamiltonian.data(), forward.data() + num_ops * dim,
                lambda_k.data(), dim);

    Complex energy_complex =
        host_inner_product(forward.data() + num_ops * dim, lambda_k.data(), dim);
    *energy_out = energy_complex.real();

    std::vector<Complex> lambda_current = lambda_k;
    std::vector<Complex> u_adj(mat_elements, Complex(0.0, 0.0));
    for (std::size_t op_index = num_ops; op_index-- > 0;) {
        std::memcpy(backward.data() + op_index * dim, lambda_current.data(),
                    sizeof(Complex) * dim);
        if (op_index != 0) {
            const auto *u = storage.gate_mats.data() + op_index * mat_elements;
            host_adjoint(u, u_adj.data(), dim);
            host_matvec(u_adj.data(), lambda_current.data(), tmp.data(), dim);
            lambda_current = tmp;
        }
    }

    std::vector<Complex> deriv(dim, Complex(0.0, 0.0));
    for (std::size_t param_index = 0; param_index < num_params; param_index++) {
        const auto gate_index = storage.param_gate_indices[param_index];
        if (gate_index < 0) {
            continue;
        }
        const auto *du = storage.dgate_mats.data() +
                         static_cast<std::size_t>(param_index) * mat_elements;
        const auto *psi_before = forward.data() + static_cast<std::size_t>(gate_index) * dim;
        const auto *lambda = backward.data() + static_cast<std::size_t>(gate_index) * dim;
        host_matvec(du, psi_before, deriv.data(), dim);
        const Complex grad = host_inner_product(lambda, deriv.data(), dim);
        gradient[param_index] = 2.0 * grad.real();
    }

    *gradient_out = std::move(gradient);
    *forward_out = std::move(forward);
    *backward_out = std::move(backward);
}

void prepare_level_indices(std::size_t span, std::size_t total_count,
                           std::vector<int> *left_indices,
                           std::vector<int> *right_indices) {
    left_indices->clear();
    right_indices->clear();
    const auto step = span * 2;
    for (std::size_t base = 0; base + step - 1 < total_count; base += step) {
        left_indices->push_back(to_int(base + span - 1, "left index"));
        right_indices->push_back(to_int(base + step - 1, "right index"));
    }
}

void upload_pointer_array(DeviceBuffer<const Complex *> &target,
                          const std::vector<const Complex *> &host_ptrs) {
    if (host_ptrs.empty()) {
        return;
    }
    detail::check_cuda(
        cudaMemcpyAsync(target.get(), host_ptrs.data(),
                        sizeof(const Complex *) * host_ptrs.size(),
                        cudaMemcpyHostToDevice, 0),
        "cudaMemcpyAsync pointer array");
}

void upload_pointer_array(DeviceBuffer<Complex *> &target,
                          const std::vector<Complex *> &host_ptrs) {
    if (host_ptrs.empty()) {
        return;
    }
    detail::check_cuda(
        cudaMemcpyAsync(target.get(), host_ptrs.data(),
                        sizeof(Complex *) * host_ptrs.size(),
                        cudaMemcpyHostToDevice, 0),
        "cudaMemcpyAsync pointer array");
}

void run_noncommutative_exclusive_scan_mats(
    cublasHandle_t handle, Complex *scan_mats, std::size_t num_mats,
    std::size_t dim, DeviceBuffer<Complex> &tmp_level,
    DeviceBuffer<Complex> &left_tmp, DeviceBuffer<int> &left_indices_dev,
    DeviceBuffer<int> &right_indices_dev, DeviceBuffer<const Complex *> &ptr_a,
    DeviceBuffer<const Complex *> &ptr_b, DeviceBuffer<Complex *> &ptr_c) {
    const auto mat_elements = dim * dim;
    const auto padded = next_power_of_two(num_mats);

    const Complex alpha(1.0, 0.0);
    const Complex beta(0.0, 0.0);
    std::vector<int> left_indices_host;
    std::vector<int> right_indices_host;
    std::vector<const Complex *> a_ptrs;
    std::vector<const Complex *> b_ptrs;
    std::vector<Complex *> c_ptrs;

    for (std::size_t span = 1; span < padded; span <<= 1) {
        prepare_level_indices(span, padded, &left_indices_host, &right_indices_host);
        const auto pairs = left_indices_host.size();
        if (pairs == 0) {
            continue;
        }
        detail::check_cuda(
            cudaMemcpyAsync(left_indices_dev.get(), left_indices_host.data(),
                            sizeof(int) * pairs, cudaMemcpyHostToDevice, 0),
            "scan up-sweep copy left indices");
        detail::check_cuda(
            cudaMemcpyAsync(right_indices_dev.get(), right_indices_host.data(),
                            sizeof(int) * pairs, cudaMemcpyHostToDevice, 0),
            "scan up-sweep copy right indices");

        a_ptrs.resize(pairs);
        b_ptrs.resize(pairs);
        c_ptrs.resize(pairs);
        for (std::size_t pair = 0; pair < pairs; pair++) {
            const auto left = static_cast<std::size_t>(left_indices_host[pair]);
            const auto right = static_cast<std::size_t>(right_indices_host[pair]);
            a_ptrs[pair] = scan_mats + right * mat_elements;
            b_ptrs[pair] = scan_mats + left * mat_elements;
            c_ptrs[pair] = tmp_level.get() + pair * mat_elements;
        }
        upload_pointer_array(ptr_a, a_ptrs);
        upload_pointer_array(ptr_b, b_ptrs);
        upload_pointer_array(ptr_c, c_ptrs);

        detail::check_cublas(
            cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"),
                               to_int(dim, "dim"), to_int(dim, "dim"),
                               reinterpret_cast<const cuDoubleComplex *>(&alpha),
                               reinterpret_cast<const cuDoubleComplex **>(ptr_a.get()),
                               to_int(dim, "dim"),
                               reinterpret_cast<const cuDoubleComplex **>(ptr_b.get()),
                               to_int(dim, "dim"),
                               reinterpret_cast<const cuDoubleComplex *>(&beta),
                               reinterpret_cast<cuDoubleComplex **>(ptr_c.get()),
                               to_int(dim, "dim"), to_int(pairs, "pairs")),
            "cublasZgemmBatched up-sweep");

        detail::launch_scatter_matrices(tmp_level.get(), right_indices_dev.get(),
                                        scan_mats, pairs, mat_elements);
    }

    detail::launch_fill_identity_matrices(
        scan_mats + (padded - 1) * mat_elements, 1, dim);

    for (std::size_t span = padded >> 1; span >= 1; span >>= 1) {
        prepare_level_indices(span, padded, &left_indices_host, &right_indices_host);
        const auto pairs = left_indices_host.size();
        if (pairs == 0) {
            if (span == 1) {
                break;
            }
            continue;
        }
        detail::check_cuda(
            cudaMemcpyAsync(left_indices_dev.get(), left_indices_host.data(),
                            sizeof(int) * pairs, cudaMemcpyHostToDevice, 0),
            "scan down-sweep copy left indices");
        detail::check_cuda(
            cudaMemcpyAsync(right_indices_dev.get(), right_indices_host.data(),
                            sizeof(int) * pairs, cudaMemcpyHostToDevice, 0),
            "scan down-sweep copy right indices");

        detail::launch_prepare_downsweep_buffers(
            scan_mats, left_indices_dev.get(), right_indices_dev.get(),
            left_tmp.get(), pairs, mat_elements);

        a_ptrs.resize(pairs);
        b_ptrs.resize(pairs);
        c_ptrs.resize(pairs);
        for (std::size_t pair = 0; pair < pairs; pair++) {
            const auto left = static_cast<std::size_t>(left_indices_host[pair]);
            a_ptrs[pair] = left_tmp.get() + pair * mat_elements;
            b_ptrs[pair] = scan_mats + left * mat_elements;
            c_ptrs[pair] = tmp_level.get() + pair * mat_elements;
        }
        upload_pointer_array(ptr_a, a_ptrs);
        upload_pointer_array(ptr_b, b_ptrs);
        upload_pointer_array(ptr_c, c_ptrs);

        detail::check_cublas(
            cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"),
                               to_int(dim, "dim"), to_int(dim, "dim"),
                               reinterpret_cast<const cuDoubleComplex *>(&alpha),
                               reinterpret_cast<const cuDoubleComplex **>(ptr_a.get()),
                               to_int(dim, "dim"),
                               reinterpret_cast<const cuDoubleComplex **>(ptr_b.get()),
                               to_int(dim, "dim"),
                               reinterpret_cast<const cuDoubleComplex *>(&beta),
                               reinterpret_cast<cuDoubleComplex **>(ptr_c.get()),
                               to_int(dim, "dim"), to_int(pairs, "pairs")),
            "cublasZgemmBatched down-sweep");

        detail::launch_scatter_matrices(tmp_level.get(), right_indices_dev.get(),
                                        scan_mats, pairs, mat_elements);
        if (span == 1) {
            break;
        }
    }
}

auto convert_states_to_ri(const std::vector<Complex> &states) -> std::vector<double> {
    std::vector<double> ri(states.size() * 2, 0.0);
    for (std::size_t idx = 0; idx < states.size(); idx++) {
        ri[idx * 2] = states[idx].real();
        ri[idx * 2 + 1] = states[idx].imag();
    }
    return ri;
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
    std::vector<Complex> dense_hamiltonian_host;
    std::vector<Complex> dense_psi0_host;
    std::vector<int> dense_reverse_index_host;

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

        if (strategy == GradientStrategy::BruteForceParallelQ6 ||
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
    std::size_t num_params, bool measure_timings) -> EnergyGradResult {
    const auto total_start = std::chrono::steady_clock::now();
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);

    const auto forward_start = std::chrono::steady_clock::now();
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
    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_save_param_states forward");
    }
    const auto forward_end = std::chrono::steady_clock::now();

    EnergyGradResult result;
    result.energy = energy_complex.real();
    result.gradient.assign(impl.expected_params, 0.0);

    double gradient_ms = 0.0;
    double back_ms = 0.0;
    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        if (op.is_parametric) {
            const auto gradient_start = std::chrono::steady_clock::now();
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
            if (measure_timings) {
                detail::check_cuda(
                    cudaDeviceSynchronize(),
                    "run_energy_and_grad_save_param_states gradient");
                gradient_ms += elapsed_ms(gradient_start,
                                          std::chrono::steady_clock::now());
            }
        }
        if (reverse_index != 0) {
            const auto back_start = std::chrono::steady_clock::now();
            apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                             impl.scratch.get(), impl.num_qubits);
            if (measure_timings) {
                detail::check_cuda(cudaDeviceSynchronize(),
                                   "run_energy_and_grad_save_param_states back");
                back_ms += elapsed_ms(back_start, std::chrono::steady_clock::now());
            }
        }
    }

    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_save_param_states completion");
    }
    result.forward_ms = measure_timings ? elapsed_ms(forward_start, forward_end) : 0.0;
    result.back_ms = measure_timings ? back_ms : 0.0;
    result.gradient_ms = measure_timings ? gradient_ms : 0.0;
    result.total_ms =
        measure_timings ? elapsed_ms(total_start, std::chrono::steady_clock::now())
                        : 0.0;
    return result;
}

auto run_energy_and_grad_checkpointed(RingIsingCudaBackend::Impl &impl,
                                      const double *params,
                                      std::size_t num_params,
                                      bool measure_timings)
    -> EnergyGradResult {
    const auto total_start = std::chrono::steady_clock::now();
    validate_num_params(impl.expected_params, num_params);

    if (impl.strategy == GradientStrategy::BruteForceParallelQ6) {
        return run_dense_scan_energy_and_grad_fast(impl, params, num_params,
                                                   measure_timings);
    }

    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);
    const auto num_ops = ops.size();

    if (impl.strategy == GradientStrategy::SaveParamStates) {
        return run_energy_and_grad_save_param_states(impl, params, num_params,
                                                     measure_timings);
    }

    const auto forward_start = std::chrono::steady_clock::now();
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
    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_checkpointed forward");
    }
    const auto forward_end = std::chrono::steady_clock::now();

    EnergyGradResult result;
    result.energy = energy_complex.real();
    result.gradient.assign(impl.expected_params, 0.0);

    double gradient_ms = 0.0;
    double back_ms = 0.0;
    for (std::size_t chunk_idx = impl.num_chunks; chunk_idx-- > 0;) {
        const auto chunk_start = chunk_idx * impl.checkpoint_interval_ops;
        const auto chunk_end =
            std::min(chunk_start + impl.checkpoint_interval_ops, num_ops);
        const auto local_count = chunk_end - chunk_start;

        const auto back_recompute_start = std::chrono::steady_clock::now();
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
        if (measure_timings) {
            detail::check_cuda(cudaDeviceSynchronize(),
                               "run_energy_and_grad_checkpointed recompute");
            back_ms += elapsed_ms(back_recompute_start,
                                  std::chrono::steady_clock::now());
        }

        for (std::size_t op_idx = chunk_end; op_idx-- > chunk_start;) {
            const auto &op = ops[op_idx];
            const auto local_before = op_idx - chunk_start;
            if (op.is_parametric) {
                const auto gradient_start = std::chrono::steady_clock::now();
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
                if (measure_timings) {
                    detail::check_cuda(cudaDeviceSynchronize(),
                                       "run_energy_and_grad_checkpointed gradient");
                    gradient_ms += elapsed_ms(gradient_start,
                                              std::chrono::steady_clock::now());
                }
            }
            if (op_idx != 0) {
                const auto back_start = std::chrono::steady_clock::now();
                apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                                 impl.scratch.get(), impl.num_qubits);
                if (measure_timings) {
                    detail::check_cuda(cudaDeviceSynchronize(),
                                       "run_energy_and_grad_checkpointed back");
                    back_ms += elapsed_ms(back_start, std::chrono::steady_clock::now());
                }
            }
        }
    }

    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_checkpointed completion");
    }
    result.forward_ms = measure_timings ? elapsed_ms(forward_start, forward_end) : 0.0;
    result.back_ms = measure_timings ? back_ms : 0.0;
    result.gradient_ms = measure_timings ? gradient_ms : 0.0;
    result.total_ms =
        measure_timings ? elapsed_ms(total_start, std::chrono::steady_clock::now())
                        : 0.0;
    return result;
}

auto run_dense_scan_gpu_pipeline(RingIsingCudaBackend::Impl &impl,
                                 const DenseGateStorage &storage,
                                 bool copy_states_to_host)
    -> DenseScanGpuRunResult {
    const auto num_ops = storage.ops.size();
    const auto dim = impl.state_size;
    const auto mat_elements = dim * dim;
    const auto padded = next_power_of_two(num_ops);
    const auto num_params_local = impl.expected_params;

    if (!impl.dense_cublas) {
        impl.dense_cublas = std::make_unique<CublasHandle>();
    }
    auto &cublas = *impl.dense_cublas;

    auto ensure_alloc = [](auto &buffer, std::size_t required_size) {
        if (buffer.size() != required_size) {
            buffer.allocate(required_size);
        }
    };

    ensure_alloc(impl.dense_gate_mats, storage.gate_mats.size());
    ensure_alloc(impl.dense_dgate_mats, storage.dgate_mats.size());
    ensure_alloc(impl.dense_hamiltonian, mat_elements);
    ensure_alloc(impl.dense_param_gate_indices, storage.param_gate_indices.size());
    ensure_alloc(impl.dense_prefix_scan, padded * mat_elements);
    ensure_alloc(impl.dense_tmp_level, (padded / 2) * mat_elements);
    ensure_alloc(impl.dense_left_tmp, (padded / 2) * mat_elements);
    ensure_alloc(impl.dense_left_indices, padded / 2);
    ensure_alloc(impl.dense_right_indices, padded / 2);
    ensure_alloc(impl.dense_ptr_a, padded / 2);
    ensure_alloc(impl.dense_ptr_b, padded / 2);
    ensure_alloc(impl.dense_ptr_c, padded / 2);
    ensure_alloc(impl.dense_ones_vector, dim);
    ensure_alloc(impl.dense_psi_before, num_ops * dim);
    ensure_alloc(impl.dense_psi_after, num_ops * dim);
    ensure_alloc(impl.dense_forward_states, (num_ops + 1) * dim);
    ensure_alloc(impl.dense_lambda_k, dim);
    ensure_alloc(impl.dense_lambda_by_op, num_ops * dim);
    ensure_alloc(impl.dense_suffix_scan, padded * mat_elements);
    ensure_alloc(impl.dense_lambda_for_params, num_params_local * dim);
    ensure_alloc(impl.dense_psi_before_for_params, num_params_local * dim);
    ensure_alloc(impl.dense_deriv_states, num_params_local * dim);
    ensure_alloc(impl.dense_gradients, num_params_local);
    ensure_alloc(impl.dense_reverse_index, num_ops);
    ensure_alloc(impl.dense_reversed_tmp, num_ops * mat_elements);
    ensure_alloc(impl.dense_suffix_for_ops, num_ops * mat_elements);

    auto &gate_mats_dev = impl.dense_gate_mats;
    auto &dgate_mats_dev = impl.dense_dgate_mats;
    auto &hamiltonian_dev = impl.dense_hamiltonian;
    auto &param_gate_indices_dev = impl.dense_param_gate_indices;
    auto &prefix_scan_dev = impl.dense_prefix_scan;
    auto &tmp_level_dev = impl.dense_tmp_level;
    auto &left_tmp_dev = impl.dense_left_tmp;
    auto &left_indices_dev = impl.dense_left_indices;
    auto &right_indices_dev = impl.dense_right_indices;
    auto &ptr_a_dev = impl.dense_ptr_a;
    auto &ptr_b_dev = impl.dense_ptr_b;
    auto &ptr_c_dev = impl.dense_ptr_c;
    auto &ones_vector_dev = impl.dense_ones_vector;
    auto &psi_before_dev = impl.dense_psi_before;
    auto &psi_after_dev = impl.dense_psi_after;
    auto &forward_states_dev = impl.dense_forward_states;
    auto &lambda_k_dev = impl.dense_lambda_k;
    auto &lambda_by_op_dev = impl.dense_lambda_by_op;
    auto &suffix_scan_dev = impl.dense_suffix_scan;
    auto &lambda_for_params_dev = impl.dense_lambda_for_params;
    auto &psi_before_for_params_dev = impl.dense_psi_before_for_params;
    auto &deriv_states_dev = impl.dense_deriv_states;
    auto &gradients_dev = impl.dense_gradients;
    auto &reverse_index_dev = impl.dense_reverse_index;
    auto &reversed_tmp_dev = impl.dense_reversed_tmp;
    auto &suffix_for_ops_dev = impl.dense_suffix_for_ops;

    detail::check_cuda(cudaMemcpyAsync(gate_mats_dev.get(), storage.gate_mats.data(),
                                       sizeof(Complex) * storage.gate_mats.size(),
                                       cudaMemcpyHostToDevice, 0),
                       "copy gate mats");
    detail::check_cuda(cudaMemcpyAsync(dgate_mats_dev.get(), storage.dgate_mats.data(),
                                       sizeof(Complex) * storage.dgate_mats.size(),
                                       cudaMemcpyHostToDevice, 0),
                       "copy derivative mats");
    detail::check_cuda(
        cudaMemcpyAsync(param_gate_indices_dev.get(),
                        storage.param_gate_indices.data(),
                        sizeof(int) * storage.param_gate_indices.size(),
                        cudaMemcpyHostToDevice, 0),
        "copy param gate indices");

    if (impl.dense_hamiltonian_host.size() != mat_elements) {
        impl.dense_hamiltonian_host.assign(mat_elements, Complex(0.0, 0.0));
        fill_ring_ising_hamiltonian_matrix(impl.dense_hamiltonian_host.data(), dim,
                                           impl.num_qubits, impl.field);
    }
    detail::check_cuda(cudaMemcpyAsync(hamiltonian_dev.get(),
                                       impl.dense_hamiltonian_host.data(),
                                       sizeof(Complex) * mat_elements,
                                       cudaMemcpyHostToDevice, 0),
                       "copy hamiltonian");

    if (impl.dense_psi0_host.size() != dim) {
        impl.dense_psi0_host.assign(dim, Complex(0.0, 0.0));
        impl.dense_psi0_host[0] = Complex(1.0, 0.0);
    }
    detail::check_cuda(cudaMemcpyAsync(ones_vector_dev.get(),
                                       impl.dense_psi0_host.data(),
                                       sizeof(Complex) * dim,
                                       cudaMemcpyHostToDevice, 0),
                       "copy psi0");

    const auto gpu_start = std::chrono::steady_clock::now();
    const auto forward_start = std::chrono::steady_clock::now();

    detail::launch_fill_identity_matrices(prefix_scan_dev.get(), padded, dim);
    detail::check_cuda(
        cudaMemcpyAsync(prefix_scan_dev.get(), gate_mats_dev.get(),
                        sizeof(Complex) * num_ops * mat_elements,
                        cudaMemcpyDeviceToDevice, 0),
        "copy gate mats to scan buffer");
    run_noncommutative_exclusive_scan_mats(
        cublas.handle, prefix_scan_dev.get(), num_ops, dim, tmp_level_dev,
        left_tmp_dev, left_indices_dev, right_indices_dev, ptr_a_dev, ptr_b_dev,
        ptr_c_dev);

    {
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"), 1,
                to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(prefix_scan_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(ones_vector_dev.get()),
                to_int(dim, "dim"), 0,
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(psi_before_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(dim),
                to_int(num_ops, "num_ops")),
            "prefix->psi_before gemmStridedBatched");

        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"), 1,
                to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(gate_mats_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(psi_before_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(dim),
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(psi_after_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(dim),
                to_int(num_ops, "num_ops")),
            "gate*psi_before gemmStridedBatched");
    }

    detail::check_cuda(cudaMemcpyAsync(forward_states_dev.get(), ones_vector_dev.get(),
                                       sizeof(Complex) * dim,
                                       cudaMemcpyDeviceToDevice, 0),
                       "copy psi0 to forward states");
    detail::check_cuda(
        cudaMemcpyAsync(forward_states_dev.get() + dim, psi_after_dev.get(),
                        sizeof(Complex) * num_ops * dim, cudaMemcpyDeviceToDevice,
                        0),
        "copy psi_after chain");

    {
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemm(cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"),
                        1, to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&alpha),
                        reinterpret_cast<const cuDoubleComplex *>(hamiltonian_dev.get()),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(
                            forward_states_dev.get() + num_ops * dim),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&beta),
                        reinterpret_cast<cuDoubleComplex *>(lambda_k_dev.get()),
                        to_int(dim, "dim")),
            "hamiltonian * psi_k");
    }

    const Complex energy_complex =
        detail::complex_inner_product(forward_states_dev.get() + num_ops * dim,
                                      lambda_k_dev.get(), dim);
    detail::check_cuda(cudaDeviceSynchronize(), "dense scan forward");
    const auto forward_end = std::chrono::steady_clock::now();

    const auto back_start = std::chrono::steady_clock::now();
    detail::launch_build_adjoint_batch(gate_mats_dev.get(), suffix_scan_dev.get(),
                                       num_ops, dim);

    if (impl.dense_reverse_index_host.size() != num_ops) {
        impl.dense_reverse_index_host.resize(num_ops);
        for (std::size_t idx = 0; idx < num_ops; idx++) {
            impl.dense_reverse_index_host[idx] =
                to_int(num_ops - 1 - idx, "reverse index");
        }
    }
    detail::check_cuda(cudaMemcpyAsync(reverse_index_dev.get(),
                                       impl.dense_reverse_index_host.data(),
                                       sizeof(int) * num_ops, cudaMemcpyHostToDevice,
                                       0),
                       "copy reverse indices");
    detail::launch_gather_vectors(suffix_scan_dev.get(), reverse_index_dev.get(),
                                  reversed_tmp_dev.get(), num_ops, mat_elements);

    detail::launch_fill_identity_matrices(suffix_scan_dev.get(), padded, dim);
    detail::check_cuda(
        cudaMemcpyAsync(suffix_scan_dev.get(), reversed_tmp_dev.get(),
                        sizeof(Complex) * num_ops * mat_elements,
                        cudaMemcpyDeviceToDevice, 0),
        "copy reversed udag");
    run_noncommutative_exclusive_scan_mats(
        cublas.handle, suffix_scan_dev.get(), num_ops, dim, tmp_level_dev,
        left_tmp_dev, left_indices_dev, right_indices_dev, ptr_a_dev, ptr_b_dev,
        ptr_c_dev);

    detail::launch_gather_vectors(suffix_scan_dev.get(), reverse_index_dev.get(),
                                  suffix_for_ops_dev.get(), num_ops,
                                  mat_elements);

    {
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"), 1,
                to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(suffix_for_ops_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(lambda_k_dev.get()),
                to_int(dim, "dim"), 0,
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(lambda_by_op_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(dim),
                to_int(num_ops, "num_ops")),
            "suffix * lambda_k");
    }
    detail::check_cuda(cudaDeviceSynchronize(), "dense scan back");
    const auto back_end = std::chrono::steady_clock::now();

    const auto gradient_start = std::chrono::steady_clock::now();
    detail::launch_gather_vectors(lambda_by_op_dev.get(), param_gate_indices_dev.get(),
                                  lambda_for_params_dev.get(), num_params_local,
                                  dim);
    detail::launch_gather_vectors(psi_before_dev.get(), param_gate_indices_dev.get(),
                                  psi_before_for_params_dev.get(),
                                  num_params_local, dim);

    {
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"), 1,
                to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(dgate_mats_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(
                    psi_before_for_params_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(dim),
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(deriv_states_dev.get()),
                to_int(dim, "dim"), static_cast<long long>(dim),
                to_int(num_params_local, "num_params")),
            "dU * psi_before gemmStridedBatched");
    }

    detail::launch_reduce_real_inner_products(
        lambda_for_params_dev.get(), deriv_states_dev.get(), gradients_dev.get(),
        num_params_local, dim, 2.0);
    detail::check_cuda(cudaDeviceSynchronize(), "dense scan completion");

    const auto gpu_end = std::chrono::steady_clock::now();
    const double gpu_scan_ms =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::vector<double> gradients(num_params_local, 0.0);
    detail::check_cuda(
        cudaMemcpy(gradients.data(), gradients_dev.get(),
                   sizeof(double) * gradients.size(), cudaMemcpyDeviceToHost),
        "copy gradients to host");
    const auto gradient_end = std::chrono::steady_clock::now();

    DenseScanGpuRunResult result;
    result.energy = energy_complex.real();
    result.gradient = std::move(gradients);
    result.num_ops = num_ops;
    result.state_size = dim;
    result.gpu_scan_ms = gpu_scan_ms;
    result.forward_ms = elapsed_ms(forward_start, forward_end);
    result.back_ms = elapsed_ms(back_start, back_end);
    result.gradient_ms = elapsed_ms(gradient_start, gradient_end);
    result.total_ms = elapsed_ms(gpu_start, gradient_end);

    if (copy_states_to_host) {
        result.forward_states.assign((num_ops + 1) * dim, Complex(0.0, 0.0));
        result.backward_states.assign(num_ops * dim, Complex(0.0, 0.0));
        detail::check_cuda(cudaMemcpy(result.forward_states.data(), forward_states_dev.get(),
                                      sizeof(Complex) * result.forward_states.size(),
                                      cudaMemcpyDeviceToHost),
                           "copy forward states to host");
        detail::check_cuda(cudaMemcpy(result.backward_states.data(), lambda_by_op_dev.get(),
                                      sizeof(Complex) * result.backward_states.size(),
                                      cudaMemcpyDeviceToHost),
                           "copy backward states to host");
    }

    return result;
}

auto run_dense_scan_energy_and_grad_fast(RingIsingCudaBackend::Impl &impl,
                                         const double *params,
                                         std::size_t num_params,
                                         bool measure_timings)
    -> EnergyGradResult {
    if (impl.num_qubits > 6) {
        throw std::invalid_argument(
            "bruteforce_parallel_q6 only supports num_qubits <= 6.");
    }
    validate_num_params(impl.expected_params, num_params);

    const auto storage = build_dense_gate_storage(
        impl.num_qubits, impl.num_layers, impl.field, params,
        impl.fuse_ring_cnot_layer);
    auto gpu = run_dense_scan_gpu_pipeline(impl, storage, false);

    EnergyGradResult result;
    result.energy = gpu.energy;
    result.gradient = std::move(gpu.gradient);
    result.forward_ms = gpu.forward_ms;
    result.back_ms = gpu.back_ms;
    result.gradient_ms = gpu.gradient_ms;
    result.total_ms = gpu.total_ms;
    if (!measure_timings) {
        result.forward_ms = 0.0;
        result.back_ms = 0.0;
        result.gradient_ms = 0.0;
        result.total_ms = 0.0;
    }
    return result;
}

auto run_dense_scan_experiment(RingIsingCudaBackend::Impl &impl,
                               const double *params, std::size_t num_params)
    -> DenseScanExperimentResult {
    if (impl.num_qubits > 6) {
        throw std::invalid_argument(
            "bruteforce_parallel_q6 only supports num_qubits <= 6.");
    }
    validate_num_params(impl.expected_params, num_params);

    const auto storage = build_dense_gate_storage(
        impl.num_qubits, impl.num_layers, impl.field, params,
        impl.fuse_ring_cnot_layer);

    const auto cpu_start = std::chrono::steady_clock::now();
    std::vector<double> cpu_grad;
    std::vector<Complex> cpu_forward;
    std::vector<Complex> cpu_backward;
    double cpu_energy = 0.0;
    run_cpu_dense_reference(storage, impl.num_qubits, impl.field, &cpu_grad,
                            &cpu_forward, &cpu_backward, &cpu_energy);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_reference_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    const auto seq_start = std::chrono::steady_clock::now();
    const auto sequential_result =
        run_energy_and_grad_save_param_states(impl, params, num_params, false);
    const auto seq_end = std::chrono::steady_clock::now();
    const double sequential_statevector_ms =
        std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

    auto gpu = run_dense_scan_gpu_pipeline(impl, storage, true);

    DenseScanExperimentResult result;
    result.energy = gpu.energy;
    result.gradient = std::move(gpu.gradient);
    result.forward_states_ri = convert_states_to_ri(gpu.forward_states);
    result.backward_states_ri = convert_states_to_ri(gpu.backward_states);
    result.cpu_reference_ms = cpu_reference_ms;
    result.gpu_scan_ms = gpu.gpu_scan_ms;
    result.sequential_statevector_ms = sequential_statevector_ms;
    result.num_forward_states = gpu.num_ops + 1;
    result.num_backward_states = gpu.num_ops;
    result.state_size = gpu.state_size;

    const double grad_tolerance = 1.0e-7;
    for (std::size_t idx = 0; idx < result.gradient.size(); idx++) {
        if (std::abs(result.gradient[idx] - cpu_grad[idx]) > grad_tolerance) {
            throw std::runtime_error(
                "bruteforce_parallel_q6 gradient mismatch against CPU reference.");
        }
    }
    if (std::abs(result.energy - cpu_energy) > 1.0e-8) {
        throw std::runtime_error(
            "bruteforce_parallel_q6 energy mismatch against CPU reference.");
    }
    for (std::size_t idx = 0; idx < result.gradient.size(); idx++) {
        if (std::abs(result.gradient[idx] - sequential_result.gradient[idx]) >
            1.0e-7) {
            throw std::runtime_error(
                "bruteforce_parallel_q6 gradient mismatch against sequential statevector.");
        }
    }
    if (std::abs(result.energy - sequential_result.energy) > 1.0e-8) {
        throw std::runtime_error(
            "bruteforce_parallel_q6 energy mismatch against sequential statevector.");
    }

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
                                           std::size_t num_params,
                                           bool measure_timings)
    -> EnergyGradResult {
    return run_energy_and_grad_checkpointed(*impl_, params, num_params,
                                            measure_timings);
}

auto RingIsingCudaBackend::dense_scan_experiment(const double *params,
                                                 std::size_t num_params)
    -> DenseScanExperimentResult {
    return run_dense_scan_experiment(*impl_, params, num_params);
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
                                 std::size_t checkpoint_interval_ops,
                                 bool measure_timings) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 gradient_strategy,
                                 fuse_ring_cnot_layer,
                                 checkpoint_interval_ops);
    return backend.energy_and_grad(params, num_params, measure_timings);
}

DenseScanExperimentResult dense_scan_experiment(
    std::size_t num_qubits, std::size_t num_layers, double field,
    bool fuse_ring_cnot_layer, const double *params, std::size_t num_params) {
    RingIsingCudaBackend backend(num_qubits, num_layers, field,
                                 "bruteforce_parallel_q6",
                                 fuse_ring_cnot_layer, 0);
    return backend.dense_scan_experiment(params, num_params);
}

} // namespace standalone_backend
