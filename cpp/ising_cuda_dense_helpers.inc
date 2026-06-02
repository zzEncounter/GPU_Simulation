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

void ensure_dense_gate_storage_layout(DenseGateStorage *storage,
                                      std::vector<std::size_t> *param_op_indices,
                                      std::size_t num_qubits,
                                      std::size_t num_layers,
                                      bool fuse_ring_cnot_layer) {
    if (storage == nullptr || param_op_indices == nullptr) {
        throw std::invalid_argument(
            "ensure_dense_gate_storage_layout requires non-null pointers.");
    }
    if (!storage->ops.empty()) {
        return;
    }

    const auto ops =
        build_ring_ops(num_qubits, num_layers, nullptr, fuse_ring_cnot_layer);
    const auto dim = std::size_t{1} << num_qubits;
    const auto mat_elements = dim * dim;
    const auto num_params = num_qubits * num_layers * 2;

    storage->ops = ops;
    storage->gate_mats.assign(ops.size() * mat_elements, Complex(0.0, 0.0));
    storage->dgate_mats.assign(num_params * mat_elements, Complex(0.0, 0.0));
    storage->param_gate_indices.assign(num_params, -1);

    param_op_indices->clear();
    param_op_indices->reserve(num_params / 2);

    for (std::size_t op_index = 0; op_index < ops.size(); op_index++) {
        const auto &op = ops[op_index];
        if (op.is_parametric) {
            storage->param_gate_indices[op.param_index0] =
                to_int(op_index, "op_index");
            storage->param_gate_indices[op.param_index1] =
                to_int(op_index, "op_index");
            param_op_indices->push_back(op_index);
            continue;
        }
        auto *u = storage->gate_mats.data() + op_index * mat_elements;
        fill_dense_gate_matrix_for_op(op, num_qubits, dim, u, nullptr, nullptr);
    }
}

void update_dense_gate_storage_params(
    DenseGateStorage *storage, const std::vector<std::size_t> &param_op_indices,
    std::size_t num_qubits, const double *params) {
    if (storage == nullptr || params == nullptr) {
        throw std::invalid_argument(
            "update_dense_gate_storage_params requires non-null inputs.");
    }

    const auto dim = std::size_t{1} << num_qubits;
    const auto mat_elements = dim * dim;
    for (const auto op_index : param_op_indices) {
        auto &op = storage->ops[op_index];
        op.theta0 = params[op.param_index0];
        op.theta1 = params[op.param_index1];
        auto *u = storage->gate_mats.data() + op_index * mat_elements;
        auto *dtheta =
            storage->dgate_mats.data() + op.param_index0 * mat_elements;
        auto *dphi = storage->dgate_mats.data() + op.param_index1 * mat_elements;
        fill_fused_ryrz_matrices(u, dtheta, dphi, dim, op.wire0, op.theta0,
                                 op.theta1);
    }
}

auto build_dense_gate_storage(std::size_t num_qubits, std::size_t num_layers,
                              double field, const double *params,
                              bool fuse_ring_cnot_layer)
    -> DenseGateStorage {
    (void)field;
    DenseGateStorage storage;
    std::vector<std::size_t> param_op_indices;
    ensure_dense_gate_storage_layout(&storage, &param_op_indices, num_qubits,
                                     num_layers, fuse_ring_cnot_layer);
    update_dense_gate_storage_params(&storage, param_op_indices, num_qubits,
                                     params);
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
