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
