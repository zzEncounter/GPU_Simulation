void fill_zero_matrix(Complex *matrix, std::size_t dim) {
    std::fill(matrix, matrix + dim * dim, Complex(0.0, 0.0));
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

struct DenseGateStorage {
    std::vector<int> param_gate_indices;
    std::vector<OpDesc> ops;
    std::size_t gate_mats_size{0};
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

    const auto dim = std::size_t{1} << num_qubits;
    const auto mat_elements = dim * dim;
    const auto num_params = num_qubits * num_layers * 2;
    (void)fuse_ring_cnot_layer;

    storage->ops.clear();
    storage->ops.reserve(num_layers * 2);
    for (std::size_t layer = 0; layer < num_layers; layer++) {
        const auto param_base = layer * num_qubits * 2;
        storage->ops.push_back({OpKind::RotationLayer, 0, 0, 0.0, 0.0,
                                param_base, param_base + num_qubits * 2 - 1,
                                true});
        storage->ops.push_back(
            {OpKind::RingCNOTLayer, 0, 0, 0.0, 0.0, 0, 0, false});
    }

    storage->gate_mats_size = storage->ops.size() * mat_elements;
    storage->param_gate_indices.assign(num_params, -1);

    param_op_indices->clear();
    param_op_indices->reserve(num_layers);

    for (std::size_t op_index = 0; op_index < storage->ops.size(); op_index++) {
        const auto &op = storage->ops[op_index];
        if (op.is_parametric) {
            for (std::size_t local = 0; local < num_qubits * 2; local++) {
                storage->param_gate_indices[op.param_index0 + local] =
                    to_int(op_index, "op_index");
            }
            param_op_indices->push_back(op_index);
            continue;
        }
    }
}
