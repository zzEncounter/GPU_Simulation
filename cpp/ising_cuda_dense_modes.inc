void ensure_dense_scan_buffers(RingIsingCudaBackend::Impl &impl,
                               const DenseGateStorage &storage) {
    const auto num_ops = storage.ops.size();
    const auto dim = impl.state_size;
    const auto mat_elements = dim * dim;
    const auto padded = next_power_of_two(num_ops);

    auto ensure_alloc = [](auto &buffer, std::size_t required_size) -> bool {
        if (buffer.size() == required_size) {
            return false;
        }
        buffer.allocate(required_size);
        return true;
    };

    ensure_alloc(impl.dense_gate_mats, storage.gate_mats.size());
    ensure_alloc(impl.dense_dgate_mats, storage.dgate_mats.size());
    const bool hamiltonian_reallocated =
        ensure_alloc(impl.dense_hamiltonian, mat_elements);
    const bool param_indices_reallocated =
        ensure_alloc(impl.dense_param_gate_indices,
                     storage.param_gate_indices.size());
    const bool psi0_reallocated = ensure_alloc(impl.dense_ones_vector, dim);
    ensure_alloc(impl.dense_prefix_scan, padded * mat_elements);
    ensure_alloc(impl.dense_suffix_scan, padded > 1 ? (padded - 1) * mat_elements : 0);
    ensure_alloc(impl.dense_psi_before, padded * dim);
    ensure_alloc(impl.dense_psi_after, padded * dim);
    ensure_alloc(impl.dense_forward_states, std::max(num_ops + 1, padded) * dim);
    ensure_alloc(impl.dense_total_matrix, mat_elements);
    ensure_alloc(impl.dense_lambda_k, dim);
    ensure_alloc(impl.dense_eta_vector, dim);
    ensure_alloc(impl.dense_eta_before, padded * dim);
    ensure_alloc(impl.dense_gradients, impl.expected_params);

    if (hamiltonian_reallocated || param_indices_reallocated || psi0_reallocated) {
        impl.dense_static_device_uploaded = false;
    }
}

void upload_dense_scan_inputs(RingIsingCudaBackend::Impl &impl,
                              const DenseGateStorage &storage) {
    const auto dim = impl.state_size;
    const auto mat_elements = dim * dim;

    detail::check_cuda(cudaMemcpyAsync(impl.dense_gate_mats.get(),
                                       storage.gate_mats.data(),
                                       sizeof(Complex) * storage.gate_mats.size(),
                                       cudaMemcpyHostToDevice, 0),
                       "copy gate mats");
    detail::check_cuda(cudaMemcpyAsync(
                           impl.dense_dgate_mats.get(), storage.dgate_mats.data(),
                           sizeof(Complex) * storage.dgate_mats.size(),
                           cudaMemcpyHostToDevice, 0),
                       "copy derivative mats");

    if (!impl.dense_static_device_uploaded) {
        detail::check_cuda(
            cudaMemcpyAsync(impl.dense_param_gate_indices.get(),
                            storage.param_gate_indices.data(),
                            sizeof(int) * storage.param_gate_indices.size(),
                            cudaMemcpyHostToDevice, 0),
            "copy param gate indices");
    }

    if (impl.dense_hamiltonian_host.size() != mat_elements) {
        impl.dense_hamiltonian_host.assign(mat_elements, Complex(0.0, 0.0));
        fill_ring_ising_hamiltonian_matrix(impl.dense_hamiltonian_host.data(), dim,
                                           impl.num_qubits, impl.field);
    }
    if (!impl.dense_static_device_uploaded) {
        detail::check_cuda(cudaMemcpyAsync(
                               impl.dense_hamiltonian.get(),
                               impl.dense_hamiltonian_host.data(),
                               sizeof(Complex) * mat_elements,
                               cudaMemcpyHostToDevice, 0),
                           "copy hamiltonian");
    }

    if (impl.dense_psi0_host.size() != dim) {
        impl.dense_psi0_host.assign(dim, Complex(0.0, 0.0));
        impl.dense_psi0_host[0] = Complex(1.0, 0.0);
    }
    if (!impl.dense_static_device_uploaded) {
        detail::check_cuda(cudaMemcpyAsync(impl.dense_ones_vector.get(),
                                           impl.dense_psi0_host.data(),
                                           sizeof(Complex) * dim,
                                           cudaMemcpyHostToDevice, 0),
                           "copy psi0");
        impl.dense_static_device_uploaded = true;
    }
}

void build_dense_product_tree(RingIsingCudaBackend::Impl &impl,
                              std::size_t num_ops, std::size_t dim) {
    const auto mat_elements = dim * dim;
    const auto padded = next_power_of_two(num_ops);
    auto &cublas = *impl.dense_cublas;
    auto &leaf_storage_dev = impl.dense_prefix_scan;
    auto &internal_storage_dev = impl.dense_suffix_scan;
    auto &gate_mats_dev = impl.dense_gate_mats;

    detail::launch_fill_identity_matrices(leaf_storage_dev.get(), padded, dim);
    if (num_ops > 0) {
        detail::check_cuda(
            cudaMemcpyAsync(leaf_storage_dev.get(), gate_mats_dev.get(),
                            sizeof(Complex) * num_ops * mat_elements,
                            cudaMemcpyDeviceToDevice, 0),
            "copy gate mats to tree leaves");
    }

    std::size_t current_count = padded;
    std::size_t write_offset = 0;
    Complex *current_level = leaf_storage_dev.get();
    while (current_count > 1) {
        const auto parent_count = current_count / 2;
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        Complex *parent_level =
            internal_storage_dev.get() + write_offset * mat_elements;
        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"),
                to_int(dim, "dim"), to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(current_level + mat_elements),
                to_int(dim, "dim"),
                static_cast<long long>(2 * mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(current_level),
                to_int(dim, "dim"),
                static_cast<long long>(2 * mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(parent_level),
                to_int(dim, "dim"), static_cast<long long>(mat_elements),
                to_int(parent_count, "parent_count")),
            "dense product tree up-sweep");
        current_level = parent_level;
        current_count = parent_count;
        write_offset += parent_count;
    }
}

auto product_tree_level_ptr(RingIsingCudaBackend::Impl &impl, std::size_t padded,
                            std::size_t level_count) -> Complex * {
    if (level_count == padded) {
        return impl.dense_prefix_scan.get();
    }
    return impl.dense_suffix_scan.get() + (padded - 2 * level_count) *
                                              (impl.state_size * impl.state_size);
}

void run_dense_vector_downsweep(RingIsingCudaBackend::Impl &impl,
                                std::size_t num_ops, std::size_t dim) {
    const auto padded = next_power_of_two(num_ops);
    const auto mat_elements = dim * dim;
    auto &cublas = *impl.dense_cublas;
    auto &ones_vector_dev = impl.dense_ones_vector;
    auto &eta_dev = impl.dense_eta_vector;
    auto &x_current_dev = impl.dense_psi_before;
    auto &x_next_dev = impl.dense_psi_after;
    auto &y_current_dev = impl.dense_eta_before;
    auto &y_next_dev = impl.dense_forward_states;

    detail::check_cuda(cudaMemcpyAsync(x_current_dev.get(), ones_vector_dev.get(),
                                       sizeof(Complex) * dim,
                                       cudaMemcpyDeviceToDevice, 0),
                       "seed x root");
    detail::check_cuda(cudaMemcpyAsync(y_current_dev.get(), eta_dev.get(),
                                       sizeof(Complex) * dim,
                                       cudaMemcpyDeviceToDevice, 0),
                       "seed y root");

    std::size_t current_count = 1;
    Complex *x_current = x_current_dev.get();
    Complex *x_next = x_next_dev.get();
    Complex *y_current = y_current_dev.get();
    Complex *y_next = y_next_dev.get();

    while (current_count < padded) {
        const auto child_count = current_count * 2;
        Complex *left_products = product_tree_level_ptr(impl, padded, child_count);
        detail::launch_scatter_parent_vectors(x_current, x_next, current_count,
                                              dim);
        detail::launch_scatter_parent_vectors(y_current, y_next, current_count,
                                              dim);

        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"), 1,
                to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(left_products),
                to_int(dim, "dim"),
                static_cast<long long>(2 * mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(x_current),
                to_int(dim, "dim"), static_cast<long long>(dim),
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(x_next + dim),
                to_int(dim, "dim"), static_cast<long long>(2 * dim),
                to_int(current_count, "current_count")),
            "dense x downsweep");
        detail::check_cublas(
            cublasZgemmStridedBatched(
                cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"), 1,
                to_int(dim, "dim"),
                reinterpret_cast<const cuDoubleComplex *>(&alpha),
                reinterpret_cast<const cuDoubleComplex *>(left_products),
                to_int(dim, "dim"),
                static_cast<long long>(2 * mat_elements),
                reinterpret_cast<const cuDoubleComplex *>(y_current),
                to_int(dim, "dim"), static_cast<long long>(dim),
                reinterpret_cast<const cuDoubleComplex *>(&beta),
                reinterpret_cast<cuDoubleComplex *>(y_next + dim),
                to_int(dim, "dim"), static_cast<long long>(2 * dim),
                to_int(current_count, "current_count")),
            "dense y downsweep");

        current_count = child_count;
        std::swap(x_current, x_next);
        std::swap(y_current, y_next);
    }

    if (x_current != x_current_dev.get()) {
        detail::check_cuda(
            cudaMemcpyAsync(x_current_dev.get(), x_current,
                            sizeof(Complex) * padded * dim,
                            cudaMemcpyDeviceToDevice, 0),
            "copy downsweep x leaves");
    }
    if (y_current != y_current_dev.get()) {
        detail::check_cuda(
            cudaMemcpyAsync(y_current_dev.get(), y_current,
                            sizeof(Complex) * padded * dim,
                            cudaMemcpyDeviceToDevice, 0),
            "copy downsweep y leaves");
    }
}

void run_dense_gradient_tail(RingIsingCudaBackend::Impl &impl,
                             std::size_t num_params_local,
                             std::size_t dim) {
    detail::launch_fused_dense_gradient_tail(
        impl.dense_gate_mats.get(), impl.dense_dgate_mats.get(),
        impl.dense_param_gate_indices.get(), impl.dense_psi_before.get(),
        impl.dense_eta_before.get(), impl.dense_gradients.get(), num_params_local,
        dim);
}

auto run_dense_scan_gpu_pipeline(RingIsingCudaBackend::Impl &impl,
                                 const DenseGateStorage &storage)
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

    ensure_dense_scan_buffers(impl, storage);
    upload_dense_scan_inputs(impl, storage);

    build_dense_product_tree(impl, num_ops, dim);
    Complex *root_matrix = product_tree_level_ptr(impl, padded, 1);
    detail::check_cuda(cudaMemcpyAsync(impl.dense_total_matrix.get(), root_matrix,
                                       sizeof(Complex) * mat_elements,
                                       cudaMemcpyDeviceToDevice, 0),
                       "copy tree root to total matrix");

    {
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemm(cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"),
                        1, to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&alpha),
                        reinterpret_cast<const cuDoubleComplex *>(
                            impl.dense_total_matrix.get()),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(
                            impl.dense_ones_vector.get()),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&beta),
                        reinterpret_cast<cuDoubleComplex *>(
                            impl.dense_forward_states.get() + num_ops * dim),
                        to_int(dim, "dim")),
            "tree psi_final = T * psi0");

        detail::check_cublas(
            cublasZgemm(cublas.handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(dim, "dim"),
                        1, to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&alpha),
                        reinterpret_cast<const cuDoubleComplex *>(
                            impl.dense_hamiltonian.get()),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(
                            impl.dense_forward_states.get() + num_ops * dim),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&beta),
                        reinterpret_cast<cuDoubleComplex *>(impl.dense_lambda_k.get()),
                        to_int(dim, "dim")),
            "tree hamiltonian * psi_final");
    }

    const Complex energy_complex =
        detail::complex_inner_product(impl.dense_forward_states.get() + num_ops * dim,
                                      impl.dense_lambda_k.get(), dim);
    {
        const Complex alpha(1.0, 0.0);
        const Complex beta(0.0, 0.0);
        detail::check_cublas(
            cublasZgemm(cublas.handle, CUBLAS_OP_C, CUBLAS_OP_N, to_int(dim, "dim"),
                        1, to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&alpha),
                        reinterpret_cast<const cuDoubleComplex *>(
                            impl.dense_total_matrix.get()),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(
                            impl.dense_lambda_k.get()),
                        to_int(dim, "dim"),
                        reinterpret_cast<const cuDoubleComplex *>(&beta),
                        reinterpret_cast<cuDoubleComplex *>(impl.dense_eta_vector.get()),
                        to_int(dim, "dim")),
            "tree eta = T^H * lambda_k");
    }
    run_dense_vector_downsweep(impl, num_ops, dim);
    run_dense_gradient_tail(impl, num_params_local, dim);

    std::vector<double> gradients(num_params_local, 0.0);
    detail::check_cuda(
        cudaMemcpy(gradients.data(), impl.dense_gradients.get(),
                   sizeof(double) * gradients.size(), cudaMemcpyDeviceToHost),
        "copy gradients to host");

    DenseScanGpuRunResult result;
    result.energy = energy_complex.real();
    result.gradient = std::move(gradients);
    return result;
}

auto run_dense_scan_energy_and_grad_fast(RingIsingCudaBackend::Impl &impl,
                                         const double *params,
                                         std::size_t num_params)
    -> EnergyGradResult {
    if (impl.num_qubits > 6) {
        throw std::invalid_argument("dense_scan only supports num_qubits <= 6.");
    }
    validate_num_params(impl.expected_params, num_params);

    ensure_dense_gate_storage_layout(&impl.dense_storage_cache,
                                     &impl.dense_param_op_indices,
                                     impl.num_qubits, impl.num_layers,
                                     impl.fuse_ring_cnot_layer);
    update_dense_gate_storage_params(&impl.dense_storage_cache,
                                     impl.dense_param_op_indices,
                                     impl.num_qubits, params);
    auto gpu = run_dense_scan_gpu_pipeline(impl, impl.dense_storage_cache);

    EnergyGradResult result;
    result.energy = gpu.energy;
    result.gradient = std::move(gpu.gradient);
    return result;
}
