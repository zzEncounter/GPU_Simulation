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
        run_energy_and_grad_save_param_states(impl, params, num_params, false,
                                              true);
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
