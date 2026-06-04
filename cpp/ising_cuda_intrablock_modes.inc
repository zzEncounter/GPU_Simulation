void ensure_intrablock_buffers(RingIsingCudaBackend::Impl &impl) {
    const auto dim = impl.state_size;
    const auto mat_elements = dim * dim;
    impl.intrablock_num_blocks =
        (impl.num_ops + impl.intrablock_block_size - 1) / impl.intrablock_block_size;

    auto ensure_alloc = [](auto &buffer, std::size_t required_size) {
        if (buffer.size() != required_size) {
            buffer.allocate(required_size);
        }
    };

    ensure_alloc(impl.intrablock_ops, impl.num_ops);
    ensure_alloc(impl.intrablock_block_mats,
                 impl.intrablock_num_blocks * mat_elements);
    ensure_alloc(impl.intrablock_block_mat_scratch,
                 impl.intrablock_num_blocks * mat_elements);
    ensure_alloc(impl.intrablock_boundary_states,
                 (impl.intrablock_num_blocks + 1) * dim);
    ensure_alloc(impl.intrablock_lambda_boundaries,
                 (impl.intrablock_num_blocks + 1) * dim);
    ensure_alloc(impl.intrablock_forward_states,
                 impl.intrablock_num_blocks * (impl.intrablock_block_size + 1) *
                     dim);
    ensure_alloc(impl.intrablock_gradients, impl.expected_params);
}

void build_intrablock_block_matrices(RingIsingCudaBackend::Impl &impl) {
    detail::launch_build_block_matrices(
        impl.intrablock_ops.get(), impl.intrablock_num_blocks,
        impl.intrablock_block_size, impl.num_ops, impl.num_qubits, impl.state_size,
        impl.intrablock_block_mats.get(), impl.intrablock_block_mat_scratch.get());
}

void apply_block_matrix(RingIsingCudaBackend::Impl &impl, const Complex *block_mat,
                        const Complex *input_state, Complex *output_state,
                        bool adjoint) {
    auto &cublas = *impl.dense_cublas;
    const auto dim = impl.state_size;
    const Complex alpha(1.0, 0.0);
    const Complex beta(0.0, 0.0);
    detail::check_cublas(
        cublasZgemm(cublas.handle, adjoint ? CUBLAS_OP_C : CUBLAS_OP_N,
                    CUBLAS_OP_N, to_int(dim, "dim"), 1, to_int(dim, "dim"),
                    reinterpret_cast<const cuDoubleComplex *>(&alpha),
                    reinterpret_cast<const cuDoubleComplex *>(block_mat),
                    to_int(dim, "dim"),
                    reinterpret_cast<const cuDoubleComplex *>(input_state),
                    to_int(dim, "dim"),
                    reinterpret_cast<const cuDoubleComplex *>(&beta),
                    reinterpret_cast<cuDoubleComplex *>(output_state),
                    to_int(dim, "dim")),
        adjoint ? "intrablock G^H * lambda" : "intrablock G * psi");
}

void propagate_intrablock_forward_boundaries(RingIsingCudaBackend::Impl &impl) {
    const auto dim = impl.state_size;
    const auto mat_elements = dim * dim;
    detail::launch_init_zero_state(impl.intrablock_boundary_states.get(), dim);
    for (std::size_t block_idx = 0; block_idx < impl.intrablock_num_blocks; block_idx++) {
        const Complex *block_mat =
            impl.intrablock_block_mats.get() + block_idx * mat_elements;
        const Complex *psi_start =
            state_slot(impl.intrablock_boundary_states.get(), dim, block_idx);
        Complex *psi_end =
            state_slot(impl.intrablock_boundary_states.get(), dim, block_idx + 1);
        apply_block_matrix(impl, block_mat, psi_start, psi_end, false);
    }
}

void propagate_intrablock_backward_boundaries(RingIsingCudaBackend::Impl &impl) {
    const auto dim = impl.state_size;
    const auto mat_elements = dim * dim;
    for (std::size_t block_idx = impl.intrablock_num_blocks; block_idx-- > 0;) {
        const Complex *block_mat =
            impl.intrablock_block_mats.get() + block_idx * mat_elements;
        const Complex *lambda_end =
            state_slot(impl.intrablock_lambda_boundaries.get(), dim, block_idx + 1);
        Complex *lambda_start =
            state_slot(impl.intrablock_lambda_boundaries.get(), dim, block_idx);
        apply_block_matrix(impl, block_mat, lambda_end, lambda_start, true);
    }
}

auto run_intrablock_parallel_energy_and_grad(RingIsingCudaBackend::Impl &impl,
                                             const double *params,
                                             std::size_t num_params,
                                             bool compute_gradient)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    if (!compute_gradient) {
        return run_energy_only(impl, params, num_params);
    }

    const auto ops =
        build_ring_ops(impl.num_qubits, impl.num_layers, params, impl.fuse_ring_cnot_layer);
    if (ops.size() != impl.num_ops) {
        throw std::runtime_error("intrablock_parallel op count mismatch.");
    }
    if (!impl.dense_cublas) {
        impl.dense_cublas = std::make_unique<CublasHandle>();
    }
    ensure_intrablock_buffers(impl);

    detail::check_cuda(cudaMemcpyAsync(impl.intrablock_ops.get(), ops.data(),
                                       sizeof(OpDesc) * ops.size(),
                                       cudaMemcpyHostToDevice, 0),
                       "copy intrablock ops");

    build_intrablock_block_matrices(impl);
    propagate_intrablock_forward_boundaries(impl);

    const auto dim = impl.state_size;
    Complex *psi_final =
        state_slot(impl.intrablock_boundary_states.get(), dim, impl.intrablock_num_blocks);
    Complex *lambda_final =
        state_slot(impl.intrablock_lambda_boundaries.get(), dim, impl.intrablock_num_blocks);
    detail::launch_apply_hamiltonian(lambda_final, psi_final, dim, impl.num_qubits,
                                     impl.field);

    EnergyGradResult result;
    result.energy = detail::complex_inner_product(psi_final, lambda_final, dim).real();

    detail::launch_simulate_blocks_forward(
        impl.intrablock_ops.get(), impl.intrablock_num_blocks,
        impl.intrablock_block_size, impl.num_ops, impl.num_qubits, dim,
        impl.intrablock_boundary_states.get(), impl.intrablock_forward_states.get());

    propagate_intrablock_backward_boundaries(impl);

    detail::check_cuda(cudaMemsetAsync(impl.intrablock_gradients.get(), 0,
                                       sizeof(double) * impl.expected_params, 0),
                       "zero intrablock gradients");
    detail::launch_simulate_blocks_backward_and_gradient(
        impl.intrablock_ops.get(), impl.intrablock_num_blocks,
        impl.intrablock_block_size, impl.num_ops, impl.num_qubits, dim,
        impl.intrablock_lambda_boundaries.get(), impl.intrablock_forward_states.get(),
        impl.intrablock_gradients.get());

    result.gradient.assign(impl.expected_params, 0.0);
    detail::check_cuda(
        cudaMemcpy(result.gradient.data(), impl.intrablock_gradients.get(),
                   sizeof(double) * result.gradient.size(), cudaMemcpyDeviceToHost),
        "copy intrablock gradients");
    return result;
}
