void ensure_intrablock_buffers(RingIsingCudaBackend::Impl &impl) {
    const auto dim = impl.state_size;
    impl.intrablock_num_blocks =
        (impl.num_ops + impl.intrablock_block_size - 1) / impl.intrablock_block_size;

    auto ensure_alloc = [](auto &buffer, std::size_t required_size) {
        if (buffer.size() != required_size) {
            buffer.allocate(required_size);
        }
    };

    ensure_alloc(impl.intrablock_ops, impl.num_ops);
    ensure_alloc(impl.intrablock_boundary_states,
                 (impl.intrablock_num_blocks + 1) * dim);
    ensure_alloc(impl.intrablock_lambda_boundaries,
                 (impl.intrablock_num_blocks + 1) * dim);
    ensure_alloc(impl.intrablock_forward_states,
                 impl.intrablock_num_blocks * (impl.intrablock_block_size + 1) *
                     dim);
    ensure_alloc(impl.intrablock_gradients, impl.expected_params);
}

void propagate_intrablock_forward_boundaries(RingIsingCudaBackend::Impl &impl,
                                             const std::vector<OpDesc> &ops) {
    const auto dim = impl.state_size;
    detail::launch_init_zero_state(impl.current.get(), dim);
    copy_device_buffer(state_slot(impl.intrablock_boundary_states.get(), dim, 0),
                       impl.current.get(), dim);
    for (std::size_t block_idx = 0; block_idx < impl.intrablock_num_blocks; block_idx++) {
        const auto block_start = block_idx * impl.intrablock_block_size;
        const auto block_end =
            std::min(block_start + impl.intrablock_block_size, impl.num_ops);
        apply_ops_range_inplace(impl.current.get(), dim, ops, block_start, block_end,
                                impl.scratch.get(), impl.num_qubits);
        Complex *psi_boundary =
            state_slot(impl.intrablock_boundary_states.get(), dim, block_idx + 1);
        copy_device_buffer(psi_boundary, impl.current.get(), dim);
    }
}

void propagate_intrablock_backward_boundaries(RingIsingCudaBackend::Impl &impl,
                                              const std::vector<OpDesc> &ops) {
    const auto dim = impl.state_size;
    copy_device_buffer(impl.lambda.get(),
                       state_slot(impl.intrablock_lambda_boundaries.get(), dim,
                                  impl.intrablock_num_blocks),
                       dim);
    for (std::size_t block_idx = impl.intrablock_num_blocks; block_idx-- > 0;) {
        const auto block_start = block_idx * impl.intrablock_block_size;
        const auto block_end =
            std::min(block_start + impl.intrablock_block_size, impl.num_ops);
        for (std::size_t op_idx = block_end; op_idx-- > block_start;) {
            apply_op_inplace(impl.lambda.get(), dim, ops[op_idx], true,
                             impl.scratch.get(), impl.num_qubits);
        }
        Complex *lambda_boundary =
            state_slot(impl.intrablock_lambda_boundaries.get(), dim, block_idx);
        copy_device_buffer(lambda_boundary, impl.lambda.get(), dim);
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
    ensure_intrablock_buffers(impl);

    detail::check_cuda(cudaMemcpyAsync(impl.intrablock_ops.get(), ops.data(),
                                       sizeof(OpDesc) * ops.size(),
                                       cudaMemcpyHostToDevice, 0),
                       "copy intrablock ops");

    propagate_intrablock_forward_boundaries(impl, ops);

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

    propagate_intrablock_backward_boundaries(impl, ops);

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
