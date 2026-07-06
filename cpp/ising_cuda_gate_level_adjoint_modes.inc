void run_inverse_walk_backward_sweep(RingIsingCudaBackend::Impl &impl,
                                     const std::vector<OpDesc> &ops) {
    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        switch (op.kind) {
        case OpKind::RY:
            detail::launch_inverse_walk_ry_step(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.theta0,
                impl.gate_level_gradients.get() + op.param_index0);
            break;
        case OpKind::RZ:
            detail::launch_inverse_walk_rz_step(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.theta0,
                impl.gate_level_gradients.get() + op.param_index0);
            break;
        case OpKind::CNOT:
            detail::launch_inverse_walk_cnot_step(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.wire1);
            break;
        case OpKind::FusedRYRZ:
            detail::launch_inverse_walk_ryrz_step(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.theta0, op.theta1,
                impl.gate_level_gradients.get() + op.param_index0,
                impl.gate_level_gradients.get() + op.param_index1);
            break;
        case OpKind::RotationLayer:
            throw std::runtime_error(
                "inverse_walk backward sweep cannot consume RotationLayer ops.");
        case OpKind::RingCNOTLayer:
            throw std::runtime_error(
                "inverse_walk backward sweep cannot consume RingCNOTLayer ops.");
        }
    }
}

auto run_energy_and_grad_gate_level_adjoint(RingIsingCudaBackend::Impl &impl,
                                            const double *params,
                                            std::size_t num_params,
                                            bool compute_gradient,
                                            bool profile)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    StageProfiler profiler(profile);
    const auto ops = cpu_stage(
        profiler, "build_ops_host_ms",
        [&]() {
            return build_ops_for_strategy(impl.num_qubits, impl.num_layers,
                                          params, impl.strategy);
        });

    gpu_stage(profiler, "zero_state_ms", [&]() {
        detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    });
    gpu_stage(profiler, "forward_apply_ms", [&]() {
        for (const auto &op : ops) {
            apply_op_inplace(impl.current.get(), impl.state_size, op, false);
        }
    });
    const auto energy_partial_count =
        detail::hamiltonian_energy_partial_count(impl.state_size);
    gpu_stage(profiler, "hamiltonian_ms", [&]() {
        detail::launch_apply_hamiltonian_energy_partials(
            impl.lambda.get(), impl.current.get(), impl.scratch.get(),
            impl.state_size, impl.num_qubits, impl.field);
    });

    EnergyGradResult result;
    result.energy = gpu_stage(
        profiler, "energy_dot_ms",
        [&]() {
            return detail::sum_real_parts(impl.scratch.get(),
                                          energy_partial_count);
        });

    if (!compute_gradient) {
        result.stage_timings_ms = std::move(profiler.timings_ms);
        return result;
    }

    gpu_stage(profiler, "gradient_zero_ms", [&]() {
        detail::check_cuda(
            cudaMemsetAsync(impl.gate_level_gradients.get(), 0,
                            sizeof(double) * impl.expected_params, 0),
            "zero gate-level adjoint gradients");
    });
    gpu_stage(profiler, "backward_sweep_ms", [&]() {
        run_inverse_walk_backward_sweep(impl, ops);
    });

    result.gradient.assign(impl.expected_params, 0.0);
    gpu_stage(profiler, "gradient_copy_ms", [&]() {
        detail::check_cuda(
            cudaMemcpy(result.gradient.data(), impl.gate_level_gradients.get(),
                       sizeof(double) * impl.expected_params,
                       cudaMemcpyDeviceToHost),
            "copy gate-level adjoint gradients");
    });
    result.stage_timings_ms = std::move(profiler.timings_ms);
    return result;
}
