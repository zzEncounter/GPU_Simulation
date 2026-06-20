auto run_energy_only(RingIsingCudaBackend::Impl &impl, const double *params,
                     std::size_t num_params) -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ops_for_strategy(impl.num_qubits, impl.num_layers,
                                            params, impl.strategy,
                                            impl.fuse_ring_cnot_layer);

    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (const auto &op : ops) {
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }
    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);

    EnergyGradResult result;
    result.energy =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size)
            .real();
    return result;
}

auto run_energy_and_grad_inverse_walk(RingIsingCudaBackend::Impl &impl,
                                      const double *params,
                                      std::size_t num_params,
                                      bool compute_gradient)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_pennylane_gate_level_ops(impl.num_qubits,
                                                    impl.num_layers, params);
    const bool record_phase_timings =
        compute_gradient && impl.strategy == GradientStrategy::InverseWalk;

    const auto forward_start =
        record_phase_timings ? Clock::now() : Clock::time_point{};
    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (const auto &op : ops) {
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }
    const auto forward_end =
        record_phase_timings
            ? sync_clock_now("inverse_walk forward timing sync")
            : Clock::time_point{};

    const auto backward_start =
        record_phase_timings ? Clock::now() : Clock::time_point{};
    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);

    EnergyGradResult result;
    result.energy =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size)
            .real();
    const auto backward_after_energy =
        record_phase_timings
            ? sync_clock_now("inverse_walk energy timing sync")
            : Clock::time_point{};

    if (!compute_gradient) {
        return result;
    }

    detail::check_cuda(
        cudaMemsetAsync(impl.gate_level_gradients.get(), 0,
                        sizeof(double) * impl.expected_params, 0),
        "zero inverse_walk gradients");
    double backward_s =
        record_phase_timings
            ? elapsed_seconds(backward_start, backward_after_energy)
            : 0.0;
    double gradient_s = 0.0;
    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        switch (op.kind) {
        case OpKind::RY:
        case OpKind::RZ: {
            const auto gradient_start =
                record_phase_timings ? Clock::now() : Clock::time_point{};
            if (op.kind == OpKind::RY) {
                detail::launch_inverse_walk_ry_step(
                    impl.current.get(), impl.lambda.get(), impl.state_size,
                    op.wire0, op.theta0,
                    impl.gate_level_gradients.get() + op.param_index0);
            } else {
                detail::launch_inverse_walk_rz_step(
                    impl.current.get(), impl.lambda.get(), impl.state_size,
                    op.wire0, op.theta0,
                    impl.gate_level_gradients.get() + op.param_index0);
            }
            if (record_phase_timings) {
                const auto gradient_end =
                    sync_clock_now("inverse_walk gradient timing sync");
                gradient_s += elapsed_seconds(gradient_start, gradient_end);
            }
            break;
        }
        case OpKind::CNOT: {
            const auto backward_step_start =
                record_phase_timings ? Clock::now() : Clock::time_point{};
            detail::launch_inverse_walk_cnot_step(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.wire1);
            if (record_phase_timings) {
                const auto backward_step_end =
                    sync_clock_now("inverse_walk backward timing sync");
                backward_s +=
                    elapsed_seconds(backward_step_start, backward_step_end);
            }
            break;
        }
        case OpKind::FusedRYRZ:
        case OpKind::RingCNOTLayer:
            throw std::runtime_error(
                "inverse_walk expects PennyLane-style gate-level ops.");
        }
    }

    result.gradient.assign(impl.expected_params, 0.0);
    detail::check_cuda(
        cudaMemcpy(result.gradient.data(), impl.gate_level_gradients.get(),
                   sizeof(double) * impl.expected_params,
                   cudaMemcpyDeviceToHost),
        "copy inverse_walk gradients");

    if (record_phase_timings) {
        result.timings_s.emplace_back(
            "forward_s", elapsed_seconds(forward_start, forward_end));
        result.timings_s.emplace_back("backward_s", backward_s);
        result.timings_s.emplace_back("gradient_s", gradient_s);
    }

    return result;
}

auto run_energy_and_grad_save_param_states(
    RingIsingCudaBackend::Impl &impl, const double *params,
    std::size_t num_params, bool compute_gradient) -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_pennylane_gate_level_ops(impl.num_qubits,
                                                    impl.num_layers, params);
    const bool record_phase_timings =
        compute_gradient && impl.strategy == GradientStrategy::SaveParamStates;

    const auto forward_start = record_phase_timings ? Clock::now() : Clock::time_point{};
    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (std::size_t op_index = 0; op_index < ops.size(); op_index++) {
        const auto &op = ops[op_index];
        if (compute_gradient && op.is_parametric) {
            copy_device_buffer(
                state_slot(impl.save_param_states.get(), impl.state_size,
                           op.param_index0),
                impl.current.get(), impl.state_size);
        }
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }
    const auto forward_end =
        record_phase_timings
            ? sync_clock_now("save_param_states forward timing sync")
            : Clock::time_point{};

    const auto backward_start =
        record_phase_timings ? Clock::now() : Clock::time_point{};
    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);

    EnergyGradResult result;
    result.energy =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size)
            .real();
    const auto backward_after_energy =
        record_phase_timings
            ? sync_clock_now("save_param_states energy timing sync")
            : Clock::time_point{};

    if (!compute_gradient) {
        return result;
    }

    detail::check_cuda(
        cudaMemsetAsync(impl.gate_level_gradients.get(), 0,
                        sizeof(double) * impl.expected_params, 0),
        "zero save_param gradients");
    double backward_s =
        record_phase_timings
            ? elapsed_seconds(backward_start, backward_after_energy)
            : 0.0;
    double gradient_s = 0.0;
    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        switch (op.kind) {
        case OpKind::RY:
        case OpKind::RZ: {
            const auto gradient_start =
                record_phase_timings ? Clock::now() : Clock::time_point{};
            if (op.kind == OpKind::RY) {
                detail::launch_save_param_ry_step(
                    state_slot(impl.save_param_states.get(), impl.state_size,
                               op.param_index0),
                    impl.lambda.get(), impl.state_size, op.wire0, op.theta0,
                    impl.gate_level_gradients.get() + op.param_index0);
            } else {
                detail::launch_save_param_rz_step(
                    state_slot(impl.save_param_states.get(), impl.state_size,
                               op.param_index0),
                    impl.lambda.get(), impl.state_size, op.wire0, op.theta0,
                    impl.gate_level_gradients.get() + op.param_index0);
            }
            if (record_phase_timings) {
                const auto gradient_end =
                    sync_clock_now("save_param_states gradient timing sync");
                gradient_s += elapsed_seconds(gradient_start, gradient_end);
            }
            break;
        }
        case OpKind::CNOT: {
            const auto backward_step_start =
                record_phase_timings ? Clock::now() : Clock::time_point{};
            detail::launch_apply_cnot(impl.lambda.get(), impl.state_size,
                                      op.wire0, op.wire1);
            if (record_phase_timings) {
                const auto backward_step_end =
                    sync_clock_now("save_param_states backward timing sync");
                backward_s +=
                    elapsed_seconds(backward_step_start, backward_step_end);
            }
            break;
        }
        case OpKind::FusedRYRZ:
        case OpKind::RingCNOTLayer:
            throw std::runtime_error(
                "save_param_states expects PennyLane-style gate-level ops.");
        }
    }

    result.gradient.assign(impl.expected_params, 0.0);
    detail::check_cuda(
        cudaMemcpy(result.gradient.data(), impl.gate_level_gradients.get(),
                   sizeof(double) * impl.expected_params,
                   cudaMemcpyDeviceToHost),
        "copy save_param gradients");

    if (record_phase_timings) {
        result.timings_s.emplace_back(
            "forward_s", elapsed_seconds(forward_start, forward_end));
        result.timings_s.emplace_back("backward_s", backward_s);
        result.timings_s.emplace_back("gradient_s", gradient_s);
    }

    return result;
}

auto run_energy_and_grad_checkpointed(RingIsingCudaBackend::Impl &impl,
                                      const double *params,
                                      std::size_t num_params,
                                      bool compute_gradient)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);

    if (!compute_gradient) {
        return run_energy_only(impl, params, num_params);
    }

    if (impl.strategy == GradientStrategy::DenseScan) {
        return run_dense_scan_energy_and_grad_fast(impl, params, num_params);
    }
    if (impl.strategy == GradientStrategy::InverseWalk) {
        return run_energy_and_grad_inverse_walk(impl, params, num_params,
                                                compute_gradient);
    }
    if (impl.strategy == GradientStrategy::IntrablockParallel) {
        return run_intrablock_parallel_energy_and_grad(impl, params, num_params,
                                                       compute_gradient);
    }

    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);
    const auto num_ops = ops.size();

    if (impl.strategy == GradientStrategy::SaveParamStates) {
        return run_energy_and_grad_save_param_states(impl, params, num_params,
                                                     true);
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

    EnergyGradResult result;
    result.energy =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size)
            .real();
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

    return result;
}
