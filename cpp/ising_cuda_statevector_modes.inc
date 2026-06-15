auto run_energy_only(RingIsingCudaBackend::Impl &impl, const double *params,
                     std::size_t num_params, bool profile = false)
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
            apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                             impl.scratch.get(), impl.num_qubits);
        }
    });
    gpu_stage(profiler, "hamiltonian_ms", [&]() {
        detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                         impl.state_size, impl.num_qubits,
                                         impl.field);
    });

    EnergyGradResult result;
    result.energy = gpu_stage(
        profiler, "energy_dot_ms",
        [&]() {
            return detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                                 impl.state_size)
                .real();
        });
    result.stage_timings_ms = std::move(profiler.timings_ms);
    return result;
}

auto run_energy_and_grad_inverse_walk(RingIsingCudaBackend::Impl &impl,
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
            return build_pennylane_gate_level_ops(impl.num_qubits,
                                                  impl.num_layers, params);
        });

    gpu_stage(profiler, "zero_state_ms", [&]() {
        detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    });
    gpu_stage(profiler, "forward_apply_ms", [&]() {
        for (const auto &op : ops) {
            apply_op_inplace(impl.current.get(), impl.state_size, op, false);
        }
    });
    gpu_stage(profiler, "hamiltonian_ms", [&]() {
        detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                         impl.state_size, impl.num_qubits,
                                         impl.field);
    });

    EnergyGradResult result;
    result.energy = gpu_stage(
        profiler, "energy_dot_ms",
        [&]() {
            return detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                                 impl.state_size)
                .real();
        });

    if (!compute_gradient) {
        result.stage_timings_ms = std::move(profiler.timings_ms);
        return result;
    }

    gpu_stage(profiler, "gradient_zero_ms", [&]() {
        detail::check_cuda(
            cudaMemsetAsync(impl.gate_level_gradients.get(), 0,
                            sizeof(double) * impl.expected_params, 0),
            "zero inverse_walk gradients");
    });
    gpu_stage(profiler, "backward_sweep_ms", [&]() {
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
            case OpKind::RingCNOTLayer:
                throw std::runtime_error(
                    "inverse_walk expects PennyLane-style gate-level ops.");
            }
        }
    });

    result.gradient.assign(impl.expected_params, 0.0);
    gpu_stage(profiler, "gradient_copy_ms", [&]() {
        detail::check_cuda(
            cudaMemcpy(result.gradient.data(), impl.gate_level_gradients.get(),
                       sizeof(double) * impl.expected_params,
                       cudaMemcpyDeviceToHost),
            "copy inverse_walk gradients");
    });
    result.stage_timings_ms = std::move(profiler.timings_ms);
    return result;
}

auto run_energy_and_grad_save_param_states(
    RingIsingCudaBackend::Impl &impl, const double *params,
    std::size_t num_params, bool compute_gradient, bool profile)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    StageProfiler profiler(profile);
    const auto ops = cpu_stage(
        profiler, "build_ops_host_ms",
        [&]() {
            return build_pennylane_gate_level_ops(impl.num_qubits,
                                                  impl.num_layers, params);
        });

    gpu_stage(profiler, "zero_state_ms", [&]() {
        detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    });
    gpu_stage(profiler, "forward_apply_and_save_ms", [&]() {
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
    });

    gpu_stage(profiler, "hamiltonian_ms", [&]() {
        detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                         impl.state_size, impl.num_qubits,
                                         impl.field);
    });

    EnergyGradResult result;
    result.energy = gpu_stage(
        profiler, "energy_dot_ms",
        [&]() {
            return detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                                 impl.state_size)
                .real();
        });

    if (!compute_gradient) {
        result.stage_timings_ms = std::move(profiler.timings_ms);
        return result;
    }

    gpu_stage(profiler, "gradient_zero_ms", [&]() {
        detail::check_cuda(
            cudaMemsetAsync(impl.gate_level_gradients.get(), 0,
                            sizeof(double) * impl.expected_params, 0),
            "zero save_param gradients");
    });
    gpu_stage(profiler, "backward_sweep_ms", [&]() {
        for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
            const auto &op = ops[reverse_index];
            switch (op.kind) {
            case OpKind::RY:
                detail::launch_save_param_ry_step(
                    state_slot(impl.save_param_states.get(), impl.state_size,
                               op.param_index0),
                    impl.lambda.get(), impl.state_size, op.wire0, op.theta0,
                    impl.gate_level_gradients.get() + op.param_index0);
                break;
            case OpKind::RZ:
                detail::launch_save_param_rz_step(
                    state_slot(impl.save_param_states.get(), impl.state_size,
                               op.param_index0),
                    impl.lambda.get(), impl.state_size, op.wire0, op.theta0,
                    impl.gate_level_gradients.get() + op.param_index0);
                break;
            case OpKind::CNOT:
                detail::launch_apply_cnot(impl.lambda.get(), impl.state_size,
                                          op.wire0, op.wire1);
                break;
            case OpKind::FusedRYRZ:
            case OpKind::RingCNOTLayer:
                throw std::runtime_error(
                    "save_param_states expects PennyLane-style gate-level ops.");
            }
        }
    });

    result.gradient.assign(impl.expected_params, 0.0);
    gpu_stage(profiler, "gradient_copy_ms", [&]() {
        detail::check_cuda(
            cudaMemcpy(result.gradient.data(), impl.gate_level_gradients.get(),
                       sizeof(double) * impl.expected_params,
                       cudaMemcpyDeviceToHost),
            "copy save_param gradients");
    });
    result.stage_timings_ms = std::move(profiler.timings_ms);
    return result;
}

auto run_energy_and_grad_baseline(RingIsingCudaBackend::Impl &impl,
                                  const double *params,
                                  std::size_t num_params,
                                  bool compute_gradient,
                                  bool profile) -> EnergyGradResult {
    if (impl.strategy == GradientStrategy::InverseWalk) {
        return run_energy_and_grad_inverse_walk(impl, params, num_params,
                                                compute_gradient, profile);
    }
    return run_energy_and_grad_save_param_states(impl, params, num_params,
                                                 compute_gradient, profile);
}
