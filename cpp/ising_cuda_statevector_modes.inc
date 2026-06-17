void apply_mode2_chunked_forward_layers(RingIsingCudaBackend::Impl &impl,
                                        const double *params,
                                        std::size_t target_chunk_width) {
    if (target_chunk_width == 0) {
        throw std::invalid_argument("target_chunk_width must be positive.");
    }
    if (target_chunk_width > MODE2_ROTATION_CHUNK_WIRES) {
        throw std::invalid_argument(
            "target_chunk_width exceeds supported maximum.");
    }

    auto *state = impl.current.get();
    auto *next = impl.scratch.get();

    for (std::size_t layer = 0; layer < impl.num_layers; layer++) {
        for (std::size_t chunk_start = 0; chunk_start < impl.num_qubits;
             chunk_start += target_chunk_width) {
            const auto chunk_width =
                std::min(target_chunk_width,
                         impl.num_qubits - chunk_start);
            std::array<double, MODE2_ROTATION_CHUNK_WIRES> theta_ry{};
            std::array<double, MODE2_ROTATION_CHUNK_WIRES> theta_rz{};
            for (std::size_t local = 0; local < chunk_width; local++) {
                const auto wire = chunk_start + local;
                const auto base = (layer * impl.num_qubits + wire) * 2;
                theta_ry[local] = params != nullptr ? params[base] : 0.0;
                theta_rz[local] = params != nullptr ? params[base + 1] : 0.0;
            }
            if (chunk_width == 1) {
                detail::launch_apply_ryrz(state, impl.state_size, chunk_start,
                                          theta_ry[0], theta_rz[0]);
            } else {
                detail::launch_apply_ryrz_rotation_chunk(
                    state, impl.state_size, chunk_start, chunk_width,
                    theta_ry.data(), theta_rz.data());
            }
        }

        detail::launch_apply_ring_cnot_layer(next, state, impl.state_size,
                                             impl.num_qubits, false);
        std::swap(state, next);
    }

    if (state != impl.current.get()) {
        copy_device_buffer(impl.current.get(), state, impl.state_size);
    }
}

void apply_mode2_fused_wire_forward_layers(RingIsingCudaBackend::Impl &impl,
                                           const double *params) {
    auto *state = impl.current.get();
    auto *next = impl.scratch.get();

    for (std::size_t layer = 0; layer < impl.num_layers; layer++) {
        for (std::size_t wire = 0; wire < impl.num_qubits; wire++) {
            const auto base = (layer * impl.num_qubits + wire) * 2;
            const double theta_ry = params != nullptr ? params[base] : 0.0;
            const double theta_rz = params != nullptr ? params[base + 1] : 0.0;
            detail::launch_apply_ryrz(state, impl.state_size, wire, theta_ry,
                                      theta_rz);
        }

        detail::launch_apply_ring_cnot_layer(next, state, impl.state_size,
                                             impl.num_qubits, false);
        std::swap(state, next);
    }

    if (state != impl.current.get()) {
        copy_device_buffer(impl.current.get(), state, impl.state_size);
    }
}

void apply_mode2_forward_layers(RingIsingCudaBackend::Impl &impl,
                                const double *params) {
    if (impl.strategy != GradientStrategy::Mode2) {
        throw std::runtime_error(
            "apply_mode2_forward_layers called for a non-mode2 strategy.");
    }
    if (impl.mode2_rotation_chunk_width == 1) {
        apply_mode2_fused_wire_forward_layers(impl, params);
        return;
    }
    apply_mode2_chunked_forward_layers(impl, params,
                                       impl.mode2_rotation_chunk_width);
}

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
        case OpKind::RingCNOTLayer:
            throw std::runtime_error(
                "inverse_walk expects PennyLane-style gate-level ops.");
        }
    }
}

auto run_energy_only(RingIsingCudaBackend::Impl &impl, const double *params,
                     std::size_t num_params, bool profile = false)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    StageProfiler profiler(profile);
    const auto ops = cpu_stage(
        profiler, "build_ops_host_ms",
        [&]() {
            if (is_mode2_family(impl.strategy)) {
                return std::vector<OpDesc>{};
            }
            return build_ops_for_strategy(impl.num_qubits, impl.num_layers,
                                          params, impl.strategy);
        });

    gpu_stage(profiler, "zero_state_ms", [&]() {
        detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    });
    gpu_stage(profiler, "forward_apply_ms", [&]() {
        if (is_mode2_family(impl.strategy)) {
            apply_mode2_forward_layers(impl, params);
            return;
        }
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
        run_inverse_walk_backward_sweep(impl, ops);
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

auto run_energy_and_grad_mode2(RingIsingCudaBackend::Impl &impl,
                               const double *params,
                               std::size_t num_params,
                               bool compute_gradient,
                               bool profile) -> EnergyGradResult {
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
        apply_mode2_forward_layers(impl, params);
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
            "zero mode2 gradients");
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
            "copy mode2 gradients");
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
    if (is_mode2_family(impl.strategy)) {
        return run_energy_and_grad_mode2(impl, params, num_params,
                                         compute_gradient, profile);
    }
    return run_energy_and_grad_save_param_states(impl, params, num_params,
                                                 compute_gradient, profile);
}
