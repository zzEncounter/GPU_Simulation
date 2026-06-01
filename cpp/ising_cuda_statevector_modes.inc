auto run_energy_only(RingIsingCudaBackend::Impl &impl, const double *params,
                     std::size_t num_params,
                     bool measure_timings) -> EnergyGradResult {
    const auto total_start = std::chrono::steady_clock::now();
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);

    const auto forward_start = std::chrono::steady_clock::now();
    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (const auto &op : ops) {
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }
    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);
    const Complex energy_complex =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size);
    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_only completion");
    }
    const auto forward_end = std::chrono::steady_clock::now();

    EnergyGradResult result;
    result.energy = energy_complex.real();
    result.forward_ms =
        measure_timings ? elapsed_ms(forward_start, forward_end) : 0.0;
    result.back_ms = 0.0;
    result.gradient_ms = 0.0;
    result.total_ms =
        measure_timings ? elapsed_ms(total_start, std::chrono::steady_clock::now())
                        : 0.0;
    return result;
}

auto run_energy_and_grad_save_param_states(
    RingIsingCudaBackend::Impl &impl, const double *params,
    std::size_t num_params, bool measure_timings,
    bool compute_gradient) -> EnergyGradResult {
    const auto total_start = std::chrono::steady_clock::now();
    validate_num_params(impl.expected_params, num_params);
    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);

    const auto forward_start = std::chrono::steady_clock::now();
    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    for (std::size_t op_index = 0; op_index < ops.size(); op_index++) {
        const auto &op = ops[op_index];
        if (compute_gradient && op.is_parametric) {
            copy_device_buffer(
                state_slot(impl.save_param_states.get(), impl.state_size,
                           op.param_index0 / 2),
                impl.current.get(), impl.state_size);
        }
        apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                         impl.scratch.get(), impl.num_qubits);
    }

    detail::launch_apply_hamiltonian(impl.lambda.get(), impl.current.get(),
                                     impl.state_size, impl.num_qubits,
                                     impl.field);
    const Complex energy_complex =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size);
    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_save_param_states forward");
    }
    const auto forward_end = std::chrono::steady_clock::now();

    EnergyGradResult result;
    result.energy = energy_complex.real();

    if (!compute_gradient) {
        result.forward_ms =
            measure_timings ? elapsed_ms(forward_start, forward_end) : 0.0;
        result.back_ms = 0.0;
        result.gradient_ms = 0.0;
        result.total_ms =
            measure_timings
                ? elapsed_ms(total_start, std::chrono::steady_clock::now())
                : 0.0;
        return result;
    }

    result.gradient.assign(impl.expected_params, 0.0);

    double gradient_ms = 0.0;
    double back_ms = 0.0;
    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        if (op.is_parametric) {
            const auto gradient_start = std::chrono::steady_clock::now();
            const Complex *state_before =
                state_slot(impl.save_param_states.get(), impl.state_size,
                           op.param_index0 / 2);
            if (op.kind == OpKind::FusedRYRZ) {
                const auto [grad_theta, grad_phi] = detail::fused_ryrz_gradients(
                    impl.lambda.get(), state_before, impl.state_size, op.wire0,
                    op.theta0, op.theta1);
                result.gradient[op.param_index0] = grad_theta;
                result.gradient[op.param_index1] = grad_phi;
            } else {
                apply_param_derivative(impl.deriv.get(), state_before,
                                       impl.state_size, op);
                const Complex grad_complex =
                    detail::complex_inner_product(impl.lambda.get(), impl.deriv.get(),
                                                  impl.state_size);
                result.gradient[op.param_index0] = 2.0 * grad_complex.real();
            }
            if (measure_timings) {
                detail::check_cuda(
                    cudaDeviceSynchronize(),
                    "run_energy_and_grad_save_param_states gradient");
                gradient_ms += elapsed_ms(gradient_start,
                                          std::chrono::steady_clock::now());
            }
        }
        if (reverse_index != 0) {
            const auto back_start = std::chrono::steady_clock::now();
            apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                             impl.scratch.get(), impl.num_qubits);
            if (measure_timings) {
                detail::check_cuda(cudaDeviceSynchronize(),
                                   "run_energy_and_grad_save_param_states back");
                back_ms += elapsed_ms(back_start, std::chrono::steady_clock::now());
            }
        }
    }

    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_save_param_states completion");
    }
    result.forward_ms = measure_timings ? elapsed_ms(forward_start, forward_end) : 0.0;
    result.back_ms = measure_timings ? back_ms : 0.0;
    result.gradient_ms = measure_timings ? gradient_ms : 0.0;
    result.total_ms =
        measure_timings ? elapsed_ms(total_start, std::chrono::steady_clock::now())
                        : 0.0;
    return result;
}

auto run_energy_and_grad_checkpointed(RingIsingCudaBackend::Impl &impl,
                                      const double *params,
                                      std::size_t num_params,
                                      bool measure_timings,
                                      bool compute_gradient)
    -> EnergyGradResult {
    const auto total_start = std::chrono::steady_clock::now();
    validate_num_params(impl.expected_params, num_params);

    if (!compute_gradient) {
        return run_energy_only(impl, params, num_params, measure_timings);
    }

    if (impl.strategy == GradientStrategy::BruteForceParallelQ6) {
        return run_dense_scan_energy_and_grad_fast(impl, params, num_params,
                                                   measure_timings);
    }

    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);
    const auto num_ops = ops.size();

    if (impl.strategy == GradientStrategy::SaveParamStates) {
        return run_energy_and_grad_save_param_states(impl, params, num_params,
                                                     measure_timings, true);
    }

    const auto forward_start = std::chrono::steady_clock::now();
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
    const Complex energy_complex =
        detail::complex_inner_product(impl.current.get(), impl.lambda.get(),
                                      impl.state_size);
    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_checkpointed forward");
    }
    const auto forward_end = std::chrono::steady_clock::now();

    EnergyGradResult result;
    result.energy = energy_complex.real();
    result.gradient.assign(impl.expected_params, 0.0);

    double gradient_ms = 0.0;
    double back_ms = 0.0;
    for (std::size_t chunk_idx = impl.num_chunks; chunk_idx-- > 0;) {
        const auto chunk_start = chunk_idx * impl.checkpoint_interval_ops;
        const auto chunk_end =
            std::min(chunk_start + impl.checkpoint_interval_ops, num_ops);
        const auto local_count = chunk_end - chunk_start;

        const auto back_recompute_start = std::chrono::steady_clock::now();
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
        if (measure_timings) {
            detail::check_cuda(cudaDeviceSynchronize(),
                               "run_energy_and_grad_checkpointed recompute");
            back_ms += elapsed_ms(back_recompute_start,
                                  std::chrono::steady_clock::now());
        }

        for (std::size_t op_idx = chunk_end; op_idx-- > chunk_start;) {
            const auto &op = ops[op_idx];
            const auto local_before = op_idx - chunk_start;
            if (op.is_parametric) {
                const auto gradient_start = std::chrono::steady_clock::now();
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
                if (measure_timings) {
                    detail::check_cuda(cudaDeviceSynchronize(),
                                       "run_energy_and_grad_checkpointed gradient");
                    gradient_ms += elapsed_ms(gradient_start,
                                              std::chrono::steady_clock::now());
                }
            }
            if (op_idx != 0) {
                const auto back_start = std::chrono::steady_clock::now();
                apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                                 impl.scratch.get(), impl.num_qubits);
                if (measure_timings) {
                    detail::check_cuda(cudaDeviceSynchronize(),
                                       "run_energy_and_grad_checkpointed back");
                    back_ms += elapsed_ms(back_start, std::chrono::steady_clock::now());
                }
            }
        }
    }

    if (measure_timings) {
        detail::check_cuda(cudaDeviceSynchronize(),
                           "run_energy_and_grad_checkpointed completion");
    }
    result.forward_ms = measure_timings ? elapsed_ms(forward_start, forward_end) : 0.0;
    result.back_ms = measure_timings ? back_ms : 0.0;
    result.gradient_ms = measure_timings ? gradient_ms : 0.0;
    result.total_ms =
        measure_timings ? elapsed_ms(total_start, std::chrono::steady_clock::now())
                        : 0.0;
    return result;
}
