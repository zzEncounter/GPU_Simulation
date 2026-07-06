auto run_energy_only(RingIsingCudaBackend::Impl &impl, const double *params,
                     std::size_t num_params, bool profile = false)
    -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    StageProfiler profiler(profile);
    std::vector<OpDesc> ops;
    std::vector<StructuredForwardRotationChunk> structured_forward_rotation_chunks;
    cpu_stage(profiler, "build_ops_host_ms", [&]() {
        if (is_structured_adjoint_family(impl.strategy)) {
            structured_forward_rotation_chunks =
                build_structured_forward_rotation_chunks(impl);
            return;
        }
        ops = build_ops_for_strategy(impl.num_qubits, impl.num_layers, params,
                                     impl.strategy);
    });

    gpu_stage(profiler, "zero_state_ms", [&]() {
        if (is_structured_adjoint_family(impl.strategy) &&
            structured_uses_product_state_init(impl)) {
            return;
        }
        detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    });
    gpu_stage(profiler, "forward_apply_ms", [&]() {
        if (is_structured_adjoint_family(impl.strategy)) {
            apply_structured_forward_layers(impl, params,
                                       structured_forward_rotation_chunks);
            return;
        }
        for (const auto &op : ops) {
            apply_op_inplace(impl.current.get(), impl.state_size, op, false,
                             impl.scratch.get(), impl.num_qubits);
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
    result.stage_timings_ms = std::move(profiler.timings_ms);
    return result;
}

auto run_energy_and_grad_baseline(RingIsingCudaBackend::Impl &impl,
                                  const double *params,
                                  std::size_t num_params,
                                  bool compute_gradient,
                                  bool profile) -> EnergyGradResult {
    if (impl.strategy == GradientStrategy::InverseWalk ||
        impl.strategy == GradientStrategy::RyrzFused) {
        return run_energy_and_grad_gate_level_adjoint(
            impl, params, num_params, compute_gradient, profile);
    }
    if (is_structured_adjoint_family(impl.strategy)) {
        return run_energy_and_grad_structured_adjoint(impl, params, num_params,
                                         compute_gradient, profile);
    }
    throw std::runtime_error("unsupported statevector gradient strategy.");
}
