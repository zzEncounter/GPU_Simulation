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

void ensure_custatevec(RingIsingCudaBackend::Impl &impl) {
    if (!impl.custatevec) {
        impl.custatevec = std::make_unique<CustatevecHandle>();
    }
}

auto custatevec_workspace(RingIsingCudaBackend::Impl &impl,
                          std::size_t required_size) -> void * {
    if (required_size == 0) {
        return nullptr;
    }
    if (impl.custatevec_workspace.size() < required_size) {
        impl.custatevec_workspace.allocate(required_size);
    }
    return impl.custatevec_workspace.get();
}

void apply_matrix_custatevec(RingIsingCudaBackend::Impl &impl, Complex *state,
                             const cuDoubleComplex *matrix,
                             const int32_t *targets, uint32_t n_targets,
                             const int32_t *controls, uint32_t n_controls,
                             bool adjoint, const char *context,
                             std::size_t *cached_workspace_size = nullptr,
                             bool *workspace_size_cached = nullptr) {
    ensure_custatevec(impl);
    size_t workspace_size = 0;
    if (cached_workspace_size != nullptr && workspace_size_cached != nullptr &&
        *workspace_size_cached) {
        workspace_size = *cached_workspace_size;
    } else {
        detail::check_custatevec(
            custatevecApplyMatrixGetWorkspaceSize(
                impl.custatevec->handle, CUDA_C_64F,
                static_cast<uint32_t>(impl.num_qubits), matrix, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint ? 1 : 0, n_targets,
                n_controls, CUSTATEVEC_COMPUTE_64F, &workspace_size),
            context);
        if (cached_workspace_size != nullptr &&
            workspace_size_cached != nullptr) {
            *cached_workspace_size = workspace_size;
            *workspace_size_cached = true;
        }
    }
    void *workspace = custatevec_workspace(impl, workspace_size);
    detail::check_custatevec(
        custatevecApplyMatrix(
            impl.custatevec->handle, state, CUDA_C_64F,
            static_cast<uint32_t>(impl.num_qubits), matrix, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint ? 1 : 0, targets, n_targets,
            controls, nullptr, n_controls, CUSTATEVEC_COMPUTE_64F, workspace,
            workspace_size),
        context);
}

void apply_pauli_rotation_custatevec(RingIsingCudaBackend::Impl &impl,
                                     Complex *state, custatevecPauli_t pauli,
                                     int32_t target, double theta,
                                     bool inverse, const char *context) {
    ensure_custatevec(impl);
    const double local_angle = inverse ? theta * 0.5 : -theta * 0.5;
    detail::check_custatevec(
        custatevecApplyPauliRotation(
            impl.custatevec->handle, state, CUDA_C_64F,
            static_cast<uint32_t>(impl.num_qubits), local_angle, &pauli,
            &target, 1, nullptr, nullptr, 0),
        context);
}

auto make_x_matrix() -> std::array<cuDoubleComplex, 4> {
    return {make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0),
            make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)};
}

void apply_op_custatevec(RingIsingCudaBackend::Impl &impl, Complex *state,
                         const OpDesc &op, bool inverse,
                         const char *context) {
    switch (op.kind) {
    case OpKind::RY: {
        const int32_t target = to_int(op.wire0, "cuStateVec RY target wire");
        apply_pauli_rotation_custatevec(impl, state, CUSTATEVEC_PAULI_Y,
                                        target, op.theta0, inverse, context);
        break;
    }
    case OpKind::RZ: {
        const int32_t target = to_int(op.wire0, "cuStateVec RZ target wire");
        apply_pauli_rotation_custatevec(impl, state, CUSTATEVEC_PAULI_Z,
                                        target, op.theta0, inverse, context);
        break;
    }
    case OpKind::CNOT: {
        const auto matrix = make_x_matrix();
        const int32_t target = to_int(op.wire1, "cuStateVec CNOT target wire");
        const int32_t control = to_int(op.wire0, "cuStateVec CNOT control wire");
        apply_matrix_custatevec(impl, state, matrix.data(), &target, 1,
                                &control, 1, false, context,
                                &impl.custatevec_cnot_workspace_size,
                                &impl.custatevec_cnot_workspace_size_cached);
        break;
    }
    case OpKind::FusedRYRZ:
    case OpKind::RotationLayer:
    case OpKind::RingCNOTLayer:
        throw std::runtime_error(
            "inverse_walk_cuQuantum expects PennyLane-style gate-level ops.");
    }
}

void run_inverse_walk_cuquantum_backward_sweep(
    RingIsingCudaBackend::Impl &impl, const std::vector<OpDesc> &ops) {
    for (std::size_t reverse_index = ops.size(); reverse_index-- > 0;) {
        const auto &op = ops[reverse_index];
        switch (op.kind) {
        case OpKind::RY:
            detail::launch_inverse_walk_ry_gradient(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.theta0,
                impl.gate_level_gradients.get() + op.param_index0);
            apply_op_custatevec(impl, impl.current.get(), op, true,
                                "inverse_walk_cuQuantum current inverse RY");
            apply_op_custatevec(impl, impl.lambda.get(), op, true,
                                "inverse_walk_cuQuantum lambda inverse RY");
            break;
        case OpKind::RZ:
            detail::launch_inverse_walk_rz_gradient(
                impl.current.get(), impl.lambda.get(), impl.state_size,
                op.wire0, op.theta0,
                impl.gate_level_gradients.get() + op.param_index0);
            apply_op_custatevec(impl, impl.current.get(), op, true,
                                "inverse_walk_cuQuantum current inverse RZ");
            apply_op_custatevec(impl, impl.lambda.get(), op, true,
                                "inverse_walk_cuQuantum lambda inverse RZ");
            break;
        case OpKind::CNOT:
            apply_op_custatevec(impl, impl.current.get(), op, true,
                                "inverse_walk_cuQuantum current inverse CNOT");
            apply_op_custatevec(impl, impl.lambda.get(), op, true,
                                "inverse_walk_cuQuantum lambda inverse CNOT");
            break;
        case OpKind::FusedRYRZ:
        case OpKind::RotationLayer:
        case OpKind::RingCNOTLayer:
            throw std::runtime_error(
                "inverse_walk_cuQuantum expects PennyLane-style gate-level ops.");
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
            if (impl.strategy == GradientStrategy::InverseWalkCuQuantum) {
                apply_op_custatevec(impl, impl.current.get(), op, false,
                                    "inverse_walk_cuQuantum forward gate");
            } else {
                apply_op_inplace(impl.current.get(), impl.state_size, op, false);
            }
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
        if (impl.strategy == GradientStrategy::InverseWalkCuQuantum) {
            run_inverse_walk_cuquantum_backward_sweep(impl, ops);
        } else {
            run_inverse_walk_backward_sweep(impl, ops);
        }
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
