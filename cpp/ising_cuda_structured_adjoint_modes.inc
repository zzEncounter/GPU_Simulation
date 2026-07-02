struct StructuredRotationChunk {
    std::size_t start;
    std::size_t width;
};

struct StructuredForwardRotationChunk {
    StructuredRotationChunk chunk;
    detail::RotationChunkKernelPreference kernel_preference{
        detail::RotationChunkKernelPreference::Cooperative};
};

auto build_structured_forward_rotation_chunks(
    const RingIsingCudaBackend::Impl &impl)
    -> std::vector<StructuredForwardRotationChunk> {
    std::vector<StructuredForwardRotationChunk> chunks;
    chunks.reserve((impl.num_qubits + 3) / 4);

    auto append_fixed_width =
        [&](std::size_t chunk_start, std::size_t max_width,
            detail::RotationChunkKernelPreference kernel_preference) {
            while (chunk_start < impl.num_qubits) {
                const auto chunk_width =
                    std::min(max_width, impl.num_qubits - chunk_start);
                chunks.push_back(
                    {{chunk_start, chunk_width}, kernel_preference});
                chunk_start += chunk_width;
            }
        };

    auto append_w8_prefix_high4 =
        [&](std::size_t prefix_end,
            detail::RotationChunkKernelPreference prefix_kernel) {
            std::size_t chunk_start = 0;
            while (chunk_start < impl.num_qubits && chunk_start < prefix_end) {
                const auto chunk_width =
                    std::min<std::size_t>(8, impl.num_qubits - chunk_start);
                chunks.push_back(
                    {{chunk_start, chunk_width}, prefix_kernel});
                chunk_start += chunk_width;
            }
            append_fixed_width(chunk_start, 4,
                               detail::RotationChunkKernelPreference::Register);
        };

    auto append_default_start_aware_hybrid = [&]() {
        if (impl.num_qubits <= 15) {
            append_fixed_width(
                0, 8, detail::RotationChunkKernelPreference::Cooperative);
            return;
        }

        const auto suffix_width = impl.num_qubits - 8;
        if (suffix_width <= 14) {
            append_w8_prefix_high4(
                16, detail::RotationChunkKernelPreference::CooperativePair512);
            return;
        }

        append_fixed_width(0, 4,
                           detail::RotationChunkKernelPreference::Register);
    };

    append_default_start_aware_hybrid();
    return chunks;
}

auto structured_param_base(const RingIsingCudaBackend::Impl &impl,
                           std::size_t layer,
                           std::size_t wire) -> std::size_t {
    return (layer * impl.num_qubits + wire) * 2;
}

void load_structured_rotation_params(
    const RingIsingCudaBackend::Impl &impl, const double *params,
    std::size_t layer, StructuredRotationChunk chunk,
    std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> &theta_ry,
    std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> &theta_rz) {
    for (std::size_t local = 0; local < chunk.width; local++) {
        const auto wire = chunk.start + local;
        const auto base = structured_param_base(impl, layer, wire);
        theta_ry[local] = params != nullptr ? params[base] : 0.0;
        theta_rz[local] = params != nullptr ? params[base + 1] : 0.0;
    }
}

void launch_structured_forward_rotation_chunk(
    Complex *state, std::size_t state_size, StructuredRotationChunk chunk,
    const std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> &theta_ry,
    const std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> &theta_rz,
    detail::RotationChunkKernelPreference kernel_preference) {
    if (chunk.width == 1) {
        detail::launch_apply_ryrz(state, state_size, chunk.start, theta_ry[0],
                                  theta_rz[0]);
        return;
    }
    detail::launch_apply_ryrz_rotation_chunk(
        state, state_size, chunk.start, chunk.width, theta_ry.data(),
        theta_rz.data(), kernel_preference);
}

void apply_structured_forward_layers(
    RingIsingCudaBackend::Impl &impl, const double *params,
    const std::vector<StructuredForwardRotationChunk> &rotation_chunks) {
    auto *state = impl.current.get();
    auto *next = impl.scratch.get();

    for (std::size_t layer = 0; layer < impl.num_layers; layer++) {
        for (const auto forward_chunk : rotation_chunks) {
            const auto chunk = forward_chunk.chunk;
            std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> theta_ry{};
            std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> theta_rz{};
            load_structured_rotation_params(impl, params, layer, chunk,
                                            theta_ry, theta_rz);
            launch_structured_forward_rotation_chunk(
                state, impl.state_size, chunk, theta_ry, theta_rz,
                forward_chunk.kernel_preference);
        }

        detail::launch_apply_ring_cnot_layer(next, state, impl.state_size,
                                             impl.num_qubits, false);
        std::swap(state, next);
    }

    if (state != impl.current.get()) {
        copy_device_buffer(impl.current.get(), state, impl.state_size);
    }
}

void launch_structured_backward_ryrz_wire(RingIsingCudaBackend::Impl &impl,
                                          Complex *current, Complex *lambda,
                                          const double *params,
                                          std::size_t layer,
                                          std::size_t wire) {
    const auto base = structured_param_base(impl, layer, wire);
    const double theta_ry = params != nullptr ? params[base] : 0.0;
    const double theta_rz = params != nullptr ? params[base + 1] : 0.0;
    detail::launch_inverse_walk_ryrz_step(
        current, lambda, impl.state_size, wire, theta_ry,
        theta_rz, impl.gate_level_gradients.get() + base,
        impl.gate_level_gradients.get() + base + 1);
}

void launch_structured_backward_ryrz_chunk(RingIsingCudaBackend::Impl &impl,
                                           Complex *current, Complex *lambda,
                                           const double *params,
                                           std::size_t layer,
                                           StructuredRotationChunk chunk) {
    std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> theta_ry{};
    std::array<double, STRUCTURED_ROTATION_CHUNK_WIRES> theta_rz{};
    load_structured_rotation_params(impl, params, layer, chunk, theta_ry,
                                    theta_rz);
    const auto base = structured_param_base(impl, layer, chunk.start);
    detail::launch_inverse_walk_ryrz_rotation_chunk(
        current, lambda, impl.state_size, chunk.start, chunk.width,
        theta_ry.data(), theta_rz.data(),
        impl.gate_level_gradients.get() + base);
}

void run_structured_backward_rotation_layer(
    RingIsingCudaBackend::Impl &impl, Complex *current, Complex *lambda,
    const double *params, std::size_t layer) {
    auto requested_chunk_width =
        std::min<std::size_t>(impl.structured_rotation_chunk_width,
                              STRUCTURED_ROTATION_CHUNK_WIRES);
    if (requested_chunk_width > 1 && impl.num_qubits <= 15) {
        requested_chunk_width =
            std::min<std::size_t>(requested_chunk_width, 4);
    }
    if (requested_chunk_width > 1 && requested_chunk_width < 8 &&
        impl.num_qubits >= 16) {
        requested_chunk_width = 1;
    }

    auto launch_chunk = [&](StructuredRotationChunk chunk) {
        launch_structured_backward_ryrz_chunk(impl, current, lambda, params,
                                              layer, chunk);
    };

    std::size_t wire = impl.num_qubits;
    if (requested_chunk_width >= 8 && impl.num_qubits >= 20) {
        while (wire > 8) {
            const auto chunk_width = std::min<std::size_t>(4, wire - 8);
            const auto chunk_start = wire - chunk_width;
            if (chunk_width >= 2) {
                launch_chunk({chunk_start, chunk_width});
            } else {
                launch_structured_backward_ryrz_wire(
                    impl, current, lambda, params, layer, chunk_start);
            }
            wire = chunk_start;
        }
    }

    while (wire > 0) {
        if (requested_chunk_width >= 2 && wire >= requested_chunk_width) {
            const auto chunk_start = wire - requested_chunk_width;
            launch_chunk({chunk_start, requested_chunk_width});
            wire = chunk_start;
            continue;
        }
        wire--;
        launch_structured_backward_ryrz_wire(impl, current, lambda, params,
                                             layer, wire);
    }
}

void run_structured_backward_sweep(RingIsingCudaBackend::Impl &impl,
                                   const double *params) {
    auto *current = impl.current.get();
    auto *next_current = impl.scratch.get();
    auto *lambda = impl.lambda.get();
    auto *next_lambda = impl.cnot_scratch.get();

    for (std::size_t layer = impl.num_layers; layer-- > 0;) {
        detail::launch_apply_ring_cnot_layer(next_current, current,
                                             impl.state_size, impl.num_qubits,
                                             true);
        detail::launch_apply_ring_cnot_layer(next_lambda, lambda,
                                             impl.state_size, impl.num_qubits,
                                             true);
        std::swap(current, next_current);
        std::swap(lambda, next_lambda);

        run_structured_backward_rotation_layer(impl, current, lambda, params,
                                               layer);
    }

    if (current != impl.current.get()) {
        copy_device_buffer(impl.current.get(), current, impl.state_size);
    }
    if (lambda != impl.lambda.get()) {
        copy_device_buffer(impl.lambda.get(), lambda, impl.state_size);
    }
}

auto run_energy_and_grad_structured_adjoint(RingIsingCudaBackend::Impl &impl,
                                            const double *params,
                                            std::size_t num_params,
                                            bool compute_gradient,
                                            bool profile) -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);
    StageProfiler profiler(profile);
    std::vector<StructuredForwardRotationChunk> forward_rotation_chunks;
    cpu_stage(
        profiler, "build_ops_host_ms",
        [&]() {
            forward_rotation_chunks =
                build_structured_forward_rotation_chunks(impl);
        });

    gpu_stage(profiler, "zero_state_ms", [&]() {
        detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    });
    gpu_stage(profiler, "forward_apply_ms", [&]() {
        apply_structured_forward_layers(impl, params, forward_rotation_chunks);
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
            "zero structured_adjoint gradients");
    });
    gpu_stage(profiler, "backward_sweep_ms", [&]() {
        run_structured_backward_sweep(impl, params);
    });

    result.gradient.assign(impl.expected_params, 0.0);
    gpu_stage(profiler, "gradient_copy_ms", [&]() {
        detail::check_cuda(
            cudaMemcpy(result.gradient.data(), impl.gate_level_gradients.get(),
                       sizeof(double) * impl.expected_params,
                       cudaMemcpyDeviceToHost),
            "copy structured_adjoint gradients");
    });
    result.stage_timings_ms = std::move(profiler.timings_ms);
    return result;
}
