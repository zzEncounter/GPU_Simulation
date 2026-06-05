__global__ void block_fused_ryrz_gradient_kernel(
    const Complex *lambda_after_ops, const Complex *local_states,
    const std::size_t *wires, const std::size_t *param_indices,
    const double *thetas, const double *phis, double *gradients,
    std::size_t run_count, std::size_t first_local_state,
    std::size_t state_size) {
    __shared__ double partial[detail::THREADS];

    const auto gradient_slot = static_cast<std::size_t>(blockIdx.x);
    if (gradient_slot >= 2 * run_count) {
        return;
    }

    const auto op_slot = gradient_slot / 2;
    const bool theta_gradient = (gradient_slot % 2) == 0;
    const auto wire = wires[op_slot];
    const auto half_stride = std::size_t{1} << wire;
    const auto low_mask = half_stride - 1;
    const auto num_pairs = state_size / 2;
    const Complex *state =
        local_states + (first_local_state + op_slot) * state_size;
    const Complex *lambda = lambda_after_ops + op_slot * state_size;
    const double theta = thetas[op_slot];
    const double phi = phis[op_slot];

    const double c = std::cos(theta * 0.5);
    const double s = std::sin(theta * 0.5);
    const double half_phi = phi * 0.5;
    const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
    const Complex phase1(std::cos(half_phi), std::sin(half_phi));

    double local_sum = 0.0;
    for (std::size_t pair_index = threadIdx.x; pair_index < num_pairs;
         pair_index += blockDim.x) {
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;

        const Complex a0 = state[i0];
        const Complex a1 = state[i1];
        const Complex l0 = lambda[i0];
        const Complex l1 = lambda[i1];

        Complex contribution;
        if (theta_gradient) {
            const Complex d0 =
                Complex(-0.5 * s, 0.0) * a0 + Complex(-0.5 * c, 0.0) * a1;
            const Complex d1 =
                Complex(0.5 * c, 0.0) * a0 + Complex(-0.5 * s, 0.0) * a1;
            contribution = thrust::conj(l0) * (phase0 * d0) +
                           thrust::conj(l1) * (phase1 * d1);
        } else {
            const Complex b0 = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
            const Complex b1 = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;
            const Complex pref0(0.0, -0.5);
            const Complex pref1(0.0, 0.5);
            contribution = thrust::conj(l0) * (pref0 * phase0 * b0) +
                           thrust::conj(l1) * (pref1 * phase1 * b1);
        }
        local_sum += contribution.real();
    }

    partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = detail::THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        gradients[param_indices[gradient_slot]] = 2.0 * partial[0];
    }
}

void copy_host_double_to_device(double *dst, double value) {
    detail::check_cuda(cudaMemcpy(dst, &value, sizeof(double),
                                  cudaMemcpyHostToDevice),
                       "copy scalar gradient to device");
}

void launch_block_fused_ryrz_gradients(
    RingIsingCudaBackend::Impl &impl, std::size_t run_start,
    std::size_t block_start, std::size_t run_count,
    const std::vector<OpDesc> &ops) {
    std::vector<std::size_t> wires(run_count);
    std::vector<std::size_t> param_indices(2 * run_count);
    std::vector<double> thetas(run_count);
    std::vector<double> phis(run_count);

    for (std::size_t slot = 0; slot < run_count; slot++) {
        const auto &op = ops[run_start + slot];
        wires[slot] = op.wire0;
        param_indices[2 * slot] = op.param_index0;
        param_indices[2 * slot + 1] = op.param_index1;
        thetas[slot] = op.theta0;
        phis[slot] = op.theta1;
    }

    detail::check_cuda(cudaMemcpy(impl.block_fused_wires.get(), wires.data(),
                                  sizeof(std::size_t) * wires.size(),
                                  cudaMemcpyHostToDevice),
                       "copy block fused wires");
    detail::check_cuda(cudaMemcpy(impl.block_fused_param_indices.get(),
                                  param_indices.data(),
                                  sizeof(std::size_t) * param_indices.size(),
                                  cudaMemcpyHostToDevice),
                       "copy block fused param indices");
    detail::check_cuda(cudaMemcpy(impl.block_fused_thetas.get(), thetas.data(),
                                  sizeof(double) * thetas.size(),
                                  cudaMemcpyHostToDevice),
                       "copy block fused thetas");
    detail::check_cuda(cudaMemcpy(impl.block_fused_phis.get(), phis.data(),
                                  sizeof(double) * phis.size(),
                                  cudaMemcpyHostToDevice),
                       "copy block fused phis");

    block_fused_ryrz_gradient_kernel<<<static_cast<int>(2 * run_count),
                                       detail::THREADS>>>(
        impl.block_fused_lambda_states.get(), impl.local_states.get(),
        impl.block_fused_wires.get(), impl.block_fused_param_indices.get(),
        impl.block_fused_thetas.get(), impl.block_fused_phis.get(),
        impl.block_fused_gradients.get(), run_count, run_start - block_start,
        impl.state_size);
    detail::check_cuda(cudaGetLastError(), "block_fused_ryrz_gradient_kernel");
    detail::maybe_synchronize_cuda("block_fused_ryrz_gradient_kernel sync");
}

void store_scalar_gradient(RingIsingCudaBackend::Impl &impl,
                           std::size_t param_index, double value) {
    copy_host_double_to_device(impl.block_fused_gradients.get() + param_index,
                               value);
}

auto run_energy_and_grad_block_fused_adjoint(
    RingIsingCudaBackend::Impl &impl, const double *params,
    std::size_t num_params, bool compute_gradient) -> EnergyGradResult {
    validate_num_params(impl.expected_params, num_params);

    if (!compute_gradient) {
        return run_energy_only(impl, params, num_params);
    }

    const auto ops = build_ring_ops(impl.num_qubits, impl.num_layers, params,
                                    impl.fuse_ring_cnot_layer);
    const auto num_ops = ops.size();

    detail::launch_init_zero_state(impl.current.get(), impl.state_size);
    copy_device_buffer(state_slot(impl.checkpoints.get(), impl.state_size, 0),
                       impl.current.get(), impl.state_size);

    for (std::size_t block_idx = 0; block_idx < impl.num_chunks; block_idx++) {
        const auto block_start = block_idx * impl.checkpoint_interval_ops;
        const auto block_end =
            std::min(block_start + impl.checkpoint_interval_ops, num_ops);
        apply_ops_range_inplace(impl.current.get(), impl.state_size, ops,
                                block_start, block_end, impl.scratch.get(),
                                impl.num_qubits);
        copy_device_buffer(
            state_slot(impl.checkpoints.get(), impl.state_size, block_idx + 1),
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
    detail::check_cuda(cudaMemsetAsync(impl.block_fused_gradients.get(), 0,
                                       sizeof(double) * impl.expected_params, 0),
                       "zero block fused gradients");

    for (std::size_t block_idx = impl.num_chunks; block_idx-- > 0;) {
        const auto block_start = block_idx * impl.checkpoint_interval_ops;
        const auto block_end =
            std::min(block_start + impl.checkpoint_interval_ops, num_ops);
        const auto local_count = block_end - block_start;

        copy_device_buffer(impl.current.get(),
                           state_slot(impl.checkpoints.get(), impl.state_size,
                                      block_idx),
                           impl.state_size);
        copy_device_buffer(state_slot(impl.local_states.get(), impl.state_size, 0),
                           impl.current.get(), impl.state_size);

        for (std::size_t local_idx = 0; local_idx < local_count; local_idx++) {
            apply_op_inplace(impl.current.get(), impl.state_size,
                             ops[block_start + local_idx], false,
                             impl.scratch.get(), impl.num_qubits);
            copy_device_buffer(
                state_slot(impl.local_states.get(), impl.state_size,
                           local_idx + 1),
                impl.current.get(), impl.state_size);
        }

        std::size_t current_op = block_end;
        while (current_op > block_start) {
            const auto op_idx = current_op - 1;
            const auto &op = ops[op_idx];

            if (op.kind == OpKind::FusedRYRZ) {
                std::size_t run_start = op_idx;
                while (run_start > block_start &&
                       ops[run_start - 1].kind == OpKind::FusedRYRZ) {
                    run_start--;
                }
                const auto run_count = current_op - run_start;

                for (std::size_t reverse_idx = current_op;
                     reverse_idx-- > run_start;) {
                    const auto slot = reverse_idx - run_start;
                    copy_device_buffer(
                        state_slot(impl.block_fused_lambda_states.get(),
                                   impl.state_size, slot),
                        impl.lambda.get(), impl.state_size);
                    if (reverse_idx != 0) {
                        apply_op_inplace(impl.lambda.get(), impl.state_size,
                                         ops[reverse_idx], true,
                                         impl.scratch.get(), impl.num_qubits);
                    }
                }

                launch_block_fused_ryrz_gradients(impl, run_start, block_start,
                                                  run_count, ops);
                current_op = run_start;
                continue;
            }

            const auto local_before = op_idx - block_start;

            if (op.is_parametric) {
                const Complex *state_before =
                    state_slot(impl.local_states.get(), impl.state_size,
                               local_before);
                if (op.kind == OpKind::FusedRYRZ) {
                    const auto [grad_theta, grad_phi] = detail::fused_ryrz_gradients(
                        impl.lambda.get(), state_before, impl.state_size,
                        op.wire0, op.theta0, op.theta1);
                    store_scalar_gradient(impl, op.param_index0, grad_theta);
                    store_scalar_gradient(impl, op.param_index1, grad_phi);
                } else {
                    apply_param_derivative(impl.deriv.get(), state_before,
                                           impl.state_size, op);
                    const Complex grad_complex =
                        detail::complex_inner_product(impl.lambda.get(),
                                                      impl.deriv.get(),
                                                      impl.state_size);
                    store_scalar_gradient(impl, op.param_index0,
                                          2.0 * grad_complex.real());
                }
            }

            if (op_idx != 0) {
                apply_op_inplace(impl.lambda.get(), impl.state_size, op, true,
                                 impl.scratch.get(), impl.num_qubits);
            }
            current_op = op_idx;
        }
    }

    detail::check_cuda(cudaMemcpy(result.gradient.data(),
                                  impl.block_fused_gradients.get(),
                                  sizeof(double) * result.gradient.size(),
                                  cudaMemcpyDeviceToHost),
                       "copy block fused gradients to host");

    return result;
}
