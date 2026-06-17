#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <thrust/complex.h>

namespace standalone_backend {
namespace detail {

__global__ void fill_identity_matrices_kernel(Complex *mats, std::size_t batch,
                                              std::size_t dim) {
    const auto mat_elements = dim * dim;
    const auto total = batch * mat_elements;
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= total) {
        return;
    }

    const auto elem = index % mat_elements;
    const auto row = elem % dim;
    const auto col = elem / dim;
    mats[index] = (row == col) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
}

__global__ void scatter_parent_vectors_kernel(const Complex *parent_vectors,
                                              Complex *child_vectors,
                                              std::size_t parent_count,
                                              std::size_t vector_size) {
    const auto total = parent_count * vector_size;
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= total) {
        return;
    }
    const auto parent_index = index / vector_size;
    const auto elem = index % vector_size;
    child_vectors[(2 * parent_index) * vector_size + elem] = parent_vectors[index];
}

__global__ void fused_dense_gradient_tail_kernel(const Complex *gate_mats,
                                                 const Complex *dgate_mats,
                                                 const int *param_gate_indices,
                                                 const Complex *psi_before,
                                                 const Complex *eta_before,
                                                 double *out,
                                                 std::size_t vector_size) {
    __shared__ Complex x_shared[DENSE_SCAN_MAX_DIM];
    __shared__ Complex y_shared[DENSE_SCAN_MAX_DIM];
    __shared__ double partial[THREADS];

    const auto param_index = static_cast<std::size_t>(blockIdx.x);
    const auto gate_index_signed = param_gate_indices[param_index];
    if (gate_index_signed < 0) {
        if (threadIdx.x == 0) {
            out[param_index] = 0.0;
        }
        return;
    }

    const auto gate_index = static_cast<std::size_t>(gate_index_signed);
    const auto mat_elements = vector_size * vector_size;
    const Complex *gate_mat = gate_mats + gate_index * mat_elements;
    const Complex *dgate_mat = dgate_mats + param_index * mat_elements;
    const Complex *x = psi_before + gate_index * vector_size;
    const Complex *y = eta_before + gate_index * vector_size;

    if (threadIdx.x < vector_size) {
        x_shared[threadIdx.x] = x[threadIdx.x];
        y_shared[threadIdx.x] = y[threadIdx.x];
    }
    __syncthreads();

    double local_sum = 0.0;
    if (threadIdx.x < vector_size) {
        const auto row = static_cast<std::size_t>(threadIdx.x);
        Complex uy(0.0, 0.0);
        Complex dx(0.0, 0.0);
        for (std::size_t col = 0; col < vector_size; col++) {
            const auto offset = row + col * vector_size;
            uy += gate_mat[offset] * y_shared[col];
            dx += dgate_mat[offset] * x_shared[col];
        }
        local_sum = (thrust::conj(uy) * dx).real();
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[param_index] = 2.0 * partial[0];
    }
}

__device__ auto apply_ring_cnot_layer_basis_device(std::size_t index,
                                                   std::size_t num_qubits,
                                                   bool inverse)
    -> std::size_t {
    auto transformed = index;
    if (!inverse) {
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            if (((transformed >> wire) & std::size_t{1}) != 0U) {
                transformed ^= (std::size_t{1} << ((wire + 1) % num_qubits));
            }
        }
    } else {
        for (std::size_t wire = num_qubits; wire-- > 0;) {
            if (((transformed >> wire) & std::size_t{1}) != 0U) {
                transformed ^= (std::size_t{1} << ((wire + 1) % num_qubits));
            }
        }
    }
    return transformed;
}

__device__ void apply_op_to_state(const Complex *in, Complex *out, std::size_t dim,
                                  std::size_t num_qubits, const OpDesc &op,
                                  bool inverse) {
    switch (op.kind) {
    case OpKind::RY: {
        const auto pairs = dim / 2;
        const auto half_stride = std::size_t{1} << op.wire0;
        const double theta = inverse ? -op.theta0 : op.theta0;
        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);
        for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
             pair_index < pairs; pair_index += static_cast<std::size_t>(blockDim.x)) {
            const auto low_mask = half_stride - 1;
            const auto base =
                ((pair_index >> op.wire0) << (op.wire0 + 1)) | (pair_index & low_mask);
            const auto i0 = base;
            const auto i1 = base | half_stride;
            const Complex a0 = in[i0];
            const Complex a1 = in[i1];
            out[i0] = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
            out[i1] = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;
        }
        return;
    }
    case OpKind::RZ: {
        const double theta = inverse ? -op.theta0 : op.theta0;
        const double half_theta = theta * 0.5;
        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
             index += static_cast<std::size_t>(blockDim.x)) {
            const bool bit = bit_is_set(index, op.wire0);
            const double phase = bit ? half_theta : -half_theta;
            const Complex factor(std::cos(phase), std::sin(phase));
            out[index] = factor * in[index];
        }
        return;
    }
    case OpKind::CNOT: {
        const auto control_mask = std::size_t{1} << op.wire0;
        const auto target_mask = std::size_t{1} << op.wire1;
        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
             index += static_cast<std::size_t>(blockDim.x)) {
            auto transformed = index;
            if ((transformed & control_mask) != 0U) {
                transformed ^= target_mask;
            }
            out[transformed] = in[index];
        }
        return;
    }
    case OpKind::FusedRYRZ: {
        const auto pairs = dim / 2;
        const auto half_stride = std::size_t{1} << op.wire0;
        if (!inverse) {
            const double c = std::cos(op.theta0 * 0.5);
            const double s = std::sin(op.theta0 * 0.5);
            const double half_phi = op.theta1 * 0.5;
            const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
            const Complex phase1(std::cos(half_phi), std::sin(half_phi));
            for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
                 pair_index < pairs;
                 pair_index += static_cast<std::size_t>(blockDim.x)) {
                const auto low_mask = half_stride - 1;
                const auto base = ((pair_index >> op.wire0) << (op.wire0 + 1)) |
                                  (pair_index & low_mask);
                const auto i0 = base;
                const auto i1 = base | half_stride;
                const Complex a0 = in[i0];
                const Complex a1 = in[i1];
                const Complex b0 = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
                const Complex b1 = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;
                out[i0] = phase0 * b0;
                out[i1] = phase1 * b1;
            }
        } else {
            const double c = std::cos(op.theta0 * 0.5);
            const double s = std::sin(op.theta0 * 0.5);
            const double half_phi = op.theta1 * 0.5;
            const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
            const Complex phase1(std::cos(half_phi), std::sin(half_phi));
            for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
                 pair_index < pairs;
                 pair_index += static_cast<std::size_t>(blockDim.x)) {
                const auto low_mask = half_stride - 1;
                const auto base = ((pair_index >> op.wire0) << (op.wire0 + 1)) |
                                  (pair_index & low_mask);
                const auto i0 = base;
                const auto i1 = base | half_stride;
                const Complex a0 = in[i0];
                const Complex a1 = in[i1];
                out[i0] = phase1 * Complex(c, 0.0) * a0 +
                          phase0 * Complex(s, 0.0) * a1;
                out[i1] = phase1 * Complex(-s, 0.0) * a0 +
                          phase0 * Complex(c, 0.0) * a1;
            }
        }
        return;
    }
    case OpKind::RingCNOTLayer: {
        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
             index += static_cast<std::size_t>(blockDim.x)) {
            const auto transformed =
                apply_ring_cnot_layer_basis_device(index, num_qubits, inverse);
            out[transformed] = in[index];
        }
        return;
    }
    }
}

__global__ void simulate_blocks_forward_kernel(const OpDesc *ops,
                                               std::size_t num_blocks,
                                               std::size_t block_size,
                                               std::size_t num_ops,
                                               std::size_t num_qubits,
                                               std::size_t dim,
                                               const Complex *boundary_states,
                                               Complex *forward_states) {
    const auto block_id = static_cast<std::size_t>(blockIdx.x);
    if (block_id >= num_blocks) {
        return;
    }

    const auto block_start = block_id * block_size;
    if (block_start >= num_ops) {
        return;
    }
    const auto remaining = num_ops - block_start;
    const auto block_length = block_size < remaining ? block_size : remaining;
    extern __shared__ unsigned char shared_raw[];
    auto *shared = reinterpret_cast<Complex *>(shared_raw);
    Complex *current = shared;
    Complex *next = shared + dim;
    const Complex *boundary = boundary_states + block_id * dim;
    Complex *block_out =
        forward_states + block_id * (block_size + 1) * dim;

    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
         index += static_cast<std::size_t>(blockDim.x)) {
        current[index] = boundary[index];
        block_out[index] = boundary[index];
    }
    __syncthreads();

    for (std::size_t local_idx = 0; local_idx < block_length; local_idx++) {
        apply_op_to_state(current, next, dim, num_qubits, ops[block_start + local_idx],
                          false);
        __syncthreads();
        Complex *state_out = block_out + (local_idx + 1) * dim;
        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
             index += static_cast<std::size_t>(blockDim.x)) {
            state_out[index] = next[index];
        }
        __syncthreads();
        Complex *tmp = current;
        current = next;
        next = tmp;
    }
}

__global__ void simulate_blocks_backward_and_gradient_kernel(
    const OpDesc *ops, std::size_t num_blocks, std::size_t block_size,
    std::size_t num_ops, std::size_t num_qubits, std::size_t dim,
    const Complex *lambda_boundaries, const Complex *forward_states,
    double *out_gradients) {
    const auto block_id = static_cast<std::size_t>(blockIdx.x);
    if (block_id >= num_blocks) {
        return;
    }

    const auto block_start = block_id * block_size;
    if (block_start >= num_ops) {
        return;
    }
    const auto remaining = num_ops - block_start;
    const auto block_length = block_size < remaining ? block_size : remaining;
    extern __shared__ unsigned char shared_raw[];
    auto *shared = reinterpret_cast<Complex *>(shared_raw);
    Complex *current = shared;
    Complex *next = shared + dim;
    __shared__ double theta_partial[THREADS];
    __shared__ double phi_partial[THREADS];

    const Complex *lambda_end = lambda_boundaries + (block_id + 1) * dim;
    const Complex *block_forward =
        forward_states + block_id * (block_size + 1) * dim;
    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
         index += static_cast<std::size_t>(blockDim.x)) {
        current[index] = lambda_end[index];
    }
    __syncthreads();

    for (std::size_t local_rev = block_length; local_rev-- > 0;) {
        const OpDesc op = ops[block_start + local_rev];
        const Complex *psi_before = block_forward + local_rev * dim;
        double theta_sum = 0.0;
        double phi_sum = 0.0;

        if (op.is_parametric) {
            switch (op.kind) {
            case OpKind::FusedRYRZ: {
                const auto pairs = dim / 2;
                const auto half_stride = std::size_t{1} << op.wire0;
                const double c = std::cos(op.theta0 * 0.5);
                const double s = std::sin(op.theta0 * 0.5);
                const double half_phi = op.theta1 * 0.5;
                const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
                const Complex phase1(std::cos(half_phi), std::sin(half_phi));
                const Complex pref0(0.0, -0.5);
                const Complex pref1(0.0, 0.5);
                for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
                     pair_index < pairs;
                     pair_index += static_cast<std::size_t>(blockDim.x)) {
                    const auto low_mask = half_stride - 1;
                    const auto base =
                        ((pair_index >> op.wire0) << (op.wire0 + 1)) |
                        (pair_index & low_mask);
                    const auto i0 = base;
                    const auto i1 = base | half_stride;
                    const Complex a0 = psi_before[i0];
                    const Complex a1 = psi_before[i1];
                    const Complex l0 = current[i0];
                    const Complex l1 = current[i1];
                    const Complex dtheta0 = phase0 * (Complex(-0.5 * s, 0.0) * a0 +
                                                      Complex(-0.5 * c, 0.0) * a1);
                    const Complex dtheta1 = phase1 * (Complex(0.5 * c, 0.0) * a0 +
                                                      Complex(-0.5 * s, 0.0) * a1);
                    theta_sum +=
                        (thrust::conj(l0) * dtheta0 + thrust::conj(l1) * dtheta1)
                            .real();

                    const Complex b0 = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
                    const Complex b1 = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;
                    const Complex dphi0 = pref0 * phase0 * b0;
                    const Complex dphi1 = pref1 * phase1 * b1;
                    phi_sum +=
                        (thrust::conj(l0) * dphi0 + thrust::conj(l1) * dphi1).real();
                }
                break;
            }
            case OpKind::RY: {
                const auto pairs = dim / 2;
                const auto half_stride = std::size_t{1} << op.wire0;
                const double c = std::cos(op.theta0 * 0.5);
                const double s = std::sin(op.theta0 * 0.5);
                for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
                     pair_index < pairs;
                     pair_index += static_cast<std::size_t>(blockDim.x)) {
                    const auto low_mask = half_stride - 1;
                    const auto base =
                        ((pair_index >> op.wire0) << (op.wire0 + 1)) |
                        (pair_index & low_mask);
                    const auto i0 = base;
                    const auto i1 = base | half_stride;
                    const Complex a0 = psi_before[i0];
                    const Complex a1 = psi_before[i1];
                    const Complex l0 = current[i0];
                    const Complex l1 = current[i1];
                    const Complex d0 =
                        Complex(-0.5 * s, 0.0) * a0 + Complex(-0.5 * c, 0.0) * a1;
                    const Complex d1 =
                        Complex(0.5 * c, 0.0) * a0 + Complex(-0.5 * s, 0.0) * a1;
                    theta_sum +=
                        (thrust::conj(l0) * d0 + thrust::conj(l1) * d1).real();
                }
                break;
            }
            case OpKind::RZ: {
                const double half_theta = op.theta0 * 0.5;
                for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dim;
                     index += static_cast<std::size_t>(blockDim.x)) {
                    const bool bit = bit_is_set(index, op.wire0);
                    const double phase = bit ? half_theta : -half_theta;
                    const Complex factor(std::cos(phase), std::sin(phase));
                    const Complex prefactor =
                        bit ? Complex(0.0, 0.5) : Complex(0.0, -0.5);
                    theta_sum +=
                        (thrust::conj(current[index]) * prefactor * factor *
                         psi_before[index])
                            .real();
                }
                break;
            }
            case OpKind::CNOT:
            case OpKind::RingCNOTLayer:
                break;
            }
        }

        theta_partial[threadIdx.x] = theta_sum;
        phi_partial[threadIdx.x] = phi_sum;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                theta_partial[threadIdx.x] += theta_partial[threadIdx.x + stride];
                phi_partial[threadIdx.x] += phi_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && op.is_parametric) {
            out_gradients[op.param_index0] = 2.0 * theta_partial[0];
            if (op.kind == OpKind::FusedRYRZ) {
                out_gradients[op.param_index1] = 2.0 * phi_partial[0];
            }
        }
        __syncthreads();

        apply_op_to_state(current, next, dim, num_qubits, op, true);
        __syncthreads();
        Complex *tmp = current;
        current = next;
        next = tmp;
    }
}

void launch_fill_identity_matrices(Complex *mats, std::size_t batch,
                                   std::size_t dim) {
    const auto total = batch * dim * dim;
    const auto blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    fill_identity_matrices_kernel<<<blocks, THREADS>>>(mats, batch, dim);
    check_cuda(cudaGetLastError(), "fill_identity_matrices_kernel");
    maybe_synchronize_cuda("fill_identity_matrices_kernel sync");
}

void launch_scatter_parent_vectors(const Complex *parent_vectors,
                                   Complex *child_vectors,
                                   std::size_t parent_count,
                                   std::size_t vector_size) {
    const auto total = parent_count * vector_size;
    const auto blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    scatter_parent_vectors_kernel<<<blocks, THREADS>>>(
        parent_vectors, child_vectors, parent_count, vector_size);
    check_cuda(cudaGetLastError(), "scatter_parent_vectors_kernel");
    maybe_synchronize_cuda("scatter_parent_vectors_kernel sync");
}

void launch_fused_dense_gradient_tail(const Complex *gate_mats,
                                      const Complex *dgate_mats,
                                      const int *param_gate_indices,
                                      const Complex *psi_before,
                                      const Complex *eta_before, double *out,
                                      std::size_t num_params,
                                      std::size_t vector_size) {
    if (vector_size > DENSE_SCAN_MAX_DIM) {
        throw std::runtime_error(
            "launch_fused_dense_gradient_tail only supports vector_size <= 64.");
    }
    fused_dense_gradient_tail_kernel<<<static_cast<int>(num_params), THREADS>>>(
        gate_mats, dgate_mats, param_gate_indices, psi_before, eta_before, out,
        vector_size);
    check_cuda(cudaGetLastError(), "fused_dense_gradient_tail_kernel");
    maybe_synchronize_cuda("fused_dense_gradient_tail_kernel sync");
}

void launch_simulate_blocks_forward(const OpDesc *ops, std::size_t num_blocks,
                                    std::size_t block_size, std::size_t num_ops,
                                    std::size_t num_qubits, std::size_t dim,
                                    const Complex *boundary_states,
                                    Complex *forward_states) {
    const auto shared_bytes = static_cast<std::size_t>(2 * dim * sizeof(Complex));
    simulate_blocks_forward_kernel<<<static_cast<unsigned int>(num_blocks), THREADS,
                                     shared_bytes>>>(
        ops, num_blocks, block_size, num_ops, num_qubits, dim, boundary_states,
        forward_states);
    check_cuda(cudaGetLastError(), "simulate_blocks_forward_kernel");
    maybe_synchronize_cuda("simulate_blocks_forward_kernel sync");
}

void launch_simulate_blocks_backward_and_gradient(
    const OpDesc *ops, std::size_t num_blocks, std::size_t block_size,
    std::size_t num_ops, std::size_t num_qubits, std::size_t dim,
    const Complex *lambda_boundaries, const Complex *forward_states,
    double *out_gradients) {
    const auto shared_bytes = static_cast<std::size_t>(2 * dim * sizeof(Complex));
    simulate_blocks_backward_and_gradient_kernel<<<
        static_cast<unsigned int>(num_blocks), THREADS, shared_bytes>>>(
        ops, num_blocks, block_size, num_ops, num_qubits, dim, lambda_boundaries,
        forward_states, out_gradients);
    check_cuda(cudaGetLastError(),
               "simulate_blocks_backward_and_gradient_kernel");
    maybe_synchronize_cuda(
        "simulate_blocks_backward_and_gradient_kernel sync");
}

} // namespace detail
} // namespace standalone_backend
