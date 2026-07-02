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

__device__ void fill_ryrz_local_u_entries_device(double theta, double phi,
                                                 Complex *u) {
    const double c = std::cos(theta * 0.5);
    const double s = std::sin(theta * 0.5);
    const double half_phi = phi * 0.5;
    const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
    const Complex phase1(std::cos(half_phi), std::sin(half_phi));

    u[0] = phase0 * Complex(c, 0.0);
    u[1] = phase1 * Complex(s, 0.0);
    u[2] = phase0 * Complex(-s, 0.0);
    u[3] = phase1 * Complex(c, 0.0);
}

__device__ void fill_ryrz_local_entries_device(double theta, double phi,
                                               Complex *u, Complex *dtheta,
                                               Complex *dphi) {
    const double c = std::cos(theta * 0.5);
    const double s = std::sin(theta * 0.5);
    const double half_phi = phi * 0.5;
    const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
    const Complex phase1(std::cos(half_phi), std::sin(half_phi));

    u[0] = phase0 * Complex(c, 0.0);
    u[1] = phase1 * Complex(s, 0.0);
    u[2] = phase0 * Complex(-s, 0.0);
    u[3] = phase1 * Complex(c, 0.0);

    dtheta[0] = phase0 * Complex(-0.5 * s, 0.0);
    dtheta[1] = phase1 * Complex(0.5 * c, 0.0);
    dtheta[2] = phase0 * Complex(-0.5 * c, 0.0);
    dtheta[3] = phase1 * Complex(-0.5 * s, 0.0);

    const Complex pref0(0.0, -0.5);
    const Complex pref1(0.0, 0.5);
    dphi[0] = pref0 * u[0];
    dphi[1] = pref1 * u[1];
    dphi[2] = pref0 * u[2];
    dphi[3] = pref1 * u[3];
}

__device__ void apply_local_matrix_to_shared_vector(const Complex *in,
                                                    Complex *out,
                                                    std::size_t dim,
                                                    std::size_t wire,
                                                    const Complex *local) {
    const auto pairs = dim / 2;
    const auto half_stride = std::size_t{1} << wire;
    const auto low_mask = half_stride - 1;
    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs; pair_index += static_cast<std::size_t>(blockDim.x)) {
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;
        const Complex a0 = in[i0];
        const Complex a1 = in[i1];
        out[i0] = local[0] * a0 + local[2] * a1;
        out[i1] = local[1] * a0 + local[3] * a1;
    }
}

__device__ auto local_entry_for_basis_bits_device(const Complex *local,
                                                  std::size_t row,
                                                  std::size_t col,
                                                  std::size_t wire) -> Complex {
    const auto row_bit = (row >> wire) & std::size_t{1};
    const auto col_bit = (col >> wire) & std::size_t{1};
    return local[col_bit * 2 + row_bit];
}

__global__ void fill_rotation_layer_matrices_kernel(
    Complex *gate_mats, const int *param_gate_indices, const double *params,
    std::size_t num_layers, std::size_t num_qubits, std::size_t dim) {
    __shared__ Complex local_u[DENSE_SCAN_MAX_QUBITS * 4];

    const auto layer = static_cast<std::size_t>(blockIdx.x);
    if (layer >= num_layers) {
        return;
    }

    const auto param_base = layer * num_qubits * 2;
    const auto gate_index_signed = param_gate_indices[param_base];
    if (gate_index_signed < 0) {
        return;
    }
    const auto gate_index = static_cast<std::size_t>(gate_index_signed);

    if (threadIdx.x < num_qubits) {
        const auto wire = static_cast<std::size_t>(threadIdx.x);
        fill_ryrz_local_u_entries_device(params[param_base + 2 * wire],
                                         params[param_base + 2 * wire + 1],
                                         local_u + 4 * wire);
    }
    __syncthreads();

    const auto mat_elements = dim * dim;
    Complex *matrix = gate_mats + gate_index * mat_elements;
    for (std::size_t elem = static_cast<std::size_t>(threadIdx.x);
         elem < mat_elements; elem += static_cast<std::size_t>(blockDim.x)) {
        const auto row = elem % dim;
        const auto col = elem / dim;
        Complex value(1.0, 0.0);
        for (std::size_t wire = 0; wire < num_qubits; wire++) {
            value *= local_entry_for_basis_bits_device(local_u + 4 * wire, row,
                                                       col, wire);
        }
        matrix[elem] = value;
    }
}

__global__ void rotation_layer_dense_gradient_tail_kernel(
    const int *param_gate_indices, const double *params,
    const Complex *psi_before, const Complex *eta_before, double *out,
    std::size_t num_layers, std::size_t num_qubits, std::size_t vector_size) {
    __shared__ Complex x_shared[DENSE_SCAN_MAX_DIM];
    __shared__ Complex y_shared[DENSE_SCAN_MAX_DIM];
    __shared__ Complex lambda_after[DENSE_SCAN_MAX_DIM];
    __shared__ Complex work_a[DENSE_SCAN_MAX_DIM];
    __shared__ Complex work_b[DENSE_SCAN_MAX_DIM];
    __shared__ Complex local_u[DENSE_SCAN_MAX_QUBITS * 4];
    __shared__ Complex local_dtheta[DENSE_SCAN_MAX_QUBITS * 4];
    __shared__ Complex local_dphi[DENSE_SCAN_MAX_QUBITS * 4];
    __shared__ double partial[THREADS];

    const auto layer = static_cast<std::size_t>(blockIdx.x);
    if (layer >= num_layers) {
        return;
    }
    const auto param_base = layer * num_qubits * 2;
    const auto gate_index_signed = param_gate_indices[param_base];
    if (gate_index_signed < 0) {
        for (std::size_t local_param = static_cast<std::size_t>(threadIdx.x);
             local_param < num_qubits * 2;
             local_param += static_cast<std::size_t>(blockDim.x)) {
            out[param_base + local_param] = 0.0;
        }
        return;
    }
    const auto gate_index = static_cast<std::size_t>(gate_index_signed);

    if (threadIdx.x < num_qubits) {
        const auto wire = static_cast<std::size_t>(threadIdx.x);
        fill_ryrz_local_entries_device(params[param_base + 2 * wire],
                                       params[param_base + 2 * wire + 1],
                                       local_u + 4 * wire,
                                       local_dtheta + 4 * wire,
                                       local_dphi + 4 * wire);
    }
    for (std::size_t index = static_cast<std::size_t>(threadIdx.x);
         index < vector_size; index += static_cast<std::size_t>(blockDim.x)) {
        x_shared[index] = psi_before[gate_index * vector_size + index];
        y_shared[index] = eta_before[gate_index * vector_size + index];
        work_a[index] = y_shared[index];
    }
    __syncthreads();

    Complex *src = work_a;
    Complex *dst = work_b;
    for (std::size_t wire = 0; wire < num_qubits; wire++) {
        apply_local_matrix_to_shared_vector(src, dst, vector_size, wire,
                                            local_u + 4 * wire);
        __syncthreads();
        Complex *tmp = src;
        src = dst;
        dst = tmp;
    }
    for (std::size_t index = static_cast<std::size_t>(threadIdx.x);
         index < vector_size; index += static_cast<std::size_t>(blockDim.x)) {
        lambda_after[index] = src[index];
    }
    __syncthreads();

    for (std::size_t target_wire = 0; target_wire < num_qubits; target_wire++) {
        for (std::size_t derivative_kind = 0; derivative_kind < 2;
             derivative_kind++) {
            for (std::size_t index = static_cast<std::size_t>(threadIdx.x);
                 index < vector_size;
                 index += static_cast<std::size_t>(blockDim.x)) {
                work_a[index] = x_shared[index];
            }
            __syncthreads();

            src = work_a;
            dst = work_b;
            for (std::size_t wire = 0; wire < num_qubits; wire++) {
                const Complex *local = local_u + 4 * wire;
                if (wire == target_wire) {
                    local = derivative_kind == 0 ? local_dtheta + 4 * wire
                                                 : local_dphi + 4 * wire;
                }
                apply_local_matrix_to_shared_vector(src, dst, vector_size, wire,
                                                    local);
                __syncthreads();
                Complex *tmp = src;
                src = dst;
                dst = tmp;
            }

            double local_sum = 0.0;
            for (std::size_t index = static_cast<std::size_t>(threadIdx.x);
                 index < vector_size;
                 index += static_cast<std::size_t>(blockDim.x)) {
                local_sum += (thrust::conj(lambda_after[index]) * src[index]).real();
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
                out[param_base + 2 * target_wire + derivative_kind] =
                    2.0 * partial[0];
            }
            __syncthreads();
        }
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

__global__ void fill_ring_cnot_layer_matrices_kernel(
    Complex *gate_mats, std::size_t num_layers, std::size_t num_qubits,
    std::size_t dim) {
    const auto layer = static_cast<std::size_t>(blockIdx.x);
    if (layer >= num_layers) {
        return;
    }

    const auto mat_elements = dim * dim;
    const auto gate_index = 2 * layer + 1;
    Complex *matrix = gate_mats + gate_index * mat_elements;
    for (std::size_t elem = static_cast<std::size_t>(threadIdx.x);
         elem < mat_elements; elem += static_cast<std::size_t>(blockDim.x)) {
        const auto row = elem % dim;
        const auto col = elem / dim;
        const auto target =
            apply_ring_cnot_layer_basis_device(col, num_qubits, false);
        matrix[elem] = row == target ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
    }
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
    case OpKind::RotationLayer:
        return;
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
            case OpKind::RotationLayer:
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
        if (threadIdx.x == 0 && op.is_parametric &&
            op.kind != OpKind::RotationLayer) {
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

void launch_fill_rotation_layer_matrices(Complex *gate_mats,
                                         const int *param_gate_indices,
                                         const double *params,
                                         std::size_t num_layers,
                                         std::size_t num_qubits,
                                         std::size_t dim) {
    if (dim > DENSE_SCAN_MAX_DIM || num_qubits > DENSE_SCAN_MAX_QUBITS) {
        throw std::runtime_error(
            "launch_fill_rotation_layer_matrices supports <= 8 qubits.");
    }
    fill_rotation_layer_matrices_kernel<<<static_cast<int>(num_layers), THREADS>>>(
        gate_mats, param_gate_indices, params, num_layers, num_qubits, dim);
    check_cuda(cudaGetLastError(), "fill_rotation_layer_matrices_kernel");
    maybe_synchronize_cuda("fill_rotation_layer_matrices_kernel sync");
}

void launch_fill_ring_cnot_layer_matrices(Complex *gate_mats,
                                          std::size_t num_layers,
                                          std::size_t num_qubits,
                                          std::size_t dim) {
    if (dim > DENSE_SCAN_MAX_DIM || num_qubits > DENSE_SCAN_MAX_QUBITS) {
        throw std::runtime_error(
            "launch_fill_ring_cnot_layer_matrices supports <= 8 qubits.");
    }
    fill_ring_cnot_layer_matrices_kernel<<<static_cast<int>(num_layers),
                                           THREADS>>>(
        gate_mats, num_layers, num_qubits, dim);
    check_cuda(cudaGetLastError(), "fill_ring_cnot_layer_matrices_kernel");
    maybe_synchronize_cuda("fill_ring_cnot_layer_matrices_kernel sync");
}

void launch_rotation_layer_dense_gradient_tail(
    const int *param_gate_indices, const double *params,
    const Complex *psi_before, const Complex *eta_before, double *out,
    std::size_t num_layers, std::size_t num_qubits, std::size_t vector_size) {
    if (vector_size > DENSE_SCAN_MAX_DIM || num_qubits > DENSE_SCAN_MAX_QUBITS) {
        throw std::runtime_error(
            "launch_rotation_layer_dense_gradient_tail supports <= 8 qubits.");
    }
    rotation_layer_dense_gradient_tail_kernel<<<static_cast<int>(num_layers),
                                                THREADS>>>(
        param_gate_indices, params, psi_before, eta_before, out, num_layers,
        num_qubits, vector_size);
    check_cuda(cudaGetLastError(), "rotation_layer_dense_gradient_tail_kernel");
    maybe_synchronize_cuda("rotation_layer_dense_gradient_tail_kernel sync");
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
