#include "ising_cuda_backend_internal.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda/std/functional>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace standalone_backend {
namespace detail {

void check_cuda(cudaError_t status, const char *context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char *context) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(context) +
                             ": cuBLAS call failed with status " +
                             std::to_string(static_cast<int>(status)));
}

void maybe_synchronize_cuda(const char *context) {
#ifdef STANDALONE_CUDA_EAGER_SYNC
    check_cuda(cudaDeviceSynchronize(), context);
#else
    (void)context;
#endif
}

struct ConjugateProduct {
    __host__ __device__ auto operator()(const thrust::tuple<Complex, Complex> &item) const
        -> Complex {
        const Complex lhs_value = thrust::get<0>(item);
        const Complex rhs_value = thrust::get<1>(item);
        return Complex(lhs_value.real(), -lhs_value.imag()) * rhs_value;
    }
};

constexpr std::size_t DENSE_SCAN_MAX_DIM = 64;

__host__ __device__ inline auto bit_is_set(std::size_t index,
                                           std::size_t wire) -> bool {
    return ((index >> wire) & 1U) != 0U;
}

__global__ void init_zero_state_kernel(Complex *state, std::size_t size) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }
    state[index] = (index == 0) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
}

__global__ void apply_ry_kernel(Complex *state, std::size_t size,
                                std::size_t wire, double theta) {
    const auto half_stride = std::size_t{1} << wire;
    const auto pair_index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                     threadIdx.x);
    if (pair_index >= size / 2) {
        return;
    }

    const auto low_mask = half_stride - 1;
    const auto base =
        ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
    const auto i0 = base;
    const auto i1 = base | half_stride;

    const double c = std::cos(theta * 0.5);
    const double s = std::sin(theta * 0.5);
    const Complex a0 = state[i0];
    const Complex a1 = state[i1];

    state[i0] = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
    state[i1] = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;
}

__global__ void apply_dry_kernel(Complex *out, const Complex *state,
                                 std::size_t size, std::size_t wire,
                                 double theta) {
    const auto half_stride = std::size_t{1} << wire;
    const auto pair_index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                     threadIdx.x);
    if (pair_index >= size / 2) {
        return;
    }

    const auto low_mask = half_stride - 1;
    const auto base =
        ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
    const auto i0 = base;
    const auto i1 = base | half_stride;

    const double c = std::cos(theta * 0.5);
    const double s = std::sin(theta * 0.5);
    const Complex a0 = state[i0];
    const Complex a1 = state[i1];

    out[i0] =
        Complex(-0.5 * s, 0.0) * a0 + Complex(-0.5 * c, 0.0) * a1;
    out[i1] = Complex(0.5 * c, 0.0) * a0 + Complex(-0.5 * s, 0.0) * a1;
}

__global__ void apply_rz_kernel(Complex *state, std::size_t size,
                                std::size_t wire, double theta) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    const double half_theta = theta * 0.5;
    const bool bit = bit_is_set(index, wire);
    const double phase = bit ? half_theta : -half_theta;
    const Complex factor(std::cos(phase), std::sin(phase));
    state[index] *= factor;
}

__global__ void apply_drz_kernel(Complex *out, const Complex *state,
                                 std::size_t size, std::size_t wire,
                                 double theta) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    const double half_theta = theta * 0.5;
    const bool bit = bit_is_set(index, wire);
    const double phase = bit ? half_theta : -half_theta;
    const Complex factor(std::cos(phase), std::sin(phase));
    const Complex prefactor = bit ? Complex(0.0, 0.5) : Complex(0.0, -0.5);
    out[index] = prefactor * factor * state[index];
}

__global__ void apply_cnot_kernel(Complex *state, std::size_t size,
                                  std::size_t control, std::size_t target) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    if (!bit_is_set(index, control) || bit_is_set(index, target)) {
        return;
    }

    const auto partner = index | (std::size_t{1} << target);
    const Complex tmp = state[index];
    state[index] = state[partner];
    state[partner] = tmp;
}

__global__ void apply_ring_cnot_layer_kernel(Complex *out, const Complex *in,
                                             std::size_t size,
                                             std::size_t num_qubits,
                                             bool inverse) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    std::size_t transformed = index;
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
    out[transformed] = in[index];
}

__global__ void apply_ryrz_kernel(Complex *state, std::size_t size,
                                  std::size_t wire, double theta_ry,
                                  double theta_rz, bool inverse) {
    const auto half_stride = std::size_t{1} << wire;
    const auto pair_index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                     threadIdx.x);
    if (pair_index >= size / 2) {
        return;
    }

    const auto low_mask = half_stride - 1;
    const auto base =
        ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
    const auto i0 = base;
    const auto i1 = base | half_stride;

    const double c = std::cos(theta_ry * 0.5);
    const double s = std::sin(theta_ry * 0.5);
    const double half_phi = theta_rz * 0.5;
    const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
    const Complex phase1(std::cos(half_phi), std::sin(half_phi));

    const Complex a0 = state[i0];
    const Complex a1 = state[i1];
    if (!inverse) {
        const Complex b0 = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
        const Complex b1 = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;
        state[i0] = phase0 * b0;
        state[i1] = phase1 * b1;
        return;
    }

    state[i0] = phase1 * Complex(c, 0.0) * a0 + phase0 * Complex(s, 0.0) * a1;
    state[i1] = phase1 * Complex(-s, 0.0) * a0 + phase0 * Complex(c, 0.0) * a1;
}

__global__ void inverse_walk_ry_step_kernel(Complex *current, Complex *lambda,
                                            std::size_t size,
                                            std::size_t wire, double theta,
                                            double *out_gradient) {
    __shared__ double partial[THREADS];

    const auto half_stride = std::size_t{1} << wire;
    const auto pair_index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                     threadIdx.x);
    double local_sum = 0.0;

    if (pair_index < size / 2) {
        const auto low_mask = half_stride - 1;
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;

        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);

        const Complex current_after0 = current[i0];
        const Complex current_after1 = current[i1];
        const Complex lambda_after0 = lambda[i0];
        const Complex lambda_after1 = lambda[i1];

        const Complex current_before0 =
            Complex(c, 0.0) * current_after0 + Complex(s, 0.0) * current_after1;
        const Complex current_before1 =
            Complex(-s, 0.0) * current_after0 + Complex(c, 0.0) * current_after1;
        current[i0] = current_before0;
        current[i1] = current_before1;

        const Complex d0 = Complex(-0.5 * s, 0.0) * current_before0 +
                           Complex(-0.5 * c, 0.0) * current_before1;
        const Complex d1 = Complex(0.5 * c, 0.0) * current_before0 +
                           Complex(-0.5 * s, 0.0) * current_before1;
        local_sum =
            (thrust::conj(lambda_after0) * d0 +
             thrust::conj(lambda_after1) * d1)
                .real();

        lambda[i0] =
            Complex(c, 0.0) * lambda_after0 + Complex(s, 0.0) * lambda_after1;
        lambda[i1] =
            Complex(-s, 0.0) * lambda_after0 + Complex(c, 0.0) * lambda_after1;
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
        atomicAdd(out_gradient, 2.0 * partial[0]);
    }
}

__global__ void inverse_walk_rz_step_kernel(Complex *current, Complex *lambda,
                                            std::size_t size,
                                            std::size_t wire, double theta,
                                            double *out_gradient) {
    __shared__ double partial[THREADS];

    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    double local_sum = 0.0;

    if (index < size) {
        const double half_theta = theta * 0.5;
        const bool bit = bit_is_set(index, wire);
        const double forward_phase = bit ? half_theta : -half_theta;
        const Complex forward_factor(std::cos(forward_phase),
                                     std::sin(forward_phase));
        const Complex inverse_factor(std::cos(-forward_phase),
                                     std::sin(-forward_phase));
        const Complex prefactor =
            bit ? Complex(0.0, 0.5) : Complex(0.0, -0.5);

        const Complex current_after = current[index];
        const Complex lambda_after = lambda[index];
        const Complex current_before = inverse_factor * current_after;
        current[index] = current_before;

        local_sum =
            (thrust::conj(lambda_after) * prefactor * forward_factor *
             current_before)
                .real();

        lambda[index] = inverse_factor * lambda_after;
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
        atomicAdd(out_gradient, 2.0 * partial[0]);
    }
}

__global__ void inverse_walk_cnot_step_kernel(Complex *current, Complex *lambda,
                                              std::size_t size,
                                              std::size_t control,
                                              std::size_t target) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    if (!bit_is_set(index, control) || bit_is_set(index, target)) {
        return;
    }

    const auto partner = index | (std::size_t{1} << target);
    const Complex current_tmp = current[index];
    current[index] = current[partner];
    current[partner] = current_tmp;

    const Complex lambda_tmp = lambda[index];
    lambda[index] = lambda[partner];
    lambda[partner] = lambda_tmp;
}

__global__ void save_param_ry_step_kernel(const Complex *state_before,
                                          Complex *lambda, std::size_t size,
                                          std::size_t wire, double theta,
                                          double *out_gradient) {
    __shared__ double partial[THREADS];

    const auto half_stride = std::size_t{1} << wire;
    const auto pair_index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                     threadIdx.x);
    double local_sum = 0.0;

    if (pair_index < size / 2) {
        const auto low_mask = half_stride - 1;
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;

        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);

        const Complex psi0 = state_before[i0];
        const Complex psi1 = state_before[i1];
        const Complex lambda_after0 = lambda[i0];
        const Complex lambda_after1 = lambda[i1];

        const Complex d0 = Complex(-0.5 * s, 0.0) * psi0 +
                           Complex(-0.5 * c, 0.0) * psi1;
        const Complex d1 = Complex(0.5 * c, 0.0) * psi0 +
                           Complex(-0.5 * s, 0.0) * psi1;
        local_sum =
            (thrust::conj(lambda_after0) * d0 +
             thrust::conj(lambda_after1) * d1)
                .real();

        lambda[i0] =
            Complex(c, 0.0) * lambda_after0 + Complex(s, 0.0) * lambda_after1;
        lambda[i1] =
            Complex(-s, 0.0) * lambda_after0 + Complex(c, 0.0) * lambda_after1;
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
        atomicAdd(out_gradient, 2.0 * partial[0]);
    }
}

__global__ void save_param_rz_step_kernel(const Complex *state_before,
                                          Complex *lambda, std::size_t size,
                                          std::size_t wire, double theta,
                                          double *out_gradient) {
    __shared__ double partial[THREADS];

    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    double local_sum = 0.0;

    if (index < size) {
        const double half_theta = theta * 0.5;
        const bool bit = bit_is_set(index, wire);
        const double phase = bit ? half_theta : -half_theta;
        const Complex forward_factor(std::cos(phase), std::sin(phase));
        const Complex inverse_factor(std::cos(-phase), std::sin(-phase));
        const Complex prefactor =
            bit ? Complex(0.0, 0.5) : Complex(0.0, -0.5);

        const Complex psi = state_before[index];
        const Complex lambda_after = lambda[index];
        local_sum =
            (thrust::conj(lambda_after) * prefactor * forward_factor * psi)
                .real();
        lambda[index] = inverse_factor * lambda_after;
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
        atomicAdd(out_gradient, 2.0 * partial[0]);
    }
}

__global__ void apply_ring_ising_hamiltonian_kernel(Complex *out,
                                                    const Complex *state,
                                                    std::size_t size,
                                                    std::size_t num_qubits,
                                                    double field) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    double diag_coeff = 0.0;
    for (std::size_t wire = 0; wire < num_qubits; wire++) {
        const auto next_wire = (wire + 1) % num_qubits;
        const double zi = bit_is_set(index, wire) ? -1.0 : 1.0;
        const double zj = bit_is_set(index, next_wire) ? -1.0 : 1.0;
        diag_coeff += -(zi * zj);
    }

    Complex value = Complex(diag_coeff, 0.0) * state[index];
    for (std::size_t wire = 0; wire < num_qubits; wire++) {
        const auto flipped = index ^ (std::size_t{1} << wire);
        value += Complex(-field, 0.0) * state[flipped];
    }
    out[index] = value;
}

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

void launch_init_zero_state(Complex *state, std::size_t size) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    init_zero_state_kernel<<<blocks, THREADS>>>(state, size);
    check_cuda(cudaGetLastError(), "init_zero_state_kernel");
    maybe_synchronize_cuda("init_zero_state_kernel sync");
}

void launch_apply_ry(Complex *state, std::size_t size, std::size_t wire,
                     double theta) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    apply_ry_kernel<<<blocks, THREADS>>>(state, size, wire, theta);
    check_cuda(cudaGetLastError(), "apply_ry_kernel");
    maybe_synchronize_cuda("apply_ry_kernel sync");
}

void launch_apply_dry(Complex *out, const Complex *state, std::size_t size,
                      std::size_t wire, double theta) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    apply_dry_kernel<<<blocks, THREADS>>>(out, state, size, wire, theta);
    check_cuda(cudaGetLastError(), "apply_dry_kernel");
    maybe_synchronize_cuda("apply_dry_kernel sync");
}

void launch_apply_rz(Complex *state, std::size_t size, std::size_t wire,
                     double theta) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    apply_rz_kernel<<<blocks, THREADS>>>(state, size, wire, theta);
    check_cuda(cudaGetLastError(), "apply_rz_kernel");
    maybe_synchronize_cuda("apply_rz_kernel sync");
}

void launch_apply_drz(Complex *out, const Complex *state, std::size_t size,
                      std::size_t wire, double theta) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    apply_drz_kernel<<<blocks, THREADS>>>(out, state, size, wire, theta);
    check_cuda(cudaGetLastError(), "apply_drz_kernel");
    maybe_synchronize_cuda("apply_drz_kernel sync");
}

void launch_apply_cnot(Complex *state, std::size_t size, std::size_t control,
                       std::size_t target) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    apply_cnot_kernel<<<blocks, THREADS>>>(state, size, control, target);
    check_cuda(cudaGetLastError(), "apply_cnot_kernel");
    maybe_synchronize_cuda("apply_cnot_kernel sync");
}

void launch_apply_ring_cnot_layer(Complex *out, const Complex *in,
                                  std::size_t size, std::size_t num_qubits,
                                  bool inverse) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    apply_ring_cnot_layer_kernel<<<blocks, THREADS>>>(out, in, size,
                                                      num_qubits, inverse);
    check_cuda(cudaGetLastError(), "apply_ring_cnot_layer_kernel");
    maybe_synchronize_cuda("apply_ring_cnot_layer_kernel sync");
}

void launch_apply_ryrz(Complex *state, std::size_t size, std::size_t wire,
                       double theta_ry, double theta_rz, bool inverse) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    apply_ryrz_kernel<<<blocks, THREADS>>>(state, size, wire, theta_ry,
                                           theta_rz, inverse);
    check_cuda(cudaGetLastError(), "apply_ryrz_kernel");
    maybe_synchronize_cuda("apply_ryrz_kernel sync");
}

void launch_inverse_walk_ry_step(Complex *current, Complex *lambda,
                                 std::size_t size, std::size_t wire,
                                 double theta, double *out_gradient) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    inverse_walk_ry_step_kernel<<<blocks, THREADS>>>(current, lambda, size,
                                                     wire, theta, out_gradient);
    check_cuda(cudaGetLastError(), "inverse_walk_ry_step_kernel");
    maybe_synchronize_cuda("inverse_walk_ry_step_kernel sync");
}

void launch_inverse_walk_rz_step(Complex *current, Complex *lambda,
                                 std::size_t size, std::size_t wire,
                                 double theta, double *out_gradient) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    inverse_walk_rz_step_kernel<<<blocks, THREADS>>>(current, lambda, size,
                                                     wire, theta, out_gradient);
    check_cuda(cudaGetLastError(), "inverse_walk_rz_step_kernel");
    maybe_synchronize_cuda("inverse_walk_rz_step_kernel sync");
}

void launch_inverse_walk_cnot_step(Complex *current, Complex *lambda,
                                   std::size_t size, std::size_t control,
                                   std::size_t target) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    inverse_walk_cnot_step_kernel<<<blocks, THREADS>>>(current, lambda, size,
                                                       control, target);
    check_cuda(cudaGetLastError(), "inverse_walk_cnot_step_kernel");
    maybe_synchronize_cuda("inverse_walk_cnot_step_kernel sync");
}

void launch_save_param_ry_step(const Complex *state_before, Complex *lambda,
                               std::size_t size, std::size_t wire,
                               double theta, double *out_gradient) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    save_param_ry_step_kernel<<<blocks, THREADS>>>(state_before, lambda, size,
                                                   wire, theta, out_gradient);
    check_cuda(cudaGetLastError(), "save_param_ry_step_kernel");
    maybe_synchronize_cuda("save_param_ry_step_kernel sync");
}

void launch_save_param_rz_step(const Complex *state_before, Complex *lambda,
                               std::size_t size, std::size_t wire,
                               double theta, double *out_gradient) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    save_param_rz_step_kernel<<<blocks, THREADS>>>(state_before, lambda, size,
                                                   wire, theta, out_gradient);
    check_cuda(cudaGetLastError(), "save_param_rz_step_kernel");
    maybe_synchronize_cuda("save_param_rz_step_kernel sync");
}

void launch_apply_hamiltonian(Complex *out, const Complex *state,
                              std::size_t size, std::size_t num_qubits,
                              double field) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    apply_ring_ising_hamiltonian_kernel<<<blocks, THREADS>>>(
        out, state, size, num_qubits, field);
    check_cuda(cudaGetLastError(), "apply_ring_ising_hamiltonian_kernel");
    maybe_synchronize_cuda("apply_ring_ising_hamiltonian_kernel sync");
}

auto complex_inner_product(const Complex *lhs, const Complex *rhs,
                           std::size_t size) -> Complex {
    auto lhs_ptr = thrust::device_pointer_cast(lhs);
    auto rhs_ptr = thrust::device_pointer_cast(rhs);
    auto first =
        thrust::make_zip_iterator(thrust::make_tuple(lhs_ptr, rhs_ptr));
    auto last = first + size;
    return thrust::transform_reduce(first, last, ConjugateProduct{},
                                    Complex(0.0, 0.0),
                                    cuda::std::plus<Complex>());
}

struct FusedRyrzThetaGradContribution {
    const Complex *state;
    const Complex *lambda;
    std::size_t wire;
    std::size_t state_size;
    double theta_ry;
    double theta_rz;

    __host__ __device__ auto operator()(std::size_t pair_index) const -> double {
        const auto half_stride = std::size_t{1} << wire;
        const auto low_mask = half_stride - 1;
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;

        const double c = std::cos(theta_ry * 0.5);
        const double s = std::sin(theta_ry * 0.5);
        const double half_phi = theta_rz * 0.5;
        const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
        const Complex phase1(std::cos(half_phi), std::sin(half_phi));

        const Complex a0 = state[i0];
        const Complex a1 = state[i1];
        const Complex l0 = lambda[i0];
        const Complex l1 = lambda[i1];

        const Complex d0 =
            Complex(-0.5 * s, 0.0) * a0 + Complex(-0.5 * c, 0.0) * a1;
        const Complex d1 =
            Complex(0.5 * c, 0.0) * a0 + Complex(-0.5 * s, 0.0) * a1;

        const Complex value =
            thrust::conj(l0) * (phase0 * d0) + thrust::conj(l1) * (phase1 * d1);
        return 2.0 * value.real();
    }
};

struct FusedRyrzPhiGradContribution {
    const Complex *state;
    const Complex *lambda;
    std::size_t wire;
    std::size_t state_size;
    double theta_ry;
    double theta_rz;

    __host__ __device__ auto operator()(std::size_t pair_index) const -> double {
        const auto half_stride = std::size_t{1} << wire;
        const auto low_mask = half_stride - 1;
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;

        const double c = std::cos(theta_ry * 0.5);
        const double s = std::sin(theta_ry * 0.5);
        const double half_phi = theta_rz * 0.5;
        const Complex phase0(std::cos(-half_phi), std::sin(-half_phi));
        const Complex phase1(std::cos(half_phi), std::sin(half_phi));

        const Complex a0 = state[i0];
        const Complex a1 = state[i1];
        const Complex l0 = lambda[i0];
        const Complex l1 = lambda[i1];

        const Complex b0 = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
        const Complex b1 = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;

        const Complex pref0(0.0, -0.5);
        const Complex pref1(0.0, 0.5);
        const Complex value = thrust::conj(l0) * (pref0 * phase0 * b0) +
                              thrust::conj(l1) * (pref1 * phase1 * b1);
        return 2.0 * value.real();
    }
};

auto fused_ryrz_gradients(const Complex *lambda, const Complex *state,
                          std::size_t state_size, std::size_t wire,
                          double theta_ry, double theta_rz)
    -> std::pair<double, double> {
    const auto num_pairs = state_size / 2;
    auto begin = thrust::make_counting_iterator<std::size_t>(0);
    auto end = begin + num_pairs;
    const double grad_theta = thrust::transform_reduce(
        begin, end,
        FusedRyrzThetaGradContribution{state, lambda, wire, state_size, theta_ry,
                                       theta_rz},
        0.0, cuda::std::plus<double>());
    const double grad_phi = thrust::transform_reduce(
        begin, end,
        FusedRyrzPhiGradContribution{state, lambda, wire, state_size, theta_ry,
                                     theta_rz},
        0.0, cuda::std::plus<double>());
    return {grad_theta, grad_phi};
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
