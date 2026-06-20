#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda/std/functional>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
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

template <bool Inverse>
__global__ void apply_ryrz_kernel(Complex *state, std::size_t size,
                                  std::size_t wire, double c, double s,
                                  double cos_half, double sin_half) {
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

    const Complex a0 = state[i0];
    const Complex a1 = state[i1];
    const double a0_r = a0.real();
    const double a0_i = a0.imag();
    const double a1_r = a1.real();
    const double a1_i = a1.imag();

    if constexpr (!Inverse) {
        const double b0_r = c * a0_r - s * a1_r;
        const double b0_i = c * a0_i - s * a1_i;
        const double b1_r = s * a0_r + c * a1_r;
        const double b1_i = s * a0_i + c * a1_i;

        state[i0] = Complex(cos_half * b0_r + sin_half * b0_i,
                            cos_half * b0_i - sin_half * b0_r);
        state[i1] = Complex(cos_half * b1_r - sin_half * b1_i,
                            cos_half * b1_i + sin_half * b1_r);
        return;
    }

    const double a0_phase1_r = cos_half * a0_r - sin_half * a0_i;
    const double a0_phase1_i = cos_half * a0_i + sin_half * a0_r;
    const double a1_phase0_r = cos_half * a1_r + sin_half * a1_i;
    const double a1_phase0_i = cos_half * a1_i - sin_half * a1_r;

    state[i0] = Complex(c * a0_phase1_r + s * a1_phase0_r,
                        c * a0_phase1_i + s * a1_phase0_i);
    state[i1] = Complex(-s * a0_phase1_r + c * a1_phase0_r,
                        -s * a0_phase1_i + c * a1_phase0_i);
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
    const double half_theta = theta_ry * 0.5;
    const double c = std::cos(half_theta);
    const double s = std::sin(half_theta);
    const double half_phi = theta_rz * 0.5;
    const double cos_half = std::cos(half_phi);
    const double sin_half = std::sin(half_phi);
    if (inverse) {
        apply_ryrz_kernel<true><<<blocks, THREADS>>>(
            state, size, wire, c, s, cos_half, sin_half);
    } else {
        apply_ryrz_kernel<false><<<blocks, THREADS>>>(
            state, size, wire, c, s, cos_half, sin_half);
    }
    check_cuda(cudaGetLastError(), "apply_ryrz_kernel");
    maybe_synchronize_cuda("apply_ryrz_kernel sync");
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

} // namespace detail
} // namespace standalone_backend
