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
                                  double theta_rz) {
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
    const Complex b0 = Complex(c, 0.0) * a0 - Complex(s, 0.0) * a1;
    const Complex b1 = Complex(s, 0.0) * a0 + Complex(c, 0.0) * a1;

    state[i0] = phase0 * b0;
    state[i1] = phase1 * b1;
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
                       double theta_ry, double theta_rz) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    apply_ryrz_kernel<<<blocks, THREADS>>>(state, size, wire, theta_ry,
                                           theta_rz);
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

} // namespace detail
} // namespace standalone_backend
