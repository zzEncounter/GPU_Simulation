#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <utility>

#include <cuda/std/functional>
#include <thrust/complex.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace standalone_backend {
namespace detail {

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
        const auto c(std::cos(forward_phase)), s(std::sin(forward_phase));
        const Complex forward_factor(c, s);
        const Complex inverse_factor(c, -s);
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
        const double c = std::cos(phase);
        const double s = std::sin(phase);
        const Complex forward_factor(c, s);
        const Complex inverse_factor(c, -s);
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
