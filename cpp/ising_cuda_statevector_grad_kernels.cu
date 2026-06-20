#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
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

__global__ void inverse_walk_ryrz_step_kernel(
    Complex *current, Complex *lambda, std::size_t size, std::size_t wire,
    double theta_ry, double theta_rz, double *out_theta_gradient,
    double *out_phi_gradient) {
    __shared__ double theta_partial[THREADS];
    __shared__ double phi_partial[THREADS];

    const auto half_stride = std::size_t{1} << wire;
    const auto pair_index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                     threadIdx.x);
    double theta_sum = 0.0;
    double phi_sum = 0.0;

    if (pair_index < size / 2) {
        const auto low_mask = half_stride - 1;
        const auto base =
            ((pair_index >> wire) << (wire + 1)) | (pair_index & low_mask);
        const auto i0 = base;
        const auto i1 = base | half_stride;

        const double half_phi = theta_rz * 0.5;
        const double rz_c = std::cos(half_phi);
        const double rz_s = std::sin(half_phi);
        const Complex rz_forward0(rz_c, -rz_s);
        const Complex rz_forward1(rz_c, rz_s);
        const Complex rz_inverse0(rz_c, rz_s);
        const Complex rz_inverse1(rz_c, -rz_s);

        const Complex current_after0 = current[i0];
        const Complex current_after1 = current[i1];
        const Complex lambda_after0 = lambda[i0];
        const Complex lambda_after1 = lambda[i1];

        const Complex current_mid0 = rz_inverse0 * current_after0;
        const Complex current_mid1 = rz_inverse1 * current_after1;
        const Complex lambda_mid0 = rz_inverse0 * lambda_after0;
        const Complex lambda_mid1 = rz_inverse1 * lambda_after1;

        phi_sum =
            (thrust::conj(lambda_after0) * Complex(0.0, -0.5) *
                 rz_forward0 * current_mid0 +
             thrust::conj(lambda_after1) * Complex(0.0, 0.5) *
                 rz_forward1 * current_mid1)
                .real();

        const double half_theta = theta_ry * 0.5;
        const double ry_c = std::cos(half_theta);
        const double ry_s = std::sin(half_theta);
        const Complex current_before0 =
            Complex(ry_c, 0.0) * current_mid0 +
            Complex(ry_s, 0.0) * current_mid1;
        const Complex current_before1 =
            Complex(-ry_s, 0.0) * current_mid0 +
            Complex(ry_c, 0.0) * current_mid1;
        current[i0] = current_before0;
        current[i1] = current_before1;

        const Complex dtheta0 =
            Complex(-0.5 * ry_s, 0.0) * current_before0 +
            Complex(-0.5 * ry_c, 0.0) * current_before1;
        const Complex dtheta1 =
            Complex(0.5 * ry_c, 0.0) * current_before0 +
            Complex(-0.5 * ry_s, 0.0) * current_before1;
        theta_sum =
            (thrust::conj(lambda_mid0) * dtheta0 +
             thrust::conj(lambda_mid1) * dtheta1)
                .real();

        lambda[i0] = Complex(ry_c, 0.0) * lambda_mid0 +
                     Complex(ry_s, 0.0) * lambda_mid1;
        lambda[i1] = Complex(-ry_s, 0.0) * lambda_mid0 +
                     Complex(ry_c, 0.0) * lambda_mid1;
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
    if (threadIdx.x == 0) {
        atomicAdd(out_theta_gradient, 2.0 * theta_partial[0]);
        atomicAdd(out_phi_gradient, 2.0 * phi_partial[0]);
    }
}

template <int W>
__global__ void inverse_walk_ryrz_rotation_chunk_kernel(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {
    static_assert(W >= 2 && W <= static_cast<int>(ROTATION_CHUNK_MAX_WIRES),
                  "unsupported rotation chunk width");

    __shared__ Complex current_tile[THREADS];
    __shared__ Complex lambda_tile[THREADS];
    __shared__ double theta_partial[W * THREADS];
    __shared__ double phi_partial[W * THREADS];

    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block =
        static_cast<std::size_t>(THREADS) / tile_dim;
    const auto local_thread = static_cast<std::size_t>(threadIdx.x);
    const auto local_tile = local_thread / tile_dim;
    const auto local_index = local_thread & (tile_dim - 1);
    const auto tile_id =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block + local_tile;
    const auto num_tiles = size / tile_dim;
    const bool active = tile_id < num_tiles;

    std::size_t state_index = 0;
    if (active) {
        const auto low_mask = (std::size_t{1} << chunk_start) - 1;
        const auto base = (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
        state_index = base | (local_index << chunk_start);
        current_tile[local_thread] = current[state_index];
        lambda_tile[local_thread] = lambda[state_index];
    }
    __syncthreads();

#pragma unroll
    for (int local_wire = W - 1; local_wire >= 0; local_wire--) {
        double theta_sum = 0.0;
        double phi_sum = 0.0;
        const auto bit = std::size_t{1} << local_wire;

        if (active && (local_index & bit) == 0U) {
            const auto partner_thread = local_thread | bit;

            const double rz_c = coeffs.cos_half[local_wire];
            const double rz_s = coeffs.sin_half[local_wire];
            const Complex rz_forward0(rz_c, -rz_s);
            const Complex rz_forward1(rz_c, rz_s);
            const Complex rz_inverse0(rz_c, rz_s);
            const Complex rz_inverse1(rz_c, -rz_s);

            const Complex current_after0 = current_tile[local_thread];
            const Complex current_after1 = current_tile[partner_thread];
            const Complex lambda_after0 = lambda_tile[local_thread];
            const Complex lambda_after1 = lambda_tile[partner_thread];

            const Complex current_mid0 = rz_inverse0 * current_after0;
            const Complex current_mid1 = rz_inverse1 * current_after1;
            const Complex lambda_mid0 = rz_inverse0 * lambda_after0;
            const Complex lambda_mid1 = rz_inverse1 * lambda_after1;

            phi_sum =
                (thrust::conj(lambda_after0) * Complex(0.0, -0.5) *
                     rz_forward0 * current_mid0 +
                 thrust::conj(lambda_after1) * Complex(0.0, 0.5) *
                     rz_forward1 * current_mid1)
                    .real();

            const double ry_c = coeffs.c[local_wire];
            const double ry_s = coeffs.s[local_wire];
            const Complex current_before0 =
                Complex(ry_c, 0.0) * current_mid0 +
                Complex(ry_s, 0.0) * current_mid1;
            const Complex current_before1 =
                Complex(-ry_s, 0.0) * current_mid0 +
                Complex(ry_c, 0.0) * current_mid1;
            current_tile[local_thread] = current_before0;
            current_tile[partner_thread] = current_before1;

            const Complex dtheta0 =
                Complex(-0.5 * ry_s, 0.0) * current_before0 +
                Complex(-0.5 * ry_c, 0.0) * current_before1;
            const Complex dtheta1 =
                Complex(0.5 * ry_c, 0.0) * current_before0 +
                Complex(-0.5 * ry_s, 0.0) * current_before1;
            theta_sum =
                (thrust::conj(lambda_mid0) * dtheta0 +
                 thrust::conj(lambda_mid1) * dtheta1)
                    .real();

            lambda_tile[local_thread] = Complex(ry_c, 0.0) * lambda_mid0 +
                                        Complex(ry_s, 0.0) * lambda_mid1;
            lambda_tile[partner_thread] = Complex(-ry_s, 0.0) * lambda_mid0 +
                                          Complex(ry_c, 0.0) * lambda_mid1;
        }

        const auto partial_offset =
            static_cast<std::size_t>(local_wire) * THREADS + threadIdx.x;
        theta_partial[partial_offset] = theta_sum;
        phi_partial[partial_offset] = phi_sum;
        __syncthreads();
    }

    if (active) {
        current[state_index] = current_tile[local_thread];
        lambda[state_index] = lambda_tile[local_thread];
    }

#pragma unroll
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int local_wire = 0; local_wire < W; local_wire++) {
                const auto partial_offset =
                    static_cast<std::size_t>(local_wire) * THREADS;
                theta_partial[partial_offset + threadIdx.x] +=
                    theta_partial[partial_offset + threadIdx.x + stride];
                phi_partial[partial_offset + threadIdx.x] +=
                    phi_partial[partial_offset + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (int local_wire = 0; local_wire < W; local_wire++) {
            const auto partial_offset =
                static_cast<std::size_t>(local_wire) * THREADS;
            atomicAdd(out_gradients + local_wire * 2,
                      2.0 * theta_partial[partial_offset]);
            atomicAdd(out_gradients + local_wire * 2 + 1,
                      2.0 * phi_partial[partial_offset]);
        }
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

void launch_inverse_walk_ryrz_step(Complex *current, Complex *lambda,
                                   std::size_t size, std::size_t wire,
                                   double theta_ry, double theta_rz,
                                   double *out_theta_gradient,
                                   double *out_phi_gradient) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    inverse_walk_ryrz_step_kernel<<<blocks, THREADS>>>(
        current, lambda, size, wire, theta_ry, theta_rz, out_theta_gradient,
        out_phi_gradient);
    check_cuda(cudaGetLastError(), "inverse_walk_ryrz_step_kernel");
    maybe_synchronize_cuda("inverse_walk_ryrz_step_kernel sync");
}

auto make_inverse_walk_rotation_chunk_coeffs(std::size_t chunk_width,
                                             const double *theta_ry,
                                             const double *theta_rz)
    -> RotationChunkCoeffs {
    RotationChunkCoeffs coeffs{};
    for (std::size_t local = 0; local < chunk_width; local++) {
        const double ry = theta_ry[local] * 0.5;
        const double rz = theta_rz[local] * 0.5;
        coeffs.c[local] = std::cos(ry);
        coeffs.s[local] = std::sin(ry);
        coeffs.cos_half[local] = std::cos(rz);
        coeffs.sin_half[local] = std::sin(rz);
    }
    return coeffs;
}

template <int W>
void launch_inverse_walk_ryrz_rotation_chunk_specialized(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {
    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block =
        static_cast<std::size_t>(THREADS) / tile_dim;
    const auto num_tiles = size / tile_dim;
    const auto blocks =
        static_cast<int>((num_tiles + tiles_per_block - 1) / tiles_per_block);
    inverse_walk_ryrz_rotation_chunk_kernel<W><<<blocks, THREADS>>>(
        current, lambda, size, chunk_start, coeffs, out_gradients);
}

void launch_inverse_walk_ryrz_rotation_chunk(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, std::size_t chunk_width,
    const double *theta_ry, const double *theta_rz, double *out_gradients) {
    if (chunk_width == 0) {
        return;
    }
    if (chunk_width == 1) {
        launch_inverse_walk_ryrz_step(current, lambda, size, chunk_start,
                                      theta_ry[0], theta_rz[0],
                                      out_gradients, out_gradients + 1);
        return;
    }
    if (chunk_width > ROTATION_CHUNK_MAX_WIRES) {
        throw std::invalid_argument(
            "inverse rotation chunk width exceeds supported maximum.");
    }
    if (chunk_start + chunk_width >= sizeof(std::size_t) * 8 ||
        (std::size_t{1} << (chunk_start + chunk_width)) > size) {
        throw std::invalid_argument(
            "inverse rotation chunk exceeds state dimension.");
    }

    const auto tile_dim = std::size_t{1} << chunk_width;
    if (tile_dim > static_cast<std::size_t>(THREADS)) {
        throw std::invalid_argument(
            "inverse rotation chunk tile_dim exceeds THREADS.");
    }
    const auto coeffs = make_inverse_walk_rotation_chunk_coeffs(
        chunk_width, theta_ry, theta_rz);

    switch (chunk_width) {
    case 2:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<2>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    case 3:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<3>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    case 4:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<4>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    case 5:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<5>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    case 6:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<6>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    case 7:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<7>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    case 8:
        launch_inverse_walk_ryrz_rotation_chunk_specialized<8>(
            current, lambda, size, chunk_start, coeffs, out_gradients);
        break;
    default:
        throw std::invalid_argument("unsupported inverse rotation chunk width.");
    }
    check_cuda(cudaGetLastError(),
               "inverse_walk_ryrz_rotation_chunk_kernel");
    maybe_synchronize_cuda(
        "inverse_walk_ryrz_rotation_chunk_kernel sync");
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
