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

__global__ void inverse_walk_ry_gradient_kernel(const Complex *current,
                                                const Complex *lambda,
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

        const Complex d0 = Complex(-0.5 * s, 0.0) * current_before0 +
                           Complex(-0.5 * c, 0.0) * current_before1;
        const Complex d1 = Complex(0.5 * c, 0.0) * current_before0 +
                           Complex(-0.5 * s, 0.0) * current_before1;
        local_sum =
            (thrust::conj(lambda_after0) * d0 +
             thrust::conj(lambda_after1) * d1)
                .real();
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

__global__ void inverse_walk_rz_gradient_kernel(const Complex *current,
                                                const Complex *lambda,
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

        const Complex current_before = inverse_factor * current[index];

        local_sum =
            (thrust::conj(lambda[index]) * prefactor * forward_factor *
             current_before)
                .real();
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

        const Complex current_after0 = current[i0];
        const Complex current_after1 = current[i1];
        const Complex lambda_after0 = lambda[i0];
        const Complex lambda_after1 = lambda[i1];
        const double ca0_r = current_after0.real();
        const double ca0_i = current_after0.imag();
        const double ca1_r = current_after1.real();
        const double ca1_i = current_after1.imag();
        const double la0_r = lambda_after0.real();
        const double la0_i = lambda_after0.imag();
        const double la1_r = lambda_after1.real();
        const double la1_i = lambda_after1.imag();

        const double current_mid0_r = rz_c * ca0_r - rz_s * ca0_i;
        const double current_mid0_i = rz_c * ca0_i + rz_s * ca0_r;
        const double current_mid1_r = rz_c * ca1_r + rz_s * ca1_i;
        const double current_mid1_i = rz_c * ca1_i - rz_s * ca1_r;
        const double lambda_mid0_r = rz_c * la0_r - rz_s * la0_i;
        const double lambda_mid0_i = rz_c * la0_i + rz_s * la0_r;
        const double lambda_mid1_r = rz_c * la1_r + rz_s * la1_i;
        const double lambda_mid1_i = rz_c * la1_i - rz_s * la1_r;

        phi_sum = 0.5 * ((la0_r * ca0_i - la0_i * ca0_r) -
                         (la1_r * ca1_i - la1_i * ca1_r));

        const double half_theta = theta_ry * 0.5;
        const double ry_c = std::cos(half_theta);
        const double ry_s = std::sin(half_theta);
        const double current_before0_r =
            ry_c * current_mid0_r + ry_s * current_mid1_r;
        const double current_before0_i =
            ry_c * current_mid0_i + ry_s * current_mid1_i;
        const double current_before1_r =
            -ry_s * current_mid0_r + ry_c * current_mid1_r;
        const double current_before1_i =
            -ry_s * current_mid0_i + ry_c * current_mid1_i;
        current[i0] = Complex(current_before0_r, current_before0_i);
        current[i1] = Complex(current_before1_r, current_before1_i);

        const double dtheta0_r =
            -0.5 * (ry_s * current_before0_r + ry_c * current_before1_r);
        const double dtheta0_i =
            -0.5 * (ry_s * current_before0_i + ry_c * current_before1_i);
        const double dtheta1_r =
            0.5 * (ry_c * current_before0_r - ry_s * current_before1_r);
        const double dtheta1_i =
            0.5 * (ry_c * current_before0_i - ry_s * current_before1_i);
        theta_sum = lambda_mid0_r * dtheta0_r +
                    lambda_mid0_i * dtheta0_i +
                    lambda_mid1_r * dtheta1_r +
                    lambda_mid1_i * dtheta1_i;

        lambda[i0] = Complex(ry_c * lambda_mid0_r + ry_s * lambda_mid1_r,
                             ry_c * lambda_mid0_i + ry_s * lambda_mid1_i);
        lambda[i1] = Complex(-ry_s * lambda_mid0_r + ry_c * lambda_mid1_r,
                             -ry_s * lambda_mid0_i + ry_c * lambda_mid1_i);
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

__device__ __forceinline__ void inverse_walk_ryrz_backward_pair_scalar(
    Complex *current_tile, Complex *lambda_tile, std::size_t i0,
    std::size_t i1, double ry_c, double ry_s, double rz_c, double rz_s,
    double *theta_sum, double *phi_sum) {
    const Complex current_after0 = current_tile[i0];
    const Complex current_after1 = current_tile[i1];
    const Complex lambda_after0 = lambda_tile[i0];
    const Complex lambda_after1 = lambda_tile[i1];
    const double ca0_r = current_after0.real();
    const double ca0_i = current_after0.imag();
    const double ca1_r = current_after1.real();
    const double ca1_i = current_after1.imag();
    const double la0_r = lambda_after0.real();
    const double la0_i = lambda_after0.imag();
    const double la1_r = lambda_after1.real();
    const double la1_i = lambda_after1.imag();

    const double current_mid0_r = rz_c * ca0_r - rz_s * ca0_i;
    const double current_mid0_i = rz_c * ca0_i + rz_s * ca0_r;
    const double current_mid1_r = rz_c * ca1_r + rz_s * ca1_i;
    const double current_mid1_i = rz_c * ca1_i - rz_s * ca1_r;
    const double lambda_mid0_r = rz_c * la0_r - rz_s * la0_i;
    const double lambda_mid0_i = rz_c * la0_i + rz_s * la0_r;
    const double lambda_mid1_r = rz_c * la1_r + rz_s * la1_i;
    const double lambda_mid1_i = rz_c * la1_i - rz_s * la1_r;

    *phi_sum += 0.5 * ((la0_r * ca0_i - la0_i * ca0_r) -
                       (la1_r * ca1_i - la1_i * ca1_r));

    const double current_before0_r =
        ry_c * current_mid0_r + ry_s * current_mid1_r;
    const double current_before0_i =
        ry_c * current_mid0_i + ry_s * current_mid1_i;
    const double current_before1_r =
        -ry_s * current_mid0_r + ry_c * current_mid1_r;
    const double current_before1_i =
        -ry_s * current_mid0_i + ry_c * current_mid1_i;
    current_tile[i0] = Complex(current_before0_r, current_before0_i);
    current_tile[i1] = Complex(current_before1_r, current_before1_i);

    const double dtheta0_r =
        -0.5 * (ry_s * current_before0_r + ry_c * current_before1_r);
    const double dtheta0_i =
        -0.5 * (ry_s * current_before0_i + ry_c * current_before1_i);
    const double dtheta1_r =
        0.5 * (ry_c * current_before0_r - ry_s * current_before1_r);
    const double dtheta1_i =
        0.5 * (ry_c * current_before0_i - ry_s * current_before1_i);
    *theta_sum += lambda_mid0_r * dtheta0_r +
                  lambda_mid0_i * dtheta0_i +
                  lambda_mid1_r * dtheta1_r +
                  lambda_mid1_i * dtheta1_i;

    lambda_tile[i0] =
        Complex(ry_c * lambda_mid0_r + ry_s * lambda_mid1_r,
                ry_c * lambda_mid0_i + ry_s * lambda_mid1_i);
    lambda_tile[i1] =
        Complex(-ry_s * lambda_mid0_r + ry_c * lambda_mid1_r,
                -ry_s * lambda_mid0_i + ry_c * lambda_mid1_i);
}

__device__ __forceinline__ void inverse_walk_ryrz_backward_pair_registers(
    double &current0_r, double &current0_i, double &current1_r,
    double &current1_i, double &lambda0_r, double &lambda0_i,
    double &lambda1_r, double &lambda1_i, double ry_c, double ry_s,
    double rz_c, double rz_s, double *theta_sum, double *phi_sum) {
    const double ca0_r = current0_r;
    const double ca0_i = current0_i;
    const double ca1_r = current1_r;
    const double ca1_i = current1_i;
    const double la0_r = lambda0_r;
    const double la0_i = lambda0_i;
    const double la1_r = lambda1_r;
    const double la1_i = lambda1_i;

    const double current_mid0_r = rz_c * ca0_r - rz_s * ca0_i;
    const double current_mid0_i = rz_c * ca0_i + rz_s * ca0_r;
    const double current_mid1_r = rz_c * ca1_r + rz_s * ca1_i;
    const double current_mid1_i = rz_c * ca1_i - rz_s * ca1_r;
    const double lambda_mid0_r = rz_c * la0_r - rz_s * la0_i;
    const double lambda_mid0_i = rz_c * la0_i + rz_s * la0_r;
    const double lambda_mid1_r = rz_c * la1_r + rz_s * la1_i;
    const double lambda_mid1_i = rz_c * la1_i - rz_s * la1_r;

    *phi_sum += 0.5 * ((la0_r * ca0_i - la0_i * ca0_r) -
                       (la1_r * ca1_i - la1_i * ca1_r));

    const double current_before0_r =
        ry_c * current_mid0_r + ry_s * current_mid1_r;
    const double current_before0_i =
        ry_c * current_mid0_i + ry_s * current_mid1_i;
    const double current_before1_r =
        -ry_s * current_mid0_r + ry_c * current_mid1_r;
    const double current_before1_i =
        -ry_s * current_mid0_i + ry_c * current_mid1_i;
    current0_r = current_before0_r;
    current0_i = current_before0_i;
    current1_r = current_before1_r;
    current1_i = current_before1_i;

    const double dtheta0_r =
        -0.5 * (ry_s * current_before0_r + ry_c * current_before1_r);
    const double dtheta0_i =
        -0.5 * (ry_s * current_before0_i + ry_c * current_before1_i);
    const double dtheta1_r =
        0.5 * (ry_c * current_before0_r - ry_s * current_before1_r);
    const double dtheta1_i =
        0.5 * (ry_c * current_before0_i - ry_s * current_before1_i);
    *theta_sum += lambda_mid0_r * dtheta0_r +
                  lambda_mid0_i * dtheta0_i +
                  lambda_mid1_r * dtheta1_r +
                  lambda_mid1_i * dtheta1_i;

    lambda0_r = ry_c * lambda_mid0_r + ry_s * lambda_mid1_r;
    lambda0_i = ry_c * lambda_mid0_i + ry_s * lambda_mid1_i;
    lambda1_r = -ry_s * lambda_mid0_r + ry_c * lambda_mid1_r;
    lambda1_i = -ry_s * lambda_mid0_i + ry_c * lambda_mid1_i;
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
            const Complex current_after0 = current_tile[local_thread];
            const Complex current_after1 = current_tile[partner_thread];
            const Complex lambda_after0 = lambda_tile[local_thread];
            const Complex lambda_after1 = lambda_tile[partner_thread];
            const double ca0_r = current_after0.real();
            const double ca0_i = current_after0.imag();
            const double ca1_r = current_after1.real();
            const double ca1_i = current_after1.imag();
            const double la0_r = lambda_after0.real();
            const double la0_i = lambda_after0.imag();
            const double la1_r = lambda_after1.real();
            const double la1_i = lambda_after1.imag();

            const double current_mid0_r = rz_c * ca0_r - rz_s * ca0_i;
            const double current_mid0_i = rz_c * ca0_i + rz_s * ca0_r;
            const double current_mid1_r = rz_c * ca1_r + rz_s * ca1_i;
            const double current_mid1_i = rz_c * ca1_i - rz_s * ca1_r;
            const double lambda_mid0_r = rz_c * la0_r - rz_s * la0_i;
            const double lambda_mid0_i = rz_c * la0_i + rz_s * la0_r;
            const double lambda_mid1_r = rz_c * la1_r + rz_s * la1_i;
            const double lambda_mid1_i = rz_c * la1_i - rz_s * la1_r;

            phi_sum = 0.5 * ((la0_r * ca0_i - la0_i * ca0_r) -
                             (la1_r * ca1_i - la1_i * ca1_r));

            const double ry_c = coeffs.c[local_wire];
            const double ry_s = coeffs.s[local_wire];
            const double current_before0_r =
                ry_c * current_mid0_r + ry_s * current_mid1_r;
            const double current_before0_i =
                ry_c * current_mid0_i + ry_s * current_mid1_i;
            const double current_before1_r =
                -ry_s * current_mid0_r + ry_c * current_mid1_r;
            const double current_before1_i =
                -ry_s * current_mid0_i + ry_c * current_mid1_i;
            current_tile[local_thread] =
                Complex(current_before0_r, current_before0_i);
            current_tile[partner_thread] =
                Complex(current_before1_r, current_before1_i);

            const double dtheta0_r =
                -0.5 * (ry_s * current_before0_r +
                        ry_c * current_before1_r);
            const double dtheta0_i =
                -0.5 * (ry_s * current_before0_i +
                        ry_c * current_before1_i);
            const double dtheta1_r =
                0.5 * (ry_c * current_before0_r -
                       ry_s * current_before1_r);
            const double dtheta1_i =
                0.5 * (ry_c * current_before0_i -
                       ry_s * current_before1_i);
            theta_sum = lambda_mid0_r * dtheta0_r +
                        lambda_mid0_i * dtheta0_i +
                        lambda_mid1_r * dtheta1_r +
                        lambda_mid1_i * dtheta1_i;

            lambda_tile[local_thread] =
                Complex(ry_c * lambda_mid0_r + ry_s * lambda_mid1_r,
                        ry_c * lambda_mid0_i + ry_s * lambda_mid1_i);
            lambda_tile[partner_thread] =
                Complex(-ry_s * lambda_mid0_r + ry_c * lambda_mid1_r,
                        -ry_s * lambda_mid0_i + ry_c * lambda_mid1_i);
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

// High-wire chunks use this mapping so adjacent lanes read adjacent tile ids
// at the same local index instead of striding through one high-wire tile.
template <int W>
__global__ void inverse_walk_ryrz_rotation_chunk_transposed_kernel(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {
    static_assert(W >= 2 && W <= 4,
                  "transposed chunks are intended for narrow high-wire chunks");

    __shared__ Complex current_tile[THREADS];
    __shared__ Complex lambda_tile[THREADS];
    __shared__ double theta_partial[W * THREADS];
    __shared__ double phi_partial[W * THREADS];

    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block =
        static_cast<std::size_t>(THREADS) / tile_dim;
    const auto local_thread = static_cast<std::size_t>(threadIdx.x);
    const auto local_tile = local_thread & (tiles_per_block - 1);
    const auto local_index = local_thread / tiles_per_block;
    const auto shared_index = local_tile * tile_dim + local_index;
    const auto tile_id =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block + local_tile;
    const auto num_tiles = size / tile_dim;
    const bool active = tile_id < num_tiles;

    std::size_t state_index = 0;
    if (active) {
        const auto low_mask = (std::size_t{1} << chunk_start) - 1;
        const auto base = (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
        state_index = base | (local_index << chunk_start);
        current_tile[shared_index] = current[state_index];
        lambda_tile[shared_index] = lambda[state_index];
    }
    __syncthreads();

#pragma unroll
    for (int local_wire = W - 1; local_wire >= 0; local_wire--) {
        double theta_sum = 0.0;
        double phi_sum = 0.0;
        const auto bit = std::size_t{1} << local_wire;

        if (active && (local_index & bit) == 0U) {
            const auto partner_shared = shared_index | bit;
            inverse_walk_ryrz_backward_pair_scalar(
                current_tile, lambda_tile, shared_index, partner_shared,
                coeffs.c[local_wire], coeffs.s[local_wire],
                coeffs.cos_half[local_wire], coeffs.sin_half[local_wire],
                &theta_sum, &phi_sum);
        }

        const auto partial_offset =
            static_cast<std::size_t>(local_wire) * THREADS + threadIdx.x;
        theta_partial[partial_offset] = theta_sum;
        phi_partial[partial_offset] = phi_sum;
        __syncthreads();
    }

    if (active) {
        current[state_index] = current_tile[shared_index];
        lambda[state_index] = lambda_tile[shared_index];
    }

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

// Fuse two adjacent wires inside each 4-amplitude cell. This keeps the
// inverse-walk order but removes the shared-memory round trip between them.
__global__ void inverse_walk_ryrz_rotation_chunk_w4_cell2_kernel(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {
    constexpr int W = 4;
    constexpr int block_threads = 128;
    constexpr int tile_dim = 1 << W;
    constexpr int cells_per_tile = 4;
    constexpr int tiles_per_block = block_threads / cells_per_tile;
    constexpr int shared_values = tiles_per_block * tile_dim;

    __shared__ Complex current_tile[shared_values];
    __shared__ Complex lambda_tile[shared_values];
    __shared__ double theta_partial[W * block_threads];
    __shared__ double phi_partial[W * block_threads];

    const int local_cell = threadIdx.x / tiles_per_block;
    const int local_tile = threadIdx.x & (tiles_per_block - 1);
    const auto tile_id =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block +
        static_cast<std::size_t>(local_tile);
    const auto num_tiles = size / static_cast<std::size_t>(tile_dim);
    const bool active = tile_id < num_tiles;

    std::size_t base = 0;
    if (active) {
        const auto low_mask = (std::size_t{1} << chunk_start) - 1;
        base = (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
    }

    const auto load_current = [&](int local_index) -> Complex {
        if (!active) {
            return Complex(0.0, 0.0);
        }
        return current[base | (static_cast<std::size_t>(local_index)
                               << chunk_start)];
    };
    const auto load_lambda = [&](int local_index) -> Complex {
        if (!active) {
            return Complex(0.0, 0.0);
        }
        return lambda[base | (static_cast<std::size_t>(local_index)
                              << chunk_start)];
    };

    const int high_i0 = local_cell;
    const int high_i1 = high_i0 | 4;
    const int high_i2 = high_i0 | 8;
    const int high_i3 = high_i0 | 12;

    Complex c0 = load_current(high_i0);
    Complex c1 = load_current(high_i1);
    Complex c2 = load_current(high_i2);
    Complex c3 = load_current(high_i3);
    Complex l0 = load_lambda(high_i0);
    Complex l1 = load_lambda(high_i1);
    Complex l2 = load_lambda(high_i2);
    Complex l3 = load_lambda(high_i3);

    double c0_r = c0.real();
    double c0_i = c0.imag();
    double c1_r = c1.real();
    double c1_i = c1.imag();
    double c2_r = c2.real();
    double c2_i = c2.imag();
    double c3_r = c3.real();
    double c3_i = c3.imag();
    double l0_r = l0.real();
    double l0_i = l0.imag();
    double l1_r = l1.real();
    double l1_i = l1.imag();
    double l2_r = l2.real();
    double l2_i = l2.imag();
    double l3_r = l3.real();
    double l3_i = l3.imag();

    double theta3 = 0.0;
    double phi3 = 0.0;
    double theta2 = 0.0;
    double phi2 = 0.0;

    inverse_walk_ryrz_backward_pair_registers(
        c0_r, c0_i, c2_r, c2_i, l0_r, l0_i, l2_r, l2_i, coeffs.c[3],
        coeffs.s[3], coeffs.cos_half[3], coeffs.sin_half[3], &theta3,
        &phi3);
    inverse_walk_ryrz_backward_pair_registers(
        c1_r, c1_i, c3_r, c3_i, l1_r, l1_i, l3_r, l3_i, coeffs.c[3],
        coeffs.s[3], coeffs.cos_half[3], coeffs.sin_half[3], &theta3,
        &phi3);
    inverse_walk_ryrz_backward_pair_registers(
        c0_r, c0_i, c1_r, c1_i, l0_r, l0_i, l1_r, l1_i, coeffs.c[2],
        coeffs.s[2], coeffs.cos_half[2], coeffs.sin_half[2], &theta2,
        &phi2);
    inverse_walk_ryrz_backward_pair_registers(
        c2_r, c2_i, c3_r, c3_i, l2_r, l2_i, l3_r, l3_i, coeffs.c[2],
        coeffs.s[2], coeffs.cos_half[2], coeffs.sin_half[2], &theta2,
        &phi2);

    const int tile_offset = local_tile * tile_dim;
    current_tile[tile_offset + high_i0] = Complex(c0_r, c0_i);
    current_tile[tile_offset + high_i1] = Complex(c1_r, c1_i);
    current_tile[tile_offset + high_i2] = Complex(c2_r, c2_i);
    current_tile[tile_offset + high_i3] = Complex(c3_r, c3_i);
    lambda_tile[tile_offset + high_i0] = Complex(l0_r, l0_i);
    lambda_tile[tile_offset + high_i1] = Complex(l1_r, l1_i);
    lambda_tile[tile_offset + high_i2] = Complex(l2_r, l2_i);
    lambda_tile[tile_offset + high_i3] = Complex(l3_r, l3_i);

    theta_partial[3 * block_threads + threadIdx.x] = theta3;
    phi_partial[3 * block_threads + threadIdx.x] = phi3;
    theta_partial[2 * block_threads + threadIdx.x] = theta2;
    phi_partial[2 * block_threads + threadIdx.x] = phi2;
    __syncthreads();

    const int low_base = local_cell << 2;
    const int low_i0 = low_base;
    const int low_i1 = low_i0 | 1;
    const int low_i2 = low_i0 | 2;
    const int low_i3 = low_i0 | 3;

    c0 = current_tile[tile_offset + low_i0];
    c1 = current_tile[tile_offset + low_i1];
    c2 = current_tile[tile_offset + low_i2];
    c3 = current_tile[tile_offset + low_i3];
    l0 = lambda_tile[tile_offset + low_i0];
    l1 = lambda_tile[tile_offset + low_i1];
    l2 = lambda_tile[tile_offset + low_i2];
    l3 = lambda_tile[tile_offset + low_i3];

    c0_r = c0.real();
    c0_i = c0.imag();
    c1_r = c1.real();
    c1_i = c1.imag();
    c2_r = c2.real();
    c2_i = c2.imag();
    c3_r = c3.real();
    c3_i = c3.imag();
    l0_r = l0.real();
    l0_i = l0.imag();
    l1_r = l1.real();
    l1_i = l1.imag();
    l2_r = l2.real();
    l2_i = l2.imag();
    l3_r = l3.real();
    l3_i = l3.imag();

    double theta1 = 0.0;
    double phi1 = 0.0;
    double theta0 = 0.0;
    double phi0 = 0.0;

    inverse_walk_ryrz_backward_pair_registers(
        c0_r, c0_i, c2_r, c2_i, l0_r, l0_i, l2_r, l2_i, coeffs.c[1],
        coeffs.s[1], coeffs.cos_half[1], coeffs.sin_half[1], &theta1,
        &phi1);
    inverse_walk_ryrz_backward_pair_registers(
        c1_r, c1_i, c3_r, c3_i, l1_r, l1_i, l3_r, l3_i, coeffs.c[1],
        coeffs.s[1], coeffs.cos_half[1], coeffs.sin_half[1], &theta1,
        &phi1);
    inverse_walk_ryrz_backward_pair_registers(
        c0_r, c0_i, c1_r, c1_i, l0_r, l0_i, l1_r, l1_i, coeffs.c[0],
        coeffs.s[0], coeffs.cos_half[0], coeffs.sin_half[0], &theta0,
        &phi0);
    inverse_walk_ryrz_backward_pair_registers(
        c2_r, c2_i, c3_r, c3_i, l2_r, l2_i, l3_r, l3_i, coeffs.c[0],
        coeffs.s[0], coeffs.cos_half[0], coeffs.sin_half[0], &theta0,
        &phi0);

    if (active) {
        const auto store_value = [&](int local_index, double current_r,
                                     double current_i, double lambda_r,
                                     double lambda_i) {
            const auto state_index =
                base | (static_cast<std::size_t>(local_index) << chunk_start);
            current[state_index] = Complex(current_r, current_i);
            lambda[state_index] = Complex(lambda_r, lambda_i);
        };
        store_value(low_i0, c0_r, c0_i, l0_r, l0_i);
        store_value(low_i1, c1_r, c1_i, l1_r, l1_i);
        store_value(low_i2, c2_r, c2_i, l2_r, l2_i);
        store_value(low_i3, c3_r, c3_i, l3_r, l3_i);
    }

    theta_partial[1 * block_threads + threadIdx.x] = theta1;
    phi_partial[1 * block_threads + threadIdx.x] = phi1;
    theta_partial[0 * block_threads + threadIdx.x] = theta0;
    phi_partial[0 * block_threads + threadIdx.x] = phi0;
    __syncthreads();

    for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int local_wire = 0; local_wire < W; local_wire++) {
                const auto partial_offset =
                    static_cast<std::size_t>(local_wire) * block_threads;
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
                static_cast<std::size_t>(local_wire) * block_threads;
            atomicAdd(out_gradients + local_wire * 2,
                      2.0 * theta_partial[partial_offset]);
            atomicAdd(out_gradients + local_wire * 2 + 1,
                      2.0 * phi_partial[partial_offset]);
        }
    }
}

__device__ __forceinline__ auto two_wire_cell_base(int cell, int low_wire)
    -> int {
    const int low_mask = (1 << low_wire) - 1;
    const int low = cell & low_mask;
    const int high = cell >> low_wire;
    return low | (high << (low_wire + 2));
}

// Low W8 chunks use the same two-wire cell idea with 128-thread blocks to stay
// under the default static shared-memory limit while reducing synchronization.
__global__ void inverse_walk_ryrz_rotation_chunk_w8_cell2_kernel(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {
    constexpr int W = 8;
    constexpr int block_threads = 128;
    constexpr int tile_dim = 1 << W;
    constexpr int cells_per_tile = tile_dim / 4;
    constexpr int tiles_per_block = block_threads / cells_per_tile;
    constexpr int shared_values = tiles_per_block * tile_dim;

    __shared__ Complex current_tile[shared_values];
    __shared__ Complex lambda_tile[shared_values];
    __shared__ double theta_partial[W * block_threads];
    __shared__ double phi_partial[W * block_threads];

    const int local_cell = threadIdx.x & (cells_per_tile - 1);
    const int local_tile = threadIdx.x / cells_per_tile;

    const auto low_mask = (std::size_t{1} << chunk_start) - 1;
    const auto num_tiles = size / static_cast<std::size_t>(tile_dim);
    const auto block_tile_base =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block;

#pragma unroll
    for (int item = 0; item < 4; item++) {
        const int shared_index = threadIdx.x + item * block_threads;
        const int tile_slot = shared_index / tile_dim;
        const int local_index = shared_index & (tile_dim - 1);
        const auto tile_id =
            block_tile_base + static_cast<std::size_t>(tile_slot);
        if (tile_id < num_tiles) {
            const auto base =
                (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
            const auto state_index =
                base | (static_cast<std::size_t>(local_index) << chunk_start);
            current_tile[shared_index] = current[state_index];
            lambda_tile[shared_index] = lambda[state_index];
        } else {
            current_tile[shared_index] = Complex(0.0, 0.0);
            lambda_tile[shared_index] = Complex(0.0, 0.0);
        }
    }
    __syncthreads();

    const auto process_group = [&](int high_wire, bool final_group) {
        const int low_wire = high_wire - 1;
        const int base_index = two_wire_cell_base(local_cell, low_wire);
        const int bit_low = 1 << low_wire;
        const int bit_high = 1 << high_wire;
        const int i0 = base_index;
        const int i1 = base_index | bit_low;
        const int i2 = base_index | bit_high;
        const int i3 = base_index | bit_low | bit_high;
        const int tile_offset = local_tile * tile_dim;

        Complex c0 = current_tile[tile_offset + i0];
        Complex c1 = current_tile[tile_offset + i1];
        Complex c2 = current_tile[tile_offset + i2];
        Complex c3 = current_tile[tile_offset + i3];
        Complex l0 = lambda_tile[tile_offset + i0];
        Complex l1 = lambda_tile[tile_offset + i1];
        Complex l2 = lambda_tile[tile_offset + i2];
        Complex l3 = lambda_tile[tile_offset + i3];

        double c0_r = c0.real();
        double c0_i = c0.imag();
        double c1_r = c1.real();
        double c1_i = c1.imag();
        double c2_r = c2.real();
        double c2_i = c2.imag();
        double c3_r = c3.real();
        double c3_i = c3.imag();
        double l0_r = l0.real();
        double l0_i = l0.imag();
        double l1_r = l1.real();
        double l1_i = l1.imag();
        double l2_r = l2.real();
        double l2_i = l2.imag();
        double l3_r = l3.real();
        double l3_i = l3.imag();

        double theta_high = 0.0;
        double phi_high = 0.0;
        double theta_low = 0.0;
        double phi_low = 0.0;

        inverse_walk_ryrz_backward_pair_registers(
            c0_r, c0_i, c2_r, c2_i, l0_r, l0_i, l2_r, l2_i,
            coeffs.c[high_wire], coeffs.s[high_wire],
            coeffs.cos_half[high_wire], coeffs.sin_half[high_wire],
            &theta_high, &phi_high);
        inverse_walk_ryrz_backward_pair_registers(
            c1_r, c1_i, c3_r, c3_i, l1_r, l1_i, l3_r, l3_i,
            coeffs.c[high_wire], coeffs.s[high_wire],
            coeffs.cos_half[high_wire], coeffs.sin_half[high_wire],
            &theta_high, &phi_high);
        inverse_walk_ryrz_backward_pair_registers(
            c0_r, c0_i, c1_r, c1_i, l0_r, l0_i, l1_r, l1_i,
            coeffs.c[low_wire], coeffs.s[low_wire],
            coeffs.cos_half[low_wire], coeffs.sin_half[low_wire],
            &theta_low, &phi_low);
        inverse_walk_ryrz_backward_pair_registers(
            c2_r, c2_i, c3_r, c3_i, l2_r, l2_i, l3_r, l3_i,
            coeffs.c[low_wire], coeffs.s[low_wire],
            coeffs.cos_half[low_wire], coeffs.sin_half[low_wire],
            &theta_low, &phi_low);

        current_tile[tile_offset + i0] = Complex(c0_r, c0_i);
        current_tile[tile_offset + i1] = Complex(c1_r, c1_i);
        current_tile[tile_offset + i2] = Complex(c2_r, c2_i);
        current_tile[tile_offset + i3] = Complex(c3_r, c3_i);
        lambda_tile[tile_offset + i0] = Complex(l0_r, l0_i);
        lambda_tile[tile_offset + i1] = Complex(l1_r, l1_i);
        lambda_tile[tile_offset + i2] = Complex(l2_r, l2_i);
        lambda_tile[tile_offset + i3] = Complex(l3_r, l3_i);

        theta_partial[high_wire * block_threads + threadIdx.x] = theta_high;
        phi_partial[high_wire * block_threads + threadIdx.x] = phi_high;
        theta_partial[low_wire * block_threads + threadIdx.x] = theta_low;
        phi_partial[low_wire * block_threads + threadIdx.x] = phi_low;
        __syncthreads();

        if (final_group) {
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const int shared_index = threadIdx.x + item * block_threads;
                const int tile_slot = shared_index / tile_dim;
                const int local_index = shared_index & (tile_dim - 1);
                const auto tile_id =
                    block_tile_base + static_cast<std::size_t>(tile_slot);
                if (tile_id < num_tiles) {
                    const auto base =
                        (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
                    const auto state_index =
                        base | (static_cast<std::size_t>(local_index)
                                << chunk_start);
                    current[state_index] = current_tile[shared_index];
                    lambda[state_index] = lambda_tile[shared_index];
                }
            }
        }
    };

    process_group(7, false);
    process_group(5, false);
    process_group(3, false);
    process_group(1, true);

    for (int stride = block_threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int local_wire = 0; local_wire < W; local_wire++) {
                const auto partial_offset =
                    static_cast<std::size_t>(local_wire) * block_threads;
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
                static_cast<std::size_t>(local_wire) * block_threads;
            atomicAdd(out_gradients + local_wire * 2,
                      2.0 * theta_partial[partial_offset]);
            atomicAdd(out_gradients + local_wire * 2 + 1,
                      2.0 * phi_partial[partial_offset]);
        }
    }
}

__global__ void inverse_walk_ryrz_rotation_chunk_pair512_kernel(
    Complex *current, Complex *lambda, std::size_t size,
    std::size_t chunk_start, RotationChunkCoeffs coeffs,
    double *out_gradients) {
    constexpr int W = 8;
    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block = std::size_t{2};

    __shared__ Complex current_tile[THREADS * 2];
    __shared__ Complex lambda_tile[THREADS * 2];
    __shared__ double theta_partial[W * THREADS];
    __shared__ double phi_partial[W * THREADS];

    const auto local_thread = static_cast<std::size_t>(threadIdx.x);
    const auto tile_id0 =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block;
    const auto tile_id1 = tile_id0 + 1;
    const auto num_tiles = size / tile_dim;
    const bool active0 = tile_id0 < num_tiles;
    const bool active1 = tile_id1 < num_tiles;

    std::size_t state_index0 = 0;
    std::size_t state_index1 = 0;
    const auto low_mask = (std::size_t{1} << chunk_start) - 1;
    if (active0) {
        const auto base =
            (tile_id0 & low_mask) | ((tile_id0 & ~low_mask) << W);
        state_index0 = base | (local_thread << chunk_start);
        current_tile[local_thread] = current[state_index0];
        lambda_tile[local_thread] = lambda[state_index0];
    }
    if (active1) {
        const auto base =
            (tile_id1 & low_mask) | ((tile_id1 & ~low_mask) << W);
        state_index1 = base | (local_thread << chunk_start);
        current_tile[tile_dim + local_thread] = current[state_index1];
        lambda_tile[tile_dim + local_thread] = lambda[state_index1];
    }
    __syncthreads();

#pragma unroll
    for (int local_wire = W - 1; local_wire >= 0; local_wire--) {
        double theta_sum = 0.0;
        double phi_sum = 0.0;
        const auto bit = std::size_t{1} << local_wire;

        if ((local_thread & bit) == 0U) {
            const double rz_c = coeffs.cos_half[local_wire];
            const double rz_s = coeffs.sin_half[local_wire];
            const double ry_c = coeffs.c[local_wire];
            const double ry_s = coeffs.s[local_wire];
            const auto partner_thread = local_thread | bit;

            if (active0) {
                inverse_walk_ryrz_backward_pair_scalar(
                    current_tile, lambda_tile, local_thread, partner_thread,
                    ry_c, ry_s, rz_c, rz_s, &theta_sum, &phi_sum);
            }
            if (active1) {
                const auto tile_offset = tile_dim;
                inverse_walk_ryrz_backward_pair_scalar(
                    current_tile, lambda_tile, tile_offset + local_thread,
                    tile_offset + partner_thread, ry_c, ry_s, rz_c, rz_s,
                    &theta_sum, &phi_sum);
            }
        }

        const auto partial_offset =
            static_cast<std::size_t>(local_wire) * THREADS + threadIdx.x;
        theta_partial[partial_offset] = theta_sum;
        phi_partial[partial_offset] = phi_sum;
        __syncthreads();
    }

    if (active0) {
        current[state_index0] = current_tile[local_thread];
        lambda[state_index0] = lambda_tile[local_thread];
    }
    if (active1) {
        current[state_index1] = current_tile[tile_dim + local_thread];
        lambda[state_index1] = lambda_tile[tile_dim + local_thread];
    }

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

void launch_inverse_walk_ry_gradient(const Complex *current,
                                     const Complex *lambda, std::size_t size,
                                     std::size_t wire, double theta,
                                     double *out_gradient) {
    const auto pairs = size / 2;
    const auto blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    inverse_walk_ry_gradient_kernel<<<blocks, THREADS>>>(
        current, lambda, size, wire, theta, out_gradient);
    check_cuda(cudaGetLastError(), "inverse_walk_ry_gradient_kernel");
    maybe_synchronize_cuda("inverse_walk_ry_gradient_kernel sync");
}

void launch_inverse_walk_rz_gradient(const Complex *current,
                                     const Complex *lambda, std::size_t size,
                                     std::size_t wire, double theta,
                                     double *out_gradient) {
    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    inverse_walk_rz_gradient_kernel<<<blocks, THREADS>>>(
        current, lambda, size, wire, theta, out_gradient);
    check_cuda(cudaGetLastError(), "inverse_walk_rz_gradient_kernel");
    maybe_synchronize_cuda("inverse_walk_rz_gradient_kernel sync");
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
    const auto num_tiles = size / tile_dim;
    if constexpr (W == 4) {
        if (chunk_start >= 8 && size >= (std::size_t{1} << 20)) {
            constexpr auto block_threads = 128;
            constexpr auto tiles_per_block = std::size_t{32};
            const auto blocks = static_cast<int>(
                (num_tiles + tiles_per_block - 1) / tiles_per_block);
            inverse_walk_ryrz_rotation_chunk_w4_cell2_kernel<<<
                blocks, block_threads>>>(
                current, lambda, size, chunk_start, coeffs, out_gradients);
            return;
        }
    }
    if constexpr (W >= 2 && W <= 4) {
        if (chunk_start >= 8 && size >= (std::size_t{1} << 20)) {
            constexpr auto tiles_per_block =
                static_cast<std::size_t>(THREADS) / tile_dim;
            const auto blocks = static_cast<int>(
                (num_tiles + tiles_per_block - 1) / tiles_per_block);
            inverse_walk_ryrz_rotation_chunk_transposed_kernel<W>
                <<<blocks, THREADS>>>(current, lambda, size, chunk_start,
                                      coeffs, out_gradients);
            return;
        }
    }
    if constexpr (W == 8) {
        if (chunk_start == 0 && size >= (std::size_t{1} << 24)) {
            constexpr auto block_threads = 128;
            constexpr auto tiles_per_block = std::size_t{2};
            const auto blocks = static_cast<int>(
                (num_tiles + tiles_per_block - 1) / tiles_per_block);
            inverse_walk_ryrz_rotation_chunk_w8_cell2_kernel<<<
                blocks, block_threads>>>(
                current, lambda, size, chunk_start, coeffs, out_gradients);
            return;
        }
        if (size >= (std::size_t{1} << 24)) {
            constexpr auto tiles_per_block = std::size_t{2};
            const auto blocks = static_cast<int>(
                (num_tiles + tiles_per_block - 1) / tiles_per_block);
            inverse_walk_ryrz_rotation_chunk_pair512_kernel<<<blocks, THREADS>>>(
                current, lambda, size, chunk_start, coeffs, out_gradients);
            return;
        }
    }

    constexpr auto tiles_per_block =
        static_cast<std::size_t>(THREADS) / tile_dim;
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

} // namespace detail
} // namespace standalone_backend
