#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace standalone_backend {
namespace detail {

__device__ inline void apply_ryrz_forward_pair(
    double c, double s, double cos_half, double sin_half, const Complex &a0,
    const Complex &a1, Complex *out0, Complex *out1) {
    const double a0_r = a0.real();
    const double a0_i = a0.imag();
    const double a1_r = a1.real();
    const double a1_i = a1.imag();
    const double b0_r = c * a0_r - s * a1_r;
    const double b0_i = c * a0_i - s * a1_i;
    const double b1_r = s * a0_r + c * a1_r;
    const double b1_i = s * a0_i + c * a1_i;

    *out0 = Complex(cos_half * b0_r + sin_half * b0_i,
                    cos_half * b0_i - sin_half * b0_r);
    *out1 = Complex(cos_half * b1_r - sin_half * b1_i,
                    cos_half * b1_i + sin_half * b1_r);
}

__device__ inline auto apply_ryrz_forward_warp_pair(
    Complex value, std::size_t local_index, int local_wire,
    const RotationChunkCoeffs &coeffs) -> Complex {
    const auto bit = std::size_t{1} << local_wire;
    const double partner_r =
        __shfl_xor_sync(0xffffffffU, value.real(), static_cast<int>(bit));
    const double partner_i =
        __shfl_xor_sync(0xffffffffU, value.imag(), static_cast<int>(bit));
    const Complex partner(partner_r, partner_i);
    const bool low = (local_index & bit) == 0U;

    Complex out0(0.0, 0.0);
    Complex out1(0.0, 0.0);
    if (low) {
        apply_ryrz_forward_pair(coeffs.c[local_wire], coeffs.s[local_wire],
                                coeffs.cos_half[local_wire],
                                coeffs.sin_half[local_wire], value, partner,
                                &out0, &out1);
    }

    const double out1_r = low ? out1.real() : 0.0;
    const double out1_i = low ? out1.imag() : 0.0;
    const double partner_out1_r =
        __shfl_xor_sync(0xffffffffU, out1_r, static_cast<int>(bit));
    const double partner_out1_i =
        __shfl_xor_sync(0xffffffffU, out1_i, static_cast<int>(bit));
    return low ? out0 : Complex(partner_out1_r, partner_out1_i);
}

template <int W>
__global__ void apply_ryrz_rotation_chunk_cooperative_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {
    static_assert(W >= 2 && W <= static_cast<int>(ROTATION_CHUNK_MAX_WIRES),
                  "unsupported rotation chunk width");

    __shared__ Complex tile[THREADS];
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
    Complex value(0.0, 0.0);
    if (active) {
        const auto low_mask = (std::size_t{1} << chunk_start) - 1;
        const auto base =
            (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
        state_index = base | (local_index << chunk_start);
        value = state[state_index];
    }

    constexpr auto warp_local_wires = W < 5 ? W : 5;
#pragma unroll
    for (int local_wire = 0; local_wire < warp_local_wires; local_wire++) {
        value = apply_ryrz_forward_warp_pair(value, local_index, local_wire,
                                             coeffs);
    }

    if constexpr (W > 5) {
        tile[local_thread] = value;
        __syncthreads();
    }

#pragma unroll
    for (int local_wire = 5; local_wire < W; local_wire++) {
        const auto bit = std::size_t{1} << local_wire;
        if (active && (local_index & bit) == 0U) {
            const auto partner_thread = local_thread | bit;
            const double c = coeffs.c[local_wire];
            const double s = coeffs.s[local_wire];
            const double cos_half = coeffs.cos_half[local_wire];
            const double sin_half = coeffs.sin_half[local_wire];

            Complex out0;
            Complex out1;
            apply_ryrz_forward_pair(c, s, cos_half, sin_half,
                                    tile[local_thread], tile[partner_thread],
                                    &out0, &out1);
            tile[local_thread] = out0;
            tile[partner_thread] = out1;
        }
        __syncthreads();
    }

    if (active) {
        if constexpr (W > 5) {
            value = tile[local_thread];
        }
        state[state_index] = value;
    }
}

template <int W>
__global__ void apply_ryrz_rotation_chunk_register_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {
    static_assert(W >= 2 && W <= static_cast<int>(ROTATION_CHUNK_MAX_WIRES),
                  "unsupported rotation chunk width");

    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto pairs_per_wire = tile_dim / 2;
    const auto tile_id = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                  threadIdx.x);
    const auto num_tiles = size / tile_dim;
    if (tile_id >= num_tiles) {
        return;
    }

    const auto low_mask = (std::size_t{1} << chunk_start) - 1;
    const auto base = (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
    Complex values[tile_dim];

#pragma unroll
    for (int local_index = 0; local_index < static_cast<int>(tile_dim);
         local_index++) {
        values[local_index] =
            state[base | (static_cast<std::size_t>(local_index)
                          << chunk_start)];
    }

#pragma unroll
    for (int local_wire = 0; local_wire < W; local_wire++) {
        const double c = coeffs.c[local_wire];
        const double s = coeffs.s[local_wire];
        const double cos_half = coeffs.cos_half[local_wire];
        const double sin_half = coeffs.sin_half[local_wire];
        const auto bit = std::size_t{1} << local_wire;
        const auto pair_low_mask = bit - 1;

#pragma unroll
        for (int pair = 0; pair < static_cast<int>(pairs_per_wire); pair++) {
            const auto pair_index = static_cast<std::size_t>(pair);
            const auto local_low = pair_index & pair_low_mask;
            const auto local_high = pair_index >> local_wire;
            const auto i0 = (local_high << (local_wire + 1)) | local_low;
            const auto i1 = i0 | bit;

            Complex out0;
            Complex out1;
            apply_ryrz_forward_pair(c, s, cos_half, sin_half, values[i0],
                                    values[i1], &out0, &out1);
            values[i0] = out0;
            values[i1] = out1;
        }
    }

#pragma unroll
    for (int local_index = 0; local_index < static_cast<int>(tile_dim);
         local_index++) {
        state[base | (static_cast<std::size_t>(local_index) << chunk_start)] =
            values[local_index];
    }
}

__global__ void apply_ryrz_rotation_chunk_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    std::size_t chunk_width, RotationChunkCoeffs coeffs) {
    __shared__ Complex tile[THREADS];

    const auto tile_dim = std::size_t{1} << chunk_width;
    const auto tiles_per_block = static_cast<std::size_t>(THREADS) / tile_dim;
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
        const auto base =
            (tile_id & low_mask) | ((tile_id & ~low_mask) << chunk_width);
        state_index = base | (local_index << chunk_start);
        tile[local_thread] = state[state_index];
    }
    __syncthreads();

    for (std::size_t local_wire = 0; local_wire < chunk_width; local_wire++) {
        const auto bit = std::size_t{1} << local_wire;
        if (active && (local_index & bit) == 0U) {
            const auto partner_thread = local_thread | bit;
            const double c = coeffs.c[local_wire];
            const double s = coeffs.s[local_wire];
            const double cos_half = coeffs.cos_half[local_wire];
            const double sin_half = coeffs.sin_half[local_wire];

            const Complex a0 = tile[local_thread];
            const Complex a1 = tile[partner_thread];
            const double a0_r = a0.real();
            const double a0_i = a0.imag();
            const double a1_r = a1.real();
            const double a1_i = a1.imag();
            const double b0_r = c * a0_r - s * a1_r;
            const double b0_i = c * a0_i - s * a1_i;
            const double b1_r = s * a0_r + c * a1_r;
            const double b1_i = s * a0_i + c * a1_i;

            tile[local_thread] =
                Complex(cos_half * b0_r + sin_half * b0_i,
                        cos_half * b0_i - sin_half * b0_r);
            tile[partner_thread] =
                Complex(cos_half * b1_r - sin_half * b1_i,
                        cos_half * b1_i + sin_half * b1_r);
        }
        __syncthreads();
    }

    if (active) {
        state[state_index] = tile[local_thread];
    }
}

__device__ inline void fused_ryrz_local_coeff(
    bool out_bit, bool in_bit, double c, double s, double cos_half,
    double sin_half, double *real, double *imag) {
    if (!out_bit && !in_bit) {
        *real = cos_half * c;
        *imag = -sin_half * c;
        return;
    }
    if (out_bit && !in_bit) {
        *real = cos_half * s;
        *imag = sin_half * s;
        return;
    }
    if (!out_bit && in_bit) {
        *real = -cos_half * s;
        *imag = sin_half * s;
        return;
    }
    *real = cos_half * c;
    *imag = sin_half * c;
}

__global__ void apply_ryrz_rotation_dense_chunk_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    std::size_t chunk_width, RotationChunkCoeffs coeffs) {
    __shared__ Complex in_tile[THREADS];
    __shared__ Complex out_tile[THREADS];

    const auto tile_dim = std::size_t{1} << chunk_width;
    const auto tiles_per_block = static_cast<std::size_t>(THREADS) / tile_dim;
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
        const auto base =
            (tile_id & low_mask) | ((tile_id & ~low_mask) << chunk_width);
        state_index = base | (local_index << chunk_start);
        in_tile[local_thread] = state[state_index];
    }
    __syncthreads();

    if (active) {
        const auto tile_offset = local_tile * tile_dim;
        double acc_r = 0.0;
        double acc_i = 0.0;
        for (std::size_t input_index = 0; input_index < tile_dim;
             input_index++) {
            double coeff_r = 1.0;
            double coeff_i = 0.0;
            for (std::size_t local_wire = 0; local_wire < chunk_width;
                 local_wire++) {
                const bool out_bit =
                    ((local_index >> local_wire) & std::size_t{1}) != 0U;
                const bool in_bit =
                    ((input_index >> local_wire) & std::size_t{1}) != 0U;
                const double c = coeffs.c[local_wire];
                const double s = coeffs.s[local_wire];
                const double cos_half = coeffs.cos_half[local_wire];
                const double sin_half = coeffs.sin_half[local_wire];
                double local_r = 0.0;
                double local_i = 0.0;
                fused_ryrz_local_coeff(out_bit, in_bit, c, s, cos_half,
                                       sin_half, &local_r, &local_i);
                const double next_r = coeff_r * local_r - coeff_i * local_i;
                const double next_i = coeff_r * local_i + coeff_i * local_r;
                coeff_r = next_r;
                coeff_i = next_i;
            }

            const Complex input = in_tile[tile_offset + input_index];
            acc_r += coeff_r * input.real() - coeff_i * input.imag();
            acc_i += coeff_r * input.imag() + coeff_i * input.real();
        }
        out_tile[local_thread] = Complex(acc_r, acc_i);
    }
    __syncthreads();

    if (active) {
        state[state_index] = out_tile[local_thread];
    }
}

template <int W>
void launch_apply_ryrz_rotation_chunk_specialized(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {
    constexpr auto tile_dim = std::size_t{1} << W;
    const auto num_tiles = size / tile_dim;
    if constexpr (W <= 4) {
        constexpr auto register_tile_min_tiles =
            W == 4 ? (std::size_t{1} << 14)
                   : ROTATION_CHUNK_REGISTER_TILE_MIN_TILES;
        if (chunk_start >= ROTATION_CHUNK_REGISTER_TILE_START &&
            num_tiles >= register_tile_min_tiles) {
            const auto blocks =
                static_cast<int>((num_tiles + THREADS - 1) / THREADS);
            apply_ryrz_rotation_chunk_register_kernel<W><<<blocks, THREADS>>>(
                state, size, chunk_start, coeffs);
            return;
        }
    }

    constexpr auto tiles_per_block =
        static_cast<std::size_t>(THREADS) / tile_dim;
    const auto blocks =
        static_cast<int>((num_tiles + tiles_per_block - 1) / tiles_per_block);
    apply_ryrz_rotation_chunk_cooperative_kernel<W><<<blocks, THREADS>>>(
        state, size, chunk_start, coeffs);
}

auto make_rotation_chunk_coeffs(std::size_t chunk_width,
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

void launch_apply_ryrz_rotation_chunk(Complex *state, std::size_t size,
                                      std::size_t chunk_start,
                                      std::size_t chunk_width,
                                      const double *theta_ry,
                                      const double *theta_rz) {
    if (chunk_width == 0) {
        return;
    }
    if (chunk_width > ROTATION_CHUNK_MAX_WIRES) {
        throw std::invalid_argument(
            "rotation chunk width exceeds supported maximum.");
    }
    if (chunk_start + chunk_width >= sizeof(std::size_t) * 8 ||
        (std::size_t{1} << (chunk_start + chunk_width)) > size) {
        throw std::invalid_argument("rotation chunk exceeds state dimension.");
    }

    const auto tile_dim = std::size_t{1} << chunk_width;
    if (tile_dim > static_cast<std::size_t>(THREADS)) {
        throw std::invalid_argument("rotation chunk tile_dim exceeds THREADS.");
    }
    const auto tiles_per_block = static_cast<std::size_t>(THREADS) / tile_dim;
    const auto num_tiles = size / tile_dim;
    const auto blocks =
        static_cast<int>((num_tiles + tiles_per_block - 1) / tiles_per_block);
    const auto coeffs =
        make_rotation_chunk_coeffs(chunk_width, theta_ry, theta_rz);

    switch (chunk_width) {
    case 2:
        launch_apply_ryrz_rotation_chunk_specialized<2>(
            state, size, chunk_start, coeffs);
        break;
    case 3:
        launch_apply_ryrz_rotation_chunk_specialized<3>(
            state, size, chunk_start, coeffs);
        break;
    case 4:
        launch_apply_ryrz_rotation_chunk_specialized<4>(
            state, size, chunk_start, coeffs);
        break;
    case 5:
        launch_apply_ryrz_rotation_chunk_specialized<5>(
            state, size, chunk_start, coeffs);
        break;
    case 6:
        launch_apply_ryrz_rotation_chunk_specialized<6>(
            state, size, chunk_start, coeffs);
        break;
    case 7:
        launch_apply_ryrz_rotation_chunk_specialized<7>(
            state, size, chunk_start, coeffs);
        break;
    case 8:
        launch_apply_ryrz_rotation_chunk_specialized<8>(
            state, size, chunk_start, coeffs);
        break;
    default:
        apply_ryrz_rotation_chunk_kernel<<<blocks, THREADS>>>(
            state, size, chunk_start, chunk_width, coeffs);
        break;
    }
    check_cuda(cudaGetLastError(), "apply_ryrz_rotation_chunk_kernel");
    maybe_synchronize_cuda("apply_ryrz_rotation_chunk_kernel sync");
}

void launch_apply_ryrz_rotation_dense_chunk(Complex *state, std::size_t size,
                                            std::size_t chunk_start,
                                            std::size_t chunk_width,
                                            const double *theta_ry,
                                            const double *theta_rz) {
    if (chunk_width == 0) {
        return;
    }
    if (chunk_width > DENSE_ROTATION_CHUNK_MAX_WIRES) {
        throw std::invalid_argument(
            "dense rotation chunk width exceeds supported maximum.");
    }
    if (chunk_start + chunk_width >= sizeof(std::size_t) * 8 ||
        (std::size_t{1} << (chunk_start + chunk_width)) > size) {
        throw std::invalid_argument(
            "dense rotation chunk exceeds state dimension.");
    }

    const auto tile_dim = std::size_t{1} << chunk_width;
    if (tile_dim > static_cast<std::size_t>(THREADS)) {
        throw std::invalid_argument(
            "dense rotation chunk tile_dim exceeds THREADS.");
    }
    const auto tiles_per_block = static_cast<std::size_t>(THREADS) / tile_dim;
    const auto num_tiles = size / tile_dim;
    const auto blocks =
        static_cast<int>((num_tiles + tiles_per_block - 1) / tiles_per_block);
    const auto coeffs =
        make_rotation_chunk_coeffs(chunk_width, theta_ry, theta_rz);

    apply_ryrz_rotation_dense_chunk_kernel<<<blocks, THREADS>>>(
        state, size, chunk_start, chunk_width, coeffs);
    check_cuda(cudaGetLastError(),
               "apply_ryrz_rotation_dense_chunk_kernel");
    maybe_synchronize_cuda(
        "apply_ryrz_rotation_dense_chunk_kernel sync");
}

} // namespace detail
} // namespace standalone_backend
