#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace standalone_backend {
namespace detail {

__device__ __forceinline__ void apply_ryrz_forward_pair(
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

__device__ __forceinline__ Complex apply_ryrz_forward_warp_pair(
    Complex value, unsigned local_index, int local_wire,
    const RotationChunkCoeffs &coeffs) {

    const unsigned bit = 1U << local_wire;
    const unsigned mask = __activemask();

    const double value_r = value.real();
    const double value_i = value.imag();

    const double partner_r =
        __shfl_xor_sync(mask, value_r, static_cast<int>(bit));
    const double partner_i =
        __shfl_xor_sync(mask, value_i, static_cast<int>(bit));

    // Low lane: -1, high lane: +1.
    const double sign = (local_index & bit) ? 1.0 : -1.0;

    const double c = coeffs.c[local_wire];
    const double s = coeffs.s[local_wire];
    const double ch = coeffs.cos_half[local_wire];
    const double sh = coeffs.sin_half[local_wire];

    const double signed_s = sign * s;
    const double signed_sh = sign * sh;

    const double b_r = fma(signed_s, partner_r, c * value_r);
    const double b_i = fma(signed_s, partner_i, c * value_i);

    return Complex(
        fma(-signed_sh, b_i, ch * b_r),
        fma( signed_sh, b_r, ch * b_i));
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

__global__ void apply_ryrz_rotation_chunk_cooperative_pair512_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {
    constexpr int W = 8;
    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block = std::size_t{2};
    constexpr auto pairs_per_tile = tile_dim / 2;

    __shared__ Complex tile[THREADS * 2];

    const auto local_thread = static_cast<std::size_t>(threadIdx.x);
    const auto num_tiles = size / tile_dim;
    std::size_t state_index0 = 0;
    std::size_t state_index1 = 0;
    Complex value0(0.0, 0.0);
    Complex value1(0.0, 0.0);

    const auto tile_id0 =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block;
    const auto tile_id1 = tile_id0 + 1;
    const bool active0 = tile_id0 < num_tiles;
    const bool active1 = tile_id1 < num_tiles;

    const auto low_mask = (std::size_t{1} << chunk_start) - 1;
    if (active0) {
        const auto base =
            (tile_id0 & low_mask) | ((tile_id0 & ~low_mask) << W);
        state_index0 = base | (local_thread << chunk_start);
        value0 = state[state_index0];
    }
    if (active1) {
        const auto base =
            (tile_id1 & low_mask) | ((tile_id1 & ~low_mask) << W);
        state_index1 = base | (local_thread << chunk_start);
        value1 = state[state_index1];
    }

#pragma unroll
    for (int local_wire = 0; local_wire < 5; local_wire++) {
        value0 = apply_ryrz_forward_warp_pair(
            value0, static_cast<unsigned>(local_thread), local_wire, coeffs);
        value1 = apply_ryrz_forward_warp_pair(
            value1, static_cast<unsigned>(local_thread), local_wire, coeffs);
    }

    tile[local_thread] = value0;
    tile[tile_dim + local_thread] = value1;
    __syncthreads();

#pragma unroll
    for (int local_wire = 5; local_wire < W; local_wire++) {
        const auto bit = std::size_t{1} << local_wire;
        const auto pair_index = local_thread & (pairs_per_tile - 1);
        const auto tile_slot = local_thread >> 7U;
        const auto local_low = pair_index & (bit - 1);
        const auto local_high = pair_index >> local_wire;
        const auto i0 = (local_high << (local_wire + 1)) | local_low;
        const auto i1 = i0 | bit;
        const auto tile_offset = tile_slot * tile_dim;
        const bool active_tile = tile_slot == 0 ? active0 : active1;

        if (active_tile) {
            Complex out0;
            Complex out1;
            apply_ryrz_forward_pair(
                coeffs.c[local_wire], coeffs.s[local_wire],
                coeffs.cos_half[local_wire], coeffs.sin_half[local_wire],
                tile[tile_offset + i0], tile[tile_offset + i1], &out0,
                &out1);
            tile[tile_offset + i0] = out0;
            tile[tile_offset + i1] = out1;
        }
        __syncthreads();
    }

    if (active0) {
        state[state_index0] = tile[local_thread];
    }
    if (active1) {
        state[state_index1] = tile[tile_dim + local_thread];
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

auto structured_state_qubits(std::size_t size) -> std::size_t {
    std::size_t qubits = 0;
    while ((std::size_t{1} << qubits) < size) {
        qubits++;
    }
    return qubits;
}

auto structured_register_threads_per_block(std::size_t size, int chunk_width)
    -> int {
    const auto num_qubits = structured_state_qubits(size);
    if (chunk_width == 2 && (num_qubits == 21 || num_qubits == 22)) {
        return 32;
    }
    if (chunk_width == 4 && num_qubits == 23) {
        return 32;
    }
    if (chunk_width == 4 && num_qubits >= 24) {
        return 96;
    }
    return THREADS;
}

template <int W>
void launch_apply_ryrz_rotation_chunk_specialized(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs,
    RotationChunkKernelPreference kernel_preference) {
    constexpr auto tile_dim = std::size_t{1} << W;
    const auto num_tiles = size / tile_dim;
    if constexpr (W <= 4) {
        if (kernel_preference == RotationChunkKernelPreference::Register) {
            const auto register_threads =
                structured_register_threads_per_block(size, W);
            const auto blocks =
                static_cast<int>((num_tiles + register_threads - 1) /
                                 register_threads);
            apply_ryrz_rotation_chunk_register_kernel<W>
                <<<blocks, register_threads>>>(state, size, chunk_start,
                                               coeffs);
            return;
        }
    }
    if constexpr (W == 8) {
        if (kernel_preference ==
            RotationChunkKernelPreference::CooperativePair512) {
            constexpr auto tiles_per_block = std::size_t{2};
            const auto blocks = static_cast<int>(
                (num_tiles + tiles_per_block - 1) / tiles_per_block);
            apply_ryrz_rotation_chunk_cooperative_pair512_kernel
                <<<blocks, THREADS>>>(state, size, chunk_start, coeffs);
            return;
        }
    }
    if (kernel_preference == RotationChunkKernelPreference::Register) {
        throw std::invalid_argument(
            "register rotation chunks are supported only for width <= 4.");
    }
    if (kernel_preference ==
        RotationChunkKernelPreference::CooperativePair512) {
        if constexpr (W > 8) {
            throw std::invalid_argument(
                "cooperative pair512 rotation chunks support only width 8.");
        }
    }
    if constexpr (W > 8) {
        throw std::invalid_argument("unsupported rotation chunk width.");
    } else {
        constexpr auto tiles_per_block =
            static_cast<std::size_t>(THREADS) / tile_dim;
        const auto blocks = static_cast<int>(
            (num_tiles + tiles_per_block - 1) / tiles_per_block);
        apply_ryrz_rotation_chunk_cooperative_kernel<W><<<blocks, THREADS>>>(
            state, size, chunk_start, coeffs);
    }
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
                                      const double *theta_rz,
                                      RotationChunkKernelPreference
                                          kernel_preference) {
    if (chunk_width == 0) {
        return;
    }
    if (chunk_width > ROTATION_CHUNK_MAX_WIRES) {
        throw std::invalid_argument(
            "rotation chunk width exceeds supported maximum.");
    }
    if (chunk_width == 1) {
        throw std::invalid_argument(
            "rotation chunk width 1 should use launch_apply_ryrz.");
    }
    if (chunk_start + chunk_width >= sizeof(std::size_t) * 8 ||
        (std::size_t{1} << (chunk_start + chunk_width)) > size) {
        throw std::invalid_argument("rotation chunk exceeds state dimension.");
    }

    const auto tile_dim = std::size_t{1} << chunk_width;
    if (tile_dim > static_cast<std::size_t>(THREADS)) {
        throw std::invalid_argument("rotation chunk tile_dim exceeds THREADS.");
    }
    const auto coeffs =
        make_rotation_chunk_coeffs(chunk_width, theta_ry, theta_rz);

    switch (chunk_width) {
    case 2:
        launch_apply_ryrz_rotation_chunk_specialized<2>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    case 3:
        launch_apply_ryrz_rotation_chunk_specialized<3>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    case 4:
        launch_apply_ryrz_rotation_chunk_specialized<4>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    case 5:
        launch_apply_ryrz_rotation_chunk_specialized<5>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    case 6:
        launch_apply_ryrz_rotation_chunk_specialized<6>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    case 7:
        launch_apply_ryrz_rotation_chunk_specialized<7>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    case 8:
        launch_apply_ryrz_rotation_chunk_specialized<8>(
            state, size, chunk_start, coeffs, kernel_preference);
        break;
    default:
        throw std::invalid_argument("unsupported rotation chunk width.");
    }
    check_cuda(cudaGetLastError(), "apply_ryrz_rotation_chunk_kernel");
    maybe_synchronize_cuda("apply_ryrz_rotation_chunk_kernel sync");
}

} // namespace detail
} // namespace standalone_backend
