#include "ising_cuda_kernel_common.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace standalone_backend {
namespace detail {

struct ProductStateInitCoeffs {
    double zero_r[PRODUCT_STATE_INIT_MAX_QUBITS]{};
    double zero_i[PRODUCT_STATE_INIT_MAX_QUBITS]{};
    double one_r[PRODUCT_STATE_INIT_MAX_QUBITS]{};
    double one_i[PRODUCT_STATE_INIT_MAX_QUBITS]{};
};

__global__ void init_ryrz_product_state_kernel(
    Complex *state, std::size_t size, std::size_t num_qubits,
    ProductStateInitCoeffs coeffs) {
    const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
    if (index >= size) {
        return;
    }

    double value_r = 1.0;
    double value_i = 0.0;
    for (std::size_t wire = 0; wire < num_qubits; wire++) {
        const bool one = bit_is_set(index, wire);
        const double factor_r = one ? coeffs.one_r[wire] : coeffs.zero_r[wire];
        const double factor_i = one ? coeffs.one_i[wire] : coeffs.zero_i[wire];
        const double next_r = value_r * factor_r - value_i * factor_i;
        const double next_i = value_r * factor_i + value_i * factor_r;
        value_r = next_r;
        value_i = next_i;
    }
    state[index] = Complex(value_r, value_i);
}

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

// ============================================================
// Phase 1: Register kernel with grid-stride double buffer (W=2/3/4)
// Each thread processes multiple tiles in a grid-stride loop.
// The next tile is loaded into values_nxt[] while values_cur[] is being
// computed, giving the hardware scheduler a chance to overlap DRAM latency
// with arithmetic (instruction-level parallelism / software pipelining).
// ============================================================

template <int W>
__global__ void apply_ryrz_rotation_chunk_register_db_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {
    static_assert(W >= 2 && W <= 4,
                  "RegisterDoubleBuffer is only supported for W <= 4");

    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto pairs_per_wire = tile_dim / 2;
    const auto num_tiles = size / tile_dim;
    const auto stride = static_cast<std::size_t>(gridDim.x) *
                        static_cast<std::size_t>(blockDim.x);
    auto tile_id = static_cast<std::size_t>(blockIdx.x * blockDim.x +
                                            threadIdx.x);
    if (tile_id >= num_tiles) {
        return;
    }

    const auto low_mask = (std::size_t{1} << chunk_start) - 1;

    // Helper: compute the base DRAM address for a given tile_id
    auto tile_base = [&](std::size_t tid) -> std::size_t {
        return (tid & low_mask) | ((tid & ~low_mask) << W);
    };

    // Helper: apply all W RyRz gates to a register array in-place
    auto compute = [&](Complex *vals) {
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
                const auto pi = static_cast<std::size_t>(pair);
                const auto lo = pi & pair_low_mask;
                const auto hi = pi >> local_wire;
                const auto i0 = (hi << (local_wire + 1)) | lo;
                const auto i1 = i0 | bit;
                Complex out0, out1;
                apply_ryrz_forward_pair(c, s, cos_half, sin_half,
                                        vals[i0], vals[i1], &out0, &out1);
                vals[i0] = out0;
                vals[i1] = out1;
            }
        }
    };

    // Prologue: load first tile into values_cur
    Complex values_cur[tile_dim];
    {
        const auto base = tile_base(tile_id);
#pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            values_cur[i] = state[base | (static_cast<std::size_t>(i) << chunk_start)];
        }
    }

    // Main loop: prefetch next tile while computing current tile
    Complex values_nxt[tile_dim];
    while (tile_id + stride < num_tiles) {
        // Prefetch next tile (issued before compute so hardware can overlap)
        const auto nxt_base = tile_base(tile_id + stride);
#pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            values_nxt[i] = state[nxt_base | (static_cast<std::size_t>(i) << chunk_start)];
        }

        // Compute current tile
        compute(values_cur);

        // Store current tile
        const auto cur_base = tile_base(tile_id);
#pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            state[cur_base | (static_cast<std::size_t>(i) << chunk_start)] = values_cur[i];
        }

        // Advance and swap buffers
        tile_id += stride;
#pragma unroll
        for (int i = 0; i < static_cast<int>(tile_dim); i++) {
            values_cur[i] = values_nxt[i];
        }
    }

    // Epilogue: compute and store the last tile
    compute(values_cur);
    const auto last_base = tile_base(tile_id);
#pragma unroll
    for (int i = 0; i < static_cast<int>(tile_dim); i++) {
        state[last_base | (static_cast<std::size_t>(i) << chunk_start)] = values_cur[i];
    }
}

// ============================================================
// Phase 2: Cooperative kernel with shared-memory double buffer (W=2..8)
// Uses cuda::pipeline<thread_scope_block> + cuda::memcpy_async to overlap
// DRAM loads of the next tile with computation of the current tile.
// The pipeline depth is 2 (double buffer).
// ============================================================

template <int W>
__global__ void apply_ryrz_rotation_chunk_cooperative_db_kernel(
    Complex *state, std::size_t size, std::size_t chunk_start,
    RotationChunkCoeffs coeffs) {
    static_assert(W >= 2 && W <= static_cast<int>(ROTATION_CHUNK_MAX_WIRES),
                  "unsupported rotation chunk width");

    // Double buffer: two shared-memory tiles
    __shared__ Complex tile[2][THREADS];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;

    constexpr auto tile_dim = std::size_t{1} << W;
    constexpr auto tiles_per_block =
        static_cast<std::size_t>(THREADS) / tile_dim;
    const auto local_thread = static_cast<std::size_t>(threadIdx.x);
    const auto local_tile = local_thread / tile_dim;
    const auto local_index = local_thread & (tile_dim - 1);
    const auto num_tiles = size / tile_dim;
    const auto low_mask = (std::size_t{1} << chunk_start) - 1;

    // Each block processes a contiguous range of tiles in steps of tiles_per_block
    const auto block_tile_base =
        static_cast<std::size_t>(blockIdx.x) * tiles_per_block;
    const auto total_block_tiles =
        (num_tiles > block_tile_base)
            ? ((num_tiles - block_tile_base + tiles_per_block - 1) /
               tiles_per_block)
            : std::size_t{0};

    if (total_block_tiles == 0) {
        return;
    }

    auto block = cooperative_groups::this_thread_block();
    auto pipeline = cuda::make_pipeline(block, &pipe_state);

    // Helper: compute state_index for (block_tile_step, local_tile, local_index)
    auto state_idx = [&](std::size_t block_step) -> std::size_t {
        const auto tile_id = block_tile_base + block_step * tiles_per_block + local_tile;
        if (tile_id >= num_tiles) {
            return static_cast<std::size_t>(-1); // sentinel: inactive
        }
        const auto base = (tile_id & low_mask) | ((tile_id & ~low_mask) << W);
        return base | (local_index << chunk_start);
    };

    // Prologue: async-load step 0 into tile[0]
    int cur_buf = 0;
    {
        const auto sidx = state_idx(0);
        pipeline.producer_acquire();
        if (sidx != static_cast<std::size_t>(-1)) {
            cuda::memcpy_async(
                &tile[cur_buf][local_thread],
                &state[sidx],
                sizeof(Complex), pipeline);
        } else {
            tile[cur_buf][local_thread] = Complex(0.0, 0.0);
        }
        pipeline.producer_commit();
    }

    for (std::size_t step = 0; step < total_block_tiles; step++) {
        // Prefetch next step into tile[1-cur_buf]
        if (step + 1 < total_block_tiles) {
            const auto sidx_nxt = state_idx(step + 1);
            pipeline.producer_acquire();
            if (sidx_nxt != static_cast<std::size_t>(-1)) {
                cuda::memcpy_async(
                    &tile[1 - cur_buf][local_thread],
                    &state[sidx_nxt],
                    sizeof(Complex), pipeline);
            } else {
                tile[1 - cur_buf][local_thread] = Complex(0.0, 0.0);
            }
            pipeline.producer_commit();
        }

        // Wait for current step data
        pipeline.consumer_wait();

        // Compute: warp-shuffle for wire < 5, shared-mem for wire >= 5
        Complex value = tile[cur_buf][local_thread];
        const auto tile_id = block_tile_base + step * tiles_per_block + local_tile;
        const bool active = tile_id < num_tiles;

        constexpr auto warp_local_wires = W < 5 ? W : 5;
#pragma unroll
        for (int local_wire = 0; local_wire < warp_local_wires; local_wire++) {
            value = apply_ryrz_forward_warp_pair(value, static_cast<unsigned>(local_index),
                                                 local_wire, coeffs);
        }

        if constexpr (W > 5) {
            tile[cur_buf][local_thread] = value;
            __syncthreads();
        }

#pragma unroll
        for (int local_wire = 5; local_wire < W; local_wire++) {
            const auto bit = std::size_t{1} << local_wire;
            if (active && (local_index & bit) == 0U) {
                const auto partner_thread = local_thread | bit;
                Complex out0, out1;
                apply_ryrz_forward_pair(
                    coeffs.c[local_wire], coeffs.s[local_wire],
                    coeffs.cos_half[local_wire], coeffs.sin_half[local_wire],
                    tile[cur_buf][local_thread], tile[cur_buf][partner_thread],
                    &out0, &out1);
                tile[cur_buf][local_thread] = out0;
                tile[cur_buf][partner_thread] = out1;
            }
            __syncthreads();
        }

        if constexpr (W > 5) {
            value = tile[cur_buf][local_thread];
        }

        pipeline.consumer_release();

        // Store result
        const auto sidx = state_idx(step);
        if (active && sidx != static_cast<std::size_t>(-1)) {
            state[sidx] = value;
        }

        cur_buf = 1 - cur_buf;
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
    RotationChunkKernelPreference kernel_preference, cudaStream_t stream) {
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
                <<<blocks, register_threads, 0, stream>>>(
                    state, size, chunk_start, coeffs);
            return;
        }
        // Phase 1: grid-stride register double buffer
        if (kernel_preference ==
            RotationChunkKernelPreference::RegisterDoubleBuffer) {
            const auto register_threads =
                structured_register_threads_per_block(size, W);
            // Use a fixed grid of 2048 blocks so each thread processes
            // multiple tiles in the grid-stride loop.
            constexpr int DB_GRID = 2048;
            const auto blocks = static_cast<int>(
                std::min(static_cast<std::size_t>(DB_GRID),
                         (num_tiles + register_threads - 1) /
                             static_cast<std::size_t>(register_threads)));
            apply_ryrz_rotation_chunk_register_db_kernel<W>
                <<<blocks, register_threads, 0, stream>>>(
                    state, size, chunk_start, coeffs);
            return;
        }
    }
    // Phase 2: cooperative shared-memory double buffer (all widths)
    if (kernel_preference ==
        RotationChunkKernelPreference::CooperativeDoubleBuffer) {
        constexpr auto tiles_per_block =
            static_cast<std::size_t>(THREADS) / tile_dim;
        const auto blocks = static_cast<int>(
            (num_tiles + tiles_per_block - 1) / tiles_per_block);
        apply_ryrz_rotation_chunk_cooperative_db_kernel<W>
            <<<blocks, THREADS, 0, stream>>>(state, size, chunk_start, coeffs);
        return;
    }
    if constexpr (W == 8) {
        if (kernel_preference ==
            RotationChunkKernelPreference::CooperativePair512) {
            constexpr auto tiles_per_block = std::size_t{2};
            const auto blocks = static_cast<int>(
                (num_tiles + tiles_per_block - 1) / tiles_per_block);
            apply_ryrz_rotation_chunk_cooperative_pair512_kernel
                <<<blocks, THREADS, 0, stream>>>(
                    state, size, chunk_start, coeffs);
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
        apply_ryrz_rotation_chunk_cooperative_kernel<W>
            <<<blocks, THREADS, 0, stream>>>(state, size, chunk_start,
                                             coeffs);
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

void launch_init_ryrz_product_state(Complex *state, std::size_t size,
                                    std::size_t num_qubits,
                                    const double *layer_params,
                                    cudaStream_t stream) {
    if (num_qubits > PRODUCT_STATE_INIT_MAX_QUBITS) {
        throw std::invalid_argument(
            "product-state initialization exceeds supported qubit count.");
    }

    ProductStateInitCoeffs coeffs{};
    for (std::size_t wire = 0; wire < num_qubits; wire++) {
        const double theta =
            layer_params != nullptr ? layer_params[wire * 2] : 0.0;
        const double phi =
            layer_params != nullptr ? layer_params[wire * 2 + 1] : 0.0;
        const double ry = theta * 0.5;
        const double rz = phi * 0.5;
        const double c = std::cos(ry);
        const double s = std::sin(ry);
        const double cos_half = std::cos(rz);
        const double sin_half = std::sin(rz);

        coeffs.zero_r[wire] = c * cos_half;
        coeffs.zero_i[wire] = -c * sin_half;
        coeffs.one_r[wire] = s * cos_half;
        coeffs.one_i[wire] = s * sin_half;
    }

    const auto blocks = static_cast<int>((size + THREADS - 1) / THREADS);
    init_ryrz_product_state_kernel<<<blocks, THREADS, 0, stream>>>(
        state, size, num_qubits, coeffs);
    check_cuda(cudaGetLastError(), "init_ryrz_product_state_kernel");
    maybe_synchronize_cuda("init_ryrz_product_state_kernel sync");
}

void launch_apply_ryrz_rotation_chunk(Complex *state, std::size_t size,
                                      std::size_t chunk_start,
                                      std::size_t chunk_width,
                                      const double *theta_ry,
                                      const double *theta_rz,
                                      RotationChunkKernelPreference
                                          kernel_preference,
                                      cudaStream_t stream) {
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
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    case 3:
        launch_apply_ryrz_rotation_chunk_specialized<3>(
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    case 4:
        launch_apply_ryrz_rotation_chunk_specialized<4>(
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    case 5:
        launch_apply_ryrz_rotation_chunk_specialized<5>(
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    case 6:
        launch_apply_ryrz_rotation_chunk_specialized<6>(
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    case 7:
        launch_apply_ryrz_rotation_chunk_specialized<7>(
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    case 8:
        launch_apply_ryrz_rotation_chunk_specialized<8>(
            state, size, chunk_start, coeffs, kernel_preference, stream);
        break;
    default:
        throw std::invalid_argument("unsupported rotation chunk width.");
    }
    check_cuda(cudaGetLastError(), "apply_ryrz_rotation_chunk_kernel");
    maybe_synchronize_cuda("apply_ryrz_rotation_chunk_kernel sync");
}

} // namespace detail
} // namespace standalone_backend
