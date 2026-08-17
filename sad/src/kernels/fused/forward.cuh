#pragma once

#include "diagonal.cuh"
#include "../ring_cnot.cuh"
#include "../rotation.cuh"

#include <algorithm>

namespace sad {

template <typename T,
          NonDiagonalGate Gate,
          FusedDiagonalMode Mode,
          bool ScatterCnot,
          bool DiagonalBefore = false,
          bool SharedParameter = false,
          bool SharedDiagonal = false>
__global__ void fused_non_diagonal_forward_kernel(
    Complex<T>* state,
    Complex<T>* output,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int parameter_offset,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    bool reverse_phases,
    bool first_call,
    bool final_call) {
#if SAD_ROTATION_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    void* mailbox = dynamic_shared;
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int step = 0; step < phase_count; ++step) {
        const int phase = reverse_phases ? phase_count - 1 - step : step;
        const bool first_phase = first_call && step == 0;
        const bool final_phase = final_call && step + 1 == phase_count;
        const int* selected = selected_maps + phase * kForwardTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base =
                    scatter_tile_assignment<kForwardTileBits>(
                        tile, qubits, selected, tile_bits);
            }
            __syncthreads();

            Complex<T> values[kForwardRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                const bool active = local < (1u << tile_bits);
                const uint64_t index =
                    tile_base |
                    scatter_local_assignment<kForwardTileBits>(
                        local, selected, tile_bits);
                values[reg] =
                    active ? state[index] : make_complex<T>(0, 0);
                if constexpr (DiagonalBefore &&
                              Mode != FusedDiagonalMode::NONE) {
                    if (active && first_phase) {
                        Complex<T> factor;
                        if constexpr (SharedDiagonal) {
                            static_assert(Mode == FusedDiagonalMode::RZZ);
                            factor = shared_ring_rzz_factor(
                                index, rzz_even_lookup, qubits);
                        } else {
                            factor = fused_diagonal_factor<T, Mode>(
                                index,
                                qubits,
                                rz_lookup,
                                rzz_even_lookup,
                                rzz_odd_lookup);
                        }
                        values[reg] = multiply(values[reg], factor);
                    }
                }
            }

            apply_phase_forward<T, Gate, SharedParameter>(
                values,
                coefficients,
                parameter_offset,
                selected,
                target_masks[phase],
                mailbox);

#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base |
                        scatter_local_assignment<kForwardTileBits>(
                            local, selected, tile_bits);
                    Complex<T> value = values[reg];
                    if (final_phase) {
                        if constexpr (!DiagonalBefore &&
                                      Mode != FusedDiagonalMode::NONE) {
                            Complex<T> factor;
                            if constexpr (SharedDiagonal) {
                                static_assert(Mode == FusedDiagonalMode::RZZ);
                                factor = shared_ring_rzz_factor(
                                    index, rzz_even_lookup, qubits);
                            } else {
                                factor = fused_diagonal_factor<T, Mode>(
                                    index,
                                    qubits,
                                    rz_lookup,
                                    rzz_even_lookup,
                                    rzz_odd_lookup);
                            }
                            value = multiply(value, factor);
                        }
                        const uint64_t output_index =
                            ScatterCnot
                                ? apply_ring_cnot_forward_to_basis(index,
                                                                   qubits)
                                : index;
                        output[output_index] = value;
                    } else {
                        state[index] = value;
                    }
                }
            }
            __syncthreads();
        }
        if (!final_phase) {
#if SAD_ROTATION_PERSISTENT
            grid.sync();
#endif
        }
    }
}
template <typename T,
          NonDiagonalGate Gate,
          FusedDiagonalMode Mode,
          bool ScatterCnot,
          bool DiagonalBefore = false,
          bool SharedParameter = false,
          bool SharedDiagonal = false>
void launch_fused_non_diagonal_forward(
    StatePair<T>* phi,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int parameter_offset,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    int multiprocessors,
    bool reverse_phases = false) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel =
        fused_non_diagonal_forward_kernel<T,
                                          Gate,
                                          Mode,
                                          ScatterCnot,
                                          DiagonalBefore,
                                          SharedParameter,
                                          SharedDiagonal>;
    constexpr size_t shared_bytes =
        forward_rotation_mailbox_bytes<T, Gate>();
    if constexpr (shared_bytes > 0) {
        SAD_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor,
        kernel,
        kForwardBlockThreads,
        shared_bytes));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
    Complex<T>* state = phi->current;
    Complex<T>* output = ScatterCnot ? phi->scratch : phi->current;
    if constexpr (kRotationPersistent) {
        bool first_call = true;
        bool final_call = true;
        void* arguments[] = {
            &state,
            &output,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &qubits,
            &parameter_offset,
            const_cast<Complex<T>**>(&rz_lookup),
            const_cast<Complex<T>**>(&rzz_even_lookup),
            const_cast<Complex<T>**>(&rzz_odd_lookup),
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &reverse_phases,
            &first_call,
            &final_call};
        SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(grid_size),
            dim3(kForwardBlockThreads),
            arguments,
            shared_bytes));
    } else {
        const int ordinary_grid_size = static_cast<int>(tile_count);
        for (int step = 0; step < phase_count; ++step) {
            const int phase = reverse_phases ? phase_count - 1 - step : step;
            const bool first_call = step == 0;
            const bool final_call = step + 1 == phase_count;
            kernel<<<ordinary_grid_size,
                     kForwardBlockThreads,
                     shared_bytes>>>(state,
                                     output,
                                     coefficients,
                                     qubits,
                                     parameter_offset,
                                     rz_lookup,
                                     rzz_even_lookup,
                                     rzz_odd_lookup,
                                     selected_maps +
                                         phase * kForwardTileBits,
                                     target_masks + phase,
                                     1,
                                     false,
                                     first_call,
                                     final_call);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
    if constexpr (ScatterCnot) {
        phi->swap();
    }
}

}  // namespace sad
