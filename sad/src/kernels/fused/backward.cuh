#pragma once

#include "diagonal.cuh"
#include "../ring_cnot.cuh"
#include "../rotation.cuh"

#include <algorithm>

namespace sad {

template <typename T, FusedDiagonalMode Mode>
__device__ __forceinline__ void accumulate_diagonal_gradients(
    const Complex<T> (&phi)[kRegisterAmplitudes],
    const Complex<T> (&lambda)[kRegisterAmplitudes],
    uint64_t tile_base,
    const int* selected,
    int target_mask,
    int phase,
    int tile_bits,
    int qubits,
    int rz_parameter_offset,
    int rzz_even_parameter_offset,
    int rzz_odd_parameter_offset,
    double* gradients,
    double* warp_partials,
    bool reverse_phases) {
    if constexpr (Mode == FusedDiagonalMode::NONE) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int component = 0;

    for (int slot = 0; slot < kTileBits; ++slot) {
        if ((target_mask & (1 << slot)) == 0) {
            continue;
        }
        const int qubit = selected[slot];
        if constexpr (Mode == FusedDiagonalMode::RZ ||
                      Mode == FusedDiagonalMode::RZ_RZZ) {
            double local_sum = 0.0;
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base |
                        scatter_local_assignment<kTileBits>(
                            local, selected, tile_bits);
                    local_sum +=
                        imag_conjugate_product(lambda[reg], phi[reg]) *
                        static_cast<double>(
                            diagonal_eigenvalue<DiagonalGate::RZ>(
                                index, qubit, 0));
                }
            }
            const double sum = warp_sum(local_sum);
            if (lane == 0) {
                warp_partials[component * kWarpsPerBlock + warp] = sum;
            }
            ++component;
        }
    }
    if constexpr (Mode == FusedDiagonalMode::RZZ ||
                  Mode == FusedDiagonalMode::RZ_RZZ) {
        for (int left = 0; left < qubits; ++left) {
            const int right = (left + 1) % qubits;
            const int left_phase = target_phase_for_qubit(
                left, kTileBits, kFixedLowLanes);
            const int right_phase = target_phase_for_qubit(
                right, kTileBits, kFixedLowLanes);
            const int owner_phase = reverse_phases
                                        ? min(left_phase, right_phase)
                                        : max(left_phase, right_phase);
            if (owner_phase != phase) {
                continue;
            }
            const bool even = (left & 1) == 0;
            const int edge = left / 2;
            double local_sum = 0.0;
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base |
                        scatter_local_assignment<kTileBits>(
                            local, selected, tile_bits);
                    const int eigenvalue =
                        even
                            ? diagonal_eigenvalue<DiagonalGate::RZZ_EVEN>(
                                  index, edge, qubits)
                            : diagonal_eigenvalue<DiagonalGate::RZZ_ODD>(
                                  index, edge, qubits);
                    local_sum +=
                        imag_conjugate_product(lambda[reg], phi[reg]) *
                        static_cast<double>(eigenvalue);
                }
            }
            const double sum = warp_sum(local_sum);
            if (lane == 0) {
                warp_partials[component * kWarpsPerBlock + warp] = sum;
            }
            ++component;
        }
    }
    __syncthreads();

    if (warp == 0) {
        component = 0;
        for (int slot = 0; slot < kTileBits; ++slot) {
            if ((target_mask & (1 << slot)) == 0) {
                continue;
            }
            const int qubit = selected[slot];
            if constexpr (Mode == FusedDiagonalMode::RZ ||
                          Mode == FusedDiagonalMode::RZ_RZZ) {
                double value =
                    lane < kWarpsPerBlock
                        ? warp_partials[component * kWarpsPerBlock + lane]
                        : 0.0;
                value = warp_sum(value);
                if (lane == 0) {
                    atomicAdd(gradients + rz_parameter_offset + qubit, value);
                }
                ++component;
            }
        }
        if constexpr (Mode == FusedDiagonalMode::RZZ ||
                      Mode == FusedDiagonalMode::RZ_RZZ) {
            for (int left = 0; left < qubits; ++left) {
                const int right = (left + 1) % qubits;
                const int left_phase = target_phase_for_qubit(
                    left, kTileBits, kFixedLowLanes);
                const int right_phase = target_phase_for_qubit(
                    right, kTileBits, kFixedLowLanes);
                const int owner_phase = reverse_phases
                                            ? min(left_phase, right_phase)
                                            : max(left_phase, right_phase);
                if (owner_phase != phase) {
                    continue;
                }
                double value =
                    lane < kWarpsPerBlock
                        ? warp_partials[component * kWarpsPerBlock + lane]
                        : 0.0;
                value = warp_sum(value);
                if (lane == 0) {
                    const int offset =
                        (left & 1) == 0 ? rzz_even_parameter_offset
                                        : rzz_odd_parameter_offset;
                    atomicAdd(gradients + offset + left / 2, value);
                }
                ++component;
            }
        }
    }
    __syncthreads();
}

template <typename T,
          NonDiagonalGate Gate,
          FusedDiagonalMode Mode,
          bool GatherCnot>
__global__ void fused_non_diagonal_backward_kernel(
    const Complex<T>* phi_input,
    const Complex<T>* lambda_input,
    Complex<T>* phi_output,
    Complex<T>* lambda_output,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int rotation_parameter_offset,
    int rz_parameter_offset,
    int rzz_even_parameter_offset,
    int rzz_odd_parameter_offset,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    bool reverse_phases,
    int single_phase_index,
    bool first_call) {
#if SAD_ROTATION_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    constexpr size_t kMailboxBytes =
        backward_rotation_mailbox_bytes<T, Gate>();
    __shared__ __align__(16)
        unsigned char mailbox[kMailboxBytes == 0 ? 16 : kMailboxBytes];
    __shared__ double reduction[kBlockThreads];
    __shared__ double diagonal_warp_partials[
        (2 * kTileBits + 1) * kWarpsPerBlock];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int step = 0; step < phase_count; ++step) {
        const int phase = reverse_phases ? step : phase_count - 1 - step;
        const int logical_phase =
            single_phase_index >= 0 ? single_phase_index : phase;
        const bool first_phase = first_call && step == 0;
        const int* selected = selected_maps + phase * kTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base =
                    scatter_tile_assignment<kTileBits>(
                        tile, qubits, selected, tile_bits);
            }
            __syncthreads();

            Complex<T> phi[kRegisterAmplitudes];
            Complex<T> lambda[kRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                const bool active = local < (1u << tile_bits);
                const uint64_t index =
                    tile_base |
                    scatter_local_assignment<kTileBits>(
                        local, selected, tile_bits);
                if (active) {
                    const uint64_t source =
                        first_phase && GatherCnot
                            ? apply_ring_cnot_forward_to_basis(index, qubits)
                            : index;
                    phi[reg] = first_phase ? phi_input[source]
                                           : phi_output[index];
                    lambda[reg] = first_phase ? lambda_input[source]
                                              : lambda_output[index];
                    if (first_phase) {
                        if constexpr (Mode != FusedDiagonalMode::NONE) {
                            const Complex<T> factor =
                                fused_diagonal_factor<T, Mode>(
                                    index,
                                    qubits,
                                    rz_lookup,
                                    rzz_even_lookup,
                                    rzz_odd_lookup);
                            const Complex<T> inverse_factor{
                                factor.real, -factor.imag};
                            phi[reg] = multiply(phi[reg], inverse_factor);
                            lambda[reg] =
                                multiply(lambda[reg], inverse_factor);
                        }
                    }
                } else {
                    phi[reg] = make_complex<T>(0, 0);
                    lambda[reg] = make_complex<T>(0, 0);
                }
            }

            accumulate_diagonal_gradients<T, Mode>(
                phi,
                lambda,
                tile_base,
                selected,
                target_masks[phase],
                logical_phase,
                tile_bits,
                qubits,
                rz_parameter_offset,
                rzz_even_parameter_offset,
                rzz_odd_parameter_offset,
                gradients,
                diagonal_warp_partials,
                reverse_phases);

            const bool thread_active =
                (lane | (warp << (kLaneBits + kRegisterBits))) <
                (1u << tile_bits);
            apply_phase_backward<T, Gate>(phi,
                                          lambda,
                                          coefficients,
                                          gradients,
                                          rotation_parameter_offset,
                                          selected,
                                          target_masks[phase],
                                          thread_active,
                                          mailbox,
                                          reduction);

#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base |
                        scatter_local_assignment<kTileBits>(
                            local, selected, tile_bits);
                    phi_output[index] = phi[reg];
                    lambda_output[index] = lambda[reg];
                }
            }
            __syncthreads();
        }
        if (step + 1 < phase_count) {
#if SAD_ROTATION_PERSISTENT
            grid.sync();
#endif
        }
    }
}
template <typename T,
          NonDiagonalGate Gate,
          FusedDiagonalMode Mode,
          bool GatherCnot>
void launch_fused_non_diagonal_backward(
    StatePair<T>* phi,
    StatePair<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int rotation_parameter_offset,
    int rz_parameter_offset,
    int rzz_even_parameter_offset,
    int rzz_odd_parameter_offset,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    int multiprocessors,
    bool reverse_phases = false) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel =
        fused_non_diagonal_backward_kernel<T, Gate, Mode, GatherCnot>;
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, kernel, kBlockThreads, 0));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
    const Complex<T>* phi_input = phi->current;
    const Complex<T>* lambda_input = lambda->current;
    Complex<T>* phi_output = phi->scratch;
    Complex<T>* lambda_output = lambda->scratch;
    if constexpr (kRotationPersistent) {
        int single_phase_index = -1;
        bool first_call = true;
        void* arguments[] = {
            const_cast<Complex<T>**>(&phi_input),
            const_cast<Complex<T>**>(&lambda_input),
            &phi_output,
            &lambda_output,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &gradients,
            &qubits,
            &rotation_parameter_offset,
            &rz_parameter_offset,
            &rzz_even_parameter_offset,
            &rzz_odd_parameter_offset,
            const_cast<Complex<T>**>(&rz_lookup),
            const_cast<Complex<T>**>(&rzz_even_lookup),
            const_cast<Complex<T>**>(&rzz_odd_lookup),
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &reverse_phases,
            &single_phase_index,
            &first_call};
        SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(grid_size),
            dim3(kBlockThreads),
            arguments));
    } else {
        const int ordinary_grid_size = static_cast<int>(tile_count);
        for (int step = 0; step < phase_count; ++step) {
            const int phase = reverse_phases ? step : phase_count - 1 - step;
            const bool first_call = step == 0;
            kernel<<<ordinary_grid_size, kBlockThreads>>>(
                phi_input,
                lambda_input,
                phi_output,
                lambda_output,
                coefficients,
                gradients,
                qubits,
                rotation_parameter_offset,
                rz_parameter_offset,
                rzz_even_parameter_offset,
                rzz_odd_parameter_offset,
                rz_lookup,
                rzz_even_lookup,
                rzz_odd_lookup,
                selected_maps + phase * kTileBits,
                target_masks + phase,
                1,
                reverse_phases,
                phase,
                first_call);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
    phi->swap();
    lambda->swap();
}

}  // namespace sad
