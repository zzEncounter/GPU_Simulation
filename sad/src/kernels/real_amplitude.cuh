#pragma once

#include "hamiltonian.cuh"
#include "ring_cnot.cuh"
#include "rotation_primitives.cuh"

#include <algorithm>

namespace sad {

template <typename T>
__device__ __forceinline__ T rotate_real(
    T self, T partner, int bit, RotationCoefficients<T> coefficients) {
    return coefficients.cosine * self +
           (bit ? coefficients.sine : -coefficients.sine) * partner;
}

template <typename T, int Slot>
__device__ __forceinline__ void apply_real_forward_slot(
    T (&values)[kForwardRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    T* mailbox) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (Slot < kLaneBits) {
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            const T partner =
                __shfl_xor_sync(0xffffffffu, values[reg], 1 << Slot);
            values[reg] = rotate_real(
                values[reg], partner, (lane >> Slot) & 1, coefficients);
        }
    } else if constexpr (Slot < kLaneBits + kForwardRegisterBits) {
        constexpr int mask = 1 << (Slot - kLaneBits);
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            if ((reg & mask) == 0) {
                const T zero = values[reg];
                const T one = values[reg | mask];
                values[reg] = rotate_real(zero, one, 0, coefficients);
                values[reg | mask] =
                    rotate_real(one, zero, 1, coefficients);
            }
        }
    } else {
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            mailbox[tid * kForwardRegisterAmplitudes + reg] = values[reg];
        }
        __syncthreads();
        constexpr int warp_bit =
            Slot - kLaneBits - kForwardRegisterBits;
        const int partner_tid = tid ^ ((1 << warp_bit) << kLaneBits);
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            values[reg] = rotate_real(
                values[reg],
                mailbox[partner_tid * kForwardRegisterAmplitudes + reg],
                (warp >> warp_bit) & 1,
                coefficients);
        }
        __syncthreads();
    }
}

template <typename T, int Slot = 0>
__device__ __forceinline__ void apply_real_forward_phase(
    T (&values)[kForwardRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    int parameter_offset,
    const int* selected,
    int target_mask,
    T* mailbox) {
    if constexpr (Slot < kForwardTileBits) {
        if (target_mask & (1 << Slot)) {
            apply_real_forward_slot<T, Slot>(
                values,
                coefficients[parameter_offset + selected[Slot]],
                mailbox);
        }
        apply_real_forward_phase<T, Slot + 1>(values,
                                               coefficients,
                                               parameter_offset,
                                               selected,
                                               target_mask,
                                               mailbox);
    }
}

template <typename T, int Slot>
__device__ __forceinline__ void apply_real_backward_slot(
    T (&phi)[kRegisterAmplitudes],
    T (&lambda)[kRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    bool active,
    T* mailbox,
    double* reduction,
    double* gradient) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    double local_overlap = 0.0;
    if constexpr (Slot < kLaneBits) {
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const T partner =
                __shfl_xor_sync(0xffffffffu, phi[reg], 1 << Slot);
            if (active) {
                local_overlap += static_cast<double>(lambda[reg]) *
                                 static_cast<double>(partner) *
                                 (((lane >> Slot) & 1) ? 1.0 : -1.0);
            }
        }
        block_atomic_sum(local_overlap, reduction, gradient);
        const RotationCoefficients<T> inverse{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const T phi_partner =
                __shfl_xor_sync(0xffffffffu, phi[reg], 1 << Slot);
            const T lambda_partner =
                __shfl_xor_sync(0xffffffffu, lambda[reg], 1 << Slot);
            phi[reg] = rotate_real(
                phi[reg], phi_partner, (lane >> Slot) & 1, inverse);
            lambda[reg] = rotate_real(
                lambda[reg], lambda_partner, (lane >> Slot) & 1, inverse);
        }
    } else if constexpr (Slot < kLaneBits + kRegisterBits) {
        constexpr int mask = 1 << (Slot - kLaneBits);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            if (active) {
                local_overlap += static_cast<double>(lambda[reg]) *
                                 static_cast<double>(phi[reg ^ mask]) *
                                 ((reg & mask) ? 1.0 : -1.0);
            }
        }
        block_atomic_sum(local_overlap, reduction, gradient);
        const RotationCoefficients<T> inverse{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            if ((reg & mask) == 0) {
                const T phi_zero = phi[reg];
                const T phi_one = phi[reg | mask];
                const T lambda_zero = lambda[reg];
                const T lambda_one = lambda[reg | mask];
                phi[reg] = rotate_real(phi_zero, phi_one, 0, inverse);
                phi[reg | mask] = rotate_real(phi_one, phi_zero, 1, inverse);
                lambda[reg] =
                    rotate_real(lambda_zero, lambda_one, 0, inverse);
                lambda[reg | mask] =
                    rotate_real(lambda_one, lambda_zero, 1, inverse);
            }
        }
    } else {
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            mailbox[tid * kRegisterAmplitudes + reg] = phi[reg];
        }
        __syncthreads();
        constexpr int warp_bit = Slot - kLaneBits - kRegisterBits;
        const int partner_tid = tid ^ ((1 << warp_bit) << kLaneBits);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const T partner =
                mailbox[partner_tid * kRegisterAmplitudes + reg];
            if (active) {
                local_overlap += static_cast<double>(lambda[reg]) *
                                 static_cast<double>(partner) *
                                 (((warp >> warp_bit) & 1) ? 1.0 : -1.0);
            }
        }
        const RotationCoefficients<T> inverse{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            phi[reg] = rotate_real(
                phi[reg],
                mailbox[partner_tid * kRegisterAmplitudes + reg],
                (warp >> warp_bit) & 1,
                inverse);
        }
        block_atomic_sum(local_overlap, reduction, gradient);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            mailbox[tid * kRegisterAmplitudes + reg] = lambda[reg];
        }
        __syncthreads();
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            lambda[reg] = rotate_real(
                lambda[reg],
                mailbox[partner_tid * kRegisterAmplitudes + reg],
                (warp >> warp_bit) & 1,
                inverse);
        }
        __syncthreads();
    }
}

template <typename T, int Slot = kTileBits - 1>
__device__ __forceinline__ void apply_real_backward_phase(
    T (&phi)[kRegisterAmplitudes],
    T (&lambda)[kRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int parameter_offset,
    const int* selected,
    int target_mask,
    bool active,
    T* mailbox,
    double* reduction) {
    if constexpr (Slot >= 0) {
        if (target_mask & (1 << Slot)) {
            apply_real_backward_slot<T, Slot>(
                phi,
                lambda,
                coefficients[parameter_offset + selected[Slot]],
                active,
                mailbox,
                reduction,
                gradients + parameter_offset + selected[Slot]);
        }
        apply_real_backward_phase<T, Slot - 1>(phi,
                                                lambda,
                                                coefficients,
                                                gradients,
                                                parameter_offset,
                                                selected,
                                                target_mask,
                                                active,
                                                mailbox,
                                                reduction);
    }
}

template <typename T>
__global__ void real_product_state_kernel(
    T* output,
    uint64_t state_size,
    int qubits,
    const Complex<T>* product_lookup) {
    const int chunk_count =
        (qubits + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        T value = static_cast<T>(1);
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const unsigned code = static_cast<unsigned>(
                (index >> (chunk * kDiagonalLookupBits)) &
                (kDiagonalLookupSize - 1));
            value *= product_lookup[chunk * kDiagonalLookupSize + code].real;
        }
        output[apply_ring_cnot_forward_to_basis(index, qubits)] = value;
    }
}

template <typename T>
__global__ void real_fused_forward_kernel(
    T* state,
    T* output,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int parameter_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    bool reverse_phases,
    bool final_call) {
#if SAD_REAL_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* mailbox = reinterpret_cast<T*>(dynamic_shared);
    __shared__ uint64_t tile_base;
    const int tile_bits = min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int step = 0; step < phase_count; ++step) {
        const int phase = reverse_phases ? phase_count - 1 - step : step;
        const bool final_phase = final_call && step + 1 == phase_count;
        const int* selected = selected_maps + phase * kForwardTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment<kForwardTileBits>(
                    tile, qubits, selected, tile_bits);
            }
            __syncthreads();
            T values[kForwardRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                const uint64_t index =
                    tile_base | scatter_local_assignment<kForwardTileBits>(
                                    local, selected, tile_bits);
                values[reg] = local < (1u << tile_bits) ? state[index]
                                                        : static_cast<T>(0);
            }
            apply_real_forward_phase(values,
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
                        tile_base | scatter_local_assignment<kForwardTileBits>(
                                        local, selected, tile_bits);
                    if (final_phase) {
                        output[apply_ring_cnot_forward_to_basis(index, qubits)] =
                            values[reg];
                    } else {
                        state[index] = values[reg];
                    }
                }
            }
            __syncthreads();
        }
        if (!final_phase) {
#if SAD_REAL_PERSISTENT
            grid.sync();
#endif
        }
    }
}

template <typename T>
__global__ void real_fused_backward_kernel(
    const T* phi_input,
    const T* lambda_input,
    T* phi_output,
    T* lambda_output,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int parameter_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    bool reverse_phases,
    bool first_call) {
#if SAD_REAL_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    __shared__ T mailbox[kTileAmplitudes];
    __shared__ double reduction[kBlockThreads];
    __shared__ uint64_t tile_base;
    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int step = 0; step < phase_count; ++step) {
        const int phase = reverse_phases ? step : phase_count - 1 - step;
        const bool first_phase = first_call && step == 0;
        const int* selected = selected_maps + phase * kTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment<kTileBits>(
                    tile, qubits, selected, tile_bits);
            }
            __syncthreads();
            T phi[kRegisterAmplitudes];
            T lambda[kRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base | scatter_local_assignment<kTileBits>(
                                        local, selected, tile_bits);
                    const uint64_t source =
                        first_phase
                            ? apply_ring_cnot_forward_to_basis(index, qubits)
                            : index;
                    phi[reg] = first_phase ? phi_input[source]
                                           : phi_output[index];
                    lambda[reg] = first_phase ? lambda_input[source]
                                              : lambda_output[index];
                } else {
                    phi[reg] = static_cast<T>(0);
                    lambda[reg] = static_cast<T>(0);
                }
            }
            const bool active =
                (lane | (warp << (kLaneBits + kRegisterBits))) <
                (1u << tile_bits);
            apply_real_backward_phase(phi,
                                      lambda,
                                      coefficients,
                                      gradients,
                                      parameter_offset,
                                      selected,
                                      target_masks[phase],
                                      active,
                                      mailbox,
                                      reduction);
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base | scatter_local_assignment<kTileBits>(
                                        local, selected, tile_bits);
                    phi_output[index] = phi[reg];
                    lambda_output[index] = lambda[reg];
                }
            }
            __syncthreads();
        }
        if (step + 1 < phase_count) {
#if SAD_REAL_PERSISTENT
            grid.sync();
#endif
        }
    }
}

template <typename T>
__global__ void real_hamiltonian_kernel(const T* phi,
                                        T* lambda,
                                        uint64_t state_size,
                                        int qubits,
                                        double* energy) {
    __shared__ double reduction[kOrdinaryBlockThreads];
    double local_energy = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int zz_sum = 0;
        for (int qubit = 0; qubit < qubits; ++qubit) {
            const int next = (qubit + 1) % qubits;
            zz_sum += ((index >> qubit) & 1ull) ==
                              ((index >> next) & 1ull)
                          ? 1
                          : -1;
        }
        const T amplitude = phi[index];
        T h_amplitude = -static_cast<T>(zz_sum) * amplitude;
        for (int qubit = 0; qubit < qubits; ++qubit) {
            h_amplitude -= phi[index ^ (1ull << qubit)];
        }
        lambda[index] = h_amplitude;
        local_energy += static_cast<double>(amplitude) *
                        static_cast<double>(h_amplitude);
    }
    block_atomic_sum(local_energy, reduction, energy);
}

template <typename T>
void launch_real_initial(StatePair<T>* phi,
                         uint64_t state_size,
                         int qubits,
                         const Complex<T>* product_lookup,
                         int grid_size) {
    real_product_state_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        reinterpret_cast<T*>(phi->current),
        state_size,
        qubits,
        product_lookup);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_real_fused_forward(
    StatePair<T>* phi,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int parameter_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    int multiprocessors,
    bool reverse_phases) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = real_fused_forward_kernel<T>;
    constexpr size_t shared_bytes = kForwardTileAmplitudes * sizeof(T);
    SAD_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor,
        kernel,
        kForwardBlockThreads,
        shared_bytes));
    const int grid_size = static_cast<int>(std::min<uint64_t>(
        tile_count,
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors));
    T* state = reinterpret_cast<T*>(phi->current);
    T* output = reinterpret_cast<T*>(phi->scratch);
    if constexpr (kRealPersistent) {
        bool final_call = true;
        void* arguments[] = {
            &state,
            &output,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &qubits,
            &parameter_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &reverse_phases,
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
            const bool final_call = step + 1 == phase_count;
            kernel<<<ordinary_grid_size,
                     kForwardBlockThreads,
                     shared_bytes>>>(state,
                                     output,
                                     coefficients,
                                     qubits,
                                     parameter_offset,
                                     selected_maps +
                                         phase * kForwardTileBits,
                                     target_masks + phase,
                                     1,
                                     false,
                                     final_call);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
    phi->swap();
}

template <typename T>
void launch_real_fused_backward(
    StatePair<T>* phi,
    StatePair<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int parameter_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    int multiprocessors,
    bool reverse_phases) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = real_fused_backward_kernel<T>;
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, kernel, kBlockThreads, 0));
    const int grid_size = static_cast<int>(std::min<uint64_t>(
        tile_count,
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors));
    const T* phi_input = reinterpret_cast<T*>(phi->current);
    const T* lambda_input = reinterpret_cast<T*>(lambda->current);
    T* phi_output = reinterpret_cast<T*>(phi->scratch);
    T* lambda_output = reinterpret_cast<T*>(lambda->scratch);
    if constexpr (kRealPersistent) {
        bool first_call = true;
        void* arguments[] = {
            const_cast<T**>(&phi_input),
            const_cast<T**>(&lambda_input),
            &phi_output,
            &lambda_output,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &gradients,
            &qubits,
            &parameter_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &reverse_phases,
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
                parameter_offset,
                selected_maps + phase * kTileBits,
                target_masks + phase,
                1,
                false,
                first_call);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
    phi->swap();
    lambda->swap();
}

}  // namespace sad
