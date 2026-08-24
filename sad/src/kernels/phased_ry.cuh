#pragma once

#include "ring_cnot.cuh"
#include "rotation_primitives.cuh"

#include <algorithm>

namespace sad {

template <typename T>
__device__ __forceinline__ Complex<T> rotate_phased_ry(
    Complex<T> self,
    Complex<T> partner,
    int bit,
    RotationCoefficients<T> ry,
    RotationCoefficients<T> rz) {
    const Complex<T> rotated = rotate_amplitude<T, NonDiagonalGate::RY>(
        self, partner, bit, ry);
    const Complex<T> phase{rz.cosine, bit ? rz.sine : -rz.sine};
    return multiply(rotated, phase);
}

template <typename T, int Slot>
__device__ __forceinline__ void apply_phased_ry_slot(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    RotationCoefficients<T> ry,
    RotationCoefficients<T> rz,
    Complex<T>* mailbox) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (Slot < kLaneBits) {
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            const Complex<T> partner =
                shuffle_xor_complex(values[reg], 1 << Slot);
            values[reg] = rotate_phased_ry(
                values[reg], partner, (lane >> Slot) & 1, ry, rz);
        }
    } else if constexpr (Slot < kLaneBits + kForwardRegisterBits) {
        constexpr int mask = 1 << (Slot - kLaneBits);
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            if ((reg & mask) == 0) {
                const Complex<T> zero = values[reg];
                const Complex<T> one = values[reg | mask];
                values[reg] = rotate_phased_ry(zero, one, 0, ry, rz);
                values[reg | mask] =
                    rotate_phased_ry(one, zero, 1, ry, rz);
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
            values[reg] = rotate_phased_ry(
                values[reg],
                mailbox[partner_tid * kForwardRegisterAmplitudes + reg],
                (warp >> warp_bit) & 1,
                ry,
                rz);
        }
        __syncthreads();
    }
}

template <typename T, int Slot = 0>
__device__ __forceinline__ void apply_phased_ry_phase(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    int ry_offset,
    int rz_offset,
    const int* selected,
    int target_mask,
    Complex<T>* mailbox) {
    if constexpr (Slot < kForwardTileBits) {
        if (target_mask & (1 << Slot)) {
            const int qubit = selected[Slot];
            apply_phased_ry_slot<T, Slot>(values,
                                          coefficients[ry_offset + qubit],
                                          coefficients[rz_offset + qubit],
                                          mailbox);
        }
        apply_phased_ry_phase<T, Slot + 1>(values,
                                            coefficients,
                                            ry_offset,
                                            rz_offset,
                                            selected,
                                            target_mask,
                                            mailbox);
    }
}

template <typename T>
__global__ void phased_ry_cnot_forward_kernel(
    Complex<T>* state,
    Complex<T>* output,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int ry_offset,
    int rz_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    bool final_call) {
#if SAD_PHASED_RY_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* mailbox = reinterpret_cast<Complex<T>*>(dynamic_shared);
    __shared__ uint64_t tile_base;
    const int tile_bits = min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int phase = 0; phase < phase_count; ++phase) {
        const bool final_phase = final_call && phase + 1 == phase_count;
        const int* selected = selected_maps + phase * kForwardTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment<kForwardTileBits>(
                    tile, qubits, selected, tile_bits);
            }
            __syncthreads();
            Complex<T> values[kForwardRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                const uint64_t index =
                    tile_base | scatter_local_assignment<kForwardTileBits>(
                                    local, selected, tile_bits);
                values[reg] = local < (1u << tile_bits)
                                  ? state[index]
                                  : make_complex<T>(0, 0);
            }
            apply_phased_ry_phase(values,
                                  coefficients,
                                  ry_offset,
                                  rz_offset,
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
#if SAD_PHASED_RY_PERSISTENT
            grid.sync();
#endif
        }
    }
}

template <typename T>
void launch_phased_ry_cnot_forward(
    StatePair<T>* phi,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int ry_offset,
    int rz_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count,
    int multiprocessors) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = phased_ry_cnot_forward_kernel<T>;
    constexpr size_t shared_bytes =
        kForwardTileAmplitudes * sizeof(Complex<T>);
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
    Complex<T>* state = phi->current;
    Complex<T>* output = phi->scratch;
    if constexpr (kPhasedRyPersistent) {
        bool final_call = true;
        void* arguments[] = {
            &state,
            &output,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &qubits,
            &ry_offset,
            &rz_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &final_call};
        SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(grid_size),
            dim3(kForwardBlockThreads),
            arguments,
            shared_bytes));
    } else {
        const int ordinary_grid_size = static_cast<int>(tile_count);
        for (int phase = 0; phase < phase_count; ++phase) {
            const bool final_call = phase + 1 == phase_count;
            kernel<<<ordinary_grid_size,
                     kForwardBlockThreads,
                     shared_bytes>>>(state,
                                     output,
                                     coefficients,
                                     qubits,
                                     ry_offset,
                                     rz_offset,
                                     selected_maps +
                                         phase * kForwardTileBits,
                                     target_masks + phase,
                                     1,
                                     final_call);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
    phi->swap();
}

template <typename T, int Slot>
__device__ __forceinline__ void apply_phased_ry_backward_slot(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int ry_offset,
    int rz_offset,
    uint64_t tile_base,
    const int* selected,
    int tile_bits,
    bool active,
    void* mailbox,
    double* reduction) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int qubit = selected[Slot];
    double rz_overlap = 0.0;
#pragma unroll
    for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
        const uint32_t local = static_cast<uint32_t>(
            lane | (reg << kLaneBits) |
            (warp << (kLaneBits + kRegisterBits)));
        if (local < (1u << tile_bits)) {
            const uint64_t index =
                tile_base | scatter_local_assignment<kTileBits>(
                                local, selected, tile_bits);
            const double eigenvalue = ((index >> qubit) & 1ull) ? -1.0 : 1.0;
            rz_overlap += imag_conjugate_product(lambda[reg], phi[reg]) *
                          eigenvalue;
        }
    }
    block_atomic_sum(rz_overlap, reduction, gradients + rz_offset + qubit);

    const RotationCoefficients<T> rz = coefficients[rz_offset + qubit];
#pragma unroll
    for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
        const uint32_t local = static_cast<uint32_t>(
            lane | (reg << kLaneBits) |
            (warp << (kLaneBits + kRegisterBits)));
        if (local < (1u << tile_bits)) {
            const uint64_t index =
                tile_base | scatter_local_assignment<kTileBits>(
                                local, selected, tile_bits);
            const Complex<T> inverse_phase{
                rz.cosine, ((index >> qubit) & 1ull) ? -rz.sine : rz.sine};
            phi[reg] = multiply(phi[reg], inverse_phase);
            lambda[reg] = multiply(lambda[reg], inverse_phase);
        }
    }

    apply_tile_gate_backward<T, NonDiagonalGate::RY, Slot>(
        phi,
        lambda,
        coefficients[ry_offset + qubit],
        active,
        mailbox,
        reduction,
        gradients + ry_offset + qubit);
}

template <typename T, int Slot = kTileBits - 1>
__device__ __forceinline__ void apply_phased_ry_backward_phase(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int ry_offset,
    int rz_offset,
    uint64_t tile_base,
    const int* selected,
    int target_mask,
    int tile_bits,
    bool active,
    void* mailbox,
    double* reduction) {
    if constexpr (Slot >= 0) {
        if (target_mask & (1 << Slot)) {
            apply_phased_ry_backward_slot<T, Slot>(phi,
                                                   lambda,
                                                   coefficients,
                                                   gradients,
                                                   ry_offset,
                                                   rz_offset,
                                                   tile_base,
                                                   selected,
                                                   tile_bits,
                                                   active,
                                                   mailbox,
                                                   reduction);
        }
        apply_phased_ry_backward_phase<T, Slot - 1>(phi,
                                                    lambda,
                                                    coefficients,
                                                    gradients,
                                                    ry_offset,
                                                    rz_offset,
                                                    tile_base,
                                                    selected,
                                                    target_mask,
                                                    tile_bits,
                                                    active,
                                                    mailbox,
                                                    reduction);
    }
}

template <typename T>
__global__ void phased_ry_cnot_backward_kernel(
    const Complex<T>* phi_input,
    const Complex<T>* lambda_input,
    Complex<T>* phi_output,
    Complex<T>* lambda_output,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int ry_offset,
    int rz_offset,
    const int* selected,
    const int* target_mask,
    bool first_call) {
    constexpr size_t kMailboxBytes =
        backward_rotation_mailbox_bytes<T, NonDiagonalGate::RY>();
    __shared__ __align__(16)
        unsigned char mailbox[kMailboxBytes == 0 ? 16 : kMailboxBytes];
    __shared__ double reduction[kBlockThreads];
    __shared__ uint64_t tile_base;
    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (uint64_t tile = blockIdx.x; tile < tile_count;
         tile += gridDim.x) {
        if (tid == 0) {
            tile_base = scatter_tile_assignment<kTileBits>(
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
            const bool amplitude_active = local < (1u << tile_bits);
            const uint64_t index =
                tile_base | scatter_local_assignment<kTileBits>(
                                local, selected, tile_bits);
            if (amplitude_active) {
                const uint64_t source =
                    first_call
                        ? apply_ring_cnot_forward_to_basis(index, qubits)
                        : index;
                phi[reg] = first_call ? phi_input[source] : phi_output[index];
                lambda[reg] =
                    first_call ? lambda_input[source] : lambda_output[index];
            } else {
                phi[reg] = make_complex<T>(0, 0);
                lambda[reg] = make_complex<T>(0, 0);
            }
        }
        const bool thread_active =
            (lane | (warp << (kLaneBits + kRegisterBits))) <
            (1u << tile_bits);
        apply_phased_ry_backward_phase(phi,
                                       lambda,
                                       coefficients,
                                       gradients,
                                       ry_offset,
                                       rz_offset,
                                       tile_base,
                                       selected,
                                       target_mask[0],
                                       tile_bits,
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
                    tile_base | scatter_local_assignment<kTileBits>(
                                    local, selected, tile_bits);
                phi_output[index] = phi[reg];
                lambda_output[index] = lambda[reg];
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_phased_ry_cnot_backward(
    StatePair<T>* phi,
    StatePair<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int ry_offset,
    int rz_offset,
    const int* selected_maps,
    const int* target_masks,
    int phase_count) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const Complex<T>* phi_input = phi->current;
    const Complex<T>* lambda_input = lambda->current;
    Complex<T>* phi_output = phi->scratch;
    Complex<T>* lambda_output = lambda->scratch;
    for (int step = 0; step < phase_count; ++step) {
        const int phase = phase_count - 1 - step;
        phased_ry_cnot_backward_kernel<T>
            <<<static_cast<int>(tile_count), kBlockThreads>>>(
                phi_input,
                lambda_input,
                phi_output,
                lambda_output,
                coefficients,
                gradients,
                qubits,
                ry_offset,
                rz_offset,
                selected_maps + phase * kTileBits,
                target_masks + phase,
                step == 0);
    }
    SAD_CUDA_CHECK(cudaGetLastError());
    phi->swap();
    lambda->swap();
}

}  // namespace sad
