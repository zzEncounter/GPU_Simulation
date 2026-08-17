#pragma once

#include "rotation_primitives.cuh"

#include <algorithm>

namespace sad {

__device__ __forceinline__ int mera_active_wire(int layer,
                                                int index,
                                                int qubits) {
    return min(((index + 1) << layer) - 1, qubits - 1);
}

__device__ __forceinline__ bool mera_wire_is_selected(int wire,
                                                       const int* selected,
                                                       int selected_count) {
    for (int slot = 0; slot < selected_count; ++slot) {
        if (selected[slot] == wire) return true;
    }
    return false;
}

__device__ __forceinline__ void mera_build_selected_map(
    int* selected,
    int tile_bits,
    int qubits,
    int layer,
    int phase_pair_offset,
    int phase_pair_count,
    bool coarse_graining) {
    for (int local_pair = 0; local_pair < phase_pair_count; ++local_pair) {
        const int pair = phase_pair_offset + local_pair;
        const int first_active = 2 * pair + (coarse_graining ? 0 : 1);
        selected[2 * local_pair] =
            mera_active_wire(layer, first_active, qubits);
        selected[2 * local_pair + 1] =
            mera_active_wire(layer, first_active + 1, qubits);
    }

    int selected_count = 2 * phase_pair_count;
    for (int wire = 0; wire < qubits && selected_count < tile_bits; ++wire) {
        if (!mera_wire_is_selected(wire, selected, selected_count)) {
            selected[selected_count++] = wire;
        }
    }
}

template <typename T, int Slot>
__device__ __forceinline__ void mera_apply_tile_ry_forward_slot(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    Complex<T>* mailbox) {
    if constexpr (Slot < kForwardTileBits) {
        apply_tile_gate_forward<T, NonDiagonalGate::RY, Slot>(
            values, coefficients, mailbox);
    }
}

template <typename T>
__device__ __forceinline__ void mera_apply_tile_ry_forward(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    int slot,
    RotationCoefficients<T> coefficients,
    Complex<T>* mailbox) {
    switch (slot) {
        case 0:
            mera_apply_tile_ry_forward_slot<T, 0>(
                values, coefficients, mailbox);
            break;
        case 1:
            mera_apply_tile_ry_forward_slot<T, 1>(
                values, coefficients, mailbox);
            break;
        case 2:
            mera_apply_tile_ry_forward_slot<T, 2>(
                values, coefficients, mailbox);
            break;
        case 3:
            mera_apply_tile_ry_forward_slot<T, 3>(
                values, coefficients, mailbox);
            break;
        case 4:
            mera_apply_tile_ry_forward_slot<T, 4>(
                values, coefficients, mailbox);
            break;
        case 5:
            mera_apply_tile_ry_forward_slot<T, 5>(
                values, coefficients, mailbox);
            break;
        case 6:
            mera_apply_tile_ry_forward_slot<T, 6>(
                values, coefficients, mailbox);
            break;
        case 7:
            mera_apply_tile_ry_forward_slot<T, 7>(
                values, coefficients, mailbox);
            break;
        default:
            break;
    }
    if constexpr (kForwardTileBits > 8) {
        if (slot == 8) {
            apply_tile_gate_forward<T, NonDiagonalGate::RY, 8>(
                values, coefficients, mailbox);
        }
    }
    if constexpr (kForwardTileBits > 9) {
        if (slot == 9) {
            apply_tile_gate_forward<T, NonDiagonalGate::RY, 9>(
                values, coefficients, mailbox);
        }
    }
    if constexpr (kForwardTileBits > 10) {
        if (slot == 10) {
            apply_tile_gate_forward<T, NonDiagonalGate::RY, 10>(
                values, coefficients, mailbox);
        }
    }
    if constexpr (kForwardTileBits > 11) {
        if (slot == 11) {
            apply_tile_gate_forward<T, NonDiagonalGate::RY, 11>(
                values, coefficients, mailbox);
        }
    }
}

template <typename T, int RegisterBits, int RegisterAmplitudes>
__device__ __forceinline__ void mera_apply_tile_cnot(
    Complex<T> (&values)[RegisterAmplitudes],
    int control_slot,
    int target_slot,
    Complex<T>* mailbox) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> kLaneBits;

#pragma unroll
    for (int reg = 0; reg < RegisterAmplitudes; ++reg) {
        mailbox[tid * RegisterAmplitudes + reg] = values[reg];
    }
    __syncthreads();

#pragma unroll
    for (int reg = 0; reg < RegisterAmplitudes; ++reg) {
        const uint32_t local = static_cast<uint32_t>(
            lane | (reg << kLaneBits) |
            (warp << (kLaneBits + RegisterBits)));
        uint32_t source = local;
        if ((local & (1u << control_slot)) != 0) {
            source ^= 1u << target_slot;
        }
        const int source_lane = source & 31;
        const int source_reg =
            (source >> kLaneBits) & (RegisterAmplitudes - 1);
        const int source_warp = source >> (kLaneBits + RegisterBits);
        const int source_tid = source_lane | (source_warp << kLaneBits);
        values[reg] = mailbox[source_tid * RegisterAmplitudes + source_reg];
    }
    __syncthreads();
}

template <typename T>
__global__ void mera_matching_forward_kernel(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    uint64_t state_size,
    int qubits,
    int layer,
    int parameter_offset,
    int pair_count,
    bool coarse_graining) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* mailbox = reinterpret_cast<Complex<T>*>(dynamic_shared);
    __shared__ int selected[kForwardTileBits];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kForwardTileBits);
    const int pairs_per_phase = tile_bits / 2;
    const int phase_count =
        (pair_count + pairs_per_phase - 1) / pairs_per_phase;
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> kLaneBits;
    (void)state_size;

    for (int phase = 0; phase < phase_count; ++phase) {
        const int phase_pair_offset = phase * pairs_per_phase;
        const int phase_pair_count =
            min(pairs_per_phase, pair_count - phase_pair_offset);
        if (tid == 0) {
            mera_build_selected_map(selected,
                                    tile_bits,
                                    qubits,
                                    layer,
                                    phase_pair_offset,
                                    phase_pair_count,
                                    coarse_graining);
        }
        __syncthreads();

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
                const bool active = local < (1u << tile_bits);
                const uint64_t index =
                    tile_base |
                    scatter_local_assignment<kForwardTileBits>(
                        local, selected, tile_bits);
                values[reg] = active ? state[index] : make_complex<T>(0, 0);
            }

            for (int local_pair = 0; local_pair < phase_pair_count;
                 ++local_pair) {
                const int pair = phase_pair_offset + local_pair;
                const int left_slot = 2 * local_pair;
                const int right_slot = left_slot + 1;
                mera_apply_tile_ry_forward(
                    values,
                    left_slot,
                    coefficients[parameter_offset + 2 * pair],
                    mailbox);
                mera_apply_tile_ry_forward(
                    values,
                    right_slot,
                    coefficients[parameter_offset + 2 * pair + 1],
                    mailbox);
                mera_apply_tile_cnot<T,
                                     kForwardRegisterBits,
                                     kForwardRegisterAmplitudes>(
                    values, left_slot, right_slot, mailbox);
            }

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
                    state[index] = values[reg];
                }
            }
            __syncthreads();
        }
        if (phase + 1 < phase_count) grid.sync();
    }
}

template <typename T>
void launch_mera_matching_forward(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    uint64_t state_size,
    int qubits,
    int layer,
    int parameter_offset,
    int pair_count,
    bool coarse_graining,
    int multiprocessors) {
    if (pair_count == 0) return;
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = mera_matching_forward_kernel<T>;
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
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min(tile_count, resident_blocks));
    void* arguments[] = {
        &state,
        const_cast<RotationCoefficients<T>**>(&coefficients),
        &state_size,
        &qubits,
        &layer,
        &parameter_offset,
        &pair_count,
        &coarse_graining};
    SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel),
        dim3(grid_size),
        dim3(kForwardBlockThreads),
        arguments,
        shared_bytes));
}

template <typename T, int Slot>
__device__ __forceinline__ void mera_apply_tile_ry_backward_slot(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    bool active,
    Complex<T>* mailbox,
    double* reduction,
    double* gradient) {
    if constexpr (Slot < kTileBits) {
        apply_tile_gate_backward<T, NonDiagonalGate::RY, Slot>(
            phi, lambda, coefficients, active, mailbox, reduction, gradient);
    }
}

template <typename T>
__device__ __forceinline__ void mera_apply_tile_ry_backward(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    int slot,
    RotationCoefficients<T> coefficients,
    bool active,
    Complex<T>* mailbox,
    double* reduction,
    double* gradient) {
    switch (slot) {
        case 0:
            mera_apply_tile_ry_backward_slot<T, 0>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 1:
            mera_apply_tile_ry_backward_slot<T, 1>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 2:
            mera_apply_tile_ry_backward_slot<T, 2>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 3:
            mera_apply_tile_ry_backward_slot<T, 3>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 4:
            mera_apply_tile_ry_backward_slot<T, 4>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 5:
            mera_apply_tile_ry_backward_slot<T, 5>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 6:
            mera_apply_tile_ry_backward_slot<T, 6>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        case 7:
            mera_apply_tile_ry_backward_slot<T, 7>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
            break;
        default:
            break;
    }
    if constexpr (kTileBits > 8) {
        if (slot == 8) {
            apply_tile_gate_backward<T, NonDiagonalGate::RY, 8>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
        }
    }
    if constexpr (kTileBits > 9) {
        if (slot == 9) {
            apply_tile_gate_backward<T, NonDiagonalGate::RY, 9>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
        }
    }
    if constexpr (kTileBits > 10) {
        if (slot == 10) {
            apply_tile_gate_backward<T, NonDiagonalGate::RY, 10>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
        }
    }
    if constexpr (kTileBits > 11) {
        if (slot == 11) {
            apply_tile_gate_backward<T, NonDiagonalGate::RY, 11>(
                phi, lambda, coefficients, active, mailbox, reduction, gradient);
        }
    }
}

template <typename T>
__global__ void mera_matching_backward_kernel(
    Complex<T>* phi,
    Complex<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    uint64_t state_size,
    int qubits,
    int layer,
    int parameter_offset,
    int pair_count,
    bool coarse_graining) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* mailbox = reinterpret_cast<Complex<T>*>(dynamic_shared);
    __shared__ double reduction[kBlockThreads];
    __shared__ int selected[kTileBits];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kTileBits);
    const int pairs_per_phase = tile_bits / 2;
    const int phase_count =
        (pair_count + pairs_per_phase - 1) / pairs_per_phase;
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> kLaneBits;
    (void)state_size;

    for (int step = 0; step < phase_count; ++step) {
        const int phase = phase_count - 1 - step;
        const int phase_pair_offset = phase * pairs_per_phase;
        const int phase_pair_count =
            min(pairs_per_phase, pair_count - phase_pair_offset);
        if (tid == 0) {
            mera_build_selected_map(selected,
                                    tile_bits,
                                    qubits,
                                    layer,
                                    phase_pair_offset,
                                    phase_pair_count,
                                    coarse_graining);
        }
        __syncthreads();

        for (uint64_t tile = blockIdx.x; tile < tile_count;
             tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment<kTileBits>(
                    tile, qubits, selected, tile_bits);
            }
            __syncthreads();

            Complex<T> phi_values[kRegisterAmplitudes];
            Complex<T> lambda_values[kRegisterAmplitudes];
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
                phi_values[reg] =
                    active ? phi[index] : make_complex<T>(0, 0);
                lambda_values[reg] =
                    active ? lambda[index] : make_complex<T>(0, 0);
            }

            const bool thread_active =
                (lane | (warp << (kLaneBits + kRegisterBits))) <
                (1u << tile_bits);
            for (int local_pair = phase_pair_count - 1; local_pair >= 0;
                 --local_pair) {
                const int pair = phase_pair_offset + local_pair;
                const int left_slot = 2 * local_pair;
                const int right_slot = left_slot + 1;
                mera_apply_tile_cnot<T, kRegisterBits, kRegisterAmplitudes>(
                    phi_values, left_slot, right_slot, mailbox);
                mera_apply_tile_cnot<T, kRegisterBits, kRegisterAmplitudes>(
                    lambda_values, left_slot, right_slot, mailbox);
                mera_apply_tile_ry_backward(
                    phi_values,
                    lambda_values,
                    right_slot,
                    coefficients[parameter_offset + 2 * pair + 1],
                    thread_active,
                    mailbox,
                    reduction,
                    gradients + parameter_offset + 2 * pair + 1);
                mera_apply_tile_ry_backward(
                    phi_values,
                    lambda_values,
                    left_slot,
                    coefficients[parameter_offset + 2 * pair],
                    thread_active,
                    mailbox,
                    reduction,
                    gradients + parameter_offset + 2 * pair);
            }

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
                    phi[index] = phi_values[reg];
                    lambda[index] = lambda_values[reg];
                }
            }
            __syncthreads();
        }
        if (step + 1 < phase_count) grid.sync();
    }
}

template <typename T>
void launch_mera_matching_backward(
    Complex<T>* phi,
    Complex<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    uint64_t state_size,
    int qubits,
    int layer,
    int parameter_offset,
    int pair_count,
    bool coarse_graining,
    int multiprocessors) {
    if (pair_count == 0) return;
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = mera_matching_backward_kernel<T>;
    constexpr size_t shared_bytes = kTileAmplitudes * sizeof(Complex<T>);
    SAD_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor,
        kernel,
        kBlockThreads,
        shared_bytes));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min(tile_count, resident_blocks));
    void* arguments[] = {
        &phi,
        &lambda,
        const_cast<RotationCoefficients<T>**>(&coefficients),
        &gradients,
        &state_size,
        &qubits,
        &layer,
        &parameter_offset,
        &pair_count,
        &coarse_graining};
    SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel),
        dim3(grid_size),
        dim3(kBlockThreads),
        arguments,
        shared_bytes));
}

}  // namespace sad
