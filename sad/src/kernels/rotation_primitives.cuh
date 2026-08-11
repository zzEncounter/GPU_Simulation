#pragma once

#include "../core/cuda_common.cuh"

namespace sad {

template <int TileBits>
__device__ inline bool physical_qubit_is_selected(int qubit,
                                                   const int* selected,
                                                   int tile_bits) {
#pragma unroll
    for (int slot = 0; slot < TileBits; ++slot) {
        if (slot < tile_bits && selected[slot] == qubit) {
            return true;
        }
    }
    return false;
}

template <int TileBits>
__device__ inline uint64_t scatter_tile_assignment(uint64_t tile,
                                                   int qubits,
                                                   const int* selected,
                                                   int tile_bits) {
    uint64_t result = 0;
    int source_bit = 0;
    for (int qubit = 0; qubit < qubits; ++qubit) {
        if (!physical_qubit_is_selected<TileBits>(qubit, selected, tile_bits)) {
            result |= ((tile >> source_bit) & 1ull) << qubit;
            ++source_bit;
        }
    }
    return result;
}

template <int TileBits>
__device__ inline uint64_t scatter_local_assignment(uint32_t local,
                                                    const int* selected,
                                                    int tile_bits) {
    uint64_t result = 0;
#pragma unroll
    for (int slot = 0; slot < TileBits; ++slot) {
        if (slot < tile_bits) {
            result |= static_cast<uint64_t>((local >> slot) & 1u) << selected[slot];
        }
    }
    return result;
}

template <typename T, NonDiagonalGate Gate, int Slot>
__device__ __forceinline__ void apply_tile_gate_forward(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    Complex<T>* mailbox) {
    static_assert(Slot >= 0 && Slot < kForwardTileBits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (Slot < kLaneBits) {
        constexpr int lane_mask = 1 << Slot;
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            const Complex<T> partner = shuffle_xor_complex(values[reg], lane_mask);
            values[reg] = rotate_amplitude<T, Gate>(
                values[reg], partner, (lane >> Slot) & 1, coefficients);
        }
    } else if constexpr (Slot < kLaneBits + kForwardRegisterBits) {
        constexpr int register_bit = Slot - kLaneBits;
        constexpr int register_mask = 1 << register_bit;
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            if ((reg & register_mask) == 0) {
                const Complex<T> zero = values[reg];
                const Complex<T> one = values[reg | register_mask];
                values[reg] =
                    rotate_amplitude<T, Gate>(zero, one, 0, coefficients);
                values[reg | register_mask] =
                    rotate_amplitude<T, Gate>(one, zero, 1, coefficients);
            }
        }
    } else {
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            mailbox[tid * kForwardRegisterAmplitudes + reg] = values[reg];
        }
        __syncthreads();
        constexpr int warp_bit = Slot - kLaneBits - kForwardRegisterBits;
        constexpr int warp_mask = 1 << warp_bit;
        const int partner_tid = tid ^ (warp_mask << 5);
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            const Complex<T> partner =
                mailbox[partner_tid * kForwardRegisterAmplitudes + reg];
            values[reg] = rotate_amplitude<T, Gate>(
                values[reg], partner, (warp >> warp_bit) & 1, coefficients);
        }
        __syncthreads();
    }
}

template <typename T, NonDiagonalGate Gate, int Slot>
__device__ __forceinline__ void apply_tile_gate_backward(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    bool active,
    Complex<T>* mailbox,
    double* reduction,
    double* gradient) {
    static_assert(Slot >= 0 && Slot < kTileBits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    double local_overlap = 0.0;

    if constexpr (Slot < kLaneBits) {
        constexpr int lane_mask = 1 << Slot;
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> partner = shuffle_xor_complex(phi[reg], lane_mask);
            if (active) {
                local_overlap += generator_overlap<T, Gate>(
                    lambda[reg], partner, (lane >> Slot) & 1);
            }
        }
        block_atomic_sum(local_overlap, reduction, gradient);
        const RotationCoefficients<T> inverse_coefficients{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> phi_partner = shuffle_xor_complex(phi[reg], lane_mask);
            const Complex<T> lambda_partner = shuffle_xor_complex(lambda[reg], lane_mask);
            phi[reg] = rotate_amplitude<T, Gate>(
                phi[reg], phi_partner, (lane >> Slot) & 1, inverse_coefficients);
            lambda[reg] = rotate_amplitude<T, Gate>(
                lambda[reg], lambda_partner, (lane >> Slot) & 1, inverse_coefficients);
        }
    } else if constexpr (Slot < kLaneBits + kRegisterBits) {
        constexpr int register_bit = Slot - kLaneBits;
        constexpr int register_mask = 1 << register_bit;
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            if (active) {
                local_overlap += generator_overlap<T, Gate>(
                    lambda[reg],
                    phi[reg ^ register_mask],
                    (reg >> register_bit) & 1);
            }
        }
        block_atomic_sum(local_overlap, reduction, gradient);
        const RotationCoefficients<T> inverse_coefficients{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            if ((reg & register_mask) == 0) {
                const Complex<T> phi_zero = phi[reg];
                const Complex<T> phi_one = phi[reg | register_mask];
                const Complex<T> lambda_zero = lambda[reg];
                const Complex<T> lambda_one = lambda[reg | register_mask];
                phi[reg] = rotate_amplitude<T, Gate>(
                    phi_zero, phi_one, 0, inverse_coefficients);
                phi[reg | register_mask] =
                    rotate_amplitude<T, Gate>(phi_one, phi_zero, 1, inverse_coefficients);
                lambda[reg] = rotate_amplitude<T, Gate>(
                    lambda_zero, lambda_one, 0, inverse_coefficients);
                lambda[reg | register_mask] =
                    rotate_amplitude<T, Gate>(lambda_one, lambda_zero, 1,
                                               inverse_coefficients);
            }
        }
    } else {
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            mailbox[tid * kRegisterAmplitudes + reg] = phi[reg];
        }
        __syncthreads();
        constexpr int warp_bit = Slot - kLaneBits - kRegisterBits;
        const int partner_tid = tid ^ ((1 << warp_bit) << 5);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            if (active) {
                const Complex<T> partner =
                    mailbox[partner_tid * kRegisterAmplitudes + reg];
                local_overlap += generator_overlap<T, Gate>(
                    lambda[reg], partner, (warp >> warp_bit) & 1);
            }
        }
        const RotationCoefficients<T> inverse_coefficients{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> phi_partner =
                mailbox[partner_tid * kRegisterAmplitudes + reg];
            phi[reg] = rotate_amplitude<T, Gate>(
                phi[reg], phi_partner, (warp >> warp_bit) & 1,
                inverse_coefficients);
        }
        block_atomic_sum(local_overlap, reduction, gradient);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            mailbox[tid * kRegisterAmplitudes + reg] = lambda[reg];
        }
        __syncthreads();
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> lambda_partner =
                mailbox[partner_tid * kRegisterAmplitudes + reg];
            lambda[reg] = rotate_amplitude<T, Gate>(
                lambda[reg], lambda_partner, (warp >> warp_bit) & 1,
                inverse_coefficients);
        }
        __syncthreads();
    }
}

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
__device__ __forceinline__ void apply_phase_forward(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    int parameter_offset,
    const int* selected,
    int target_mask,
    Complex<T>* mailbox) {
#define SAD_ROTATION_PARAMETER(slot) \
    (parameter_offset + (SharedParameter ? 0 : selected[slot]))
    if (target_mask & (1 << 0))
        apply_tile_gate_forward<T, Gate, 0>(
            values, coefficients[SAD_ROTATION_PARAMETER(0)], mailbox);
    if (target_mask & (1 << 1))
        apply_tile_gate_forward<T, Gate, 1>(
            values, coefficients[SAD_ROTATION_PARAMETER(1)], mailbox);
    if (target_mask & (1 << 2))
        apply_tile_gate_forward<T, Gate, 2>(
            values, coefficients[SAD_ROTATION_PARAMETER(2)], mailbox);
    if (target_mask & (1 << 3))
        apply_tile_gate_forward<T, Gate, 3>(
            values, coefficients[SAD_ROTATION_PARAMETER(3)], mailbox);
    if (target_mask & (1 << 4))
        apply_tile_gate_forward<T, Gate, 4>(
            values, coefficients[SAD_ROTATION_PARAMETER(4)], mailbox);
    if (target_mask & (1 << 5))
        apply_tile_gate_forward<T, Gate, 5>(
            values, coefficients[SAD_ROTATION_PARAMETER(5)], mailbox);
    if (target_mask & (1 << 6))
        apply_tile_gate_forward<T, Gate, 6>(
            values, coefficients[SAD_ROTATION_PARAMETER(6)], mailbox);
    if (target_mask & (1 << 7))
        apply_tile_gate_forward<T, Gate, 7>(
            values, coefficients[SAD_ROTATION_PARAMETER(7)], mailbox);
    if constexpr (kForwardTileBits > 8) {
        if (target_mask & (1 << 8))
            apply_tile_gate_forward<T, Gate, 8>(
                values, coefficients[SAD_ROTATION_PARAMETER(8)], mailbox);
    }
    if constexpr (kForwardTileBits > 9) {
        if (target_mask & (1 << 9))
            apply_tile_gate_forward<T, Gate, 9>(
                values, coefficients[SAD_ROTATION_PARAMETER(9)], mailbox);
    }
    if constexpr (kForwardTileBits > 10) {
        if (target_mask & (1 << 10))
            apply_tile_gate_forward<T, Gate, 10>(
                values, coefficients[SAD_ROTATION_PARAMETER(10)], mailbox);
    }
    if constexpr (kForwardTileBits > 11) {
        if (target_mask & (1 << 11))
            apply_tile_gate_forward<T, Gate, 11>(
                values, coefficients[SAD_ROTATION_PARAMETER(11)], mailbox);
    }
#undef SAD_ROTATION_PARAMETER
}

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
__device__ __forceinline__ void apply_phase_backward(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    double* gradient_accumulator,
    int parameter_offset,
    const int* selected,
    int target_mask,
    bool active,
    Complex<T>* mailbox,
    double* reduction) {
#define SAD_APPLY_BACKWARD_SLOT(slot)                                                   \
    if (target_mask & (1 << slot))                                                     \
        apply_tile_gate_backward<T, Gate, slot>(                                       \
            phi, lambda,                                                               \
            coefficients[parameter_offset +                                           \
                         (SharedParameter ? 0 : selected[slot])],                       \
            active,                                                                    \
            mailbox, reduction,                                                        \
            gradient_accumulator + parameter_offset +                                  \
                (SharedParameter ? 0 : selected[slot]))
    if constexpr (kTileBits > 11) {
        SAD_APPLY_BACKWARD_SLOT(11);
    }
    if constexpr (kTileBits > 10) {
        SAD_APPLY_BACKWARD_SLOT(10);
    }
    if constexpr (kTileBits > 9) {
        SAD_APPLY_BACKWARD_SLOT(9);
    }
    if constexpr (kTileBits > 8) {
        SAD_APPLY_BACKWARD_SLOT(8);
    }
    SAD_APPLY_BACKWARD_SLOT(7);
    SAD_APPLY_BACKWARD_SLOT(6);
    SAD_APPLY_BACKWARD_SLOT(5);
    SAD_APPLY_BACKWARD_SLOT(4);
    SAD_APPLY_BACKWARD_SLOT(3);
    SAD_APPLY_BACKWARD_SLOT(2);
    SAD_APPLY_BACKWARD_SLOT(1);
    SAD_APPLY_BACKWARD_SLOT(0);
#undef SAD_APPLY_BACKWARD_SLOT
}

}  // namespace sad
