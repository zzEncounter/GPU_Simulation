#pragma once

#include "../core/cuda_common.cuh"

namespace sad {

template <typename T, NonDiagonalGate Gate>
__host__ __device__ constexpr size_t forward_rotation_mailbox_bytes() {
    if constexpr (kForwardWarpBits == 0) {
        return 0;
    } else if constexpr (kRyScalarMailbox && Gate == NonDiagonalGate::RY) {
        return kForwardTileAmplitudes * sizeof(T);
    } else {
        return (kForwardTileAmplitudes / kMailboxChunks) *
               sizeof(Complex<T>);
    }
}

template <typename T, NonDiagonalGate Gate>
__host__ __device__ constexpr size_t backward_rotation_mailbox_bytes() {
    if constexpr (kWarpBits == 0) {
        return 0;
    } else if constexpr (kRyScalarMailbox && Gate == NonDiagonalGate::RY) {
        return kTileAmplitudes * sizeof(T);
    } else {
        return (kTileAmplitudes / kMailboxChunks) * sizeof(Complex<T>);
    }
}

__device__ inline void rotation_gradient_sum(double value,
                                              double* reduction,
                                              double* destination) {
    if constexpr (kRotationWarpAtomic) {
        value = warp_sum(value);
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(destination, value);
        }
    } else {
        block_atomic_sum(value, reduction, destination);
    }
}

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
    void* mailbox_storage) {
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
        constexpr int warp_bit = Slot - kLaneBits - kForwardRegisterBits;
        constexpr int warp_mask = 1 << warp_bit;
        const int partner_tid = tid ^ (warp_mask << 5);
        const int bit = (warp >> warp_bit) & 1;
        if constexpr (kRyScalarMailbox && Gate == NonDiagonalGate::RY) {
            auto* mailbox = reinterpret_cast<T*>(mailbox_storage);
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                mailbox[tid * kForwardRegisterAmplitudes + reg] =
                    values[reg].real;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const T partner =
                    mailbox[partner_tid * kForwardRegisterAmplitudes + reg];
                values[reg].real = rotate_amplitude<T, Gate>(
                    {values[reg].real, 0}, {partner, 0}, bit, coefficients).real;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                mailbox[tid * kForwardRegisterAmplitudes + reg] =
                    values[reg].imag;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const T partner =
                    mailbox[partner_tid * kForwardRegisterAmplitudes + reg];
                values[reg].imag = rotate_amplitude<T, Gate>(
                    {values[reg].imag, 0}, {partner, 0}, bit, coefficients).real;
            }
            __syncthreads();
        } else {
            auto* mailbox = reinterpret_cast<Complex<T>*>(mailbox_storage);
            constexpr int registers_per_chunk =
                kForwardRegisterAmplitudes / kMailboxChunks;
#pragma unroll
            for (int chunk = 0; chunk < kMailboxChunks; ++chunk) {
#pragma unroll
                for (int offset = 0; offset < registers_per_chunk; ++offset) {
                    const int reg = chunk * registers_per_chunk + offset;
                    mailbox[tid * registers_per_chunk + offset] = values[reg];
                }
                __syncthreads();
#pragma unroll
                for (int offset = 0; offset < registers_per_chunk; ++offset) {
                    const int reg = chunk * registers_per_chunk + offset;
                    const Complex<T> partner =
                        mailbox[partner_tid * registers_per_chunk + offset];
                    values[reg] = rotate_amplitude<T, Gate>(
                        values[reg], partner, bit, coefficients);
                }
                __syncthreads();
            }
        }
    }
}

template <typename T, NonDiagonalGate Gate, int Slot>
__device__ __forceinline__ void apply_tile_gate_backward(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    bool active,
    void* mailbox_storage,
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
        rotation_gradient_sum(local_overlap, reduction, gradient);
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
        rotation_gradient_sum(local_overlap, reduction, gradient);
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
        constexpr int warp_bit = Slot - kLaneBits - kRegisterBits;
        const int partner_tid = tid ^ ((1 << warp_bit) << 5);
        const int bit = (warp >> warp_bit) & 1;
        const RotationCoefficients<T> inverse_coefficients{
            -coefficients.sine, coefficients.cosine};
        if constexpr (kRyScalarMailbox && Gate == NonDiagonalGate::RY) {
            auto* mailbox = reinterpret_cast<T*>(mailbox_storage);
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                mailbox[tid * kRegisterAmplitudes + reg] = phi[reg].real;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const T partner =
                    mailbox[partner_tid * kRegisterAmplitudes + reg];
                if (active) {
                    local_overlap += generator_overlap<T, Gate>(
                        {lambda[reg].real, 0}, {partner, 0}, bit);
                }
                phi[reg].real = rotate_amplitude<T, Gate>(
                    {phi[reg].real, 0}, {partner, 0}, bit,
                    inverse_coefficients).real;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                mailbox[tid * kRegisterAmplitudes + reg] = phi[reg].imag;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const T partner =
                    mailbox[partner_tid * kRegisterAmplitudes + reg];
                if (active) {
                    local_overlap += generator_overlap<T, Gate>(
                        {0, lambda[reg].imag}, {0, partner}, bit);
                }
                phi[reg].imag = rotate_amplitude<T, Gate>(
                    {phi[reg].imag, 0}, {partner, 0}, bit,
                    inverse_coefficients).real;
            }
            rotation_gradient_sum(local_overlap, reduction, gradient);
            if constexpr (kRotationWarpAtomic) {
                __syncthreads();
            }
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                mailbox[tid * kRegisterAmplitudes + reg] = lambda[reg].real;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const T partner =
                    mailbox[partner_tid * kRegisterAmplitudes + reg];
                lambda[reg].real = rotate_amplitude<T, Gate>(
                    {lambda[reg].real, 0}, {partner, 0}, bit,
                    inverse_coefficients).real;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                mailbox[tid * kRegisterAmplitudes + reg] = lambda[reg].imag;
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const T partner =
                    mailbox[partner_tid * kRegisterAmplitudes + reg];
                lambda[reg].imag = rotate_amplitude<T, Gate>(
                    {lambda[reg].imag, 0}, {partner, 0}, bit,
                    inverse_coefficients).real;
            }
            __syncthreads();
        } else {
            auto* mailbox = reinterpret_cast<Complex<T>*>(mailbox_storage);
            constexpr int registers_per_chunk =
                kRegisterAmplitudes / kMailboxChunks;
#pragma unroll
            for (int chunk = 0; chunk < kMailboxChunks; ++chunk) {
#pragma unroll
                for (int offset = 0; offset < registers_per_chunk; ++offset) {
                    const int reg = chunk * registers_per_chunk + offset;
                    mailbox[tid * registers_per_chunk + offset] = phi[reg];
                }
                __syncthreads();
#pragma unroll
                for (int offset = 0; offset < registers_per_chunk; ++offset) {
                    const int reg = chunk * registers_per_chunk + offset;
                    const Complex<T> partner =
                        mailbox[partner_tid * registers_per_chunk + offset];
                    if (active) {
                        local_overlap += generator_overlap<T, Gate>(
                            lambda[reg], partner, bit);
                    }
                    phi[reg] = rotate_amplitude<T, Gate>(
                        phi[reg], partner, bit, inverse_coefficients);
                }
                if (chunk + 1 < kMailboxChunks) {
                    __syncthreads();
                }
            }
            rotation_gradient_sum(local_overlap, reduction, gradient);
            if constexpr (kRotationWarpAtomic) {
                __syncthreads();
            }
#pragma unroll
            for (int chunk = 0; chunk < kMailboxChunks; ++chunk) {
#pragma unroll
                for (int offset = 0; offset < registers_per_chunk; ++offset) {
                    const int reg = chunk * registers_per_chunk + offset;
                    mailbox[tid * registers_per_chunk + offset] = lambda[reg];
                }
                __syncthreads();
#pragma unroll
                for (int offset = 0; offset < registers_per_chunk; ++offset) {
                    const int reg = chunk * registers_per_chunk + offset;
                    const Complex<T> partner =
                        mailbox[partner_tid * registers_per_chunk + offset];
                    lambda[reg] = rotate_amplitude<T, Gate>(
                        lambda[reg], partner, bit, inverse_coefficients);
                }
                __syncthreads();
            }
        }
    }
}

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
__device__ __forceinline__ void apply_phase_forward(
    Complex<T> (&values)[kForwardRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    int parameter_offset,
    const int* selected,
    int target_mask,
    void* mailbox) {
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
    if constexpr (kForwardTileBits > 7) {
        if (target_mask & (1 << 7))
            apply_tile_gate_forward<T, Gate, 7>(
                values, coefficients[SAD_ROTATION_PARAMETER(7)], mailbox);
    }
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
    void* mailbox,
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
    if constexpr (kTileBits > 7) {
        SAD_APPLY_BACKWARD_SLOT(7);
    }
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
