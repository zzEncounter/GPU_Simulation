#include "sad_api.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cg = cooperative_groups;

namespace sad {

constexpr int kBlockThreads = 128;
constexpr int kWarpsPerBlock = 4;
constexpr int kRegisterBits = 3;
constexpr int kRegisterAmplitudes = 1 << kRegisterBits;
constexpr int kLaneBits = 5;
constexpr int kWarpBits = 2;
constexpr int kTileBits = kLaneBits + kRegisterBits + kWarpBits;
constexpr int kTileAmplitudes = 1 << kTileBits;
constexpr int kMaxQubits = 30;
constexpr int kDiagonalLookupBits = 8;
constexpr int kDiagonalLookupSize = 1 << kDiagonalLookupBits;

enum class NonDiagonalGate : int { RX = 0, RY = 1 };
enum class DiagonalGate : int { RZ = 0, RZZ_EVEN = 1, RZZ_ODD = 2 };

inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed at " + file + ":" +
                                 std::to_string(line) + ": " + cudaGetErrorString(status));
    }
}

#define SAD_CUDA_CHECK(expr) ::sad::cuda_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
struct alignas(sizeof(T) * 2) Complex {
    T real;
    T imag;
};

template <typename T>
__host__ __device__ inline Complex<T> make_complex(T real, T imag) {
    return {real, imag};
}

template <typename T>
__device__ inline Complex<T> add(Complex<T> lhs, Complex<T> rhs) {
    return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

template <typename T>
__device__ inline Complex<T> sub(Complex<T> lhs, Complex<T> rhs) {
    return {lhs.real - rhs.real, lhs.imag - rhs.imag};
}

template <typename T>
__device__ inline Complex<T> scale(Complex<T> value, T factor) {
    return {value.real * factor, value.imag * factor};
}

template <typename T>
__host__ __device__ inline Complex<T> multiply(Complex<T> lhs, Complex<T> rhs) {
    return {lhs.real * rhs.real - lhs.imag * rhs.imag,
            lhs.real * rhs.imag + lhs.imag * rhs.real};
}

template <typename T>
__device__ inline Complex<T> multiply_phase(Complex<T> value, T angle) {
    T sine;
    T cosine;
    if constexpr (std::is_same_v<T, float>) {
        sincosf(angle, &sine, &cosine);
    } else {
        sincos(angle, &sine, &cosine);
    }
    return {cosine * value.real - sine * value.imag,
            sine * value.real + cosine * value.imag};
}

template <typename T>
__device__ inline double imag_conjugate_product(Complex<T> bra, Complex<T> ket) {
    return static_cast<double>(bra.real) * static_cast<double>(ket.imag) -
           static_cast<double>(bra.imag) * static_cast<double>(ket.real);
}

template <typename T>
__device__ inline double real_conjugate_product(Complex<T> bra, Complex<T> ket) {
    return static_cast<double>(bra.real) * static_cast<double>(ket.real) +
           static_cast<double>(bra.imag) * static_cast<double>(ket.imag);
}

template <typename T>
__device__ inline Complex<T> shuffle_xor_complex(Complex<T> value, int lane_mask) {
    return {__shfl_xor_sync(0xffffffffu, value.real, lane_mask),
            __shfl_xor_sync(0xffffffffu, value.imag, lane_mask)};
}

template <typename T>
struct RotationCoefficients {
    T sine;
    T cosine;
};

template <typename T, NonDiagonalGate Gate>
__device__ __forceinline__ Complex<T> rotate_amplitude(
    Complex<T> self,
    Complex<T> partner,
    int bit,
    RotationCoefficients<T> coefficients) {
    if constexpr (Gate == NonDiagonalGate::RY) {
        if (bit == 0) {
            return sub(scale(self, coefficients.cosine),
                       scale(partner, coefficients.sine));
        }
        return add(scale(self, coefficients.cosine),
                   scale(partner, coefficients.sine));
    } else {
        // RX: c*self - i*s*partner.
        return {coefficients.cosine * self.real + coefficients.sine * partner.imag,
                coefficients.cosine * self.imag - coefficients.sine * partner.real};
    }
}

template <typename T, NonDiagonalGate Gate>
__device__ inline Complex<T> apply_generator(Complex<T> partner, int bit) {
    if constexpr (Gate == NonDiagonalGate::RX) {
        return partner;  // X|b> flips b.
    } else {
        // Y|0> = i|1>, Y|1> = -i|0>; as an output amplitude this is
        // (Y phi)_0=-i phi_1 and (Y phi)_1=+i phi_0.
        if (bit == 0) {
            return {partner.imag, -partner.real};
        }
        return {-partner.imag, partner.real};
    }
}

__device__ inline double warp_sum(double value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ inline void block_atomic_sum(double value,
                                         double* reduction,
                                         double* destination) {
    const int tid = threadIdx.x;
    reduction[tid] = value;
    __syncthreads();
    for (int stride = kBlockThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(destination, reduction[0]);
    }
    __syncthreads();
}

__device__ inline bool physical_qubit_is_selected(int qubit,
                                                   const int* selected,
                                                   int tile_bits) {
#pragma unroll
    for (int slot = 0; slot < kTileBits; ++slot) {
        if (slot < tile_bits && selected[slot] == qubit) {
            return true;
        }
    }
    return false;
}

__device__ inline uint64_t scatter_tile_assignment(uint64_t tile,
                                                   int qubits,
                                                   const int* selected,
                                                   int tile_bits) {
    uint64_t result = 0;
    int source_bit = 0;
    for (int qubit = 0; qubit < qubits; ++qubit) {
        if (!physical_qubit_is_selected(qubit, selected, tile_bits)) {
            result |= ((tile >> source_bit) & 1ull) << qubit;
            ++source_bit;
        }
    }
    return result;
}

__device__ inline uint64_t scatter_local_assignment(uint32_t local,
                                                    const int* selected,
                                                    int tile_bits) {
    uint64_t result = 0;
#pragma unroll
    for (int slot = 0; slot < kTileBits; ++slot) {
        if (slot < tile_bits) {
            result |= static_cast<uint64_t>((local >> slot) & 1u) << selected[slot];
        }
    }
    return result;
}

template <typename T, NonDiagonalGate Gate, int Slot>
__device__ __forceinline__ void apply_tile_gate_forward(
    Complex<T> (&values)[kRegisterAmplitudes],
    RotationCoefficients<T> coefficients,
    Complex<T>* mailbox) {
    static_assert(Slot >= 0 && Slot < kTileBits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (Slot < kLaneBits) {
        constexpr int lane_mask = 1 << Slot;
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> partner = shuffle_xor_complex(values[reg], lane_mask);
            values[reg] = rotate_amplitude<T, Gate>(
                values[reg], partner, (lane >> Slot) & 1, coefficients);
        }
    } else if constexpr (Slot < kLaneBits + kRegisterBits) {
        constexpr int register_bit = Slot - kLaneBits;
        constexpr int register_mask = 1 << register_bit;
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
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
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            mailbox[tid * kRegisterAmplitudes + reg] = values[reg];
        }
        __syncthreads();
        constexpr int warp_bit = Slot - kLaneBits - kRegisterBits;
        constexpr int warp_mask = 1 << warp_bit;
        const int partner_tid = tid ^ (warp_mask << 5);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> partner = mailbox[partner_tid * kRegisterAmplitudes + reg];
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
    Complex<T>* phi_mailbox,
    Complex<T>* lambda_mailbox,
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
                local_overlap += imag_conjugate_product(
                    lambda[reg], apply_generator<T, Gate>(partner, (lane >> Slot) & 1));
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
                local_overlap += imag_conjugate_product(
                    lambda[reg], apply_generator<T, Gate>(phi[reg ^ register_mask],
                                                           (reg >> register_bit) & 1));
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
            phi_mailbox[tid * kRegisterAmplitudes + reg] = phi[reg];
            lambda_mailbox[tid * kRegisterAmplitudes + reg] = lambda[reg];
        }
        __syncthreads();
        constexpr int warp_bit = Slot - kLaneBits - kRegisterBits;
        const int partner_tid = tid ^ ((1 << warp_bit) << 5);
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            if (active) {
                const Complex<T> partner =
                    phi_mailbox[partner_tid * kRegisterAmplitudes + reg];
                local_overlap += imag_conjugate_product(
                    lambda[reg],
                    apply_generator<T, Gate>(partner, (warp >> warp_bit) & 1));
            }
        }
        block_atomic_sum(local_overlap, reduction, gradient);
        const RotationCoefficients<T> inverse_coefficients{
            -coefficients.sine, coefficients.cosine};
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const Complex<T> phi_partner =
                phi_mailbox[partner_tid * kRegisterAmplitudes + reg];
            const Complex<T> lambda_partner =
                lambda_mailbox[partner_tid * kRegisterAmplitudes + reg];
            phi[reg] = rotate_amplitude<T, Gate>(
                phi[reg], phi_partner, (warp >> warp_bit) & 1,
                inverse_coefficients);
            lambda[reg] = rotate_amplitude<T, Gate>(
                lambda[reg], lambda_partner, (warp >> warp_bit) & 1,
                inverse_coefficients);
        }
        __syncthreads();
    }
}

template <typename T, NonDiagonalGate Gate>
__device__ __forceinline__ void apply_phase_forward(
    Complex<T> (&values)[kRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    int parameter_offset,
    const int* selected,
    int target_count,
    Complex<T>* mailbox) {
    if (target_count > 0)
        apply_tile_gate_forward<T, Gate, 0>(
            values, coefficients[parameter_offset + selected[0]], mailbox);
    if (target_count > 1)
        apply_tile_gate_forward<T, Gate, 1>(
            values, coefficients[parameter_offset + selected[1]], mailbox);
    if (target_count > 2)
        apply_tile_gate_forward<T, Gate, 2>(
            values, coefficients[parameter_offset + selected[2]], mailbox);
    if (target_count > 3)
        apply_tile_gate_forward<T, Gate, 3>(
            values, coefficients[parameter_offset + selected[3]], mailbox);
    if (target_count > 4)
        apply_tile_gate_forward<T, Gate, 4>(
            values, coefficients[parameter_offset + selected[4]], mailbox);
    if (target_count > 5)
        apply_tile_gate_forward<T, Gate, 5>(
            values, coefficients[parameter_offset + selected[5]], mailbox);
    if (target_count > 6)
        apply_tile_gate_forward<T, Gate, 6>(
            values, coefficients[parameter_offset + selected[6]], mailbox);
    if (target_count > 7)
        apply_tile_gate_forward<T, Gate, 7>(
            values, coefficients[parameter_offset + selected[7]], mailbox);
    if (target_count > 8)
        apply_tile_gate_forward<T, Gate, 8>(
            values, coefficients[parameter_offset + selected[8]], mailbox);
    if (target_count > 9)
        apply_tile_gate_forward<T, Gate, 9>(
            values, coefficients[parameter_offset + selected[9]], mailbox);
}

template <typename T, NonDiagonalGate Gate>
__device__ __forceinline__ void apply_phase_backward(
    Complex<T> (&phi)[kRegisterAmplitudes],
    Complex<T> (&lambda)[kRegisterAmplitudes],
    const RotationCoefficients<T>* coefficients,
    double* gradient_accumulator,
    int parameter_offset,
    const int* selected,
    int target_count,
    bool active,
    Complex<T>* phi_mailbox,
    Complex<T>* lambda_mailbox,
    double* reduction) {
#define SAD_APPLY_BACKWARD_SLOT(slot)                                                   \
    if (target_count > slot)                                                           \
        apply_tile_gate_backward<T, Gate, slot>(                                       \
            phi, lambda, coefficients[parameter_offset + selected[slot]], active,      \
            phi_mailbox, lambda_mailbox, reduction,                                    \
            gradient_accumulator + parameter_offset + selected[slot])
    SAD_APPLY_BACKWARD_SLOT(9);
    SAD_APPLY_BACKWARD_SLOT(8);
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

template <typename T, NonDiagonalGate Gate>
__global__ void non_diagonal_forward_kernel(Complex<T>* state,
                                            const RotationCoefficients<T>* coefficients,
                                            int qubits,
                                            int parameter_offset,
                                            const int* selected_maps,
                                            const int* target_counts,
                                            int phase_count) {
    cg::grid_group grid = cg::this_grid();
    __shared__ Complex<T> mailbox[kTileAmplitudes];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int phase = 0; phase < phase_count; ++phase) {
        const int* selected = selected_maps + phase * kTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment(tile, qubits, selected, tile_bits);
            }
            __syncthreads();

            Complex<T> values[kRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(lane | (reg << 5) | (warp << 8));
                const bool active = local < (1u << tile_bits);
                const uint64_t index =
                    tile_base | scatter_local_assignment(local, selected, tile_bits);
                values[reg] = active ? state[index] : make_complex<T>(0, 0);
            }

            apply_phase_forward<T, Gate>(values,
                                         coefficients,
                                         parameter_offset,
                                         selected,
                                         target_counts[phase],
                                         mailbox);

#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(lane | (reg << 5) | (warp << 8));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base | scatter_local_assignment(local, selected, tile_bits);
                    state[index] = values[reg];
                }
            }
            __syncthreads();
        }
        grid.sync();
    }
}

template <typename T, NonDiagonalGate Gate>
__global__ void non_diagonal_backward_gradient_kernel(Complex<T>* phi_state,
                                                      Complex<T>* lambda_state,
                                                      const RotationCoefficients<T>* coefficients,
                                                      double* gradient_accumulator,
                                                      int qubits,
                                                      int parameter_offset,
                                                      const int* selected_maps,
                                                      const int* target_counts,
                                                      int phase_count) {
    cg::grid_group grid = cg::this_grid();
    __shared__ Complex<T> phi_mailbox[kTileAmplitudes];
    __shared__ Complex<T> lambda_mailbox[kTileAmplitudes];
    __shared__ double reduction[kBlockThreads];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int phase = phase_count - 1; phase >= 0; --phase) {
        const int* selected = selected_maps + phase * kTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
            if (tid == 0) {
                tile_base = scatter_tile_assignment(tile, qubits, selected, tile_bits);
            }
            __syncthreads();

            Complex<T> phi[kRegisterAmplitudes];
            Complex<T> lambda[kRegisterAmplitudes];
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(lane | (reg << 5) | (warp << 8));
                const bool active = local < (1u << tile_bits);
                const uint64_t index =
                    tile_base | scatter_local_assignment(local, selected, tile_bits);
                phi[reg] = active ? phi_state[index] : make_complex<T>(0, 0);
                lambda[reg] = active ? lambda_state[index] : make_complex<T>(0, 0);
            }

            const bool thread_active = (lane | (warp << 8)) < (1u << tile_bits);
            apply_phase_backward<T, Gate>(phi,
                                          lambda,
                                          coefficients,
                                          gradient_accumulator,
                                          parameter_offset,
                                          selected,
                                          target_counts[phase],
                                          thread_active,
                                          phi_mailbox,
                                          lambda_mailbox,
                                          reduction);

#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(lane | (reg << 5) | (warp << 8));
                if (local < (1u << tile_bits)) {
                    const uint64_t index =
                        tile_base | scatter_local_assignment(local, selected, tile_bits);
                    phi_state[index] = phi[reg];
                    lambda_state[index] = lambda[reg];
                }
            }
            __syncthreads();
        }
        grid.sync();
    }
}

__device__ inline uint64_t apply_ring_cnot_forward_to_basis(uint64_t basis, int qubits) {
    for (int control = 0; control < qubits; ++control) {
        const int target = (control + 1) % qubits;
        if ((basis >> control) & 1ull) {
            basis ^= 1ull << target;
        }
    }
    return basis;
}

__device__ inline uint64_t apply_ring_cnot_inverse_to_basis(uint64_t basis, int qubits) {
    for (int control = qubits - 1; control >= 0; --control) {
        const int target = (control + 1) % qubits;
        if ((basis >> control) & 1ull) {
            basis ^= 1ull << target;
        }
    }
    return basis;
}

template <typename T>
__global__ void ring_cnot_permutation_kernel(const Complex<T>* phi_input,
                                             Complex<T>* phi_output,
                                             const Complex<T>* lambda_input,
                                             Complex<T>* lambda_output,
                                             uint64_t state_size,
                                             int qubits,
                                             bool adjoint) {
    for (uint64_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
         output_index < state_size;
         output_index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        // Coalesced output writes. Forward U reads P^-1(y); adjoint U^dagger
        // reads P(y). The two full state buffers are swapped by the host.
        const uint64_t input_index =
            adjoint ? apply_ring_cnot_forward_to_basis(output_index, qubits)
                    : apply_ring_cnot_inverse_to_basis(output_index, qubits);
        phi_output[output_index] = phi_input[input_index];
        if (lambda_input != nullptr) {
            lambda_output[output_index] = lambda_input[input_index];
        }
    }
}

template <DiagonalGate Gate>
__device__ inline int diagonal_eigenvalue(uint64_t basis,
                                          int gate_index,
                                          int qubits) {
    if constexpr (Gate == DiagonalGate::RZ) {
        return ((basis >> gate_index) & 1ull) ? -1 : 1;
    } else {
        const int left =
            Gate == DiagonalGate::RZZ_EVEN ? 2 * gate_index : 2 * gate_index + 1;
        const int right = (left + 1) % qubits;
        const int left_bit = (basis >> left) & 1ull;
        const int right_bit = (basis >> right) & 1ull;
        return left_bit == right_bit ? 1 : -1;
    }
}

template <typename T, DiagonalGate Gate>
__global__ void diagonal_forward_kernel(Complex<T>* state,
                                        const Complex<T>* phase_lookup,
                                        uint64_t state_size,
                                        int qubits,
                                        int gate_count) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex<T> factor = make_complex<T>(static_cast<T>(1), static_cast<T>(0));
        const int chunk_count =
            (gate_count + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            unsigned code = 0;
#pragma unroll
            for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
                const int gate = chunk * kDiagonalLookupBits + bit;
                if (gate < gate_count &&
                    diagonal_eigenvalue<Gate>(index, gate, qubits) < 0) {
                    code |= 1u << bit;
                }
            }
            factor = multiply(factor,
                              phase_lookup[chunk * kDiagonalLookupSize + code]);
        }
        state[index] = multiply(state[index], factor);
    }
}

template <typename T, DiagonalGate Gate>
__global__ void diagonal_backward_kernel(Complex<T>* phi_state,
                                         Complex<T>* lambda_state,
                                         const Complex<T>* phase_lookup,
                                         double* gradient_accumulator,
                                         uint64_t state_size,
                                         int qubits,
                                         int parameter_offset,
                                         int gate_count) {
    __shared__ double overlaps[kMaxQubits * kBlockThreads];
    __shared__ double warp_partials[kMaxQubits * kWarpsPerBlock];
    const int tid = threadIdx.x;
    for (int gate = 0; gate < gate_count; ++gate) {
        overlaps[gate * kBlockThreads + tid] = 0.0;
    }
    __syncthreads();

    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex<T> phi = phi_state[index];
        Complex<T> lambda = lambda_state[index];
        const double base_overlap = imag_conjugate_product(lambda, phi);
        Complex<T> factor = make_complex<T>(static_cast<T>(1), static_cast<T>(0));
        const int chunk_count =
            (gate_count + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            unsigned code = 0;
#pragma unroll
            for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
                const int gate = chunk * kDiagonalLookupBits + bit;
                if (gate < gate_count) {
                    const int eigenvalue =
                        diagonal_eigenvalue<Gate>(index, gate, qubits);
                    overlaps[gate * kBlockThreads + tid] +=
                        base_overlap * static_cast<double>(eigenvalue);
                    if (eigenvalue < 0) {
                        code |= 1u << bit;
                    }
                }
            }
            factor = multiply(factor,
                              phase_lookup[chunk * kDiagonalLookupSize + code]);
        }
        const Complex<T> inverse_factor{factor.real, -factor.imag};
        phi_state[index] = multiply(phi, inverse_factor);
        lambda_state[index] = multiply(lambda, inverse_factor);
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int gate = 0; gate < gate_count; ++gate) {
        const double sum = warp_sum(overlaps[gate * kBlockThreads + tid]);
        if (lane == 0) {
            warp_partials[gate * kWarpsPerBlock + warp] = sum;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int gate = 0; gate < gate_count; ++gate) {
            double value = lane < kWarpsPerBlock
                               ? warp_partials[gate * kWarpsPerBlock + lane]
                               : 0.0;
            value = warp_sum(value);
            if (lane == 0) {
                atomicAdd(gradient_accumulator + parameter_offset + gate, value);
            }
        }
    }
}

template <typename T>
__global__ void initialise_zero_state_kernel(Complex<T>* state) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        state[0] = make_complex<T>(static_cast<T>(1), static_cast<T>(0));
    }
}

template <typename T>
__global__ void hamiltonian_kernel(const Complex<T>* phi,
                                   Complex<T>* lambda,
                                   uint64_t state_size,
                                   int qubits,
                                   double* energy) {
    __shared__ double reduction[kBlockThreads];
    double local_energy = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int zz_sum = 0;
        for (int qubit = 0; qubit < qubits; ++qubit) {
            const int next = (qubit + 1) % qubits;
            const int bit = (index >> qubit) & 1ull;
            const int next_bit = (index >> next) & 1ull;
            zz_sum += bit == next_bit ? 1 : -1;
        }

        const Complex<T> amplitude = phi[index];
        Complex<T> h_amplitude = scale(amplitude, static_cast<T>(-zz_sum));
        for (int qubit = 0; qubit < qubits; ++qubit) {
            h_amplitude = sub(h_amplitude, phi[index ^ (1ull << qubit)]);
        }
        lambda[index] = h_amplitude;
        local_energy += real_conjugate_product(amplitude, h_amplitude);
    }
    block_atomic_sum(local_energy, reduction, energy);
}

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    void allocate(size_t count) {
        release();
        count_ = count;
        SAD_CUDA_CHECK(cudaMalloc(&pointer_, count * sizeof(T)));
    }
    T* get() const { return pointer_; }
    size_t bytes() const { return count_ * sizeof(T); }

  private:
    void release() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
            pointer_ = nullptr;
        }
    }
    T* pointer_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(size_t count) : count_(count) {
        SAD_CUDA_CHECK(cudaMallocHost(&pointer_, count * sizeof(T)));
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() {
        if (pointer_ != nullptr) {
            cudaFreeHost(pointer_);
        }
    }
    T* get() const { return pointer_; }

  private:
    T* pointer_ = nullptr;
    size_t count_ = 0;
};

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    EventPair() {
        SAD_CUDA_CHECK(cudaEventCreate(&start));
        SAD_CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~EventPair() {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
    }
    template <typename Function>
    double measure(Function&& function) {
        const auto wall_start = std::chrono::steady_clock::now();
        SAD_CUDA_CHECK(cudaEventRecord(start));
        function();
        SAD_CUDA_CHECK(cudaEventRecord(stop));
        SAD_CUDA_CHECK(cudaEventSynchronize(stop));
        const auto wall_stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(wall_stop - wall_start).count();
    }
};

template <typename T>
struct DiagonalLookupData {
    std::vector<Complex<T>> factors;
    std::vector<size_t> offsets_by_parameter;
};

template <typename T>
void append_diagonal_lookup_group(const T* parameters,
                                  size_t parameter_offset,
                                  int gate_count,
                                  DiagonalLookupData<T>* data) {
    data->offsets_by_parameter[parameter_offset] = data->factors.size();
    const int chunk_count =
        (gate_count + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        for (int code = 0; code < kDiagonalLookupSize; ++code) {
            T phase = 0;
            for (int bit = 0; bit < kDiagonalLookupBits; ++bit) {
                const int gate = chunk * kDiagonalLookupBits + bit;
                if (gate < gate_count) {
                    const int eigenvalue = ((code >> bit) & 1) ? -1 : 1;
                    phase -= static_cast<T>(0.5) *
                             parameters[parameter_offset + gate] *
                             static_cast<T>(eigenvalue);
                }
            }
            data->factors.push_back(
                {static_cast<T>(std::cos(phase)), static_cast<T>(std::sin(phase))});
        }
    }
}

inline void build_phase_maps(int qubits,
                             std::vector<int>* selected_maps,
                             std::vector<int>* target_counts) {
    const int tile_bits = std::min(qubits, kTileBits);
    const int phase_count = (qubits + kTileBits - 1) / kTileBits;
    selected_maps->assign(phase_count * kTileBits, -1);
    target_counts->assign(phase_count, 0);
    for (int phase = 0; phase < phase_count; ++phase) {
        const int first = phase * kTileBits;
        const int count = std::min(kTileBits, qubits - first);
        (*target_counts)[phase] = count;
        int filled = 0;
        for (int qubit = first; qubit < first + count; ++qubit) {
            (*selected_maps)[phase * kTileBits + filled++] = qubit;
        }
        for (int qubit = 0; filled < tile_bits && qubit < qubits; ++qubit) {
            if (qubit < first || qubit >= first + count) {
                (*selected_maps)[phase * kTileBits + filled++] = qubit;
            }
        }
    }
}

inline int ordinary_grid_size(uint64_t state_size, int multiprocessors) {
    const uint64_t required =
        (state_size + static_cast<uint64_t>(kBlockThreads) - 1) / kBlockThreads;
    return static_cast<int>(std::min<uint64_t>(required, multiprocessors * 4ull));
}

template <typename T, NonDiagonalGate Gate>
void launch_non_diagonal_forward(Complex<T>* state,
                                 const RotationCoefficients<T>* coefficients,
                                 int qubits,
                                 int parameter_offset,
                                 const int* selected_maps,
                                 const int* target_counts,
                                 int phase_count,
                                 int multiprocessors) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = non_diagonal_forward_kernel<T, Gate>;
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, kernel, kBlockThreads, 0));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
    void* arguments[] = {&state,
                         const_cast<RotationCoefficients<T>**>(&coefficients),
                         &qubits,
                         &parameter_offset,
                         const_cast<int**>(&selected_maps),
                         const_cast<int**>(&target_counts),
                         &phase_count};
    SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel),
        dim3(grid_size),
        dim3(kBlockThreads),
        arguments));
}

template <typename T, NonDiagonalGate Gate>
void launch_non_diagonal_backward(Complex<T>* phi,
                                  Complex<T>* lambda,
                                  const RotationCoefficients<T>* coefficients,
                                  double* gradients,
                                  int qubits,
                                  int parameter_offset,
                                  const int* selected_maps,
                                  const int* target_counts,
                                  int phase_count,
                                  int multiprocessors) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = non_diagonal_backward_gradient_kernel<T, Gate>;
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, kernel, kBlockThreads, 0));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
    void* arguments[] = {&phi,
                         &lambda,
                         const_cast<RotationCoefficients<T>**>(&coefficients),
                         &gradients,
                         &qubits,
                         &parameter_offset,
                         const_cast<int**>(&selected_maps),
                         const_cast<int**>(&target_counts),
                         &phase_count};
    SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel),
        dim3(grid_size),
        dim3(kBlockThreads),
        arguments));
}

template <typename T, DiagonalGate Gate>
void launch_diagonal_forward(Complex<T>* state,
                             const Complex<T>* phase_lookup,
                             uint64_t state_size,
                             int qubits,
                             int gate_count,
                             int grid_size) {
    diagonal_forward_kernel<T, Gate><<<grid_size, kBlockThreads>>>(
        state, phase_lookup, state_size, qubits, gate_count);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T, DiagonalGate Gate>
void launch_diagonal_backward(Complex<T>* phi,
                              Complex<T>* lambda,
                              const Complex<T>* phase_lookup,
                              double* gradients,
                              uint64_t state_size,
                              int qubits,
                              int parameter_offset,
                              int gate_count,
                              int grid_size) {
    diagonal_backward_kernel<T, Gate><<<grid_size, kBlockThreads>>>(phi,
                                                                    lambda,
                                                                    phase_lookup,
                                                                   gradients,
                                                                   state_size,
                                                                   qubits,
                                                                   parameter_offset,
                                                                   gate_count);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
struct StatePair {
    Complex<T>* current;
    Complex<T>* scratch;
    void swap() { std::swap(current, scratch); }
};

template <typename T>
void launch_cnot(StatePair<T>* phi,
                 StatePair<T>* lambda,
                 uint64_t state_size,
                 int qubits,
                 bool adjoint,
                 int grid_size) {
    ring_cnot_permutation_kernel<T><<<grid_size, kBlockThreads>>>(
        phi->current,
        phi->scratch,
        lambda == nullptr ? nullptr : lambda->current,
        lambda == nullptr ? nullptr : lambda->scratch,
        state_size,
        qubits,
        adjoint);
    SAD_CUDA_CHECK(cudaGetLastError());
    phi->swap();
    if (lambda != nullptr) {
        lambda->swap();
    }
}

template <typename T>
struct ForwardCircuitContext {
    int qubits;
    uint64_t state_size;
    const RotationCoefficients<T>* rotation_coefficients;
    const Complex<T>* diagonal_lookup;
    const size_t* diagonal_lookup_offsets;
    const int* selected_maps;
    const int* target_counts;
    int phase_count;
    int multiprocessors;
    int ordinary_grid;
    StatePair<T>* phi;

    auto diagonal_lookup_at(size_t parameter_offset) const -> const Complex<T>* {
        return diagonal_lookup + diagonal_lookup_offsets[parameter_offset];
    }
};

template <typename T>
struct BackwardCircuitContext {
    int qubits;
    uint64_t state_size;
    const RotationCoefficients<T>* rotation_coefficients;
    const Complex<T>* diagonal_lookup;
    const size_t* diagonal_lookup_offsets;
    double* gradients;
    const int* selected_maps;
    const int* target_counts;
    int phase_count;
    int multiprocessors;
    int ordinary_grid;
    StatePair<T>* phi;
    StatePair<T>* lambda;

    auto diagonal_lookup_at(size_t parameter_offset) const -> const Complex<T>* {
        return diagonal_lookup + diagonal_lookup_offsets[parameter_offset];
    }
};

template <int Circuit, typename T>
struct CircuitExecutor;

}  // namespace sad

#include "circuits/ra_hea.cuh"
#include "circuits/rzz_hea.cuh"
#include "circuits/su2_hea.cuh"

namespace sad {

template <int Circuit, typename T>
auto build_circuit_diagonal_lookups(int qubits,
                                    int layers,
                                    const T* parameters,
                                    size_t parameter_count) -> DiagonalLookupData<T> {
    DiagonalLookupData<T> data;
    data.offsets_by_parameter.assign(parameter_count,
                                     std::numeric_limits<size_t>::max());
    CircuitExecutor<Circuit, T>::append_diagonal_lookups(
        qubits, layers, parameters, &data);
    return data;
}

template <typename T>
auto build_diagonal_lookups(int circuit,
                            int qubits,
                            int layers,
                            const T* parameters,
                            size_t parameter_count) -> DiagonalLookupData<T> {
    switch (circuit) {
        case SAD_CIRCUIT_RA_HEA:
            return build_circuit_diagonal_lookups<SAD_CIRCUIT_RA_HEA>(
                qubits, layers, parameters, parameter_count);
        case SAD_CIRCUIT_SU2_HEA:
            return build_circuit_diagonal_lookups<SAD_CIRCUIT_SU2_HEA>(
                qubits, layers, parameters, parameter_count);
        case SAD_CIRCUIT_RZZ_HEA:
            return build_circuit_diagonal_lookups<SAD_CIRCUIT_RZZ_HEA>(
                qubits, layers, parameters, parameter_count);
        default:
            throw std::invalid_argument("unknown circuit id");
    }
}

template <int Circuit>
inline size_t expected_parameter_count(int qubits, int layers) {
    return static_cast<size_t>(
               CircuitExecutor<Circuit, double>::kParametersPerQubitLayer) *
           qubits * layers;
}

inline void validate_inputs(int circuit,
                            int qubits,
                            int layers,
                            int steps,
                            int warmup_steps,
                            size_t parameter_count) {
    if (qubits < 2 || qubits > kMaxQubits) {
        throw std::invalid_argument("qubits must be in [2, 30]");
    }
    if (layers < 1 || steps < 1 || warmup_steps < 0) {
        throw std::invalid_argument(
            "layers/steps must be positive and warmup non-negative");
    }

    size_t expected = 0;
    switch (circuit) {
        case SAD_CIRCUIT_RA_HEA:
            CircuitExecutor<SAD_CIRCUIT_RA_HEA, double>::validate(qubits);
            expected = expected_parameter_count<SAD_CIRCUIT_RA_HEA>(qubits, layers);
            break;
        case SAD_CIRCUIT_SU2_HEA:
            CircuitExecutor<SAD_CIRCUIT_SU2_HEA, double>::validate(qubits);
            expected = expected_parameter_count<SAD_CIRCUIT_SU2_HEA>(qubits, layers);
            break;
        case SAD_CIRCUIT_RZZ_HEA:
            CircuitExecutor<SAD_CIRCUIT_RZZ_HEA, double>::validate(qubits);
            expected = expected_parameter_count<SAD_CIRCUIT_RZZ_HEA>(qubits, layers);
            break;
        default:
            throw std::invalid_argument("unknown circuit id");
    }
    if (parameter_count != expected) {
        throw std::invalid_argument("parameter_count mismatch: expected " +
                                    std::to_string(expected) + ", got " +
                                    std::to_string(parameter_count));
    }
}

template <typename T>
void run_forward(int circuit,
                 int qubits,
                 int layers,
                 uint64_t state_size,
                 const RotationCoefficients<T>* rotation_coefficients,
                 const Complex<T>* diagonal_lookup,
                 const size_t* diagonal_lookup_offsets,
                 const int* selected_maps,
                 const int* target_counts,
                 int phase_count,
                 int multiprocessors,
                 int ordinary_grid,
                 StatePair<T>* phi) {
    SAD_CUDA_CHECK(cudaMemset(phi->current, 0, state_size * sizeof(Complex<T>)));
    SAD_CUDA_CHECK(cudaMemset(phi->scratch, 0, state_size * sizeof(Complex<T>)));
    initialise_zero_state_kernel<T><<<1, 1>>>(phi->current);
    SAD_CUDA_CHECK(cudaGetLastError());

    const ForwardCircuitContext<T> context{
        qubits,
        state_size,
        rotation_coefficients,
        diagonal_lookup,
        diagonal_lookup_offsets,
        selected_maps,
        target_counts,
        phase_count,
        multiprocessors,
        ordinary_grid,
        phi};
    const auto run_layers = [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        for (int layer = 0; layer < layers; ++layer) {
            CircuitExecutor<Circuit, T>::forward_layer(layer, context);
        }
    };
    switch (circuit) {
        case SAD_CIRCUIT_RA_HEA:
            run_layers(std::integral_constant<int, SAD_CIRCUIT_RA_HEA>{});
            break;
        case SAD_CIRCUIT_SU2_HEA:
            run_layers(std::integral_constant<int, SAD_CIRCUIT_SU2_HEA>{});
            break;
        case SAD_CIRCUIT_RZZ_HEA:
            run_layers(std::integral_constant<int, SAD_CIRCUIT_RZZ_HEA>{});
            break;
        default:
            throw std::invalid_argument("unknown circuit id");
    }
}

template <typename T>
void run_backward(int circuit,
                  int qubits,
                  int layers,
                  uint64_t state_size,
                  const RotationCoefficients<T>* rotation_coefficients,
                  const Complex<T>* diagonal_lookup,
                  const size_t* diagonal_lookup_offsets,
                  double* gradients,
                  const int* selected_maps,
                  const int* target_counts,
                  int phase_count,
                  int multiprocessors,
                  int ordinary_grid,
                  StatePair<T>* phi,
                  StatePair<T>* lambda) {
    const BackwardCircuitContext<T> context{
        qubits,
        state_size,
        rotation_coefficients,
        diagonal_lookup,
        diagonal_lookup_offsets,
        gradients,
        selected_maps,
        target_counts,
        phase_count,
        multiprocessors,
        ordinary_grid,
        phi,
        lambda};
    const auto run_layers = [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        for (int layer = layers - 1; layer >= 0; --layer) {
            CircuitExecutor<Circuit, T>::backward_layer(layer, context);
        }
    };
    switch (circuit) {
        case SAD_CIRCUIT_RA_HEA:
            run_layers(std::integral_constant<int, SAD_CIRCUIT_RA_HEA>{});
            break;
        case SAD_CIRCUIT_SU2_HEA:
            run_layers(std::integral_constant<int, SAD_CIRCUIT_SU2_HEA>{});
            break;
        case SAD_CIRCUIT_RZZ_HEA:
            run_layers(std::integral_constant<int, SAD_CIRCUIT_RZZ_HEA>{});
            break;
        default:
            throw std::invalid_argument("unknown circuit id");
    }
}

template <typename T>
void run_typed(int circuit,
               int qubits,
               int layers,
               int steps,
               int warmup_steps,
               int device_id,
               const T* host_parameters,
               size_t parameter_count,
               double* out_energy,
               T* out_gradient,
               double* out_forward_times,
               double* out_hamiltonian_times,
               double* out_backward_times,
               SadMemoryInfo* out_memory) {
    validate_inputs(circuit, qubits, layers, steps, warmup_steps, parameter_count);
    SAD_CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp properties{};
    SAD_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));
    if (!properties.cooperativeLaunch) {
        throw std::runtime_error("selected CUDA device does not support cooperative launch");
    }

    const uint64_t state_size = 1ull << qubits;
    size_t free_before = 0;
    size_t total_memory = 0;
    SAD_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_memory));

    DeviceBuffer<Complex<T>> phi_a(state_size);
    DeviceBuffer<Complex<T>> phi_b(state_size);
    DeviceBuffer<Complex<T>> lambda_a(state_size);
    DeviceBuffer<Complex<T>> lambda_b(state_size);
    DeviceBuffer<RotationCoefficients<T>> device_rotation_coefficients(parameter_count);
    DeviceBuffer<Complex<T>> device_diagonal_lookup;
    DeviceBuffer<double> device_gradients(parameter_count);
    DeviceBuffer<double> device_energy(1);

    std::vector<int> selected_maps;
    std::vector<int> target_counts;
    auto diagonal_lookup_data = build_diagonal_lookups(
        circuit, qubits, layers, host_parameters, parameter_count);
    if (!diagonal_lookup_data.factors.empty()) {
        device_diagonal_lookup.allocate(diagonal_lookup_data.factors.size());
    }
    std::vector<RotationCoefficients<T>> host_rotation_coefficients(parameter_count);
    for (size_t parameter = 0; parameter < parameter_count; ++parameter) {
        const T half_angle = host_parameters[parameter] * static_cast<T>(0.5);
        host_rotation_coefficients[parameter] = {
            static_cast<T>(std::sin(half_angle)),
            static_cast<T>(std::cos(half_angle))};
    }
    build_phase_maps(qubits, &selected_maps, &target_counts);
    DeviceBuffer<int> device_selected_maps(selected_maps.size());
    DeviceBuffer<int> device_target_counts(target_counts.size());
    SAD_CUDA_CHECK(cudaMemcpy(device_rotation_coefficients.get(),
                              host_rotation_coefficients.data(),
                              parameter_count * sizeof(RotationCoefficients<T>),
                              cudaMemcpyHostToDevice));
    if (!diagonal_lookup_data.factors.empty()) {
        SAD_CUDA_CHECK(cudaMemcpy(device_diagonal_lookup.get(),
                                  diagonal_lookup_data.factors.data(),
                                  device_diagonal_lookup.bytes(),
                                  cudaMemcpyHostToDevice));
    }
    SAD_CUDA_CHECK(cudaMemcpy(device_selected_maps.get(),
                              selected_maps.data(),
                              selected_maps.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    SAD_CUDA_CHECK(cudaMemcpy(device_target_counts.get(),
                              target_counts.data(),
                              target_counts.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    size_t free_after_alloc = 0;
    size_t ignored_total = 0;
    SAD_CUDA_CHECK(cudaMemGetInfo(&free_after_alloc, &ignored_total));
    out_memory->state_vector_bytes = state_size * sizeof(Complex<T>);
    out_memory->total_workspace_bytes =
        4 * out_memory->state_vector_bytes + device_rotation_coefficients.bytes() +
        device_diagonal_lookup.bytes() + device_gradients.bytes() + device_energy.bytes() +
        device_selected_maps.bytes() + device_target_counts.bytes();
    out_memory->device_free_before_bytes = free_before;
    out_memory->device_free_after_alloc_bytes = free_after_alloc;
    out_memory->device_total_bytes = total_memory;

    PinnedBuffer<double> host_energy(1);
    PinnedBuffer<double> host_gradients(parameter_count);
    EventPair timer;
    const int ordinary_grid = ordinary_grid_size(state_size, properties.multiProcessorCount);
    const int phase_count = static_cast<int>(target_counts.size());

    auto run_once = [&](double* forward_time,
                        double* hamiltonian_time,
                        double* backward_time) {
        StatePair<T> phi{phi_a.get(), phi_b.get()};
        StatePair<T> lambda{lambda_a.get(), lambda_b.get()};

        const double measured_forward = timer.measure([&]() {
            run_forward(circuit,
                        qubits,
                        layers,
                        state_size,
                        device_rotation_coefficients.get(),
                        device_diagonal_lookup.get(),
                        diagonal_lookup_data.offsets_by_parameter.data(),
                        device_selected_maps.get(),
                        device_target_counts.get(),
                        phase_count,
                        properties.multiProcessorCount,
                        ordinary_grid,
                        &phi);
        });

        const double measured_hamiltonian = timer.measure([&]() {
            SAD_CUDA_CHECK(cudaMemset(device_energy.get(), 0, sizeof(double)));
            hamiltonian_kernel<T><<<ordinary_grid, kBlockThreads>>>(
                phi.current, lambda.current, state_size, qubits, device_energy.get());
            SAD_CUDA_CHECK(cudaGetLastError());
            SAD_CUDA_CHECK(cudaMemcpyAsync(host_energy.get(),
                                           device_energy.get(),
                                           sizeof(double),
                                           cudaMemcpyDeviceToHost));
        });

        const double measured_backward = timer.measure([&]() {
            SAD_CUDA_CHECK(
                cudaMemset(device_gradients.get(), 0, parameter_count * sizeof(double)));
            run_backward(circuit,
                         qubits,
                         layers,
                         state_size,
                         device_rotation_coefficients.get(),
                         device_diagonal_lookup.get(),
                         diagonal_lookup_data.offsets_by_parameter.data(),
                         device_gradients.get(),
                         device_selected_maps.get(),
                         device_target_counts.get(),
                         phase_count,
                         properties.multiProcessorCount,
                         ordinary_grid,
                         &phi,
                         &lambda);
            SAD_CUDA_CHECK(cudaMemcpyAsync(host_gradients.get(),
                                           device_gradients.get(),
                                           parameter_count * sizeof(double),
                                           cudaMemcpyDeviceToHost));
        });

        if (forward_time != nullptr) *forward_time = measured_forward;
        if (hamiltonian_time != nullptr) *hamiltonian_time = measured_hamiltonian;
        if (backward_time != nullptr) *backward_time = measured_backward;
    };

    for (int warmup = 0; warmup < warmup_steps; ++warmup) {
        run_once(nullptr, nullptr, nullptr);
    }
    for (int step = 0; step < steps; ++step) {
        run_once(out_forward_times + step,
                 out_hamiltonian_times + step,
                 out_backward_times + step);
    }

    *out_energy = host_energy.get()[0];
    for (size_t parameter = 0; parameter < parameter_count; ++parameter) {
        out_gradient[parameter] = static_cast<T>(host_gradients.get()[parameter]);
    }
}

inline void write_error(char* destination, size_t size, const std::string& message) {
    if (destination == nullptr || size == 0) return;
    std::snprintf(destination, size, "%s", message.c_str());
}

}  // namespace sad

extern "C" int sad_energy_and_grad(int precision,
                                    int circuit,
                                    int qubits,
                                    int layers,
                                    int steps,
                                    int warmup_steps,
                                    int device,
                                    const void* params,
                                    size_t parameter_count,
                                    double* out_energy,
                                    void* out_grad,
                                    double* out_forward_times_s,
                                    double* out_hamiltonian_times_s,
                                    double* out_backward_times_s,
                                    SadMemoryInfo* out_memory,
                                    char* error_message,
                                    size_t error_message_size) {
    try {
        if (params == nullptr || out_energy == nullptr || out_grad == nullptr ||
            out_forward_times_s == nullptr || out_hamiltonian_times_s == nullptr ||
            out_backward_times_s == nullptr || out_memory == nullptr) {
            throw std::invalid_argument("null pointer passed to sad_energy_and_grad");
        }
        if (precision == SAD_PRECISION_FLOAT32) {
            sad::run_typed<float>(circuit,
                                  qubits,
                                  layers,
                                  steps,
                                  warmup_steps,
                                  device,
                                  static_cast<const float*>(params),
                                  parameter_count,
                                  out_energy,
                                  static_cast<float*>(out_grad),
                                  out_forward_times_s,
                                  out_hamiltonian_times_s,
                                  out_backward_times_s,
                                  out_memory);
        } else if (precision == SAD_PRECISION_FLOAT64) {
            sad::run_typed<double>(circuit,
                                   qubits,
                                   layers,
                                   steps,
                                   warmup_steps,
                                   device,
                                   static_cast<const double*>(params),
                                   parameter_count,
                                   out_energy,
                                   static_cast<double*>(out_grad),
                                   out_forward_times_s,
                                   out_hamiltonian_times_s,
                                   out_backward_times_s,
                                   out_memory);
        } else {
            throw std::invalid_argument("unknown precision id");
        }
        sad::write_error(error_message, error_message_size, "");
        return 0;
    } catch (const std::exception& exception) {
        sad::write_error(error_message, error_message_size, exception.what());
        return 1;
    } catch (...) {
        sad::write_error(error_message, error_message_size, "unknown C++ exception");
        return 2;
    }
}

extern "C" const char* sad_version(void) {
    return "0.1.0";
}
