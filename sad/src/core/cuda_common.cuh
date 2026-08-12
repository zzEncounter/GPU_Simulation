#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace cg = cooperative_groups;

namespace sad {

#ifndef SAD_BLOCK_THREADS
#define SAD_BLOCK_THREADS 128
#endif
#ifndef SAD_REGISTER_BITS
#define SAD_REGISTER_BITS 2
#endif
#ifndef SAD_FORWARD_BLOCK_THREADS
#define SAD_FORWARD_BLOCK_THREADS SAD_BLOCK_THREADS
#endif
#ifndef SAD_FORWARD_REGISTER_BITS
#define SAD_FORWARD_REGISTER_BITS SAD_REGISTER_BITS
#endif
#ifndef SAD_FIXED_LOW_LANES
#define SAD_FIXED_LOW_LANES 0
#endif
#ifndef SAD_FORWARD_FIXED_LOW_LANES
#define SAD_FORWARD_FIXED_LOW_LANES SAD_FIXED_LOW_LANES
#endif
#ifndef SAD_ALTERNATE_PHASES
#define SAD_ALTERNATE_PHASES 0
#endif
#ifndef SAD_ORDINARY_BLOCK_THREADS
#define SAD_ORDINARY_BLOCK_THREADS 128
#endif
#ifndef SAD_DIAGONAL_BLOCK_THREADS
#define SAD_DIAGONAL_BLOCK_THREADS 64
#endif
#ifndef SAD_SHARED_DIAGONAL_BLOCK_THREADS
#define SAD_SHARED_DIAGONAL_BLOCK_THREADS 128
#endif
#ifndef SAD_DIAGONAL_LOOKUP_BITS
#define SAD_DIAGONAL_LOOKUP_BITS 8
#endif
#ifndef SAD_MAILBOX_CHUNKS
#define SAD_MAILBOX_CHUNKS 1
#endif
#ifndef SAD_RY_SCALAR_MAILBOX
#define SAD_RY_SCALAR_MAILBOX 0
#endif
#ifndef SAD_ROTATION_PERSISTENT
#define SAD_ROTATION_PERSISTENT 0
#endif
#ifndef SAD_LEGACY_BLOCK_REDUCTION
#define SAD_LEGACY_BLOCK_REDUCTION 1
#endif
#ifndef SAD_ROTATION_WARP_ATOMIC
#define SAD_ROTATION_WARP_ATOMIC 0
#endif
#ifndef SAD_DIAGONAL_WARP_ATOMIC
#define SAD_DIAGONAL_WARP_ATOMIC 0
#endif
#ifndef SAD_CNOT_FORWARD_SCATTER
#define SAD_CNOT_FORWARD_SCATTER 1
#endif
#ifndef SAD_XXZ_PERSISTENT
#define SAD_XXZ_PERSISTENT 0
#endif
#ifndef SAD_REAL_PERSISTENT
#define SAD_REAL_PERSISTENT 0
#endif
#ifndef SAD_PHASED_RY_PERSISTENT
#define SAD_PHASED_RY_PERSISTENT 0
#endif
#ifndef SAD_SU2_PHASED_BACKWARD
#define SAD_SU2_PHASED_BACKWARD 0
#endif
#ifndef SAD_XXZ_CROSS_MATCHING
#define SAD_XXZ_CROSS_MATCHING 1
#endif
#ifndef SAD_QAOA_COMPACT_LOOKUP
#define SAD_QAOA_COMPACT_LOOKUP 1
#endif
#ifndef SAD_QAOA_FUSED_BACKWARD
#define SAD_QAOA_FUSED_BACKWARD 1
#endif

constexpr int exact_log2(int value) {
    return value == 1 ? 0 : 1 + exact_log2(value / 2);
}

constexpr int kBlockThreads = SAD_BLOCK_THREADS;
constexpr int kWarpsPerBlock = kBlockThreads / 32;
constexpr int kRegisterBits = SAD_REGISTER_BITS;
constexpr int kRegisterAmplitudes = 1 << kRegisterBits;
constexpr int kLaneBits = 5;
constexpr int kWarpBits = exact_log2(kWarpsPerBlock);
constexpr int kTileBits = kLaneBits + kRegisterBits + kWarpBits;
constexpr int kTileAmplitudes = 1 << kTileBits;
constexpr int kForwardBlockThreads = SAD_FORWARD_BLOCK_THREADS;
constexpr int kForwardWarpsPerBlock = kForwardBlockThreads / 32;
constexpr int kForwardRegisterBits = SAD_FORWARD_REGISTER_BITS;
constexpr int kForwardRegisterAmplitudes = 1 << kForwardRegisterBits;
constexpr int kForwardWarpBits = exact_log2(kForwardWarpsPerBlock);
constexpr int kForwardTileBits =
    kLaneBits + kForwardRegisterBits + kForwardWarpBits;
constexpr int kForwardTileAmplitudes = 1 << kForwardTileBits;
constexpr int kOrdinaryBlockThreads = SAD_ORDINARY_BLOCK_THREADS;
constexpr int kOrdinaryWarpsPerBlock = kOrdinaryBlockThreads / 32;
constexpr int kDiagonalBlockThreads = SAD_DIAGONAL_BLOCK_THREADS;
constexpr int kDiagonalWarpsPerBlock = kDiagonalBlockThreads / 32;
constexpr int kSharedDiagonalBlockThreads = SAD_SHARED_DIAGONAL_BLOCK_THREADS;
constexpr bool kFixedLowLanes = SAD_FIXED_LOW_LANES != 0;
constexpr bool kForwardFixedLowLanes =
    SAD_FORWARD_FIXED_LOW_LANES != 0;
constexpr bool kAlternatePhases = SAD_ALTERNATE_PHASES != 0;

__host__ __device__ constexpr int target_phase_for_qubit(
    int qubit,
    int tile_bits,
    bool fixed_low_lanes) {
    if (!fixed_low_lanes || qubit < tile_bits) {
        return qubit / tile_bits;
    }
    return 1 + (qubit - tile_bits) / (tile_bits - kLaneBits);
}
constexpr int kMaxQubits = 30;
constexpr int kDiagonalLookupBits = SAD_DIAGONAL_LOOKUP_BITS;
constexpr int kDiagonalLookupSize = 1 << kDiagonalLookupBits;
constexpr int kMailboxChunks = SAD_MAILBOX_CHUNKS;
constexpr bool kRyScalarMailbox = SAD_RY_SCALAR_MAILBOX != 0;
constexpr bool kRotationPersistent = SAD_ROTATION_PERSISTENT != 0;
constexpr bool kLegacyBlockReduction = SAD_LEGACY_BLOCK_REDUCTION != 0;
constexpr bool kRotationWarpAtomic = SAD_ROTATION_WARP_ATOMIC != 0;
constexpr bool kDiagonalWarpAtomic = SAD_DIAGONAL_WARP_ATOMIC != 0;
constexpr bool kCnotForwardScatter = SAD_CNOT_FORWARD_SCATTER != 0;
constexpr bool kXxzPersistent = SAD_XXZ_PERSISTENT != 0;
constexpr bool kRealPersistent = SAD_REAL_PERSISTENT != 0;
constexpr bool kPhasedRyPersistent = SAD_PHASED_RY_PERSISTENT != 0;
constexpr bool kSu2PhasedBackward = SAD_SU2_PHASED_BACKWARD != 0;
constexpr bool kXxzCrossMatching = SAD_XXZ_CROSS_MATCHING != 0;
constexpr bool kQaoaCompactLookup = SAD_QAOA_COMPACT_LOOKUP != 0;
constexpr bool kQaoaFusedBackward = SAD_QAOA_FUSED_BACKWARD != 0;

static_assert(kBlockThreads == 32 || kBlockThreads == 64 ||
              kBlockThreads == 128 ||
              kBlockThreads == 256 || kBlockThreads == 512);
static_assert(kWarpsPerBlock == (1 << kWarpBits));
static_assert(kRegisterBits >= 2 && kRegisterBits <= 6);
static_assert(kTileBits <= 12);
static_assert(kForwardBlockThreads == 32 || kForwardBlockThreads == 64 ||
              kForwardBlockThreads == 128 ||
              kForwardBlockThreads == 256 || kForwardBlockThreads == 512);
static_assert(kForwardWarpsPerBlock == (1 << kForwardWarpBits));
static_assert(kForwardRegisterBits >= 2 && kForwardRegisterBits <= 6);
static_assert(kForwardTileBits <= 12);
static_assert(kOrdinaryBlockThreads == 64 ||
              kOrdinaryBlockThreads == 128 ||
              kOrdinaryBlockThreads == 256 ||
              kOrdinaryBlockThreads == 512);
static_assert(kDiagonalBlockThreads == 64 ||
              kDiagonalBlockThreads == 128 ||
              kDiagonalBlockThreads == 256 ||
              kDiagonalBlockThreads == 512);
static_assert(kSharedDiagonalBlockThreads == 64 ||
              kSharedDiagonalBlockThreads == 128 ||
              kSharedDiagonalBlockThreads == 256 ||
              kSharedDiagonalBlockThreads == 512);
static_assert(kDiagonalLookupBits >= 1 && kDiagonalLookupBits <= 12);
static_assert(kMailboxChunks == 1 || kMailboxChunks == 2 ||
              kMailboxChunks == 4 || kMailboxChunks == 8 ||
              kMailboxChunks == 16 || kMailboxChunks == 32 ||
              kMailboxChunks == 64);
static_assert(kRegisterAmplitudes % kMailboxChunks == 0);
static_assert(kForwardRegisterAmplitudes % kMailboxChunks == 0);

enum class NonDiagonalGate : int { RX = 0, RY = 1 };
enum class DiagonalGate : int { RZ = 0, RZZ_EVEN = 1, RZZ_ODD = 2 };
enum class FusedDiagonalMode : int {
    NONE = 0,
    RZ = 1,
    RZZ = 2,
    RZ_RZZ = 3,
};

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
        T signed_sine;
        if constexpr (sizeof(T) == sizeof(double)) {
            const auto bits = static_cast<unsigned long long>(
                __double_as_longlong(static_cast<double>(coefficients.sine)));
            signed_sine = static_cast<T>(__longlong_as_double(
                bits ^ (static_cast<unsigned long long>(bit ^ 1) << 63)));
        } else {
            const int bits = __float_as_int(
                static_cast<float>(coefficients.sine));
            signed_sine = static_cast<T>(
                __int_as_float(bits ^ ((bit ^ 1) << 31)));
        }
        return {coefficients.cosine * self.real + signed_sine * partner.real,
                coefficients.cosine * self.imag + signed_sine * partner.imag};
    } else {
        // RX: c*self - i*s*partner.
        return {coefficients.cosine * self.real + coefficients.sine * partner.imag,
                coefficients.cosine * self.imag - coefficients.sine * partner.real};
    }
}

template <typename T, NonDiagonalGate Gate>
__device__ inline double generator_overlap(Complex<T> bra,
                                           Complex<T> partner,
                                           int bit) {
    if constexpr (Gate == NonDiagonalGate::RX) {
        return imag_conjugate_product(bra, partner);
    } else {
        const double value = real_conjugate_product(bra, partner);
        return bit ? value : -value;
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
    if constexpr (kLegacyBlockReduction) {
        reduction[tid] = value;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(destination, reduction[0]);
        }
        __syncthreads();
    } else {
        const int lane = tid & 31;
        const int warp = tid >> 5;
        value = warp_sum(value);
        if (lane == 0) {
            reduction[warp] = value;
        }
        __syncthreads();
        if (warp == 0) {
            value = lane < blockDim.x / 32 ? reduction[lane] : 0.0;
            value = warp_sum(value);
            if (lane == 0) {
                atomicAdd(destination, value);
            }
        }
        __syncthreads();
    }
}


}  // namespace sad
