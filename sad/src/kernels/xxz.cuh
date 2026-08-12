#pragma once

#include "rotation_primitives.cuh"

#include <algorithm>

namespace sad {

template <typename T>
__device__ __forceinline__ Complex<T> apply_xxz_bond(
    Complex<T> self,
    Complex<T> partner,
    int z_eigenvalue,
    RotationCoefficients<T> x,
    RotationCoefficients<T> y,
    RotationCoefficients<T> z) {
    // RXX and RYY couple 00<->11 and 01<->10.  YY contributes -Z_i Z_j
    // times the XX-coupled partner, so the three commuting rotations can be
    // collapsed into one two-amplitude update and one diagonal phase.
    const T zz = static_cast<T>(z_eigenvalue);
    const T self_coefficient =
        x.cosine * y.cosine + x.sine * y.sine * zz;
    const T partner_coefficient =
        x.sine * y.cosine - x.cosine * y.sine * zz;
    const Complex<T> mixed{
        self_coefficient * self.real + partner_coefficient * partner.imag,
        self_coefficient * self.imag - partner_coefficient * partner.real};
    const Complex<T> rzz_factor{
        z.cosine, -z.sine * zz};
    return multiply(mixed, rzz_factor);
}

template <typename T>
__global__ void xxz_matching_forward_kernel(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* pair_counts,
    int phase_count) {
#if SAD_XXZ_PERSISTENT
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
        const int* selected = selected_maps + phase * kForwardTileBits;
        const int pair_count = pair_counts[phase];
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
            for (int pair = 0; pair < pair_count; ++pair) {
#pragma unroll
                for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kForwardRegisterBits)));
                    mailbox[local] = values[reg];
                }
                __syncthreads();
                const int first_slot = 2 * pair;
                const uint32_t pair_mask =
                    (1u << first_slot) | (1u << (first_slot + 1));
                const int edge = selected[first_slot];
                const auto x = coefficients[x_parameter_offset + edge];
                const auto y = coefficients[y_parameter_offset + edge];
                const auto z = coefficients[z_parameter_offset + edge];
#pragma unroll
                for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kForwardRegisterBits)));
                    if (local < (1u << tile_bits)) {
                        const int z_eigenvalue =
                            ((local >> first_slot) & 1u) ==
                                    ((local >> (first_slot + 1)) & 1u)
                                ? 1
                                : -1;
                        values[reg] = apply_xxz_bond(
                            values[reg],
                            mailbox[local ^ pair_mask],
                            z_eigenvalue,
                            x,
                            y,
                            z);
                    }
                }
                __syncthreads();
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
        if (phase + 1 < phase_count) {
#if SAD_XXZ_PERSISTENT
            grid.sync();
#endif
        }
    }
}

template <typename T>
void launch_xxz_matching_forward(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* pair_counts,
    int phase_count,
    int multiprocessors) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = xxz_matching_forward_kernel<T>;
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
    if constexpr (kXxzPersistent) {
        const int grid_size =
            static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
        void* arguments[] = {
            &state,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &qubits,
            &x_parameter_offset,
            &y_parameter_offset,
            &z_parameter_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&pair_counts),
            &phase_count};
        SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(grid_size),
            dim3(kForwardBlockThreads),
            arguments,
            shared_bytes));
    } else {
        const int ordinary_grid_size = static_cast<int>(tile_count);
        for (int phase = 0; phase < phase_count; ++phase) {
            kernel<<<ordinary_grid_size,
                     kForwardBlockThreads,
                     shared_bytes>>>(state,
                                     coefficients,
                                     qubits,
                                     x_parameter_offset,
                                     y_parameter_offset,
                                     z_parameter_offset,
                                     selected_maps +
                                         phase * kForwardTileBits,
                                     pair_counts + phase,
                                     1);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
__global__ void xxz_matching_backward_kernel(
    Complex<T>* phi_state,
    Complex<T>* lambda_state,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* pair_counts,
    int phase_count) {
#if SAD_XXZ_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    __shared__ Complex<T> mailbox[kTileAmplitudes];
    __shared__ double reduction[kBlockThreads];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int step = 0; step < phase_count; ++step) {
        const int phase = phase_count - 1 - step;
        const int* selected = selected_maps + phase * kTileBits;
        const int pair_count = pair_counts[phase];
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
                const bool active = local < (1u << tile_bits);
                const uint64_t index =
                    tile_base |
                    scatter_local_assignment<kTileBits>(
                        local, selected, tile_bits);
                phi[reg] = active ? phi_state[index] : make_complex<T>(0, 0);
                lambda[reg] =
                    active ? lambda_state[index] : make_complex<T>(0, 0);
            }
            for (int pair = pair_count - 1; pair >= 0; --pair) {
                const int first_slot = 2 * pair;
                const uint32_t pair_mask =
                    (1u << first_slot) | (1u << (first_slot + 1));
                const int edge = selected[first_slot];
#pragma unroll
                for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kRegisterBits)));
                    mailbox[local] = phi[reg];
                }
                __syncthreads();
                double x_overlap = 0.0;
                double y_overlap = 0.0;
                double z_overlap = 0.0;
#pragma unroll
                for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kRegisterBits)));
                    if (local < (1u << tile_bits)) {
                        const int z_eigenvalue =
                            ((local >> first_slot) & 1u) ==
                                    ((local >> (first_slot + 1)) & 1u)
                                ? 1
                                : -1;
                        const Complex<T> partner = mailbox[local ^ pair_mask];
                        x_overlap += imag_conjugate_product(lambda[reg], partner);
                        y_overlap += imag_conjugate_product(
                            lambda[reg],
                            scale(partner, static_cast<T>(-z_eigenvalue)));
                        z_overlap += imag_conjugate_product(
                            lambda[reg],
                            scale(phi[reg], static_cast<T>(z_eigenvalue)));
                    }
                }
                block_atomic_sum(x_overlap,
                                 reduction,
                                 gradients + x_parameter_offset + edge);
                block_atomic_sum(y_overlap,
                                 reduction,
                                 gradients + y_parameter_offset + edge);
                block_atomic_sum(z_overlap,
                                 reduction,
                                 gradients + z_parameter_offset + edge);
                RotationCoefficients<T> x =
                    coefficients[x_parameter_offset + edge];
                RotationCoefficients<T> y =
                    coefficients[y_parameter_offset + edge];
                RotationCoefficients<T> z =
                    coefficients[z_parameter_offset + edge];
                x.sine = -x.sine;
                y.sine = -y.sine;
                z.sine = -z.sine;
#pragma unroll
                for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kRegisterBits)));
                    if (local < (1u << tile_bits)) {
                        const int z_eigenvalue =
                            ((local >> first_slot) & 1u) ==
                                    ((local >> (first_slot + 1)) & 1u)
                                ? 1
                                : -1;
                        phi[reg] = apply_xxz_bond(
                            phi[reg],
                            mailbox[local ^ pair_mask],
                            z_eigenvalue,
                            x,
                            y,
                            z);
                    }
                }
                __syncthreads();
#pragma unroll
                for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kRegisterBits)));
                    mailbox[local] = lambda[reg];
                }
                __syncthreads();
#pragma unroll
                for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                    const uint32_t local = static_cast<uint32_t>(
                        lane | (reg << kLaneBits) |
                        (warp << (kLaneBits + kRegisterBits)));
                    if (local < (1u << tile_bits)) {
                        const int z_eigenvalue =
                            ((local >> first_slot) & 1u) ==
                                    ((local >> (first_slot + 1)) & 1u)
                                ? 1
                                : -1;
                        lambda[reg] = apply_xxz_bond(
                            lambda[reg],
                            mailbox[local ^ pair_mask],
                            z_eigenvalue,
                            x,
                            y,
                            z);
                    }
                }
                __syncthreads();
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
                    phi_state[index] = phi[reg];
                    lambda_state[index] = lambda[reg];
                }
            }
            __syncthreads();
        }
        if (step + 1 < phase_count) {
#if SAD_XXZ_PERSISTENT
            grid.sync();
#endif
        }
    }
}

template <typename T>
void launch_xxz_matching_backward(
    Complex<T>* phi,
    Complex<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* pair_counts,
    int phase_count,
    int multiprocessors) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel = xxz_matching_backward_kernel<T>;
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, kernel, kBlockThreads, 0));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    if constexpr (kXxzPersistent) {
        const int grid_size =
            static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
        void* arguments[] = {
            &phi,
            &lambda,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &gradients,
            &qubits,
            &x_parameter_offset,
            &y_parameter_offset,
            &z_parameter_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&pair_counts),
            &phase_count};
        SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(grid_size),
            dim3(kBlockThreads),
            arguments));
    } else {
        const int ordinary_grid_size = static_cast<int>(tile_count);
        for (int step = 0; step < phase_count; ++step) {
            const int phase = phase_count - 1 - step;
            kernel<<<ordinary_grid_size, kBlockThreads>>>(
                phi,
                lambda,
                coefficients,
                gradients,
                qubits,
                x_parameter_offset,
                y_parameter_offset,
                z_parameter_offset,
                selected_maps + phase * kTileBits,
                pair_counts + phase,
                1);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T>
__global__ void xxz_cross_matching_forward_kernel(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected,
    const int* bond_offsets,
    const int* slot_pairs,
    const int* edges) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* mailbox = reinterpret_cast<Complex<T>*>(dynamic_shared);
    __shared__ uint64_t tile_base;
    const int tile_bits = min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
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
        for (int bond = bond_offsets[0]; bond < bond_offsets[1]; ++bond) {
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                mailbox[local] = values[reg];
            }
            __syncthreads();
            const int first_slot = slot_pairs[2 * bond];
            const int second_slot = slot_pairs[2 * bond + 1];
            const uint32_t pair_mask =
                (1u << first_slot) | (1u << second_slot);
            const int edge = edges[bond];
            const auto x = coefficients[x_parameter_offset + edge];
            const auto y = coefficients[y_parameter_offset + edge];
            const auto z = coefficients[z_parameter_offset + edge];
#pragma unroll
            for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kForwardRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    values[reg] = apply_xxz_bond(
                        values[reg],
                        mailbox[local ^ pair_mask],
                        z_eigenvalue,
                        x,
                        y,
                        z);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int reg = 0; reg < kForwardRegisterAmplitudes; ++reg) {
            const uint32_t local = static_cast<uint32_t>(
                lane | (reg << kLaneBits) |
                (warp << (kLaneBits + kForwardRegisterBits)));
            if (local < (1u << tile_bits)) {
                const uint64_t index =
                    tile_base | scatter_local_assignment<kForwardTileBits>(
                                    local, selected, tile_bits);
                state[index] = values[reg];
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_xxz_cross_matching_forward(
    Complex<T>* state,
    const RotationCoefficients<T>* coefficients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* bond_offsets,
    const int* slot_pairs,
    const int* edges,
    int phase_count) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    constexpr size_t shared_bytes =
        kForwardTileAmplitudes * sizeof(Complex<T>);
    const auto kernel = xxz_cross_matching_forward_kernel<T>;
    SAD_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    for (int phase = 0; phase < phase_count; ++phase) {
        kernel<<<static_cast<int>(tile_count),
                 kForwardBlockThreads,
                 shared_bytes>>>(state,
                                 coefficients,
                                 qubits,
                                 x_parameter_offset,
                                 y_parameter_offset,
                                 z_parameter_offset,
                                 selected_maps + phase * kForwardTileBits,
                                 bond_offsets + phase,
                                 slot_pairs,
                                 edges);
    }
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void xxz_cross_matching_backward_kernel(
    Complex<T>* phi_state,
    Complex<T>* lambda_state,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected,
    const int* bond_offsets,
    const int* slot_pairs,
    const int* edges) {
    __shared__ Complex<T> mailbox[kTileAmplitudes];
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
            const bool active = local < (1u << tile_bits);
            const uint64_t index =
                tile_base | scatter_local_assignment<kTileBits>(
                                local, selected, tile_bits);
            phi[reg] = active ? phi_state[index] : make_complex<T>(0, 0);
            lambda[reg] =
                active ? lambda_state[index] : make_complex<T>(0, 0);
        }
        for (int bond = bond_offsets[1] - 1; bond >= bond_offsets[0]; --bond) {
            const int first_slot = slot_pairs[2 * bond];
            const int second_slot = slot_pairs[2 * bond + 1];
            const uint32_t pair_mask =
                (1u << first_slot) | (1u << second_slot);
            const int edge = edges[bond];
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                mailbox[local] = phi[reg];
            }
            __syncthreads();
            double x_overlap = 0.0;
            double y_overlap = 0.0;
            double z_overlap = 0.0;
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    const Complex<T> partner = mailbox[local ^ pair_mask];
                    x_overlap +=
                        imag_conjugate_product(lambda[reg], partner);
                    y_overlap += imag_conjugate_product(
                        lambda[reg],
                        scale(partner, static_cast<T>(-z_eigenvalue)));
                    z_overlap += imag_conjugate_product(
                        lambda[reg],
                        scale(phi[reg], static_cast<T>(z_eigenvalue)));
                }
            }
            block_atomic_sum(x_overlap,
                             reduction,
                             gradients + x_parameter_offset + edge);
            block_atomic_sum(y_overlap,
                             reduction,
                             gradients + y_parameter_offset + edge);
            block_atomic_sum(z_overlap,
                             reduction,
                             gradients + z_parameter_offset + edge);
            RotationCoefficients<T> x = coefficients[x_parameter_offset + edge];
            RotationCoefficients<T> y = coefficients[y_parameter_offset + edge];
            RotationCoefficients<T> z = coefficients[z_parameter_offset + edge];
            x.sine = -x.sine;
            y.sine = -y.sine;
            z.sine = -z.sine;
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    phi[reg] = apply_xxz_bond(phi[reg],
                                               mailbox[local ^ pair_mask],
                                               z_eigenvalue,
                                               x,
                                               y,
                                               z);
                }
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                mailbox[local] = lambda[reg];
            }
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
                const uint32_t local = static_cast<uint32_t>(
                    lane | (reg << kLaneBits) |
                    (warp << (kLaneBits + kRegisterBits)));
                if (local < (1u << tile_bits)) {
                    const int z_eigenvalue =
                        ((local >> first_slot) & 1u) ==
                                ((local >> second_slot) & 1u)
                            ? 1
                            : -1;
                    lambda[reg] = apply_xxz_bond(lambda[reg],
                                                  mailbox[local ^ pair_mask],
                                                  z_eigenvalue,
                                                  x,
                                                  y,
                                                  z);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int reg = 0; reg < kRegisterAmplitudes; ++reg) {
            const uint32_t local = static_cast<uint32_t>(
                lane | (reg << kLaneBits) |
                (warp << (kLaneBits + kRegisterBits)));
            if (local < (1u << tile_bits)) {
                const uint64_t index =
                    tile_base | scatter_local_assignment<kTileBits>(
                                    local, selected, tile_bits);
                phi_state[index] = phi[reg];
                lambda_state[index] = lambda[reg];
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_xxz_cross_matching_backward(
    Complex<T>* phi,
    Complex<T>* lambda,
    const RotationCoefficients<T>* coefficients,
    double* gradients,
    int qubits,
    int x_parameter_offset,
    int y_parameter_offset,
    int z_parameter_offset,
    const int* selected_maps,
    const int* bond_offsets,
    const int* slot_pairs,
    const int* edges,
    int phase_count) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    for (int step = 0; step < phase_count; ++step) {
        const int phase = phase_count - 1 - step;
        xxz_cross_matching_backward_kernel<T>
            <<<static_cast<int>(tile_count), kBlockThreads>>>(
                phi,
                lambda,
                coefficients,
                gradients,
                qubits,
                x_parameter_offset,
                y_parameter_offset,
                z_parameter_offset,
                selected_maps + phase * kTileBits,
                bond_offsets + phase,
                slot_pairs,
                edges);
    }
    SAD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sad
