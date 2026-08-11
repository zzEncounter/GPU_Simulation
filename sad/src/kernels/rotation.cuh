#pragma once

#include "rotation_primitives.cuh"

#include <algorithm>

namespace sad {

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
__global__ void non_diagonal_forward_kernel(Complex<T>* state,
                                            const RotationCoefficients<T>* coefficients,
                                            int qubits,
                                            int parameter_offset,
                                            const int* selected_maps,
                                            const int* target_masks,
                                            int phase_count,
                                            bool reverse_phases) {
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
        const int* selected = selected_maps + phase * kForwardTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
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
                    tile_base | scatter_local_assignment<kForwardTileBits>(
                                    local, selected, tile_bits);
                values[reg] = active ? state[index] : make_complex<T>(0, 0);
            }

            apply_phase_forward<T, Gate, SharedParameter>(values,
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
                    state[index] = values[reg];
                }
            }
            __syncthreads();
        }
#if SAD_ROTATION_PERSISTENT
        grid.sync();
#endif
    }
}

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
__global__ void non_diagonal_backward_gradient_kernel(Complex<T>* phi_state,
                                                      Complex<T>* lambda_state,
                                                      const RotationCoefficients<T>* coefficients,
                                                      double* gradient_accumulator,
                                                      int qubits,
                                                      int parameter_offset,
                                                      const int* selected_maps,
                                                      const int* target_masks,
                                                      int phase_count,
                                                      bool reverse_phases) {
#if SAD_ROTATION_PERSISTENT
    cg::grid_group grid = cg::this_grid();
#endif
    constexpr size_t kMailboxBytes =
        backward_rotation_mailbox_bytes<T, Gate>();
    __shared__ __align__(16)
        unsigned char mailbox[kMailboxBytes == 0 ? 16 : kMailboxBytes];
    __shared__ double reduction[kBlockThreads];
    __shared__ uint64_t tile_base;

    const int tile_bits = min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int step = 0; step < phase_count; ++step) {
        const int phase = reverse_phases ? step : phase_count - 1 - step;
        const int* selected = selected_maps + phase * kTileBits;
        for (uint64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
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
                lambda[reg] = active ? lambda_state[index] : make_complex<T>(0, 0);
            }

            const bool thread_active =
                (lane | (warp << (kLaneBits + kRegisterBits))) <
                (1u << tile_bits);
            apply_phase_backward<T, Gate, SharedParameter>(phi,
                                          lambda,
                                          coefficients,
                                          gradient_accumulator,
                                          parameter_offset,
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
                        tile_base | scatter_local_assignment<kTileBits>(
                                        local, selected, tile_bits);
                    phi_state[index] = phi[reg];
                    lambda_state[index] = lambda[reg];
                }
            }
            __syncthreads();
        }
#if SAD_ROTATION_PERSISTENT
        grid.sync();
#endif
    }
}

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
void launch_non_diagonal_forward(Complex<T>* state,
                                 const RotationCoefficients<T>* coefficients,
                                 int qubits,
                                 int parameter_offset,
                                 const int* selected_maps,
                                 const int* target_masks,
                                 int phase_count,
                                 int multiprocessors,
                                 bool reverse_phases = false) {
    const int tile_bits = std::min(qubits, kForwardTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel =
        non_diagonal_forward_kernel<T, Gate, SharedParameter>;
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
    if constexpr (kRotationPersistent) {
        void* arguments[] = {
            &state,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &qubits,
            &parameter_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &reverse_phases};
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
            kernel<<<ordinary_grid_size,
                     kForwardBlockThreads,
                     shared_bytes>>>(state,
                                     coefficients,
                                     qubits,
                                     parameter_offset,
                                     selected_maps + phase * kForwardTileBits,
                                     target_masks + phase,
                                     1,
                                     false);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename T, NonDiagonalGate Gate, bool SharedParameter = false>
void launch_non_diagonal_backward(Complex<T>* phi,
                                  Complex<T>* lambda,
                                  const RotationCoefficients<T>* coefficients,
                                  double* gradients,
                                  int qubits,
                                  int parameter_offset,
                                  const int* selected_maps,
                                  const int* target_masks,
                                  int phase_count,
                                  int multiprocessors,
                                  bool reverse_phases = false) {
    const int tile_bits = std::min(qubits, kTileBits);
    const uint64_t tile_count = 1ull << (qubits - tile_bits);
    const auto kernel =
        non_diagonal_backward_gradient_kernel<T, Gate, SharedParameter>;
    int blocks_per_multiprocessor = 0;
    SAD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, kernel, kBlockThreads, 0));
    const uint64_t resident_blocks =
        static_cast<uint64_t>(blocks_per_multiprocessor) * multiprocessors;
    const int grid_size =
        static_cast<int>(std::min<uint64_t>(tile_count, resident_blocks));
    if constexpr (kRotationPersistent) {
        void* arguments[] = {
            &phi,
            &lambda,
            const_cast<RotationCoefficients<T>**>(&coefficients),
            &gradients,
            &qubits,
            &parameter_offset,
            const_cast<int**>(&selected_maps),
            const_cast<int**>(&target_masks),
            &phase_count,
            &reverse_phases};
        SAD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(grid_size),
            dim3(kBlockThreads),
            arguments));
    } else {
        const int ordinary_grid_size = static_cast<int>(tile_count);
        for (int step = 0; step < phase_count; ++step) {
            const int phase = reverse_phases ? step : phase_count - 1 - step;
            kernel<<<ordinary_grid_size, kBlockThreads>>>(
                phi,
                lambda,
                coefficients,
                gradients,
                qubits,
                parameter_offset,
                selected_maps + phase * kTileBits,
                target_masks + phase,
                1,
                false);
        }
        SAD_CUDA_CHECK(cudaGetLastError());
    }
}


}  // namespace sad
