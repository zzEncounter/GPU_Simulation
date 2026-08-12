#pragma once

#include "../core/cuda_common.cuh"

#include <algorithm>

namespace sad {

__device__ __forceinline__ int ring_domain_wall_count(uint64_t index,
                                                       int qubits) {
    const uint64_t mask = (1ull << qubits) - 1ull;
    const uint64_t rotated = ((index << 1) & mask) |
                             (index >> (qubits - 1));
    return __popcll(index ^ rotated);
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
__device__ __forceinline__ Complex<T> diagonal_lookup_factor(
    uint64_t index,
    const Complex<T>* phase_lookup,
    int qubits,
    int gate_count) {
    Complex<T> factor =
        make_complex<T>(static_cast<T>(1), static_cast<T>(0));
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
        factor = multiply(
            factor, phase_lookup[chunk * kDiagonalLookupSize + code]);
    }
    return factor;
}

template <typename T>
__device__ __forceinline__ Complex<T> shared_ring_rzz_factor(
    uint64_t index,
    const Complex<T>* phase_lookup,
    int qubits) {
    if constexpr (kQaoaCompactLookup) {
        return phase_lookup[ring_domain_wall_count(index, qubits) / 2];
    } else {
        Complex<T> factor =
            diagonal_lookup_factor<T, DiagonalGate::RZZ_EVEN>(
                index, phase_lookup, qubits, qubits / 2);
        return multiply(
            factor,
            diagonal_lookup_factor<T, DiagonalGate::RZZ_ODD>(
                index, phase_lookup, qubits, qubits / 2));
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
        const Complex<T> factor = diagonal_lookup_factor<T, Gate>(
            index, phase_lookup, qubits, gate_count);
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
    __shared__ double overlaps[kMaxQubits * kDiagonalBlockThreads];
    __shared__ double warp_partials[
        kMaxQubits * kDiagonalWarpsPerBlock];
    const int tid = threadIdx.x;
    for (int gate = 0; gate < gate_count; ++gate) {
        overlaps[gate * kDiagonalBlockThreads + tid] = 0.0;
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
                    overlaps[gate * kDiagonalBlockThreads + tid] +=
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
        const double sum =
            warp_sum(overlaps[gate * kDiagonalBlockThreads + tid]);
        if constexpr (kDiagonalWarpAtomic) {
            if (lane == 0) {
                atomicAdd(gradient_accumulator + parameter_offset + gate,
                          sum);
            }
        } else if (lane == 0) {
            warp_partials[gate * kDiagonalWarpsPerBlock + warp] = sum;
        }
    }
    if constexpr (kDiagonalWarpAtomic) {
        return;
    }
    __syncthreads();
    if (warp == 0) {
        for (int gate = 0; gate < gate_count; ++gate) {
            double value = lane < kDiagonalWarpsPerBlock
                               ? warp_partials[
                                     gate * kDiagonalWarpsPerBlock + lane]
                               : 0.0;
            value = warp_sum(value);
            if (lane == 0) {
                atomicAdd(gradient_accumulator + parameter_offset + gate, value);
            }
        }
    }
}

template <typename T, DiagonalGate Gate>
void launch_diagonal_forward(Complex<T>* state,
                             const Complex<T>* phase_lookup,
                             uint64_t state_size,
                             int qubits,
                             int gate_count,
                             int grid_size) {
    diagonal_forward_kernel<T, Gate><<<grid_size, kDiagonalBlockThreads>>>(
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
    diagonal_backward_kernel<T, Gate><<<grid_size, kDiagonalBlockThreads>>>(phi,
                                                                    lambda,
                                                                    phase_lookup,
                                                                   gradients,
                                                                   state_size,
                                                                   qubits,
                                                                   parameter_offset,
                                                                   gate_count);
    SAD_CUDA_CHECK(cudaGetLastError());
}

constexpr int kCombinedDiagonalThreads = 64;
constexpr int kCombinedDiagonalWarps = kCombinedDiagonalThreads / 32;

template <typename T>
__global__ void rz_rzz_backward_kernel(
    Complex<T>* phi_state,
    Complex<T>* lambda_state,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    double* gradients,
    uint64_t state_size,
    int qubits,
    int rz_parameter_offset,
    int rzz_even_parameter_offset,
    int rzz_odd_parameter_offset) {
    __shared__ double overlaps[
        2 * kMaxQubits * kCombinedDiagonalThreads];
    __shared__ double warp_partials[
        2 * kMaxQubits * kCombinedDiagonalWarps];
    const int tid = threadIdx.x;
    const int generator_count = 2 * qubits;
    for (int generator = 0; generator < generator_count; ++generator) {
        overlaps[generator * kCombinedDiagonalThreads + tid] = 0.0;
    }
    __syncthreads();

    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const Complex<T> phi = phi_state[index];
        const Complex<T> lambda = lambda_state[index];
        const double base_overlap = imag_conjugate_product(lambda, phi);
        for (int qubit = 0; qubit < qubits; ++qubit) {
            overlaps[qubit * kCombinedDiagonalThreads + tid] +=
                base_overlap * static_cast<double>(
                    diagonal_eigenvalue<DiagonalGate::RZ>(index,
                                                          qubit,
                                                          qubits));
        }
        for (int left = 0; left < qubits; ++left) {
            const bool even = (left & 1) == 0;
            const int eigenvalue =
                even ? diagonal_eigenvalue<DiagonalGate::RZZ_EVEN>(
                           index, left / 2, qubits)
                     : diagonal_eigenvalue<DiagonalGate::RZZ_ODD>(
                           index, left / 2, qubits);
            overlaps[(qubits + left) * kCombinedDiagonalThreads + tid] +=
                base_overlap * static_cast<double>(eigenvalue);
        }
        Complex<T> factor = diagonal_lookup_factor<T, DiagonalGate::RZ>(
            index, rz_lookup, qubits, qubits);
        factor = multiply(
            factor,
            diagonal_lookup_factor<T, DiagonalGate::RZZ_EVEN>(
                index, rzz_even_lookup, qubits, qubits / 2));
        factor = multiply(
            factor,
            diagonal_lookup_factor<T, DiagonalGate::RZZ_ODD>(
                index, rzz_odd_lookup, qubits, qubits / 2));
        const Complex<T> inverse_factor{factor.real, -factor.imag};
        phi_state[index] = multiply(phi, inverse_factor);
        lambda_state[index] = multiply(lambda, inverse_factor);
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int generator = 0; generator < generator_count; ++generator) {
        const double sum = warp_sum(
            overlaps[generator * kCombinedDiagonalThreads + tid]);
        if constexpr (kDiagonalWarpAtomic) {
            if (lane == 0) {
                int parameter = 0;
                if (generator < qubits) {
                    parameter = rz_parameter_offset + generator;
                } else {
                    const int left = generator - qubits;
                    parameter = (left & 1) == 0
                                    ? rzz_even_parameter_offset + left / 2
                                    : rzz_odd_parameter_offset + left / 2;
                }
                atomicAdd(gradients + parameter, sum);
            }
        } else if (lane == 0) {
            warp_partials[generator * kCombinedDiagonalWarps + warp] = sum;
        }
    }
    if constexpr (kDiagonalWarpAtomic) {
        return;
    }
    __syncthreads();
    if (warp == 0) {
        for (int generator = 0; generator < generator_count; ++generator) {
            double value = lane < kCombinedDiagonalWarps
                               ? warp_partials[
                                     generator * kCombinedDiagonalWarps + lane]
                               : 0.0;
            value = warp_sum(value);
            if (lane == 0) {
                int parameter = 0;
                if (generator < qubits) {
                    parameter = rz_parameter_offset + generator;
                } else {
                    const int left = generator - qubits;
                    parameter = (left & 1) == 0
                                    ? rzz_even_parameter_offset + left / 2
                                    : rzz_odd_parameter_offset + left / 2;
                }
                atomicAdd(gradients + parameter, value);
            }
        }
    }
}

template <typename T>
void launch_rz_rzz_backward(
    Complex<T>* phi,
    Complex<T>* lambda,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    double* gradients,
    uint64_t state_size,
    int qubits,
    int rz_parameter_offset,
    int rzz_even_parameter_offset,
    int rzz_odd_parameter_offset,
    int multiprocessors) {
    const uint64_t required =
        (state_size + kCombinedDiagonalThreads - 1) /
        kCombinedDiagonalThreads;
    const int grid_size = static_cast<int>(
        std::min<uint64_t>(required, multiprocessors * 8ull));
    rz_rzz_backward_kernel<T>
        <<<grid_size, kCombinedDiagonalThreads>>>(phi,
                                                  lambda,
                                                  rz_lookup,
                                                  rzz_even_lookup,
                                                  rzz_odd_lookup,
                                                  gradients,
                                                  state_size,
                                                  qubits,
                                                  rz_parameter_offset,
                                                  rzz_even_parameter_offset,
                                                  rzz_odd_parameter_offset);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void shared_ring_rzz_forward_kernel(
    Complex<T>* state,
    const Complex<T>* phase_lookup,
    uint64_t state_size,
    int qubits) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const Complex<T> factor =
            shared_ring_rzz_factor(index, phase_lookup, qubits);
        state[index] = multiply(state[index], factor);
    }
}

template <typename T>
void launch_shared_ring_rzz_forward(Complex<T>* state,
                                    const Complex<T>* phase_lookup,
                                    uint64_t state_size,
                                    int qubits,
                                    int grid_size) {
    shared_ring_rzz_forward_kernel<T>
        <<<grid_size, kSharedDiagonalBlockThreads>>>(
            state, phase_lookup, state_size, qubits);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void shared_ring_rzz_backward_kernel(
    Complex<T>* phi_state,
    Complex<T>* lambda_state,
    const Complex<T>* phase_lookup,
    double* gradients,
    uint64_t state_size,
    int qubits,
    int parameter_offset) {
    __shared__ double reduction[kSharedDiagonalBlockThreads];
    double local_gradient = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const Complex<T> phi = phi_state[index];
        const Complex<T> lambda = lambda_state[index];
        const int eigenvalue_sum =
            qubits - 2 * ring_domain_wall_count(index, qubits);
        local_gradient += imag_conjugate_product(lambda, phi) *
                          static_cast<double>(eigenvalue_sum);
        const Complex<T> factor =
            shared_ring_rzz_factor(index, phase_lookup, qubits);
        const Complex<T> inverse_factor{factor.real, -factor.imag};
        phi_state[index] = multiply(phi, inverse_factor);
        lambda_state[index] = multiply(lambda, inverse_factor);
    }
    block_atomic_sum(local_gradient,
                     reduction,
                     gradients + parameter_offset);
}

template <typename T>
void launch_shared_ring_rzz_backward(Complex<T>* phi,
                                     Complex<T>* lambda,
                                     const Complex<T>* phase_lookup,
                                     double* gradients,
                                     uint64_t state_size,
                                     int qubits,
                                     int parameter_offset,
                                     int grid_size) {
    shared_ring_rzz_backward_kernel<T>
        <<<grid_size, kSharedDiagonalBlockThreads>>>(phi,
                                               lambda,
                                               phase_lookup,
                                               gradients,
                                               state_size,
                                               qubits,
                                               parameter_offset);
    SAD_CUDA_CHECK(cudaGetLastError());
}


}  // namespace sad
