#pragma once

#include "../core/cuda_common.cuh"

#include <algorithm>

namespace sad {

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

template <typename T>
__global__ void ring_cnot_forward_scatter_kernel(
    const Complex<T>* phi_input,
    Complex<T>* phi_output,
    const Complex<T>* lambda_input,
    Complex<T>* lambda_output,
    uint64_t state_size,
    int qubits) {
    for (uint64_t input_index = blockIdx.x * blockDim.x + threadIdx.x;
         input_index < state_size;
         input_index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        // This ring's forward map preserves pairs of complex amplitudes inside
        // 32-byte sectors.  Consecutive input reads plus permuted writes are
        // therefore better coalesced than gathering P^-1 into linear output.
        const uint64_t output_index =
            apply_ring_cnot_forward_to_basis(input_index, qubits);
        phi_output[output_index] = phi_input[input_index];
        if (lambda_input != nullptr) {
            lambda_output[output_index] = lambda_input[input_index];
        }
    }
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
    if (kCnotForwardScatter && !adjoint) {
        ring_cnot_forward_scatter_kernel<T>
            <<<grid_size, kOrdinaryBlockThreads>>>(
                phi->current,
                phi->scratch,
                lambda == nullptr ? nullptr : lambda->current,
                lambda == nullptr ? nullptr : lambda->scratch,
                state_size,
                qubits);
    } else {
        ring_cnot_permutation_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
            phi->current,
            phi->scratch,
            lambda == nullptr ? nullptr : lambda->current,
            lambda == nullptr ? nullptr : lambda->scratch,
            state_size,
            qubits,
            adjoint);
    }
    SAD_CUDA_CHECK(cudaGetLastError());
    phi->swap();
    if (lambda != nullptr) {
        lambda->swap();
    }
}


}  // namespace sad
