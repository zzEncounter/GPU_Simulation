#pragma once

#include "../core/cuda_common.cuh"

namespace sad {

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
    __shared__ double reduction[kOrdinaryBlockThreads];
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

// Ring MaxCut cost Hamiltonian used by standard QAOA:
// H_C = 1/2 sum_(i,j) (Z_i Z_j - I).  Its expectation is minus the cut size.
template <typename T>
__global__ void qaoa_cost_hamiltonian_kernel(const Complex<T>* phi,
                                             Complex<T>* lambda,
                                             uint64_t state_size,
                                             int qubits,
                                             double* energy) {
    __shared__ double reduction[kOrdinaryBlockThreads];
    double local_energy = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int zz_sum = 0;
        for (int left = 0; left < qubits; ++left) {
            const int right = (left + 1) % qubits;
            zz_sum += ((index >> left) & 1ull) ==
                              ((index >> right) & 1ull)
                          ? 1
                          : -1;
        }
        const T eigenvalue = static_cast<T>(0.5) *
                             static_cast<T>(zz_sum - qubits);
        const Complex<T> amplitude = phi[index];
        const Complex<T> h_amplitude = scale(amplitude, eigenvalue);
        lambda[index] = h_amplitude;
        local_energy += real_conjugate_product(amplitude, h_amplitude);
    }
    block_atomic_sum(local_energy, reduction, energy);
}

// Periodic antiferromagnetic XXZ target used by XXZ-HVA.
// H = sum_i (X_i X_(i+1) + Y_i Y_(i+1) + Delta Z_i Z_(i+1)), Delta=1/2.
template <typename T>
__global__ void xxz_hamiltonian_kernel(const Complex<T>* phi,
                                       Complex<T>* lambda,
                                       uint64_t state_size,
                                       int qubits,
                                       double* energy) {
    __shared__ double reduction[kOrdinaryBlockThreads];
    double local_energy = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int zz_sum = 0;
        Complex<T> h_amplitude = make_complex<T>(0, 0);
        for (int left = 0; left < qubits; ++left) {
            const int right = (left + 1) % qubits;
            const bool unequal = ((index >> left) & 1ull) !=
                                 ((index >> right) & 1ull);
            zz_sum += unequal ? -1 : 1;
            if (unequal) {
                const uint64_t partner =
                    index ^ (1ull << left) ^ (1ull << right);
                h_amplitude = add(
                    h_amplitude,
                    scale(phi[partner], static_cast<T>(2)));
            }
        }
        h_amplitude = add(
            h_amplitude,
            scale(phi[index], static_cast<T>(0.5 * zz_sum)));
        lambda[index] = h_amplitude;
        local_energy += real_conjugate_product(phi[index], h_amplitude);
    }
    block_atomic_sum(local_energy, reduction, energy);
}


}  // namespace sad
