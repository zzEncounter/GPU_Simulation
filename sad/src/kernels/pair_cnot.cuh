#pragma once

#include "../core/cuda_common.cuh"
#include "ring_cnot.cuh"

namespace sad {

#ifndef SAD_QAOA_BD_FUSION
#define SAD_QAOA_BD_FUSION 1
#endif

template <typename T>
__global__ void rz_single_forward_kernel(Complex<T>* state,
                                         const Complex<T>* phase_lookup,
                                         uint64_t state_size,
                                         int wire) {
    const uint64_t mask = 1ull << wire;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        state[index] = multiply(state[index], phase_lookup[(index & mask) ? 1 : 0]);
    }
}

template <typename T>
void launch_rz_single_forward(Complex<T>* state,
                              const Complex<T>* phase_lookup,
                              uint64_t state_size,
                              int wire,
                              int grid_size) {
    rz_single_forward_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        state, phase_lookup, state_size, wire);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void rz_single_backward_kernel(Complex<T>* phi_state,
                                          Complex<T>* lambda_state,
                                          const Complex<T>* phase_lookup,
                                          double* gradient,
                                          int gradient_offset,
                                          uint64_t state_size,
                                          int wire) {
    const uint64_t mask = 1ull << wire;
    double local = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const int eigenvalue = (index & mask) ? -1 : 1;
        local += static_cast<double>(eigenvalue) *
                 imag_conjugate_product(lambda_state[index], phi_state[index]);
        const Complex<T> phase = phase_lookup[(index & mask) ? 1 : 0];
        const Complex<T> inverse_phase{phase.real, -phase.imag};
        phi_state[index] = multiply(phi_state[index], inverse_phase);
        lambda_state[index] = multiply(lambda_state[index], inverse_phase);
    }
    atomicAdd(gradient + gradient_offset, local);
}

template <typename T>
void launch_rz_single_backward(Complex<T>* phi,
                               Complex<T>* lambda,
                               const Complex<T>* phase_lookup,
                               double* gradient,
                               int gradient_offset,
                               uint64_t state_size,
                               int wire,
                               int grid_size) {
    rz_single_backward_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        phi, lambda, phase_lookup, gradient, gradient_offset, state_size, wire);
    SAD_CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ uint64_t apply_pair_cnot(uint64_t basis,
                                                    int control,
                                                    int target) {
    if (((basis >> control) & 1ull) != 0) {
        basis ^= 1ull << target;
    }
    return basis;
}

__device__ __forceinline__ uint64_t apply_matching_cnot(uint64_t basis,
                                                        int qubits,
                                                        int parity) {
    // For an even ring, both parity matchings are disjoint, including the
    // wrap-around pair in the odd matching. Applying each pair in order is
    // therefore equivalent to one matching permutation.
    for (int control = parity; control < qubits; control += 2) {
        const int target = (control + 1) % qubits;
        basis = apply_pair_cnot(basis, control, target);
    }
    return basis;
}

template <typename T>
__device__ __forceinline__ Complex<T> matching_rz_factor(
    uint64_t basis,
    const Complex<T>* phase_lookup,
    int qubits,
    int parity) {
    Complex<T> factor =
        make_complex<T>(static_cast<T>(1), static_cast<T>(0));
    for (int control = parity; control < qubits; control += 2) {
        const int target = (control + 1) % qubits;
        factor = multiply(factor,
                          phase_lookup[(basis >> target) & 1ull ? 1 : 0]);
    }
    return factor;
}

template <typename T>
__global__ void matching_cnot_rz_cnot_forward_kernel(
    Complex<T>* state,
    const Complex<T>* phase_lookup,
    uint64_t state_size,
    int qubits,
    int parity) {
    for (uint64_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
         output_index < state_size;
         output_index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t transformed =
            apply_matching_cnot(output_index, qubits, parity);
        const Complex<T> factor = matching_rz_factor(
            transformed, phase_lookup, qubits, parity);
        state[output_index] = multiply(state[output_index], factor);
    }
}

template <typename T>
void launch_matching_cnot_rz_cnot_forward(Complex<T>* state,
                                          const Complex<T>* phase_lookup,
                                          uint64_t state_size,
                                          int qubits,
                                          int parity,
                                          int grid_size) {
    matching_cnot_rz_cnot_forward_kernel<T>
        <<<grid_size, kOrdinaryBlockThreads>>>(
            state, phase_lookup, state_size, qubits, parity);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void matching_cnot_rz_cnot_backward_kernel(
    Complex<T>* phi_state,
    Complex<T>* lambda_state,
    const Complex<T>* phase_lookup,
    double* gradient,
    int gradient_offset,
    uint64_t state_size,
    int qubits,
    int parity) {
    double local = 0.0;
    for (uint64_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
         output_index < state_size;
         output_index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t transformed =
            apply_matching_cnot(output_index, qubits, parity);
        int eigenvalue_sum = 0;
        for (int control = parity; control < qubits; control += 2) {
            const int target = (control + 1) % qubits;
            eigenvalue_sum += ((transformed >> target) & 1ull) ? -1 : 1;
        }
        const Complex<T> factor = matching_rz_factor(
            transformed, phase_lookup, qubits, parity);
        local += static_cast<double>(eigenvalue_sum) *
                 imag_conjugate_product(lambda_state[output_index],
                                        phi_state[output_index]);
        const Complex<T> inverse_factor{factor.real, -factor.imag};
        phi_state[output_index] =
            multiply(phi_state[output_index], inverse_factor);
        lambda_state[output_index] =
            multiply(lambda_state[output_index], inverse_factor);
    }
    atomicAdd(gradient + gradient_offset, local);
}

template <typename T>
void launch_matching_cnot_rz_cnot_backward(
    Complex<T>* phi,
    Complex<T>* lambda,
    const Complex<T>* phase_lookup,
    double* gradient,
    int gradient_offset,
    uint64_t state_size,
    int qubits,
    int parity,
    int grid_size) {
    matching_cnot_rz_cnot_backward_kernel<T>
        <<<grid_size, kOrdinaryBlockThreads>>>(
            phi,
            lambda,
            phase_lookup,
            gradient,
            gradient_offset,
            state_size,
            qubits,
            parity);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void matching_cnot_permutation_kernel(
    const Complex<T>* phi_input,
    Complex<T>* phi_output,
    const Complex<T>* lambda_input,
    Complex<T>* lambda_output,
    uint64_t state_size,
    int qubits,
    int parity) {
    for (uint64_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
         output_index < state_size;
         output_index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t input_index =
            apply_matching_cnot(output_index, qubits, parity);
        phi_output[output_index] = phi_input[input_index];
        if (lambda_input != nullptr) {
            lambda_output[output_index] = lambda_input[input_index];
        }
    }
}

template <typename T>
void launch_matching_cnot(StatePair<T>* phi,
                          StatePair<T>* lambda,
                          uint64_t state_size,
                          int qubits,
                          int parity,
                          int grid_size) {
    matching_cnot_permutation_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        phi->current,
        phi->scratch,
        lambda == nullptr ? nullptr : lambda->current,
        lambda == nullptr ? nullptr : lambda->scratch,
        state_size,
        qubits,
        parity);
    SAD_CUDA_CHECK(cudaGetLastError());
    phi->swap();
    if (lambda != nullptr) {
        lambda->swap();
    }
}

template <typename T>
__global__ void matching_rz_forward_kernel(Complex<T>* state,
                                           const Complex<T>* phase_lookup,
                                           uint64_t state_size,
                                           int qubits,
                                           int parity) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex<T> factor = make_complex<T>(static_cast<T>(1), static_cast<T>(0));
        for (int control = parity; control < qubits; control += 2) {
            const int target = (control + 1) % qubits;
            factor = multiply(factor,
                              phase_lookup[(index >> target) & 1ull ? 1 : 0]);
        }
        state[index] = multiply(state[index], factor);
    }
}

template <typename T>
void launch_matching_rz_forward(Complex<T>* state,
                                const Complex<T>* phase_lookup,
                                uint64_t state_size,
                                int qubits,
                                int parity,
                                int grid_size) {
    matching_rz_forward_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        state, phase_lookup, state_size, qubits, parity);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void matching_rz_backward_kernel(Complex<T>* phi_state,
                                            Complex<T>* lambda_state,
                                            const Complex<T>* phase_lookup,
                                            double* gradient,
                                            int gradient_offset,
                                            uint64_t state_size,
                                            int qubits,
                                            int parity) {
    double local = 0.0;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        int eigenvalue_sum = 0;
        Complex<T> factor = make_complex<T>(static_cast<T>(1), static_cast<T>(0));
        for (int control = parity; control < qubits; control += 2) {
            const int target = (control + 1) % qubits;
            const int eigenvalue = ((index >> target) & 1ull) ? -1 : 1;
            eigenvalue_sum += eigenvalue;
            factor = multiply(factor, phase_lookup[eigenvalue < 0 ? 1 : 0]);
        }
        local += static_cast<double>(eigenvalue_sum) *
                 imag_conjugate_product(lambda_state[index], phi_state[index]);
        const Complex<T> inverse_factor{factor.real, -factor.imag};
        phi_state[index] = multiply(phi_state[index], inverse_factor);
        lambda_state[index] = multiply(lambda_state[index], inverse_factor);
    }
    atomicAdd(gradient + gradient_offset, local);
}

template <typename T>
void launch_matching_rz_backward(Complex<T>* phi,
                                 Complex<T>* lambda,
                                 const Complex<T>* phase_lookup,
                                 double* gradient,
                                 int gradient_offset,
                                 uint64_t state_size,
                                 int qubits,
                                 int parity,
                                 int grid_size) {
    matching_rz_backward_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        phi,
        lambda,
        phase_lookup,
        gradient,
        gradient_offset,
        state_size,
        qubits,
        parity);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void pair_cnot_permutation_kernel(
    const Complex<T>* phi_input,
    Complex<T>* phi_output,
    const Complex<T>* lambda_input,
    Complex<T>* lambda_output,
    uint64_t state_size,
    int control,
    int target) {
    for (uint64_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
         output_index < state_size;
         output_index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        // CNOT is self-inverse, so gather from the same pair permutation for
        // both forward and adjoint application.
        const uint64_t input_index =
            apply_pair_cnot(output_index, control, target);
        phi_output[output_index] = phi_input[input_index];
        if (lambda_input != nullptr) {
            lambda_output[output_index] = lambda_input[input_index];
        }
    }
}

template <typename T>
void launch_pair_cnot(StatePair<T>* phi,
                      StatePair<T>* lambda,
                      uint64_t state_size,
                      int control,
                      int target,
                      int grid_size) {
    pair_cnot_permutation_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        phi->current,
        phi->scratch,
        lambda == nullptr ? nullptr : lambda->current,
        lambda == nullptr ? nullptr : lambda->scratch,
        state_size,
        control,
        target);
    SAD_CUDA_CHECK(cudaGetLastError());
    phi->swap();
    if (lambda != nullptr) {
        lambda->swap();
    }
}

}  // namespace sad
