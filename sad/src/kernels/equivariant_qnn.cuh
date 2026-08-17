#pragma once

#include "rotation.cuh"

namespace sad {

__device__ __forceinline__ int equivariant_qnn_pair_eigenvalue_sum(
    uint64_t index, int qubits) {
    const int zsum = qubits - 2 * __popcll(index);
    return (zsum * zsum - qubits) / 2;
}

template <typename T>
__device__ __forceinline__ Complex<T> equivariant_qnn_integer_power(
    T cosine, T sine, int exponent) {
    Complex<T> result =
        make_complex<T>(static_cast<T>(1), static_cast<T>(0));
    if (exponent == 0) return result;

    Complex<T> base = make_complex<T>(
        cosine, exponent < 0 ? -sine : sine);
    unsigned int remaining = static_cast<unsigned int>(
        exponent < 0 ? -exponent : exponent);
    while (remaining != 0) {
        if (remaining & 1u) result = multiply(result, base);
        remaining >>= 1u;
        if (remaining != 0) base = multiply(base, base);
    }
    return result;
}

template <typename T>
__global__ void equivariant_qnn_forward_kernel(
    Complex<T>* state, const RotationCoefficients<T>* coefficients,
    uint64_t state_size, int qubits, int parameter_offset) {
    const T sine = coefficients[parameter_offset + 2].sine;
    const T cosine = coefficients[parameter_offset + 2].cosine;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const int pair_sum =
            equivariant_qnn_pair_eigenvalue_sum(index, qubits);
        const Complex<T> factor = equivariant_qnn_integer_power(
            cosine, -sine, pair_sum);
        state[index] = multiply(state[index], factor);
    }
}

template <typename T>
void launch_equivariant_qnn_forward(
    Complex<T>* state, const RotationCoefficients<T>* coefficients,
    uint64_t state_size, int qubits, int parameter_offset, int grid_size) {
    equivariant_qnn_forward_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        state, coefficients, state_size, qubits, parameter_offset);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void equivariant_qnn_backward_kernel(
    Complex<T>* phi_state, Complex<T>* lambda_state,
    const RotationCoefficients<T>* coefficients, double* gradients,
    uint64_t state_size, int qubits, int parameter_offset) {
    __shared__ double reduction[kOrdinaryBlockThreads];
    double local = 0.0;
    const T sine = coefficients[parameter_offset + 2].sine;
    const T cosine = coefficients[parameter_offset + 2].cosine;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const int pair_sum =
            equivariant_qnn_pair_eigenvalue_sum(index, qubits);
        const Complex<T> phi = phi_state[index];
        const Complex<T> lambda = lambda_state[index];
        local += imag_conjugate_product(lambda, phi) *
                 static_cast<double>(pair_sum);
        const Complex<T> inverse_factor = equivariant_qnn_integer_power(
            cosine, sine, pair_sum);
        phi_state[index] = multiply(phi, inverse_factor);
        lambda_state[index] = multiply(lambda, inverse_factor);
    }
    block_atomic_sum(local, reduction, gradients + parameter_offset + 2);
}

template <typename T>
void launch_equivariant_qnn_backward(
    Complex<T>* phi, Complex<T>* lambda,
    const RotationCoefficients<T>* coefficients, double* gradients,
    uint64_t state_size, int qubits, int parameter_offset, int grid_size) {
    equivariant_qnn_backward_kernel<T><<<grid_size, kOrdinaryBlockThreads>>>(
        phi, lambda, coefficients, gradients, state_size, qubits,
        parameter_offset);
    SAD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sad
