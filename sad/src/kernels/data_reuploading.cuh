#pragma once

#include "../core/cuda_common.cuh"

namespace sad {

__device__ __forceinline__ bool data_reuploading_cz_sign(uint64_t index,
                                                         int qubits,
                                                         int parity) {
    bool odd = false;
    for (int pair = 0; pair < qubits / 2; ++pair) {
        const int left = parity == 0 ? 2 * pair : 2 * pair + 1;
        const int right = (left + 1) % qubits;
        odd ^= (((index >> left) & 1ull) != 0) &&
               (((index >> right) & 1ull) != 0);
    }
    return odd;
}

template <typename T>
__global__ void data_reuploading_brickwork_forward_kernel(Complex<T>* state,
                                                           uint64_t state_size,
                                                           int qubits,
                                                           int parity) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        if (data_reuploading_cz_sign(index, qubits, parity)) {
            state[index].real = -state[index].real;
            state[index].imag = -state[index].imag;
        }
    }
}

template <typename T>
__global__ void data_reuploading_brickwork_backward_kernel(
    Complex<T>* phi, Complex<T>* lambda, uint64_t state_size, int qubits,
    int parity) {
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        if (data_reuploading_cz_sign(index, qubits, parity)) {
            phi[index].real = -phi[index].real;
            phi[index].imag = -phi[index].imag;
            lambda[index].real = -lambda[index].real;
            lambda[index].imag = -lambda[index].imag;
        }
    }
}

template <typename T>
void launch_data_reuploading_brickwork_forward(Complex<T>* state,
                                               uint64_t state_size,
                                               int qubits,
                                               int parity,
                                               int grid_size) {
    data_reuploading_brickwork_forward_kernel<T>
        <<<grid_size, kOrdinaryBlockThreads>>>(state, state_size, qubits, parity);
    SAD_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_data_reuploading_brickwork_backward(Complex<T>* phi,
                                                Complex<T>* lambda,
                                                uint64_t state_size,
                                                int qubits,
                                                int parity,
                                                int grid_size) {
    data_reuploading_brickwork_backward_kernel<T>
        <<<grid_size, kOrdinaryBlockThreads>>>(phi, lambda, state_size, qubits,
                                                parity);
    SAD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sad
