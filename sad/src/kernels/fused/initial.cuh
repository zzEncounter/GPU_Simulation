#pragma once

#include "diagonal.cuh"
#include "../ring_cnot.cuh"

namespace sad {

template <typename T, FusedDiagonalMode Mode, bool ScatterCnot>
__global__ void product_state_initialization_kernel(
    Complex<T>* output,
    uint64_t state_size,
    int qubits,
    const Complex<T>* product_lookup,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup) {
    const int chunk_count =
        (qubits + kDiagonalLookupBits - 1) / kDiagonalLookupBits;
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        Complex<T> value{static_cast<T>(1), static_cast<T>(0)};
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const unsigned code = static_cast<unsigned>(
                (index >> (chunk * kDiagonalLookupBits)) &
                (kDiagonalLookupSize - 1));
            value = multiply(
                value, product_lookup[chunk * kDiagonalLookupSize + code]);
        }
        if constexpr (Mode != FusedDiagonalMode::NONE) {
            value = multiply(
                value,
                fused_diagonal_factor<T, Mode>(index,
                                                qubits,
                                                rz_lookup,
                                                rzz_even_lookup,
                                                rzz_odd_lookup));
        }
        const uint64_t output_index =
            ScatterCnot ? apply_ring_cnot_forward_to_basis(index, qubits)
                        : index;
        output[output_index] = value;
    }
}
template <typename T, FusedDiagonalMode Mode, bool ScatterCnot>
void launch_product_state_initialization(
    Complex<T>* output,
    uint64_t state_size,
    int qubits,
    const Complex<T>* product_lookup,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup,
    int grid_size) {
    product_state_initialization_kernel<T, Mode, ScatterCnot>
        <<<grid_size, kOrdinaryBlockThreads>>>(output,
                                       state_size,
                                       qubits,
                                       product_lookup,
                                       rz_lookup,
                                       rzz_even_lookup,
                                       rzz_odd_lookup);
    SAD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sad
