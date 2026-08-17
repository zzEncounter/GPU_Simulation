#pragma once

#include "../diagonal.cuh"

namespace sad {

template <typename T, FusedDiagonalMode Mode>
__device__ __forceinline__ Complex<T> fused_diagonal_factor(
    uint64_t index,
    int qubits,
    const Complex<T>* rz_lookup,
    const Complex<T>* rzz_even_lookup,
    const Complex<T>* rzz_odd_lookup) {
    Complex<T> factor{static_cast<T>(1), static_cast<T>(0)};
    if constexpr (Mode == FusedDiagonalMode::RZ ||
                  Mode == FusedDiagonalMode::RZ_RZZ) {
        factor = multiply(
            factor,
            diagonal_lookup_factor<T, DiagonalGate::RZ>(
                index, rz_lookup, qubits, qubits));
    }
    if constexpr (Mode == FusedDiagonalMode::RZZ ||
                  Mode == FusedDiagonalMode::RZ_RZZ) {
        factor = multiply(
            factor,
            diagonal_lookup_factor<T, DiagonalGate::RZZ_EVEN>(
                index, rzz_even_lookup, qubits, qubits / 2));
        factor = multiply(
            factor,
            diagonal_lookup_factor<T, DiagonalGate::RZZ_ODD>(
                index, rzz_odd_lookup, qubits, qubits / 2));
    }
    return factor;
}

}  // namespace sad
