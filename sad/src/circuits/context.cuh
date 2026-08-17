#pragma once

#include "../core/cuda_common.cuh"
#include "../kernels/ring_cnot.cuh"

#include <cstddef>
#include <cstdint>

namespace sad {

template <typename T>
struct ForwardCircuitContext {
    int qubits;
    uint64_t state_size;
    const RotationCoefficients<T>* rotation_coefficients;
    const Complex<T>* initial_state_lookup;
    const Complex<T>* diagonal_lookup;
    const size_t* diagonal_lookup_offsets;
    const int* selected_maps;
    const int* target_masks;
    int phase_count;
    const int* xxz_even_selected_maps;
    const int* xxz_even_pair_counts;
    int xxz_even_phase_count;
    const int* xxz_odd_selected_maps;
    const int* xxz_odd_pair_counts;
    int xxz_odd_phase_count;
    int multiprocessors;
    int ordinary_grid;
    StatePair<T>* phi;

    auto diagonal_lookup_at(size_t parameter_offset) const -> const Complex<T>* {
        return diagonal_lookup + diagonal_lookup_offsets[parameter_offset];
    }
};

template <typename T>
struct BackwardCircuitContext {
    int qubits;
    uint64_t state_size;
    const RotationCoefficients<T>* rotation_coefficients;
    const Complex<T>* diagonal_lookup;
    const size_t* diagonal_lookup_offsets;
    double* gradients;
    const int* selected_maps;
    const int* target_masks;
    const int* target_phases;
    int phase_count;
    const int* xxz_even_selected_maps;
    const int* xxz_even_pair_counts;
    int xxz_even_phase_count;
    const int* xxz_odd_selected_maps;
    const int* xxz_odd_pair_counts;
    int xxz_odd_phase_count;
    int multiprocessors;
    int ordinary_grid;
    StatePair<T>* phi;
    StatePair<T>* lambda;

    auto diagonal_lookup_at(size_t parameter_offset) const -> const Complex<T>* {
        return diagonal_lookup + diagonal_lookup_offsets[parameter_offset];
    }
};

template <int Circuit, typename T>
struct CircuitExecutor;

}  // namespace sad
