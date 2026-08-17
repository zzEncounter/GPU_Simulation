#pragma once

#include "sad_api.h"

#include "../circuits/context.cuh"
#include "../core/cuda_common.cuh"
#include "../kernels/hamiltonian.cuh"
#include "../kernels/real_amplitude.cuh"
#include "circuit_dispatch.cuh"
#include "lookups.cuh"
#include "options.cuh"

#include "../circuits/ra_hea.cuh"
#include "../circuits/rzz_hea.cuh"
#include "../circuits/su2_hea.cuh"
#include "../circuits/qaoa.cuh"
#include "../circuits/qaoa_ns.cuh"
#include "../circuits/xxz_hva.cuh"
#include "../circuits/mera.cuh"
#include "../circuits/equivariant_qnn.cuh"
#include "../circuits/data_reuploading.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sad {

template <int Circuit, typename T>
auto build_circuit_diagonal_lookups(int qubits,
                                    int layers,
                                    const T* parameters,
                                    size_t parameter_count) -> DiagonalLookupData<T> {
    DiagonalLookupData<T> data;
    data.offsets_by_parameter.assign(parameter_count,
                                     std::numeric_limits<size_t>::max());
    CircuitExecutor<Circuit, T>::append_diagonal_lookups(
        qubits, layers, parameters, &data);
    return data;
}

template <typename T>
auto build_diagonal_lookups(int circuit,
                            int qubits,
                            int layers,
                            const T* parameters,
                            size_t parameter_count) -> DiagonalLookupData<T> {
    return visit_circuit(circuit, [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        return build_circuit_diagonal_lookups<Circuit>(
            qubits, layers, parameters, parameter_count);
    });
}

template <typename T>
auto build_initial_state_lookup(int circuit,
                                int qubits,
                                const T* parameters)
    -> InitialStateLookupData<T> {
    return visit_circuit(circuit, [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        return CircuitExecutor<Circuit, T>::build_initial_state_lookup(
            qubits, parameters);
    });
}

template <int Circuit>
inline size_t expected_parameter_count(int qubits, int layers) {
    if constexpr (Circuit == SAD_CIRCUIT_QAOA) {
        return static_cast<size_t>(2) * layers;
    }
    if constexpr (Circuit == SAD_CIRCUIT_QAOA_NS) {
        return static_cast<size_t>(2) * qubits * layers;
    }
    if constexpr (Circuit == SAD_CIRCUIT_MERA) {
        int expected_layers = 0;
        for (int active = qubits; active > 1; active = (active + 1) / 2) {
            ++expected_layers;
        }
        if (layers != expected_layers) {
            throw std::invalid_argument(
                "MERA layers must equal ceil(log2(qubits))");
        }
        int set_bits = 0;
        for (int value = qubits - 1; value != 0; value >>= 1) {
            set_bits += value & 1;
        }
        return static_cast<size_t>(4 * (qubits - 1) - 2 * set_bits);
    }
    if constexpr (Circuit == SAD_CIRCUIT_EQUIVARIANT_QNN) {
        return static_cast<size_t>(3) * layers;
    }
    if constexpr (Circuit == SAD_CIRCUIT_DATA_REUPLOADING) {
        return static_cast<size_t>(3) * qubits * layers;
    }
    return static_cast<size_t>(
               CircuitExecutor<Circuit, double>::kParametersPerQubitLayer) *
           qubits * layers;
}

inline void validate_inputs(int circuit,
                            int qubits,
                            int layers,
                            int steps,
                            int warmup_steps,
                            size_t parameter_count) {
    if (qubits < 2 || qubits > kMaxQubits) {
        throw std::invalid_argument("qubits must be in [2, 30]");
    }
    if (layers < 1 || steps < 1 || warmup_steps < 0) {
        throw std::invalid_argument(
            "layers/steps must be positive and warmup non-negative");
    }

    const size_t expected = visit_circuit(circuit, [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        CircuitExecutor<Circuit, double>::validate(qubits);
        return expected_parameter_count<Circuit>(qubits, layers);
    });
    if (parameter_count != expected) {
        throw std::invalid_argument("parameter_count mismatch: expected " +
                                    std::to_string(expected) + ", got " +
                                    std::to_string(parameter_count));
    }
}

inline int ordinary_grid_size(uint64_t state_size, int multiprocessors) {
    const uint64_t required =
        (state_size + static_cast<uint64_t>(kOrdinaryBlockThreads) - 1) /
        kOrdinaryBlockThreads;
    return static_cast<int>(
        std::min<uint64_t>(required, multiprocessors * 4ull));
}

template <typename T>
void run_forward(int circuit,
                 int qubits,
                 int layers,
                 uint64_t state_size,
                 const RotationCoefficients<T>* rotation_coefficients,
                 const Complex<T>* initial_state_lookup,
                 const Complex<T>* diagonal_lookup,
                 const size_t* diagonal_lookup_offsets,
                 const int* selected_maps,
                 const int* target_masks,
                 int phase_count,
                 const int* xxz_even_selected_maps,
                 const int* xxz_even_pair_counts,
                 int xxz_even_phase_count,
                 const int* xxz_odd_selected_maps,
                 const int* xxz_odd_pair_counts,
                 int xxz_odd_phase_count,
                 int multiprocessors,
                 int ordinary_grid,
                 ExecutionMode execution_mode,
                 StatePair<T>* phi) {
    const ForwardCircuitContext<T> context{
        qubits,
        state_size,
        rotation_coefficients,
        initial_state_lookup,
        diagonal_lookup,
        diagonal_lookup_offsets,
        selected_maps,
        target_masks,
        phase_count,
        xxz_even_selected_maps,
        xxz_even_pair_counts,
        xxz_even_phase_count,
        xxz_odd_selected_maps,
        xxz_odd_pair_counts,
        xxz_odd_phase_count,
        multiprocessors,
        ordinary_grid,
        phi};
    const auto run_layers = [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        if constexpr (Circuit == SAD_CIRCUIT_RA_HEA) {
            if (use_real_amplitude_state(circuit, execution_mode)) {
                launch_real_initial(context.phi,
                                    context.state_size,
                                    context.qubits,
                                    context.initial_state_lookup,
                                    context.ordinary_grid);
                for (int layer = 1; layer < layers; ++layer) {
                    launch_real_fused_forward(
                        context.phi,
                        context.rotation_coefficients,
                        context.qubits,
                        layer * context.qubits,
                        context.selected_maps,
                        context.target_masks,
                        context.phase_count,
                        context.multiprocessors,
                        kAlternatePhases && (layer & 1));
                }
                return;
            }
        }
        if (execution_mode == ExecutionMode::LEGACY) {
            SAD_CUDA_CHECK(
                cudaMemset(phi->current, 0, state_size * sizeof(Complex<T>)));
            SAD_CUDA_CHECK(
                cudaMemset(phi->scratch, 0, state_size * sizeof(Complex<T>)));
            initialise_zero_state_kernel<T><<<1, 1>>>(phi->current);
            SAD_CUDA_CHECK(cudaGetLastError());
            for (int layer = 0; layer < layers; ++layer) {
                CircuitExecutor<Circuit, T>::forward_layer(layer, context);
            }
            return;
        }

        CircuitExecutor<Circuit, T>::forward_initial(context);
        for (int layer = 1; layer < layers; ++layer) {
            if (execution_mode == ExecutionMode::INITIAL_ONLY) {
                CircuitExecutor<Circuit, T>::forward_layer(layer, context);
            } else if constexpr (Circuit == SAD_CIRCUIT_SU2_HEA) {
                if (execution_mode == ExecutionMode::PHASED_FORWARD) {
                    CircuitExecutor<Circuit, T>::forward_layer_phased(layer,
                                                                       context);
                } else {
                    CircuitExecutor<Circuit, T>::forward_layer_optimized(layer,
                                                                          context);
                }
            } else {
                CircuitExecutor<Circuit, T>::forward_layer_optimized(layer,
                                                                      context);
            }
        }
    };
    visit_circuit(circuit, run_layers);
}

template <typename T>
void run_backward(int circuit,
                  int qubits,
                  int layers,
                  uint64_t state_size,
                  const RotationCoefficients<T>* rotation_coefficients,
                  const Complex<T>* diagonal_lookup,
                  const size_t* diagonal_lookup_offsets,
                  double* gradients,
                  const int* selected_maps,
                  const int* target_masks,
                  const int* target_phases,
                  int phase_count,
                  const int* xxz_even_selected_maps,
                  const int* xxz_even_pair_counts,
                  int xxz_even_phase_count,
                  const int* xxz_odd_selected_maps,
                  const int* xxz_odd_pair_counts,
                  int xxz_odd_phase_count,
                  int multiprocessors,
                  int ordinary_grid,
                  ExecutionMode execution_mode,
                  StatePair<T>* phi,
                  StatePair<T>* lambda) {
    const BackwardCircuitContext<T> context{
        qubits,
        state_size,
        rotation_coefficients,
        diagonal_lookup,
        diagonal_lookup_offsets,
        gradients,
        selected_maps,
        target_masks,
        target_phases,
        phase_count,
        xxz_even_selected_maps,
        xxz_even_pair_counts,
        xxz_even_phase_count,
        xxz_odd_selected_maps,
        xxz_odd_pair_counts,
        xxz_odd_phase_count,
        multiprocessors,
        ordinary_grid,
        phi,
        lambda};
    const auto run_layers = [&](auto circuit_tag) {
        constexpr int Circuit = decltype(circuit_tag)::value;
        if constexpr (Circuit == SAD_CIRCUIT_RA_HEA) {
            if (use_real_amplitude_state(circuit, execution_mode)) {
                for (int layer = layers - 1; layer >= 0; --layer) {
                    launch_real_fused_backward(
                        context.phi,
                        context.lambda,
                        context.rotation_coefficients,
                        context.gradients,
                        context.qubits,
                        layer * context.qubits,
                        context.selected_maps,
                        context.target_masks,
                        context.phase_count,
                        context.multiprocessors,
                        kAlternatePhases && (layer & 1));
                }
                return;
            }
        }
        for (int layer = layers - 1; layer >= 0; --layer) {
            if (execution_mode == ExecutionMode::LEGACY ||
                execution_mode == ExecutionMode::INITIAL_ONLY ||
                execution_mode == ExecutionMode::FUSED_FORWARD ||
                execution_mode == ExecutionMode::PHASED_FORWARD) {
                CircuitExecutor<Circuit, T>::backward_layer(layer, context);
            } else if (execution_mode == ExecutionMode::ALL_FUSED) {
                CircuitExecutor<Circuit, T>::backward_layer_fused(layer,
                                                                   context);
            } else {
                CircuitExecutor<Circuit, T>::backward_layer_optimized(layer,
                                                                       context);
            }
        }
    };
    visit_circuit(circuit, run_layers);
}

}  // namespace sad
