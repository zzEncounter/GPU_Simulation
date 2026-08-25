#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "../kernels/xxz.cuh"

#include <stdexcept>

namespace sad {

struct XxzLayerLayout {
    int x;
    int y;
    int z;

    static auto at(int layer, int qubits) -> XxzLayerLayout {
        const int base = layer * 3 * qubits;
        return {base, base + qubits, base + 2 * qubits};
    }
};

template <typename T>
__global__ void initialise_neel_state_kernel(Complex<T>* state,
                                             uint64_t state_size,
                                             int qubits) {
    uint64_t neel_index = 0;
    for (int qubit = 1; qubit < qubits; qubit += 2) {
        neel_index |= 1ull << qubit;
    }
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        state[index] = index == neel_index
                           ? make_complex<T>(static_cast<T>(1),
                                             static_cast<T>(0))
                           : make_complex<T>(static_cast<T>(0),
                                             static_cast<T>(0));
    }
}

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_XXZ_HVA, T> {
    static constexpr int kParametersPerQubitLayer = 3;

    static void validate(int qubits) {
        if (qubits < 4 || (qubits & 1)) {
            throw std::invalid_argument(
                "XXZ-HVA ring requires an even qubit count of at least four");
        }
    }

    static void append_diagonal_lookups(int,
                                        int,
                                        const T*,
                                        DiagonalLookupData<T>*) {}

    static auto build_initial_state_lookup(int, const T*)
        -> InitialStateLookupData<T> {
        return {{{static_cast<T>(1), static_cast<T>(0)}}};
    }

    static void apply_layer(int layer,
                            const ForwardCircuitContext<T>& context) {
        const auto layout = XxzLayerLayout::at(layer, context.qubits);
        if constexpr (kXxzCrossMatching) {
            launch_xxz_cross_matching_forward(
                context.phi->current,
                context.rotation_coefficients,
                context.qubits,
                layout.x,
                layout.y,
                layout.z,
                context.xxz_even_selected_maps,
                context.xxz_even_pair_counts,
                context.xxz_odd_selected_maps,
                context.xxz_odd_pair_counts,
                context.xxz_even_phase_count);
            return;
        }
        launch_xxz_matching_forward(context.phi->current,
                                    context.rotation_coefficients,
                                    context.qubits,
                                    layout.x,
                                    layout.y,
                                    layout.z,
                                    context.xxz_even_selected_maps,
                                    context.xxz_even_pair_counts,
                                    context.xxz_even_phase_count,
                                    context.multiprocessors);
        launch_xxz_matching_forward(context.phi->current,
                                    context.rotation_coefficients,
                                    context.qubits,
                                    layout.x,
                                    layout.y,
                                    layout.z,
                                    context.xxz_odd_selected_maps,
                                    context.xxz_odd_pair_counts,
                                    context.xxz_odd_phase_count,
                                    context.multiprocessors);
    }

    static void forward_initial(const ForwardCircuitContext<T>& context) {
        initialise_neel_state_kernel<T>
            <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                context.phi->current, context.state_size, context.qubits);
        SAD_CUDA_CHECK(cudaGetLastError());
        apply_layer(0, context);
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        apply_layer(layer, context);
    }

    static void forward_layer(int layer,
                              const ForwardCircuitContext<T>& context) {
        if (layer == 0) {
            initialise_neel_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current, context.state_size, context.qubits);
            SAD_CUDA_CHECK(cudaGetLastError());
        }
        apply_layer(layer, context);
    }

    static void backward_layer(int layer,
                               const BackwardCircuitContext<T>& context) {
        const auto layout = XxzLayerLayout::at(layer, context.qubits);
        if constexpr (kXxzCrossMatching) {
            launch_xxz_cross_matching_backward(
                context.phi->current,
                context.lambda->current,
                context.rotation_coefficients,
                context.gradients,
                context.qubits,
                layout.x,
                layout.y,
                layout.z,
                context.xxz_even_selected_maps,
                context.xxz_even_pair_counts,
                context.xxz_odd_selected_maps,
                context.xxz_odd_pair_counts,
                context.xxz_even_phase_count);
            return;
        }
        launch_xxz_matching_backward(context.phi->current,
                                     context.lambda->current,
                                     context.rotation_coefficients,
                                     context.gradients,
                                     context.qubits,
                                     layout.x,
                                     layout.y,
                                     layout.z,
                                     context.xxz_odd_selected_maps,
                                     context.xxz_odd_pair_counts,
                                     context.xxz_odd_phase_count,
                                     context.multiprocessors);
        launch_xxz_matching_backward(context.phi->current,
                                     context.lambda->current,
                                     context.rotation_coefficients,
                                     context.gradients,
                                     context.qubits,
                                     layout.x,
                                     layout.y,
                                     layout.z,
                                     context.xxz_even_selected_maps,
                                     context.xxz_even_pair_counts,
                                     context.xxz_even_phase_count,
                                     context.multiprocessors);
    }

    static void backward_layer_optimized(
        int layer, const BackwardCircuitContext<T>& context) {
        backward_layer(layer, context);
    }

    static void backward_layer_fused(
        int layer, const BackwardCircuitContext<T>& context) {
        backward_layer(layer, context);
    }
};

}  // namespace sad
