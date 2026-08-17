#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "qaoa.cuh"
#include "../kernels/diagonal.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

#include <stdexcept>

namespace sad {

struct QaoaNsLayerLayout {
    int beta;
    int gamma;

    static auto at(int layer, int qubits) -> QaoaNsLayerLayout {
        const int base = 2 * layer * qubits;
        return {base, base + qubits};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_QAOA_NS, T> {
    static constexpr int kParametersPerQubitLayer = 0;

    static void validate(int qubits) {
        if (qubits < 4 || (qubits & 1)) {
            throw std::invalid_argument(
                "QAOA-NS ring requires an even qubit count of at least four");
        }
    }

    static void append_diagonal_lookups(int qubits,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto layout = QaoaNsLayerLayout::at(layer, qubits);
            for (int edge = 0; edge < qubits; ++edge) {
                append_nonshared_ring_rzz_lookup_group(
                    parameters, layout.gamma + edge, data);
            }
        }
    }

    static auto build_initial_state_lookup(int, const T*)
        -> InitialStateLookupData<T> {
        return {{{static_cast<T>(1), static_cast<T>(0)}}};
    }

    static void apply_cost(const QaoaNsLayerLayout& layout,
                           const ForwardCircuitContext<T>& context) {
        launch_nonshared_ring_rzz_forward_tile(
            context.phi->current,
            context.diagonal_lookup_at(layout.gamma),
            context.state_size,
            context.qubits,
            context.multiprocessors);
    }

    static void apply_mixer(const QaoaNsLayerLayout& layout,
                            const ForwardCircuitContext<T>& context,
                            int layer) {
        launch_non_diagonal_forward<T, NonDiagonalGate::RX, false>(
            context.phi->current,
            context.rotation_coefficients,
            context.qubits,
            layout.beta,
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void forward_initial(const ForwardCircuitContext<T>& context) {
        initialise_plus_state_kernel<T>
            <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                context.phi->current, context.state_size);
        SAD_CUDA_CHECK(cudaGetLastError());
        const auto layout = QaoaNsLayerLayout::at(0, context.qubits);
        apply_cost(layout, context);
        apply_mixer(layout, context, 0);
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = QaoaNsLayerLayout::at(layer, context.qubits);
        apply_cost(layout, context);
        apply_mixer(layout, context, layer);
    }

    static void forward_layer(int layer,
                              const ForwardCircuitContext<T>& context) {
        if (layer == 0) {
            initialise_plus_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current, context.state_size);
            SAD_CUDA_CHECK(cudaGetLastError());
        }
        const auto layout = QaoaNsLayerLayout::at(layer, context.qubits);
        apply_cost(layout, context);
        apply_mixer(layout, context, layer);
    }

    static void backward_layer(int layer,
                               const BackwardCircuitContext<T>& context) {
        const auto layout = QaoaNsLayerLayout::at(layer, context.qubits);
        launch_non_diagonal_backward<T, NonDiagonalGate::RX, false>(
            context.phi->current,
            context.lambda->current,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.beta,
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
        launch_nonshared_ring_rzz_backward(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(layout.gamma),
            context.gradients,
            context.state_size,
            context.qubits,
            layout.gamma,
            context.ordinary_grid);
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
