#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "qaoa.cuh"
#include "../kernels/diagonal.cuh"
#include "../kernels/pair_cnot.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

#include <stdexcept>

namespace sad {

struct QaoaBdLayerLayout {
    int beta;
    int gamma;

    static auto at(int layer) -> QaoaBdLayerLayout {
        return {2 * layer, 2 * layer + 1};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_QAOA_BD, T> {
    static constexpr int kParametersPerQubitLayer = 0;

    static void validate(int qubits) {
        if (qubits < 4 || (qubits & 1)) {
            throw std::invalid_argument(
                "QAOA-BD ring requires an even qubit count of at least four");
        }
    }

    static void append_diagonal_lookups(int,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            append_diagonal_lookup_group(parameters,
                                          QaoaBdLayerLayout::at(layer).gamma,
                                          1,
                                          data);
        }
    }

    static auto build_initial_state_lookup(int, const T*)
        -> InitialStateLookupData<T> {
        return {{{static_cast<T>(1), static_cast<T>(0)}}};
    }

    static void apply_cost(const QaoaBdLayerLayout& layout,
                           const ForwardCircuitContext<T>& context) {
#if SAD_QAOA_BD_FUSION
        for (int parity = 0; parity < 2; ++parity) {
            launch_matching_cnot_rz_cnot_forward(
                context.phi->current,
                context.diagonal_lookup_at(layout.gamma),
                context.state_size,
                context.qubits,
                parity,
                context.ordinary_grid);
        }
        return;
#else
        for (int parity = 0; parity < 2; ++parity) {
            for (int left = parity; left < context.qubits; left += 2) {
                launch_pair_cnot(context.phi,
                                 static_cast<StatePair<T>*>(nullptr),
                                 context.state_size, left,
                                 (left + 1) % context.qubits,
                                 context.ordinary_grid);
            }
            launch_matching_rz_forward(context.phi->current,
                                       context.diagonal_lookup_at(layout.gamma),
                                       context.state_size,
                                       context.qubits,
                                       parity,
                                       context.ordinary_grid);
            for (int left = parity; left < context.qubits; left += 2) {
                launch_pair_cnot(context.phi,
                                 static_cast<StatePair<T>*>(nullptr),
                                 context.state_size, left,
                                 (left + 1) % context.qubits,
                                 context.ordinary_grid);
            }
        }
#endif
    }

    static void apply_mixer(const QaoaBdLayerLayout& layout,
                            const ForwardCircuitContext<T>& context,
                            int layer) {
        launch_non_diagonal_forward<T, NonDiagonalGate::RX, true>(
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
        const auto layout = QaoaBdLayerLayout::at(0);
        apply_cost(layout, context);
        apply_mixer(layout, context, 0);
    }

    static void forward_layer(int layer,
                              const ForwardCircuitContext<T>& context) {
        if (layer == 0) {
            initialise_plus_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current, context.state_size);
            SAD_CUDA_CHECK(cudaGetLastError());
        }
        const auto layout = QaoaBdLayerLayout::at(layer);
        apply_cost(layout, context);
        apply_mixer(layout, context, layer);
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        forward_layer(layer, context);
    }

    static void backward_layer(int layer,
                               const BackwardCircuitContext<T>& context) {
        const auto layout = QaoaBdLayerLayout::at(layer);
        launch_non_diagonal_backward<T, NonDiagonalGate::RX, true>(
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
#if SAD_QAOA_BD_FUSION
        for (int parity = 1; parity >= 0; --parity) {
            launch_matching_cnot_rz_cnot_backward(
                context.phi->current,
                context.lambda->current,
                context.diagonal_lookup_at(layout.gamma),
                context.gradients,
                layout.gamma,
                context.state_size,
                context.qubits,
                parity,
                context.ordinary_grid);
        }
        return;
#else
        for (int parity = 1; parity >= 0; --parity) {
            for (int left = parity == 0 ? context.qubits - 2 : context.qubits - 1;
                 left >= parity; left -= 2) {
                launch_pair_cnot(context.phi, context.lambda, context.state_size,
                                 left, (left + 1) % context.qubits,
                                 context.ordinary_grid);
            }
            launch_matching_rz_backward(context.phi->current,
                                        context.lambda->current,
                                        context.diagonal_lookup_at(layout.gamma),
                                        context.gradients,
                                        layout.gamma,
                                        context.state_size,
                                        context.qubits,
                                        parity,
                                        context.ordinary_grid);
            for (int left = parity == 0 ? context.qubits - 2 : context.qubits - 1;
                 left >= parity; left -= 2) {
                launch_pair_cnot(context.phi, context.lambda, context.state_size,
                                 left, (left + 1) % context.qubits,
                                 context.ordinary_grid);
            }
        }
#endif
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
