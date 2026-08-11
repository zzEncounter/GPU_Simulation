#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "../kernels/diagonal.cuh"
#include "../kernels/fused_layers.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

#include <stdexcept>

namespace sad {

struct RzzLayerLayout {
    int rx;
    int rz;
    int rzz_even;
    int rzz_odd;

    static auto at(int layer, int qubits) -> RzzLayerLayout {
        const int base = layer * 3 * qubits;
        const int rzz_even = base + 2 * qubits;
        return {base, base + qubits, rzz_even, rzz_even + qubits / 2};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_RZZ_HEA, T> {
    static constexpr int kParametersPerQubitLayer = 3;

    static void validate(int qubits) {
        if (qubits & 1) {
            throw std::invalid_argument("RZZ-HEA requires an even qubit count");
        }
    }

    static void append_diagonal_lookups(int qubits,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto layout = RzzLayerLayout::at(layer, qubits);
            append_diagonal_lookup_group(parameters, layout.rz, qubits, data);
            append_diagonal_lookup_group(
                parameters, layout.rzz_even, qubits / 2, data);
            append_diagonal_lookup_group(
                parameters, layout.rzz_odd, qubits / 2, data);
        }
    }

    static auto build_initial_state_lookup(int qubits, const T* parameters)
        -> InitialStateLookupData<T> {
        const auto layout = RzzLayerLayout::at(0, qubits);
        return build_initial_product_lookup<T, NonDiagonalGate::RX>(
            qubits, parameters, layout.rx, layout.rz);
    }

    static void forward_initial(const ForwardCircuitContext<T>& context) {
        const auto layout = RzzLayerLayout::at(0, context.qubits);
        launch_product_state_initialization<
            T, FusedDiagonalMode::RZZ, false>(
            context.phi->current,
            context.state_size,
            context.qubits,
            context.initial_state_lookup,
            static_cast<const Complex<T>*>(nullptr),
            context.diagonal_lookup_at(layout.rzz_even),
            context.diagonal_lookup_at(layout.rzz_odd),
            context.ordinary_grid);
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = RzzLayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_forward<
            T, NonDiagonalGate::RX, FusedDiagonalMode::RZ_RZZ, false>(
            context.phi,
            context.rotation_coefficients,
            context.qubits,
            layout.rx,
            context.diagonal_lookup_at(layout.rz),
            context.diagonal_lookup_at(layout.rzz_even),
            context.diagonal_lookup_at(layout.rzz_odd),
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = RzzLayerLayout::at(layer, context.qubits);
        launch_non_diagonal_forward<T, NonDiagonalGate::RX>(
            context.phi->current,
            context.rotation_coefficients,
            context.qubits,
            layout.rx,
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
        launch_diagonal_forward<T, DiagonalGate::RZ>(
            context.phi->current,
            context.diagonal_lookup_at(layout.rz),
            context.state_size,
            context.qubits,
            context.qubits,
            context.ordinary_grid);
        launch_diagonal_forward<T, DiagonalGate::RZZ_EVEN>(
            context.phi->current,
            context.diagonal_lookup_at(layout.rzz_even),
            context.state_size,
            context.qubits,
            context.qubits / 2,
            context.ordinary_grid);
        launch_diagonal_forward<T, DiagonalGate::RZZ_ODD>(
            context.phi->current,
            context.diagonal_lookup_at(layout.rzz_odd),
            context.state_size,
            context.qubits,
            context.qubits / 2,
            context.ordinary_grid);
    }

    static void backward_layer(int layer, const BackwardCircuitContext<T>& context) {
        const auto layout = RzzLayerLayout::at(layer, context.qubits);
        launch_diagonal_backward<T, DiagonalGate::RZZ_ODD>(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(layout.rzz_odd),
            context.gradients,
            context.state_size,
            context.qubits,
            layout.rzz_odd,
            context.qubits / 2,
            context.ordinary_grid);
        launch_diagonal_backward<T, DiagonalGate::RZZ_EVEN>(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(layout.rzz_even),
            context.gradients,
            context.state_size,
            context.qubits,
            layout.rzz_even,
            context.qubits / 2,
            context.ordinary_grid);
        launch_diagonal_backward<T, DiagonalGate::RZ>(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(layout.rz),
            context.gradients,
            context.state_size,
            context.qubits,
            layout.rz,
            context.qubits,
            context.ordinary_grid);
        launch_non_diagonal_backward<T, NonDiagonalGate::RX>(
            context.phi->current,
            context.lambda->current,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.rx,
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void backward_layer_optimized(
        int layer, const BackwardCircuitContext<T>& context) {
        if (context.qubits <= 20 || context.qubits >= 26) {
            backward_layer_fused(layer, context);
            return;
        }
        const auto layout = RzzLayerLayout::at(layer, context.qubits);
        launch_rz_rzz_backward(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(layout.rz),
            context.diagonal_lookup_at(layout.rzz_even),
            context.diagonal_lookup_at(layout.rzz_odd),
            context.gradients,
            context.state_size,
            context.qubits,
            layout.rz,
            layout.rzz_even,
            layout.rzz_odd,
            context.multiprocessors);
        launch_non_diagonal_backward<T, NonDiagonalGate::RX>(
            context.phi->current,
            context.lambda->current,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.rx,
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void backward_layer_fused(
        int layer, const BackwardCircuitContext<T>& context) {
        const auto layout = RzzLayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_backward<
            T, NonDiagonalGate::RX, FusedDiagonalMode::RZ_RZZ, false>(
            context.phi,
            context.lambda,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.rx,
            layout.rz,
            layout.rzz_even,
            layout.rzz_odd,
            context.diagonal_lookup_at(layout.rz),
            context.diagonal_lookup_at(layout.rzz_even),
            context.diagonal_lookup_at(layout.rzz_odd),
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }
};

}  // namespace sad
