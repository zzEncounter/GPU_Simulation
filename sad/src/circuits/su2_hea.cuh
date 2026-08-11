#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "../kernels/diagonal.cuh"
#include "../kernels/fused_layers.cuh"
#include "../kernels/phased_ry.cuh"
#include "../kernels/ring_cnot.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

namespace sad {

struct Su2LayerLayout {
    int ry;
    int rz;

    static auto at(int layer, int qubits) -> Su2LayerLayout {
        const int base = layer * 2 * qubits;
        return {base, base + qubits};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_SU2_HEA, T> {
    static constexpr int kParametersPerQubitLayer = 2;

    static void validate(int) {}

    static void append_diagonal_lookups(int qubits,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto layout = Su2LayerLayout::at(layer, qubits);
            append_diagonal_lookup_group(parameters, layout.rz, qubits, data);
        }
    }

    static auto build_initial_state_lookup(int qubits, const T* parameters)
        -> InitialStateLookupData<T> {
        const auto layout = Su2LayerLayout::at(0, qubits);
        return build_initial_product_lookup<T, NonDiagonalGate::RY>(
            qubits, parameters, layout.ry, layout.rz);
    }

    static void forward_initial(const ForwardCircuitContext<T>& context) {
        launch_product_state_initialization<
            T, FusedDiagonalMode::NONE, true>(
            context.phi->current,
            context.state_size,
            context.qubits,
            context.initial_state_lookup,
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            context.ordinary_grid);
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        if (context.qubits >= 28) {
            forward_layer_phased(layer, context);
            return;
        }
        const auto layout = Su2LayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_forward<
            T, NonDiagonalGate::RY, FusedDiagonalMode::RZ, true>(
            context.phi,
            context.rotation_coefficients,
            context.qubits,
            layout.ry,
            context.diagonal_lookup_at(layout.rz),
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void forward_layer_phased(
        int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = Su2LayerLayout::at(layer, context.qubits);
        launch_phased_ry_cnot_forward(context.phi,
                                      context.rotation_coefficients,
                                      context.qubits,
                                      layout.ry,
                                      layout.rz,
                                      context.selected_maps,
                                      context.target_masks,
                                      context.phase_count,
                                      context.multiprocessors);
    }

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = Su2LayerLayout::at(layer, context.qubits);
        launch_non_diagonal_forward<T, NonDiagonalGate::RY>(
            context.phi->current,
            context.rotation_coefficients,
            context.qubits,
            layout.ry,
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
        launch_cnot(context.phi,
                    static_cast<StatePair<T>*>(nullptr),
                    context.state_size,
                    context.qubits,
                    false,
                    context.ordinary_grid);
    }

    static void backward_layer(int layer, const BackwardCircuitContext<T>& context) {
        const auto layout = Su2LayerLayout::at(layer, context.qubits);
        launch_cnot(context.phi,
                    context.lambda,
                    context.state_size,
                    context.qubits,
                    true,
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
        launch_non_diagonal_backward<T, NonDiagonalGate::RY>(
            context.phi->current,
            context.lambda->current,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.ry,
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void backward_layer_optimized(
        int layer, const BackwardCircuitContext<T>& context) {
        if (context.qubits == 20 || context.qubits >= 28) {
            backward_layer(layer, context);
            return;
        }
        const auto layout = Su2LayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_backward<
            T, NonDiagonalGate::RY, FusedDiagonalMode::RZ, true>(
            context.phi,
            context.lambda,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.ry,
            layout.rz,
            0,
            0,
            context.diagonal_lookup_at(layout.rz),
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void backward_layer_fused(
        int layer, const BackwardCircuitContext<T>& context) {
        backward_layer_optimized(layer, context);
    }
};

}  // namespace sad
