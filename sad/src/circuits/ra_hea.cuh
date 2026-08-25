#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "../kernels/fused_layers.cuh"
#include "../kernels/ring_cnot.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

#ifndef SAD_RA_FORWARD_FUSED
#define SAD_RA_FORWARD_FUSED -1
#endif
#ifndef SAD_RA_BACKWARD_FUSED
#define SAD_RA_BACKWARD_FUSED -1
#endif

namespace sad {

struct RaLayerLayout {
    int ry;

    static auto at(int layer, int qubits) -> RaLayerLayout {
        return {layer * qubits};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_RA_HEA, T> {
    static constexpr int kParametersPerQubitLayer = 1;

    static void validate(int) {}

    static void append_diagonal_lookups(int,
                                        int,
                                        const T*,
                                        DiagonalLookupData<T>*) {}

    static auto build_initial_state_lookup(int qubits, const T* parameters)
        -> InitialStateLookupData<T> {
        return build_initial_product_lookup<T, NonDiagonalGate::RY>(
            qubits, parameters, 0);
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
#if SAD_RA_FORWARD_FUSED == 0
        forward_layer(layer, context);
        return;
#endif
#if SAD_RA_FORWARD_FUSED < 0
        // Complex-amplitude measurements are tied through 22q, while fusion
        // wins decisively from 24q onward.  The real RA fast path does not pass
        // through this branch.
        if (context.qubits < 24) {
            forward_layer(layer, context);
            return;
        }
#endif
        const auto layout = RaLayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_forward<
            T, NonDiagonalGate::RY, FusedDiagonalMode::NONE, true>(
            context.phi,
            context.rotation_coefficients,
            context.qubits,
            layout.ry,
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            context.selected_maps,
            context.target_masks,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = RaLayerLayout::at(layer, context.qubits);
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
        launch_cnot(context.phi,
                    static_cast<StatePair<T>*>(nullptr),
                    context.state_size,
                    context.qubits,
                    false,
                    context.ordinary_grid);
    }

    static void backward_layer(int layer, const BackwardCircuitContext<T>& context) {
        const auto layout = RaLayerLayout::at(layer, context.qubits);
        launch_cnot(context.phi,
                    context.lambda,
                    context.state_size,
                    context.qubits,
                    true,
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
#if SAD_RA_BACKWARD_FUSED == 0
        backward_layer(layer, context);
        return;
#endif
#if SAD_RA_BACKWARD_FUSED < 0
        if (context.qubits < 22) {
            backward_layer(layer, context);
            return;
        }
#endif
        const auto layout = RaLayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_backward<
            T, NonDiagonalGate::RY, FusedDiagonalMode::NONE, true>(
            context.phi,
            context.lambda,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.ry,
            0,
            0,
            0,
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            context.selected_maps,
            context.target_masks,
            context.target_phases,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }

    static void backward_layer_fused(
        int layer, const BackwardCircuitContext<T>& context) {
        const auto layout = RaLayerLayout::at(layer, context.qubits);
        launch_fused_non_diagonal_backward<
            T, NonDiagonalGate::RY, FusedDiagonalMode::NONE, true>(
            context.phi,
            context.lambda,
            context.rotation_coefficients,
            context.gradients,
            context.qubits,
            layout.ry,
            0,
            0,
            0,
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            static_cast<const Complex<T>*>(nullptr),
            context.selected_maps,
            context.target_masks,
            context.target_phases,
            context.phase_count,
            context.multiprocessors,
            kAlternatePhases && (layer & 1));
    }
};

}  // namespace sad
