#pragma once

#include "sad_api.h"
#include "context.cuh"
#include "../kernels/equivariant_qnn.cuh"
#include "../kernels/rotation.cuh"

#include <stdexcept>

namespace sad {

struct EquivariantQNNLayerLayout {
    int parameter_offset;
    int phase_count;
    static auto at(int layer, int qubits) -> EquivariantQNNLayerLayout {
        const int participant_count = qubits + (qubits & 1);
        return {3 * layer, participant_count - 1};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_EQUIVARIANT_QNN, T> {
    static constexpr int kParametersPerQubitLayer = 0;
    static void validate(int qubits) {
        if (qubits < 2) throw std::invalid_argument("EQNN requires at least two qubits");
    }
    static void append_diagonal_lookups(int, int, const T*, DiagonalLookupData<T>*) {}
    static auto build_initial_state_lookup(int, const T*) -> InitialStateLookupData<T> {
        return {{{static_cast<T>(1), static_cast<T>(0)}}};
    }
    static void initialise(const ForwardCircuitContext<T>& context) {
        SAD_CUDA_CHECK(cudaMemset(context.phi->current, 0,
                                  context.state_size * sizeof(Complex<T>)));
        initialise_zero_state_kernel<T><<<1, 1>>>(context.phi->current);
        SAD_CUDA_CHECK(cudaGetLastError());
    }
    static void apply_layer(int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = EquivariantQNNLayerLayout::at(layer, context.qubits);
        launch_non_diagonal_forward<T, NonDiagonalGate::RX, true>(
            context.phi->current, context.rotation_coefficients, context.qubits,
            layout.parameter_offset, context.selected_maps, context.target_masks,
            context.phase_count, context.multiprocessors);
        launch_non_diagonal_forward<T, NonDiagonalGate::RY, true>(
            context.phi->current, context.rotation_coefficients, context.qubits,
            layout.parameter_offset + 1, context.selected_maps, context.target_masks,
            context.phase_count, context.multiprocessors);
        launch_equivariant_qnn_forward(context.phi->current,
            context.rotation_coefficients, context.state_size, context.qubits,
            layout.parameter_offset, context.ordinary_grid);
    }
    static void forward_initial(const ForwardCircuitContext<T>& context) {
        initialise(context); apply_layer(0, context);
    }
    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        if (layer == 0) initialise(context);
        apply_layer(layer, context);
    }
    static void forward_layer_optimized(int layer, const ForwardCircuitContext<T>& context) { apply_layer(layer, context); }
    static void backward_layer(int layer, const BackwardCircuitContext<T>& context) {
        const auto layout = EquivariantQNNLayerLayout::at(layer, context.qubits);
        launch_equivariant_qnn_backward(context.phi->current, context.lambda->current,
            context.rotation_coefficients, context.gradients, context.state_size,
            context.qubits, layout.parameter_offset, context.ordinary_grid);
        launch_non_diagonal_backward<T, NonDiagonalGate::RY, true>(
            context.phi->current, context.lambda->current, context.rotation_coefficients,
            context.gradients, context.qubits, layout.parameter_offset + 1,
            context.selected_maps, context.target_masks, context.phase_count,
            context.multiprocessors);
        launch_non_diagonal_backward<T, NonDiagonalGate::RX, true>(
            context.phi->current, context.lambda->current, context.rotation_coefficients,
            context.gradients, context.qubits, layout.parameter_offset,
            context.selected_maps, context.target_masks, context.phase_count,
            context.multiprocessors);
    }
    static void backward_layer_optimized(int layer, const BackwardCircuitContext<T>& context) { backward_layer(layer, context); }
    static void backward_layer_fused(int layer, const BackwardCircuitContext<T>& context) { backward_layer(layer, context); }
};

}  // namespace sad
