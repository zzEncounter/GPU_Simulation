#pragma once

#include "sad_api.h"
#include "context.cuh"
#include "../kernels/data_reuploading.cuh"
#include "../kernels/diagonal.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

#include <stdexcept>

namespace sad {

struct DataReuploadingLayerLayout {
    int rz_z_offset;
    int ry_offset;
    int rz_x_offset;
    int parity;

    static auto at(int layer, int qubits) -> DataReuploadingLayerLayout {
        const int base = 3 * layer * qubits;
        return {base, base + qubits, base + 2 * qubits, layer & 1};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_DATA_REUPLOADING, T> {
    static constexpr int kParametersPerQubitLayer = 3;

    static void validate(int qubits) {
        if (qubits < 4 || (qubits & 1)) {
            throw std::invalid_argument(
                "Data Re-uploading requires an even qubit count >= 4");
        }
    }

    static void append_diagonal_lookups(int qubits,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto layout = DataReuploadingLayerLayout::at(layer, qubits);
            append_diagonal_lookup_group(parameters, layout.rz_z_offset, qubits,
                                         data);
            append_diagonal_lookup_group(parameters, layout.rz_x_offset, qubits,
                                         data);
        }
    }

    static auto build_initial_state_lookup(int, const T*)
        -> InitialStateLookupData<T> {
        return {{{static_cast<T>(1), static_cast<T>(0)}}};
    }

    static void initialise(const ForwardCircuitContext<T>& context) {
        SAD_CUDA_CHECK(cudaMemset(context.phi->current, 0,
                                  context.state_size * sizeof(Complex<T>)));
        initialise_zero_state_kernel<T><<<1, 1>>>(context.phi->current);
        SAD_CUDA_CHECK(cudaGetLastError());
    }

    static void apply_forward_layer(int layer,
                                    const ForwardCircuitContext<T>& context) {
        const auto layout = DataReuploadingLayerLayout::at(layer, context.qubits);
        launch_diagonal_forward<T, DiagonalGate::RZ>(
            context.phi->current, context.diagonal_lookup_at(layout.rz_z_offset),
            context.state_size, context.qubits, context.qubits,
            context.ordinary_grid);
        launch_non_diagonal_forward<T, NonDiagonalGate::RY>(
            context.phi->current, context.rotation_coefficients, context.qubits,
            layout.ry_offset, context.selected_maps, context.target_masks,
            context.phase_count, context.multiprocessors);
        launch_diagonal_forward<T, DiagonalGate::RZ>(
            context.phi->current, context.diagonal_lookup_at(layout.rz_x_offset),
            context.state_size, context.qubits, context.qubits,
            context.ordinary_grid);
        launch_data_reuploading_brickwork_forward(
            context.phi->current, context.state_size, context.qubits,
            layout.parity, context.ordinary_grid);
    }

    static void forward_initial(const ForwardCircuitContext<T>& context) {
        initialise(context);
        apply_forward_layer(0, context);
    }

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        if (layer == 0) initialise(context);
        apply_forward_layer(layer, context);
    }

    static void forward_layer_optimized(int layer,
                                        const ForwardCircuitContext<T>& context) {
        apply_forward_layer(layer, context);
    }

    static void backward_layer(int layer,
                               const BackwardCircuitContext<T>& context) {
        const auto layout = DataReuploadingLayerLayout::at(layer, context.qubits);
        launch_data_reuploading_brickwork_backward(
            context.phi->current, context.lambda->current, context.state_size,
            context.qubits, layout.parity, context.ordinary_grid);
        launch_diagonal_backward<T, DiagonalGate::RZ>(
            context.phi->current, context.lambda->current,
            context.diagonal_lookup_at(layout.rz_x_offset), context.gradients,
            context.state_size, context.qubits, layout.rz_x_offset, context.qubits,
            context.ordinary_grid);
        launch_non_diagonal_backward<T, NonDiagonalGate::RY>(
            context.phi->current, context.lambda->current,
            context.rotation_coefficients, context.gradients, context.qubits,
            layout.ry_offset, context.selected_maps, context.target_masks,
            context.phase_count, context.multiprocessors);
        launch_diagonal_backward<T, DiagonalGate::RZ>(
            context.phi->current, context.lambda->current,
            context.diagonal_lookup_at(layout.rz_z_offset), context.gradients,
            context.state_size, context.qubits, layout.rz_z_offset, context.qubits,
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
