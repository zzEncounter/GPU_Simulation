#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "../kernels/mera.cuh"

namespace sad {

struct MeraLayerLayout {
    int active_count;
    int d_pair_count;
    int d_parameter_offset;
    int u_pair_count;
    int u_parameter_offset;

    static auto at(int layer, int qubits) -> MeraLayerLayout {
        int active_count = qubits;
        int parameter_offset = 0;
        for (int stage = 0; stage < layer; ++stage) {
            const int d_pair_count = (active_count - 1) / 2;
            const int u_pair_count = active_count / 2;
            parameter_offset += 2 * (d_pair_count + u_pair_count);
            active_count = (active_count + 1) / 2;
        }
        const int d_pair_count = (active_count - 1) / 2;
        const int u_pair_count = active_count / 2;
        return {active_count,
                d_pair_count,
                parameter_offset,
                u_pair_count,
                parameter_offset + 2 * d_pair_count};
    }
};

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_MERA, T> {
    static constexpr int kParametersPerQubitLayer = 0;

    static void validate(int) {}

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
        const auto layout = MeraLayerLayout::at(layer, context.qubits);
        launch_mera_matching_forward(context.phi->current,
                                     context.rotation_coefficients,
                                     context.state_size,
                                     context.qubits,
                                     layer,
                                     layout.d_parameter_offset,
                                     layout.d_pair_count,
                                     false,
                                     context.multiprocessors);
        launch_mera_matching_forward(context.phi->current,
                                     context.rotation_coefficients,
                                     context.state_size,
                                     context.qubits,
                                     layer,
                                     layout.u_parameter_offset,
                                     layout.u_pair_count,
                                     true,
                                     context.multiprocessors);
    }

    static void initialise(const ForwardCircuitContext<T>& context) {
        SAD_CUDA_CHECK(cudaMemset(
            context.phi->current, 0, context.state_size * sizeof(Complex<T>)));
        SAD_CUDA_CHECK(cudaMemset(
            context.phi->scratch, 0, context.state_size * sizeof(Complex<T>)));
        initialise_zero_state_kernel<T><<<1, 1>>>(context.phi->current);
        SAD_CUDA_CHECK(cudaGetLastError());
    }

    static void forward_initial(const ForwardCircuitContext<T>& context) {
        initialise(context);
        apply_layer(0, context);
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        apply_layer(layer, context);
    }

    static void forward_layer(int layer,
                              const ForwardCircuitContext<T>& context) {
        if (layer == 0) initialise(context);
        apply_layer(layer, context);
    }

    static void backward_layer(int layer,
                               const BackwardCircuitContext<T>& context) {
        const auto layout = MeraLayerLayout::at(layer, context.qubits);
        launch_mera_matching_backward(context.phi->current,
                                      context.lambda->current,
                                      context.rotation_coefficients,
                                      context.gradients,
                                      context.state_size,
                                      context.qubits,
                                      layer,
                                      layout.u_parameter_offset,
                                      layout.u_pair_count,
                                      true,
                                      context.multiprocessors);
        launch_mera_matching_backward(context.phi->current,
                                      context.lambda->current,
                                      context.rotation_coefficients,
                                      context.gradients,
                                      context.state_size,
                                      context.qubits,
                                      layer,
                                      layout.d_parameter_offset,
                                      layout.d_pair_count,
                                      false,
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
