#pragma once

#include "sad_api.h"

#include "context.cuh"
#include "../kernels/diagonal.cuh"
#include "../kernels/fused_layers.cuh"
#include "../kernels/rotation.cuh"
#include "../runtime/lookups.cuh"

#include <stdexcept>

#ifndef SAD_QAOA_FUSE_COST_RX
#define SAD_QAOA_FUSE_COST_RX -1
#endif
#ifndef SAD_QAOA_INITIAL_STRATEGY
// -1 auto, 0 combine |+> initialization with cost, 1 split, 2 fuse cost+RX.
#define SAD_QAOA_INITIAL_STRATEGY -1
#endif

static_assert(SAD_QAOA_INITIAL_STRATEGY >= -1 &&
              SAD_QAOA_INITIAL_STRATEGY <= 2);

namespace sad {

// Standard ring-MaxCut QAOA.  Parameters are stored as the gate angles
// [beta_0, gamma_0, beta_1, gamma_1, ...] and are shared by every gate in the
// corresponding mixer/cost layer.
struct QaoaLayerLayout {
    int beta;
    int gamma;

    static auto at(int layer) -> QaoaLayerLayout {
        return {2 * layer, 2 * layer + 1};
    }
};

template <typename T>
__global__ void initialise_plus_state_kernel(Complex<T>* state,
                                             uint64_t state_size) {
    const T value = static_cast<T>(1) /
                    static_cast<T>(sqrt(static_cast<double>(state_size)));
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        state[index] = {value, static_cast<T>(0)};
    }
}

template <typename T>
__global__ void initialise_plus_cost_state_kernel(
    Complex<T>* state,
    uint64_t state_size,
    int qubits,
    const Complex<T>* phase_lookup) {
    const T plus_amplitude = static_cast<T>(1) /
                             static_cast<T>(
                                 sqrt(static_cast<double>(state_size)));
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < state_size;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const Complex<T> factor =
            shared_ring_rzz_factor(index, phase_lookup, qubits);
        state[index] = scale(factor, plus_amplitude);
    }
}

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_QAOA, T> {
    // expected_parameter_count handles QAOA separately: two shared values/layer.
    static constexpr int kParametersPerQubitLayer = 0;

    static void validate(int qubits) {
        if (qubits < 4 || (qubits & 1)) {
            throw std::invalid_argument(
                "QAOA ring requires an even qubit count of at least four");
        }
    }

    static void append_diagonal_lookups(int qubits,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto layout = QaoaLayerLayout::at(layer);
            if constexpr (kQaoaCompactLookup) {
                append_ring_rzz_compact_lookup_group(
                    parameters, layout.gamma, qubits, data);
            } else {
                append_shared_diagonal_lookup_group(
                    parameters, layout.gamma, qubits / 2, data);
            }
        }
    }

    static auto build_initial_state_lookup(int, const T*)
        -> InitialStateLookupData<T> {
        return {{{static_cast<T>(1), static_cast<T>(0)}}};
    }

    static void apply_cost(const QaoaLayerLayout& layout,
                           const ForwardCircuitContext<T>& context) {
        launch_shared_ring_rzz_forward(
            context.phi->current,
            context.diagonal_lookup_at(layout.gamma),
            context.state_size,
            context.qubits,
            context.ordinary_grid);
    }

    static void apply_mixer(const QaoaLayerLayout& layout,
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
        const auto layout = QaoaLayerLayout::at(0);
        const auto combined = [&]() {
            initialise_plus_cost_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current,
                    context.state_size,
                    context.qubits,
                    context.diagonal_lookup_at(layout.gamma));
            SAD_CUDA_CHECK(cudaGetLastError());
            apply_mixer(layout, context, 0);
        };
        const auto split = [&]() {
            initialise_plus_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current, context.state_size);
            SAD_CUDA_CHECK(cudaGetLastError());
            apply_cost(layout, context);
            apply_mixer(layout, context, 0);
        };
        const auto fused = [&]() {
            initialise_plus_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current, context.state_size);
            SAD_CUDA_CHECK(cudaGetLastError());
            forward_layer_optimized(0, context);
        };
#if SAD_QAOA_INITIAL_STRATEGY == 0
        combined();
#elif SAD_QAOA_INITIAL_STRATEGY == 1
        split();
#elif SAD_QAOA_INITIAL_STRATEGY == 2
        fused();
#else
        // The three variants are within 1% end to end at 20q--28q.  Keep the
        // combined initializer at every size to avoid an unsupported boundary.
        combined();
#endif
    }

    static void forward_layer_optimized(
        int layer, const ForwardCircuitContext<T>& context) {
        const auto layout = QaoaLayerLayout::at(layer);
#if SAD_QAOA_FUSE_COST_RX == 0
        apply_cost(layout, context);
        apply_mixer(layout, context, layer);
        return;
#elif SAD_QAOA_FUSE_COST_RX < 0
        // The fused path is never more than 2% slower and wins decisively at
        // several small shapes as well as at the largest sizes.
#endif
        launch_fused_non_diagonal_forward<
            T,
            NonDiagonalGate::RX,
            FusedDiagonalMode::RZZ,
            false,
            true,
            true,
            kQaoaCompactLookup>(context.phi,
                  context.rotation_coefficients,
                  context.qubits,
                  layout.beta,
                  static_cast<const Complex<T>*>(nullptr),
                  context.diagonal_lookup_at(layout.gamma),
                  context.diagonal_lookup_at(layout.gamma),
                  context.selected_maps,
                  context.target_masks,
                  context.phase_count,
                  context.multiprocessors,
                  kAlternatePhases && (layer & 1));
    }

    static void forward_layer(int layer,
                              const ForwardCircuitContext<T>& context) {
        if (layer == 0) {
            initialise_plus_state_kernel<T>
                <<<context.ordinary_grid, kOrdinaryBlockThreads>>>(
                    context.phi->current, context.state_size);
            SAD_CUDA_CHECK(cudaGetLastError());
        }
        const auto layout = QaoaLayerLayout::at(layer);
        apply_cost(layout, context);
        apply_mixer(layout, context, layer);
    }

    static void backward_layer(int layer,
                               const BackwardCircuitContext<T>& context) {
        const auto layout = QaoaLayerLayout::at(layer);
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
        launch_shared_ring_rzz_backward(
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
        if constexpr (kQaoaCompactLookup && kQaoaFusedBackward) {
            const auto layout = QaoaLayerLayout::at(layer);
            launch_qaoa_mixer_cost_backward(
                context.phi->current,
                context.lambda->current,
                context.rotation_coefficients,
                context.diagonal_lookup_at(layout.gamma),
                context.gradients,
                context.qubits,
                layout.beta,
                layout.gamma,
                context.selected_maps,
                context.target_masks,
                context.phase_count);
            return;
        }
        backward_layer(layer, context);
    }

    static void backward_layer_fused(
        int layer, const BackwardCircuitContext<T>& context) {
        if constexpr (kQaoaCompactLookup) {
            const auto layout = QaoaLayerLayout::at(layer);
            launch_qaoa_mixer_cost_backward(
                context.phi->current,
                context.lambda->current,
                context.rotation_coefficients,
                context.diagonal_lookup_at(layout.gamma),
                context.gradients,
                context.qubits,
                layout.beta,
                layout.gamma,
                context.selected_maps,
                context.target_masks,
                context.phase_count);
            return;
        }
        backward_layer(layer, context);
    }
};

}  // namespace sad
