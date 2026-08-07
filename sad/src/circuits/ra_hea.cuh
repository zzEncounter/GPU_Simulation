#pragma once

namespace sad {

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_RA_HEA, T> {
    static constexpr int kParametersPerQubitLayer = 1;

    static void validate(int) {}

    static void append_diagonal_lookups(int,
                                        int,
                                        const T*,
                                        DiagonalLookupData<T>*) {}

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        const int base = layer * context.qubits;
        launch_non_diagonal_forward<T, NonDiagonalGate::RY>(
            context.phi->current,
            context.rotation_coefficients,
            context.qubits,
            base,
            context.selected_maps,
            context.target_counts,
            context.phase_count,
            context.multiprocessors);
        launch_cnot(context.phi,
                    static_cast<StatePair<T>*>(nullptr),
                    context.state_size,
                    context.qubits,
                    false,
                    context.ordinary_grid);
    }

    static void backward_layer(int layer, const BackwardCircuitContext<T>& context) {
        const int base = layer * context.qubits;
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
            base,
            context.selected_maps,
            context.target_counts,
            context.phase_count,
            context.multiprocessors);
    }
};

}  // namespace sad
