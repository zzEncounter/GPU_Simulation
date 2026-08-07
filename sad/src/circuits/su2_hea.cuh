#pragma once

namespace sad {

template <typename T>
struct CircuitExecutor<SAD_CIRCUIT_SU2_HEA, T> {
    static constexpr int kParametersPerQubitLayer = 2;

    static void validate(int) {}

    static void append_diagonal_lookups(int qubits,
                                        int layers,
                                        const T* parameters,
                                        DiagonalLookupData<T>* data) {
        for (int layer = 0; layer < layers; ++layer) {
            const size_t base = static_cast<size_t>(layer) * 2 * qubits;
            append_diagonal_lookup_group(
                parameters, base + qubits, qubits, data);
        }
    }

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        const int base = layer * 2 * context.qubits;
        launch_non_diagonal_forward<T, NonDiagonalGate::RY>(
            context.phi->current,
            context.rotation_coefficients,
            context.qubits,
            base,
            context.selected_maps,
            context.target_counts,
            context.phase_count,
            context.multiprocessors);
        launch_diagonal_forward<T, DiagonalGate::RZ>(
            context.phi->current,
            context.diagonal_lookup_at(base + context.qubits),
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
        const int base = layer * 2 * context.qubits;
        launch_cnot(context.phi,
                    context.lambda,
                    context.state_size,
                    context.qubits,
                    true,
                    context.ordinary_grid);
        launch_diagonal_backward<T, DiagonalGate::RZ>(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(base + context.qubits),
            context.gradients,
            context.state_size,
            context.qubits,
            base + context.qubits,
            context.qubits,
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
