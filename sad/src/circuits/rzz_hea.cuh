#pragma once

namespace sad {

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
            const size_t base = static_cast<size_t>(layer) * 3 * qubits;
            append_diagonal_lookup_group(
                parameters, base + qubits, qubits, data);
            append_diagonal_lookup_group(
                parameters, base + 2 * qubits, qubits / 2, data);
            append_diagonal_lookup_group(
                parameters, base + 2 * qubits + qubits / 2, qubits / 2, data);
        }
    }

    static void forward_layer(int layer, const ForwardCircuitContext<T>& context) {
        const int base = layer * 3 * context.qubits;
        const int rzz_even = base + 2 * context.qubits;
        const int rzz_odd = rzz_even + context.qubits / 2;
        launch_non_diagonal_forward<T, NonDiagonalGate::RX>(
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
        launch_diagonal_forward<T, DiagonalGate::RZZ_EVEN>(
            context.phi->current,
            context.diagonal_lookup_at(rzz_even),
            context.state_size,
            context.qubits,
            context.qubits / 2,
            context.ordinary_grid);
        launch_diagonal_forward<T, DiagonalGate::RZZ_ODD>(
            context.phi->current,
            context.diagonal_lookup_at(rzz_odd),
            context.state_size,
            context.qubits,
            context.qubits / 2,
            context.ordinary_grid);
    }

    static void backward_layer(int layer, const BackwardCircuitContext<T>& context) {
        const int base = layer * 3 * context.qubits;
        const int rzz_even = base + 2 * context.qubits;
        const int rzz_odd = rzz_even + context.qubits / 2;
        launch_diagonal_backward<T, DiagonalGate::RZZ_ODD>(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(rzz_odd),
            context.gradients,
            context.state_size,
            context.qubits,
            rzz_odd,
            context.qubits / 2,
            context.ordinary_grid);
        launch_diagonal_backward<T, DiagonalGate::RZZ_EVEN>(
            context.phi->current,
            context.lambda->current,
            context.diagonal_lookup_at(rzz_even),
            context.gradients,
            context.state_size,
            context.qubits,
            rzz_even,
            context.qubits / 2,
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
        launch_non_diagonal_backward<T, NonDiagonalGate::RX>(
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
