"""Parity checks between the standalone CUDA backend and PennyLane."""

from __future__ import annotations

import unittest

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ising_model import build_ring_ising_hamiltonian, hardware_efficient_ring
from standalone_backend import RingIsingAdjointBackend, RingIsingConfig


class StandaloneBackendParityTest(unittest.TestCase):
    def test_energy_and_gradient_match_pennylane(self) -> None:
        num_qubits = 4
        layers = 2
        field = 0.7
        rng = np.random.default_rng(123)
        params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

        backend = RingIsingAdjointBackend(
            RingIsingConfig(num_qubits=num_qubits, layers=layers, field=field)
        )
        standalone_energy, standalone_grad = backend.energy_and_grad(params)

        dev = qml.device("lightning.gpu", wires=num_qubits)
        hamiltonian = build_ring_ising_hamiltonian(num_qubits, field)

        @qml.qnode(dev, diff_method="adjoint")
        def energy_qnode(qnode_params):
            hardware_efficient_ring(qnode_params)
            return qml.expval(hamiltonian)

        qnode_params = pnp.array(params, requires_grad=True)
        pennylane_energy = float(energy_qnode(qnode_params))
        pennylane_grad = np.asarray(qml.grad(energy_qnode)(qnode_params), dtype=np.float64)

        np.testing.assert_allclose(standalone_energy, pennylane_energy, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(standalone_grad, pennylane_grad, atol=1e-8, rtol=1e-8)

    def test_checkpoint_strategy_matches_save_param_states(self) -> None:
        num_qubits = 6
        layers = 2
        field = 1.0
        rng = np.random.default_rng(321)
        params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

        save_param_backend = RingIsingAdjointBackend(
            RingIsingConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="save_param_states",
            )
        )
        checkpoint_backend = RingIsingAdjointBackend(
            RingIsingConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="checkpoint",
                checkpoint_interval_ops=5,
            )
        )

        save_param_energy, save_param_grad = save_param_backend.energy_and_grad(params)
        checkpoint_energy, checkpoint_grad = checkpoint_backend.energy_and_grad(params)

        np.testing.assert_allclose(checkpoint_energy, save_param_energy, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(checkpoint_grad, save_param_grad, atol=1e-8, rtol=1e-8)

    def test_auto_strategy_matches_explicit_resolved_strategy(self) -> None:
        num_qubits = 6
        layers = 2
        field = 1.0
        rng = np.random.default_rng(654)
        params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

        auto_backend = RingIsingAdjointBackend(
            RingIsingConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="auto",
            )
        )

        resolved = auto_backend.strategy_resolution.resolved_strategy
        self.assertIn(resolved, {"save_param_states", "checkpoint"})
        explicit_backend = RingIsingAdjointBackend(
            RingIsingConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy=resolved,
                checkpoint_interval_ops=auto_backend.strategy_resolution.checkpoint_interval_ops
                if resolved == "checkpoint"
                else None,
            )
        )

        auto_energy, auto_grad = auto_backend.energy_and_grad(params)
        explicit_energy, explicit_grad = explicit_backend.energy_and_grad(params)
        np.testing.assert_allclose(auto_energy, explicit_energy, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(auto_grad, explicit_grad, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
