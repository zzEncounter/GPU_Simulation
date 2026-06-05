"""Parity checks between the standalone CUDA backend and PennyLane."""

from __future__ import annotations

import unittest

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ring_ising.backends.standalone import RingIsingAdjointBackend, StandaloneBackendConfig
from ring_ising.models import build_ring_ising_hamiltonian, hardware_efficient_ring


class StandaloneBackendParityTest(unittest.TestCase):
    def test_energy_and_gradient_match_pennylane(self) -> None:
        num_qubits = 4
        layers = 2
        field = 0.7
        rng = np.random.default_rng(123)
        params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

        backend = RingIsingAdjointBackend(
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="save_param_states",
                checkpoint_interval_ops=None,
                gate_fusion=True,
            )
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
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="save_param_states",
                checkpoint_interval_ops=None,
                gate_fusion=True,
            )
        )
        checkpoint_backend = RingIsingAdjointBackend(
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="checkpoint",
                checkpoint_interval_ops=5,
                gate_fusion=True,
            )
        )

        save_param_energy, save_param_grad = save_param_backend.energy_and_grad(params)
        checkpoint_energy, checkpoint_grad = checkpoint_backend.energy_and_grad(params)

        np.testing.assert_allclose(checkpoint_energy, save_param_energy, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(checkpoint_grad, save_param_grad, atol=1e-8, rtol=1e-8)

    def test_block_fused_adjoint_strategy_matches_save_param_states(self) -> None:
        num_qubits = 7
        layers = 3
        field = 0.8
        rng = np.random.default_rng(2028)
        params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

        save_param_backend = RingIsingAdjointBackend(
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="save_param_states",
                checkpoint_interval_ops=None,
                gate_fusion=True,
            )
        )
        block_fused_backend = RingIsingAdjointBackend(
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=layers,
                field=field,
                gradient_strategy="block_fused_adjoint",
                checkpoint_interval_ops=6,
                gate_fusion=True,
            )
        )

        save_param_energy, save_param_grad = save_param_backend.energy_and_grad(params)
        block_energy, block_grad = block_fused_backend.energy_and_grad(params)

        np.testing.assert_allclose(block_energy, save_param_energy, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(block_grad, save_param_grad, atol=1e-8, rtol=1e-8)

    def test_dense_scan_matches_save_param_states(self) -> None:
        for num_qubits, layers, field, seed in (
            (5, 2, 0.9, 2026),
            (6, 2, 1.1, 2027),
        ):
            with self.subTest(num_qubits=num_qubits, layers=layers):
                rng = np.random.default_rng(seed)
                params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

                reference_backend = RingIsingAdjointBackend(
                    StandaloneBackendConfig(
                        num_qubits=num_qubits,
                        layers=layers,
                        field=field,
                        gradient_strategy="save_param_states",
                        checkpoint_interval_ops=None,
                        gate_fusion=True,
                    )
                )
                dense_backend = RingIsingAdjointBackend(
                    StandaloneBackendConfig(
                        num_qubits=num_qubits,
                        layers=layers,
                        field=field,
                        gradient_strategy="dense_scan",
                        checkpoint_interval_ops=None,
                        gate_fusion=True,
                    )
                )

                ref_energy, ref_grad = reference_backend.energy_and_grad(params)
                dense_energy, dense_grad = dense_backend.energy_and_grad(params)
                np.testing.assert_allclose(
                    dense_energy, ref_energy, atol=1e-8, rtol=1e-8
                )
                np.testing.assert_allclose(
                    dense_grad, ref_grad, atol=1e-7, rtol=1e-7
                )



if __name__ == "__main__":
    unittest.main()
