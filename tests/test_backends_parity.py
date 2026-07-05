"""Parity checks between standalone baseline modes and the PennyLane reference."""

from __future__ import annotations

import unittest

from autograd import value_and_grad
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ring_ising.backends.standalone import RingIsingAdjointBackend, StandaloneBackendConfig
from ring_ising.models import build_ring_ising_hamiltonian, hardware_efficient_ring

LAYERS = 2
CASES = (
    {"num_qubits": 2, "field": 0.7},
    {"num_qubits": 4, "field": 1.0},
    {"num_qubits": 6, "field": 1.1},
)
SEEDS = (123, 321, 2026)
MODES = (
    "inverse_walk_cuQuantum",
    "structured_adjoint",
    "dense_scan",
)
TOLERANCES = {
    "inverse_walk_cuQuantum": {"energy_atol": 1e-9, "grad_atol": 1e-8},
    "structured_adjoint": {"energy_atol": 1e-9, "grad_atol": 1e-8},
    "dense_scan": {"energy_atol": 1e-8, "grad_atol": 1e-7},
}
REMOVED_MODES = (
    "inverse_walk",
    "ryrz_fused",
    "mode2",
    "save_param_states",
    "checkpoint",
    "block_fused_adjoint",
    "intrablock_parallel",
    "mode2_chunk2",
    "mode2_chunk3",
    "mode2_dense2",
    "mode2_dense3",
)


class StandaloneBackendParityTest(unittest.TestCase):
    def _make_params(self, *, seed: int, num_qubits: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return 0.2 * rng.standard_normal((LAYERS, num_qubits, 2))

    def _pennylane_reference(
        self,
        *,
        num_qubits: int,
        field: float,
        params: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        dev = qml.device("lightning.gpu", wires=num_qubits)
        hamiltonian = build_ring_ising_hamiltonian(num_qubits, field)

        @qml.qnode(dev, diff_method="adjoint")
        def energy_qnode(qnode_params):
            hardware_efficient_ring(qnode_params)
            return qml.expval(hamiltonian)

        qnode_params = pnp.array(params, requires_grad=True)
        energy, grad = value_and_grad(energy_qnode)(qnode_params)
        return float(energy), np.asarray(grad, dtype=np.float64)

    def _standalone_result(
        self,
        *,
        mode: str,
        num_qubits: int,
        field: float,
        params: np.ndarray,
        structured_width: int = 1,
    ) -> tuple[float, np.ndarray]:
        backend = RingIsingAdjointBackend(
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=LAYERS,
                field=field,
                gradient_strategy=mode,
                structured_rotation_chunk_width=structured_width,
            )
        )
        return backend.energy_and_grad(params)

    def test_each_mode_matches_pennylane_reference(self) -> None:
        for case in CASES:
            num_qubits = int(case["num_qubits"])
            field = float(case["field"])
            for seed in SEEDS:
                params = self._make_params(seed=seed, num_qubits=num_qubits)
                ref_energy, ref_grad = self._pennylane_reference(
                    num_qubits=num_qubits,
                    field=field,
                    params=params,
                )
                for mode in MODES:
                    tolerances = TOLERANCES[mode]
                    with self.subTest(
                        mode=mode,
                        num_qubits=num_qubits,
                        layers=LAYERS,
                        field=field,
                        seed=seed,
                    ):
                        energy, grad = self._standalone_result(
                            mode=mode,
                            num_qubits=num_qubits,
                            field=field,
                            params=params,
                        )
                        np.testing.assert_allclose(
                            energy,
                            ref_energy,
                            atol=tolerances["energy_atol"],
                            rtol=tolerances["energy_atol"],
                        )
                        np.testing.assert_allclose(
                            grad,
                            ref_grad,
                            atol=tolerances["grad_atol"],
                            rtol=tolerances["grad_atol"],
                        )

    def test_cuquantum_and_structured_match_beyond_small_reference_cases(self) -> None:
        num_qubits = 8
        field = 0.9
        params = self._make_params(seed=2028, num_qubits=num_qubits)

        reference_energy, reference_grad = self._standalone_result(
            mode="structured_adjoint",
            num_qubits=num_qubits,
            field=field,
            params=params,
        )
        for mode in ("inverse_walk_cuQuantum", "structured_adjoint"):
            with self.subTest(mode=mode):
                energy, grad = self._standalone_result(
                    mode=mode,
                    num_qubits=num_qubits,
                    field=field,
                    params=params,
                )

                np.testing.assert_allclose(
                    energy,
                    reference_energy,
                    atol=TOLERANCES[mode]["energy_atol"],
                    rtol=TOLERANCES[mode]["energy_atol"],
                )
                np.testing.assert_allclose(
                    grad,
                    reference_grad,
                    atol=TOLERANCES[mode]["grad_atol"],
                    rtol=TOLERANCES[mode]["grad_atol"],
                )

    def test_structured_rotation_chunk_widths_match_reference(self) -> None:
        num_qubits = 10
        field = 0.9
        params = self._make_params(seed=2030, num_qubits=num_qubits)
        reference_energy, reference_grad = self._standalone_result(
            mode="structured_adjoint",
            num_qubits=num_qubits,
            field=field,
            params=params,
            structured_width=1,
        )

        for width in (2, 3, 4, 8):
            with self.subTest(structured_width=width):
                energy, grad = self._standalone_result(
                    mode="structured_adjoint",
                    num_qubits=num_qubits,
                    field=field,
                    params=params,
                    structured_width=width,
                )
                np.testing.assert_allclose(
                    energy,
                    reference_energy,
                    atol=TOLERANCES["structured_adjoint"]["energy_atol"],
                    rtol=TOLERANCES["structured_adjoint"]["energy_atol"],
                )
                np.testing.assert_allclose(
                    grad,
                    reference_grad,
                    atol=TOLERANCES["structured_adjoint"]["grad_atol"],
                    rtol=TOLERANCES["structured_adjoint"]["grad_atol"],
                )

    def test_dense_scan_matches_structured_for_deeper_small_qubit_circuit(
        self,
    ) -> None:
        field = 1.0
        for num_qubits, layers, seed in ((6, 32, 2032), (8, 8, 2033)):
            rng = np.random.default_rng(seed)
            params = 0.2 * rng.standard_normal((layers, num_qubits, 2))

            reference_backend = RingIsingAdjointBackend(
                StandaloneBackendConfig(
                    num_qubits=num_qubits,
                    layers=layers,
                    field=field,
                    gradient_strategy="structured_adjoint",
                )
            )
            dense_backend = RingIsingAdjointBackend(
                StandaloneBackendConfig(
                    num_qubits=num_qubits,
                    layers=layers,
                    field=field,
                    gradient_strategy="dense_scan",
                )
            )

            reference_energy, reference_grad = reference_backend.energy_and_grad(params)
            dense_energy, dense_grad = dense_backend.energy_and_grad(params)
            np.testing.assert_allclose(
                dense_energy,
                reference_energy,
                atol=TOLERANCES["dense_scan"]["energy_atol"],
                rtol=TOLERANCES["dense_scan"]["energy_atol"],
            )
            np.testing.assert_allclose(
                dense_grad,
                reference_grad,
                atol=TOLERANCES["dense_scan"]["grad_atol"],
                rtol=TOLERANCES["dense_scan"]["grad_atol"],
            )

    def test_removed_legacy_modes_are_rejected(self) -> None:
        for mode in REMOVED_MODES:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    StandaloneBackendConfig(
                        num_qubits=4,
                        layers=LAYERS,
                        field=1.0,
                        gradient_strategy=mode,
                    ).validate()

    def test_dense_scan_rejects_more_than_eight_qubits(self) -> None:
        with self.assertRaises(ValueError):
            StandaloneBackendConfig(
                num_qubits=9,
                layers=LAYERS,
                field=1.0,
                gradient_strategy="dense_scan",
            ).validate()

    def test_dense_scan_workspace_estimate_uses_layer_level_ops(self) -> None:
        config = StandaloneBackendConfig(
            num_qubits=6,
            layers=32,
            field=1.0,
            gradient_strategy="dense_scan",
        )
        num_ops = config.layers * 2
        padded = 1 << (num_ops - 1).bit_length()
        matrix_count = num_ops + padded + max(0, padded - 1) + 2
        vector_count = 3 * padded + max(num_ops + 1, padded) + 3
        expected_workspace_gib = (
            matrix_count * config.dense_matrix_nbytes
            + vector_count * config.statevector_nbytes
            + config.num_params * (8 + 8 + 4)
        ) / (1024**3)
        self.assertEqual(
            config.estimated_gradient_state_buffers_for("dense_scan"),
            3 * padded + max(num_ops + 1, padded) + 3,
        )
        self.assertAlmostEqual(
            config.estimated_gradient_workspace_gib_for("dense_scan"),
            expected_workspace_gib,
        )

    def test_structured_rotation_chunk_width_is_validated(self) -> None:
        for width in (0, 9):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    StandaloneBackendConfig(
                        num_qubits=4,
                        layers=LAYERS,
                        field=1.0,
                        gradient_strategy="structured_adjoint",
                        structured_rotation_chunk_width=width,
                    ).validate()


if __name__ == "__main__":
    unittest.main()
