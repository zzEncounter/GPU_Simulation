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
    "inverse_walk",
    "mode2",
    "save_param_states",
    "dense_scan",
)
TOLERANCES = {
    "inverse_walk": {"energy_atol": 1e-9, "grad_atol": 1e-8},
    "mode2": {"energy_atol": 1e-9, "grad_atol": 1e-8},
    "save_param_states": {"energy_atol": 1e-9, "grad_atol": 1e-8},
    "dense_scan": {"energy_atol": 1e-8, "grad_atol": 1e-7},
}
REMOVED_MODES = (
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
        mode2_width: int = 1,
    ) -> tuple[float, np.ndarray]:
        backend = RingIsingAdjointBackend(
            StandaloneBackendConfig(
                num_qubits=num_qubits,
                layers=LAYERS,
                field=field,
                gradient_strategy=mode,
                mode2_rotation_chunk_width=mode2_width,
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

    def test_inverse_walk_variants_match_save_param_states_beyond_small_reference_cases(self) -> None:
        num_qubits = 8
        field = 0.9
        params = self._make_params(seed=2028, num_qubits=num_qubits)

        reference_energy, reference_grad = self._standalone_result(
            mode="save_param_states",
            num_qubits=num_qubits,
            field=field,
            params=params,
        )
        for mode in ("inverse_walk", "mode2"):
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

    def test_mode2_rotation_chunk_widths_match_reference(self) -> None:
        num_qubits = 8
        field = 0.9
        params = self._make_params(seed=2030, num_qubits=num_qubits)
        reference_energy, reference_grad = self._standalone_result(
            mode="mode2",
            num_qubits=num_qubits,
            field=field,
            params=params,
            mode2_width=1,
        )

        for width in (2, 3, 4, 8):
            with self.subTest(mode2_width=width):
                energy, grad = self._standalone_result(
                    mode="mode2",
                    num_qubits=num_qubits,
                    field=field,
                    params=params,
                    mode2_width=width,
                )
                np.testing.assert_allclose(
                    energy,
                    reference_energy,
                    atol=TOLERANCES["mode2"]["energy_atol"],
                    rtol=TOLERANCES["mode2"]["energy_atol"],
                )
                np.testing.assert_allclose(
                    grad,
                    reference_grad,
                    atol=TOLERANCES["mode2"]["grad_atol"],
                    rtol=TOLERANCES["mode2"]["grad_atol"],
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

    def test_dense_scan_rejects_more_than_six_qubits(self) -> None:
        with self.assertRaises(ValueError):
            StandaloneBackendConfig(
                num_qubits=7,
                layers=LAYERS,
                field=1.0,
                gradient_strategy="dense_scan",
            ).validate()

    def test_mode2_rotation_chunk_width_is_validated(self) -> None:
        for width in (0, 9):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    StandaloneBackendConfig(
                        num_qubits=4,
                        layers=LAYERS,
                        field=1.0,
                        gradient_strategy="mode2",
                        mode2_rotation_chunk_width=width,
                    ).validate()


if __name__ == "__main__":
    unittest.main()
