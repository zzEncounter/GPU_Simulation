"""Reference-level checks for the intrablock_parallel algorithm."""

from __future__ import annotations

import unittest

import numpy as np

from ring_ising.backends.standalone.reference import (
    intrablock_reference_trace,
    ordinary_forward,
    sequential_adjoint_trace,
    zero_state,
)


class IntrablockReferenceTest(unittest.TestCase):
    def test_block_propagator_matches_ordinary_forward(self) -> None:
        rng = np.random.default_rng(99)
        num_qubits = 5
        layers = 2
        params = 0.2 * rng.standard_normal((layers, num_qubits, 2))
        block_size = 3

        trace = intrablock_reference_trace(
            num_qubits=num_qubits,
            layers=layers,
            field=0.8,
            params=params,
            gate_fusion=True,
            block_size=block_size,
            dtype=np.complex128,
        )

        boundary = zero_state(num_qubits, np.complex128)
        for block_idx, block_propagator in enumerate(trace.block_propagators):
            start = block_idx * block_size
            stop = min(start + block_size, len(trace.ops))
            block_ops = trace.ops[start:stop]
            ordinary = ordinary_forward(block_ops, boundary, num_qubits=num_qubits)
            matrix_based = block_propagator @ boundary
            np.testing.assert_allclose(ordinary, matrix_based, atol=1e-10, rtol=1e-10)
            boundary = ordinary

    def test_intrablock_trace_matches_sequential_reference(self) -> None:
        cases = (
            {"num_qubits": 4, "layers": 2, "field": 0.7, "block_size": 1, "gate_fusion": True},
            {"num_qubits": 4, "layers": 2, "field": 1.0, "block_size": 3, "gate_fusion": True},
            {"num_qubits": 5, "layers": 2, "field": 1.2, "block_size": 5, "gate_fusion": False},
        )
        dtype_tolerances = {
            np.complex64: 1e-4,
            np.complex128: 1e-10,
        }
        rng = np.random.default_rng(2026)

        for dtype, atol in dtype_tolerances.items():
            for case in cases:
                params = 0.15 * rng.standard_normal(
                    (case["layers"], case["num_qubits"], 2)
                )
                with self.subTest(dtype=np.dtype(dtype).name, **case):
                    sequential = sequential_adjoint_trace(
                        num_qubits=case["num_qubits"],
                        layers=case["layers"],
                        field=case["field"],
                        params=params,
                        gate_fusion=case["gate_fusion"],
                        dtype=dtype,
                    )
                    intrablock = intrablock_reference_trace(
                        num_qubits=case["num_qubits"],
                        layers=case["layers"],
                        field=case["field"],
                        params=params,
                        gate_fusion=case["gate_fusion"],
                        block_size=case["block_size"],
                        dtype=dtype,
                    )

                    np.testing.assert_allclose(
                        intrablock.forward_states,
                        sequential.forward_states,
                        atol=atol,
                        rtol=atol,
                    )
                    np.testing.assert_allclose(
                        intrablock.backward_states,
                        sequential.backward_states,
                        atol=atol,
                        rtol=atol,
                    )
                    np.testing.assert_allclose(
                        intrablock.gradient,
                        sequential.gradient,
                        atol=atol,
                        rtol=atol,
                    )
                    self.assertAlmostEqual(intrablock.energy, sequential.energy, delta=atol)


if __name__ == "__main__":
    unittest.main()
