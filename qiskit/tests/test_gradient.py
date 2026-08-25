"""Gradient correctness tests: compare Qiskit PSR gradients against PennyLane adjoint.

Run with:
    pytest qiskit/tests/test_gradient.py -v

Requirements: both qiskit-aer (GPU optional) and pennylane-lightning must be installed.

Each test:
  1. Samples a random parameter vector with a fixed seed.
  2. Computes ⟨H⟩ and ∇⟨H⟩ via PennyLane adjoint diff (ground truth).
  3. Computes ⟨H⟩ and ∇⟨H⟩ via our Qiskit PSR implementation.
  4. Asserts that energy and gradient match within tolerance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the qiskit_baseline package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit_baseline.circuits import build_circuit, build_hamiltonian as qiskit_hamiltonian
from qiskit_baseline.runner import _EvaluatorRaw, _parameter_shift_grad_v2

# Also make the pennylane baseline importable for ground-truth comparison
_PL_SRC = Path(__file__).resolve().parents[2] / "pennylane-lightning" / "src"
sys.path.insert(0, str(_PL_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL_ENERGY = 1e-6
ATOL_GRAD = 1e-5

# Small circuits for fast testing
_TEST_CASES = [
    ("ra-hea",          4, 2),
    ("su2-hea",         4, 2),
    ("rzz-hea",         4, 2),
    ("qaoa",            4, 2),
    ("qaoa-bd",         4, 2),
    ("qaoa-ns",         4, 2),
    ("qaoa-ns-bd",      4, 2),
    ("xxz-hva",         4, 2),
    ("xxz-hva-bd",      4, 2),
    ("equivariant-qnn", 4, 2),
    ("data-reuploading",4, 2),
]


def _pennylane_energy_and_grad(circuit: str, qubits: int, layers: int, params: np.ndarray):
    """Compute energy and gradient using PennyLane adjoint diff (ground truth)."""
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane_lightning_baseline.circuits import (
        get_circuit as pl_get_circuit,
        build_hamiltonian as pl_build_hamiltonian,
    )

    circuit_spec = pl_get_circuit(circuit)
    hamiltonian = pl_build_hamiltonian(qubits, circuit_spec)
    pl_params = pnp.array(params.astype(np.float64), requires_grad=True)

    device = qml.device("default.qubit", wires=qubits, shots=None)

    @qml.qnode(device, interface="autograd", diff_method="adjoint")
    def qnode(values):
        circuit_spec.apply(values, qubits, layers)
        return qml.expval(hamiltonian)

    grad_fn = qml.grad(qnode)
    grad = grad_fn(pl_params)
    energy = float(grad_fn.forward)
    return energy, np.asarray(grad, dtype=np.float64)


def _qiskit_energy_and_grad(circuit: str, qubits: int, layers: int, params: np.ndarray,
                             gpu: bool = False):
    """Compute energy and gradient using Qiskit PSR."""
    from qiskit_aer import AerSimulator

    bundle = build_circuit(circuit, qubits, layers)
    hamiltonian = qiskit_hamiltonian(circuit, qubits)

    if gpu:
        backend = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=True)
    else:
        backend = AerSimulator(method="statevector")

    evaluator = _EvaluatorRaw(bundle, hamiltonian, backend)
    model_values = params.astype(np.float64).copy()
    energy, grad = _parameter_shift_grad_v2(evaluator, model_values, bundle)
    return energy, np.asarray(grad, dtype=np.float64)


def _sample_params(circuit: str, qubits: int, layers: int, seed: int = 42) -> np.ndarray:
    bundle = build_circuit(circuit, qubits, layers)
    rng = np.random.default_rng(seed)
    params = rng.uniform(-math.pi, math.pi, bundle.n_model_params)
    # data-reuploading: embed features (matching PennyLane runner)
    if circuit in {"data-reuploading", "data-reupload", "drqnn"}:
        for layer in range(layers):
            base = 3 * layer * qubits
            for wire in range(qubits):
                feature = 2.0 * (wire + 1) / (qubits + 1) - 1.0
                params[base + wire] += feature
                params[base + qubits + wire] += 0.5 * feature
                params[base + 2 * qubits + wire] -= 0.5 * feature
    return params.astype(np.float64)


# ---------------------------------------------------------------------------
# Parametrised tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("circuit,qubits,layers", _TEST_CASES)
def test_energy_matches_pennylane(circuit: str, qubits: int, layers: int) -> None:
    """Qiskit PSR energy must match PennyLane adjoint energy."""
    params = _sample_params(circuit, qubits, layers)
    pl_energy, _ = _pennylane_energy_and_grad(circuit, qubits, layers, params)
    qk_energy, _ = _qiskit_energy_and_grad(circuit, qubits, layers, params)
    assert np.isclose(qk_energy, pl_energy, atol=ATOL_ENERGY), (
        f"{circuit} {qubits}q: energy mismatch — "
        f"PennyLane={pl_energy:.8f}, Qiskit={qk_energy:.8f}, "
        f"diff={abs(qk_energy - pl_energy):.2e}"
    )


@pytest.mark.parametrize("circuit,qubits,layers", _TEST_CASES)
def test_gradient_matches_pennylane(circuit: str, qubits: int, layers: int) -> None:
    """Qiskit PSR gradient must match PennyLane adjoint gradient element-wise."""
    params = _sample_params(circuit, qubits, layers)
    pl_energy, pl_grad = _pennylane_energy_and_grad(circuit, qubits, layers, params)
    qk_energy, qk_grad = _qiskit_energy_and_grad(circuit, qubits, layers, params)

    assert pl_grad.shape == qk_grad.shape, (
        f"{circuit} {qubits}q: gradient shape mismatch — "
        f"PennyLane={pl_grad.shape}, Qiskit={qk_grad.shape}"
    )
    max_diff = float(np.max(np.abs(qk_grad - pl_grad)))
    assert np.allclose(qk_grad, pl_grad, atol=ATOL_GRAD), (
        f"{circuit} {qubits}q: gradient mismatch — max |diff|={max_diff:.2e} "
        f"(atol={ATOL_GRAD:.0e})\n"
        f"  PL:    {pl_grad}\n"
        f"  Qiskit:{qk_grad}"
    )


@pytest.mark.parametrize("circuit,qubits,layers", _TEST_CASES)
def test_gradient_l2_norm_close(circuit: str, qubits: int, layers: int) -> None:
    """Gradient L2 norms must be within 1% of each other."""
    params = _sample_params(circuit, qubits, layers)
    _, pl_grad = _pennylane_energy_and_grad(circuit, qubits, layers, params)
    _, qk_grad = _qiskit_energy_and_grad(circuit, qubits, layers, params)
    pl_norm = float(np.linalg.norm(pl_grad))
    qk_norm = float(np.linalg.norm(qk_grad))
    if pl_norm > 1e-8:
        rel_diff = abs(qk_norm - pl_norm) / pl_norm
        assert rel_diff < 0.01, (
            f"{circuit} {qubits}q: L2 norm relative diff={rel_diff:.2%} "
            f"(PL={pl_norm:.6f}, Qiskit={qk_norm:.6f})"
        )
