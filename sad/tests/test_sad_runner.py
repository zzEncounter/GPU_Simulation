from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sad_baseline import energy_and_grad

REFERENCE_SRC = Path(__file__).resolve().parents[2] / "pennylane-lightning" / "src"
sys.path.insert(0, str(REFERENCE_SRC))
pennylane = pytest.importorskip("pennylane")
from pennylane_lightning_baseline import energy_and_grad as reference_energy_and_grad


@pytest.mark.parametrize(
    ("circuit", "parameter_multiplier"),
    [("ra-hea", 1), ("su2-hea", 2), ("rzz-hea", 3)],
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_matches_lightning_gpu(circuit, parameter_multiplier, precision):
    kwargs = {
        "circuit": circuit,
        "random_seed": 42,
        "scalability": (4, 2),
        "precision": precision,
        "steps": 1,
        "warmup_steps": 0,
    }
    result = energy_and_grad(**kwargs)
    reference = reference_energy_and_grad(**kwargs, device_name="lightning.gpu")
    tolerance = 3e-5 if precision == "float32" else 1e-10
    assert result.energy == pytest.approx(reference.energy, abs=tolerance)
    np.testing.assert_allclose(result.grad, reference.grad, rtol=0, atol=tolerance)
    assert result.parameter_count == parameter_multiplier * 4 * 2


@pytest.mark.parametrize("circuit", ["ra-hea", "su2-hea", "rzz-hea"])
def test_cross_phase_path_matches_reference(circuit):
    kwargs = {
        "circuit": circuit,
        "random_seed": 42,
        "scalability": (12, 2),
        "precision": "float64",
        "steps": 1,
        "warmup_steps": 0,
    }
    result = energy_and_grad(**kwargs)
    reference = reference_energy_and_grad(**kwargs, device_name="lightning.gpu")
    assert result.energy == pytest.approx(reference.energy, abs=1e-10)
    np.testing.assert_allclose(result.grad, reference.grad, rtol=0, atol=1e-9)


def test_split_times_sum_to_total():
    result = energy_and_grad(
        circuit="su2-hea", scalability=(4, 1), steps=3, warmup_steps=1
    )
    expected = (
        np.asarray(result.forward_times_s)
        + np.asarray(result.hamiltonian_times_s)
        + np.asarray(result.backward_times_s)
    )
    np.testing.assert_allclose(result.step_times_s, expected, rtol=0, atol=1e-15)
    assert all(value > 0 for value in result.step_times_s)
    assert result.memory.total_workspace_mib >= 4 * result.memory.state_vector_mib


def test_validation():
    with pytest.raises(ValueError, match="must be 1"):
        energy_and_grad(batches=2, steps=1)
    with pytest.raises(ValueError, match="even"):
        energy_and_grad(circuit="rzz-hea", scalability=(5, 1), steps=1)
