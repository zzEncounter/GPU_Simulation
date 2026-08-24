from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sad_baseline import energy_and_grad
from sad_baseline import runner as sad_runner

REFERENCE_SRC = Path(__file__).resolve().parents[2] / "pennylane-lightning" / "src"
sys.path.insert(0, str(REFERENCE_SRC))
pennylane = pytest.importorskip("pennylane")
from pennylane_lightning_baseline import energy_and_grad as reference_energy_and_grad


@pytest.mark.parametrize(
    ("circuit", "expected_parameters"),
    [
        ("ra-hea", 8),
        ("su2-hea", 16),
        ("rzz-hea", 24),
        ("qaoa", 4),
        ("qaoa-ns", 16),
        ("xxz-hva", 24),
        ("mera", 8),
        ("eqnn", 6),
    ],
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_matches_lightning_gpu(circuit, expected_parameters, precision):
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
    assert result.parameter_count == expected_parameters


@pytest.mark.parametrize(
    "circuit", ["ra-hea", "su2-hea", "rzz-hea", "qaoa", "qaoa-ns", "xxz-hva"]
)
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


@pytest.mark.parametrize(("qubits", "layers"), [(3, 2), (6, 3), (8, 3)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_mera_matches_lightning_gpu(qubits, layers, precision):
    kwargs = {
        "circuit": "mera",
        "random_seed": 42,
        "scalability": (qubits, layers),
        "precision": precision,
        "steps": 1,
        "warmup_steps": 0,
    }
    result = energy_and_grad(**kwargs)
    reference = reference_energy_and_grad(**kwargs, device_name="lightning.gpu")
    tolerance = 3e-5 if precision == "float32" else 1e-10
    assert result.energy == pytest.approx(reference.energy, abs=tolerance)
    np.testing.assert_allclose(result.grad, reference.grad, rtol=0, atol=tolerance)


@pytest.mark.parametrize("mode", ["optimized", "all-fused"])
@pytest.mark.parametrize(
    "circuit", ["ra-hea", "su2-hea", "rzz-hea", "qaoa", "qaoa-ns", "xxz-hva"]
)
def test_optimized_paths_match_retained_legacy_path(circuit, mode, monkeypatch):
    kwargs = {
        "circuit": circuit,
        "random_seed": 42,
        "scalability": (12, 2),
        "precision": "float64",
        "steps": 1,
        "warmup_steps": 0,
    }
    monkeypatch.setenv("SAD_EXECUTION_MODE", "legacy")
    legacy = energy_and_grad(**kwargs)
    monkeypatch.setenv("SAD_EXECUTION_MODE", mode)
    optimized = energy_and_grad(**kwargs)
    assert optimized.energy == pytest.approx(legacy.energy, abs=1e-11)
    np.testing.assert_allclose(optimized.grad, legacy.grad, rtol=0, atol=1e-10)


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
    assert result.kernel_variant == "f32r2_b32r2"


@pytest.mark.parametrize(
    ("circuit_id", "qubits", "expected"),
    [
        (0, 18, "f128r2_b128r2"),
        (0, 20, "f64r4_b64r4"),
        (0, 22, "f128r2_b64r4"),
        (0, 28, "f64r3_b64r4"),
        (1, 20, "f128r2_b128r2"),
        (1, 22, "f128r2_b64r4"),
        (1, 24, "f128r2_b64r4"),
        (2, 10, "f32r2_b32r2_d6"),
        (2, 20, "f128r2_b128r2_d10"),
        (2, 22, "f128r2_b128r2"),
        (2, 24, "f128r2_b64r4"),
        (2, 26, "f128r2_b128r3"),
        (3, 18, "f128r2_b32r2"),
        (3, 26, "f128r2_b128r3"),
        (3, 28, "f64r4_b128r3"),
        (4, 20, "f128r2_b128r2"),
        (4, 28, "f128r3_b32r3"),
    ],
)
def test_size_dependent_kernel_variant(
    circuit_id, qubits, expected, monkeypatch
):
    monkeypatch.delenv("SAD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("SAD_DISABLE_VARIANT_DISPATCH", raising=False)
    name, _ = sad_runner._select_library(circuit_id, qubits, "optimized")
    assert name == expected


def test_fixed_parameter_policy_disables_variant_dispatch(monkeypatch):
    monkeypatch.delenv("SAD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("SAD_DISABLE_VARIANT_DISPATCH", "1")
    for circuit_id in range(5):
        name, _ = sad_runner._select_library(circuit_id, 28, "optimized")
        assert name == "f128r2_b128r2"
    assert sad_runner._select_library(2, 10, "optimized")[0] == (
        "f128r2_b128r2_d6"
    )
    assert sad_runner._select_library(2, 20, "optimized")[0] == (
        "f128r2_b128r2_d10"
    )
    assert sad_runner._select_library(4, 6, "optimized")[0] == (
        "f128r2_b128r2_xsep"
    )


def test_trainable_phased_ry_rz_keeps_fused_cnot(monkeypatch):
    kwargs = {
        "circuit": "su2-hea",
        "random_seed": 17,
        "scalability": (12, 3),
        "precision": "float64",
        "steps": 1,
        "warmup_steps": 0,
    }
    monkeypatch.setenv("SAD_EXECUTION_MODE", "legacy")
    legacy = energy_and_grad(**kwargs)
    monkeypatch.setenv("SAD_EXECUTION_MODE", "phased-forward")
    phased = energy_and_grad(**kwargs)
    assert phased.energy == pytest.approx(legacy.energy, abs=1e-11)
    np.testing.assert_allclose(phased.grad, legacy.grad, rtol=0, atol=1e-10)


@pytest.mark.parametrize("circuit", ["ra-hea", "su2-hea", "rzz-hea", "qaoa"])
def test_nonuniform_phase_plan_matches_reference(circuit):
    base_kwargs = {
        "circuit": circuit,
        "random_seed": 23,
        "scalability": (12, 2),
        "precision": "float64",
        "steps": 1,
        "warmup_steps": 0,
    }
    result = energy_and_grad(
        **base_kwargs,
        forward_phase_plan="compact:L4R2W0-L4R2W0",
        backward_phase_plan="compact:L4R2W0-L4R2W0",
    )
    reference = reference_energy_and_grad(
        **base_kwargs, device_name="lightning.gpu"
    )
    assert result.energy == pytest.approx(reference.energy, abs=1e-10)
    np.testing.assert_allclose(result.grad, reference.grad, rtol=0, atol=1e-9)


def test_invalid_nonuniform_phase_plan_is_rejected():
    with pytest.raises(RuntimeError, match="invalid forward phase plan"):
        energy_and_grad(
            circuit="su2-hea",
            scalability=(12, 1),
            steps=1,
            forward_phase_plan="compact:L5R2W2",
        )


def test_small_qubit_shape_and_phase_dispatch(monkeypatch):
    monkeypatch.delenv("SAD_DISABLE_VARIANT_DISPATCH", raising=False)
    assert sad_runner._select_library(0, 8, "optimized")[0] == "f64r2_b64r2"
    assert sad_runner._select_library(0, 12, "optimized")[0] == "f32r2_b32r2"
    assert sad_runner._select_library(4, 6, "optimized")[0] == "f32r2_b32r2_xsep"
    assert sad_runner._select_library(0, 18, "optimized")[0] == "f128r2_b128r2"
    assert sad_runner._select_library(4, 18, "optimized")[0] == "f32r2_b32r2"
    assert sad_runner._select_library(1, 20, "optimized")[0] == "f128r2_b128r2"
    assert sad_runner._select_library(2, 10, "optimized")[0] == "f32r2_b32r2_d6"
    assert sad_runner._select_library(2, 20, "optimized")[0] == "f128r2_b128r2_d10"
    assert sad_runner._select_library(4, 20, "optimized")[0] == "f128r2_b128r2"
    assert sad_runner._select_library(3, 24, "optimized")[0] == "f128r2_b128r2"
    assert sad_runner._select_phase_plans(3, 10, "optimized") == (
        "compact:L2R2W0-L4R2W0",
        "compact:L4R2W0-L2R2W0",
    )
    assert sad_runner._select_phase_plans(2, 12, "optimized") == ("", "")


def test_conservative_policy_disables_shape_and_phase_dispatch(monkeypatch):
    monkeypatch.setenv("SAD_DISABLE_VARIANT_DISPATCH", "1")
    assert sad_runner._select_library(3, 10, "optimized")[0] == (
        "f128r2_b128r2"
    )
    assert sad_runner._select_phase_plans(3, 10, "optimized") == ("", "")


def test_automatic_phase_plans_apply_only_to_production_dispatch(monkeypatch):
    monkeypatch.delenv("SAD_LIBRARY_PATH", raising=False)
    production = energy_and_grad(
        circuit="qaoa", scalability=(10, 1), steps=1, warmup_steps=0
    )
    assert production.forward_phase_plan == "compact:L2R2W0-L4R2W0"
    assert production.backward_phase_plan == "compact:L4R2W0-L2R2W0"

    monkeypatch.setenv("SAD_LIBRARY_PATH", str(sad_runner._DEFAULT_LIBRARY))
    custom = energy_and_grad(
        circuit="qaoa", scalability=(10, 1), steps=1, warmup_steps=0
    )
    assert custom.kernel_variant.startswith("custom:")
    assert custom.forward_phase_plan == ""
    assert custom.backward_phase_plan == ""


def test_validation():
    with pytest.raises(ValueError, match="must be 1"):
        energy_and_grad(batches=2, steps=1)
    with pytest.raises(ValueError, match="even"):
        energy_and_grad(circuit="rzz-hea", scalability=(5, 1), steps=1)
    with pytest.raises(ValueError, match="at least four"):
        energy_and_grad(circuit="qaoa", scalability=(2, 1), steps=1)
    with pytest.raises(ValueError, match="at least four"):
        energy_and_grad(circuit="qaoa-ns", scalability=(2, 1), steps=1)
    with pytest.raises(ValueError, match="even"):
        energy_and_grad(circuit="xxz-hva", scalability=(5, 1), steps=1)
    with pytest.raises(ValueError, match="ceil"):
        energy_and_grad(circuit="mera", scalability=(6, 2), steps=1)
