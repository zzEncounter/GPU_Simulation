import numpy as np
import pytest

from pennylane_lightning_baseline import energy_and_grad, native_energy_and_grad


@pytest.mark.parametrize(
    ("circuit", "expected_parameters"),
    [
        ("ra-hea", 8),
        ("su2-hea", 16),
        ("rzz-hea", 24),
        ("qaoa", 4),
        ("xxz-hva", 24),
    ],
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_native_matches_qnode_lightning_gpu(circuit, expected_parameters, precision):
    kwargs = {
        "circuit": circuit,
        "random_seed": 42,
        "scalability": (4, 2),
        "precision": precision,
        "steps": 1,
        "warmup_steps": 0,
    }
    native = native_energy_and_grad(**kwargs)
    qnode = energy_and_grad(**kwargs, device_name="lightning.gpu")
    tolerance = 3e-5 if precision == "float32" else 1e-10
    assert native.energy == pytest.approx(qnode.energy, abs=tolerance)
    np.testing.assert_allclose(native.grad, qnode.grad, rtol=0, atol=tolerance)
    assert native.parameter_count == expected_parameters
    assert native.grad.dtype == (np.float32 if precision == "float32" else np.float64)


def test_native_phase_times_sum_to_total():
    result = native_energy_and_grad(
        circuit="su2-hea", scalability=(4, 1), steps=3, warmup_steps=1
    )
    expected = (
        np.asarray(result.forward_times_s)
        + np.asarray(result.hamiltonian_times_s)
        + np.asarray(result.backward_times_s)
    )
    np.testing.assert_allclose(result.step_times_s, expected, rtol=0, atol=1e-15)
    assert all(value > 0 for value in result.step_times_s)


def test_native_validation():
    with pytest.raises(ValueError, match="must be 1"):
        native_energy_and_grad(batches=2, steps=1)
    with pytest.raises(ValueError, match="device_name"):
        native_energy_and_grad(device_name="lightning.gpu", steps=1)
