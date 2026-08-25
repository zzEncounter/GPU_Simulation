import numpy as np
import pytest
from pennylane_lightning_baseline import energy_and_grad


@pytest.mark.parametrize(
    ("circuit", "expected_parameters"),
    [("ra-hea", 8), ("su2-hea", 16), ("rzz-hea", 24), ("mera", 8)],
)
def test_energy_and_grad_on_cpu_lightning(circuit, expected_parameters):
    result = energy_and_grad(
        circuit=circuit,
        random_seed=42,
        scalability=(4, 2),
        precision="float64",
        steps=2,
        warmup_steps=1,
        device_name="lightning.qubit",
    )
    assert np.isfinite(result.energy)
    assert result.grad.shape == (expected_parameters,)
    assert np.all(np.isfinite(result.grad))
    assert len(result.step_times_s) == 2
    assert all(value > 0 for value in result.step_times_s)


def test_seed_and_precision_are_deterministic():
    kwargs = {
        "circuit": "su2-hea",
        "random_seed": 42,
        "scalability": (4, 1),
        "precision": "float32",
        "steps": 1,
        "warmup_steps": 0,
        "device_name": "lightning.qubit",
    }
    first = energy_and_grad(**kwargs)
    second = energy_and_grad(**kwargs)
    assert first.energy == pytest.approx(second.energy, rel=1e-6)
    np.testing.assert_allclose(first.grad, second.grad, rtol=1e-6, atol=1e-6)
    assert first.grad.dtype == np.float32


def test_batches_must_be_one():
    with pytest.raises(ValueError, match="must be 1"):
        energy_and_grad(batches=2, steps=1, device_name="lightning.qubit")
