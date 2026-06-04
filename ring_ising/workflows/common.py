"""Shared workflow helpers for PennyLane and standalone runs."""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

from ring_ising.config import RunConfig
from ring_ising.params import make_initial_params_array
from ring_ising.training import TrainingRunResult


def make_initial_params(config: RunConfig) -> np.ndarray:
    """Create the shared initial parameter tensor for one run."""
    return make_initial_params_array(
        num_qubits=config.num_qubits,
        layers=config.layers,
        seed=config.seed,
        init_scale=config.init_scale,
    )


def apply_gradient_step(params: np.ndarray, grad: np.ndarray, stepsize: float) -> np.ndarray:
    """Apply one plain gradient-descent update in float64."""
    return np.asarray(params, dtype=np.float64) - stepsize * np.asarray(grad, dtype=np.float64)


def as_pennylane_params(params: np.ndarray) -> pnp.ndarray:
    """Convert shared parameters into a PennyLane trainable tensor."""
    return pnp.array(np.asarray(params, dtype=np.float64), requires_grad=True)


def print_problem_summary(config: RunConfig, params: np.ndarray) -> None:
    """Print the common problem-definition summary."""
    print("Problem:")
    print(f"  Qubits: {config.num_qubits}")
    print(f"  Layers: {config.layers}")
    print(f"  Field strength: {config.field}")
    print(f"  Parameter tensor shape: {tuple(np.asarray(params).shape)}")
    print()


def print_result_summary(result: TrainingRunResult, config: RunConfig) -> None:
    """Print the common run summary shared by both backends."""
    avg_step_s = result.timings.wall_s / config.steps if config.steps else 0.0

    print()
    print("Summary:")
    print(f"  Final energy: {result.final_energy:.10f}")
    if config.steps:
        print(f"  Average step time: {1000.0 * avg_step_s:.3f} ms")
    print(f"  Total wall time: {result.timings.wall_s:.4f} s")
