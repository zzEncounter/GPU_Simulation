"""Shared workflow helpers for PennyLane and standalone runs."""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

from ring_ising.config import RunConfig
from ring_ising.params import make_initial_params_array


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
