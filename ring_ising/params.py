"""Shared parameter-initialization helpers."""

from __future__ import annotations

import numpy as np


def make_initial_params_array(
    *,
    num_qubits: int,
    layers: int,
    seed: int,
    init_scale: float,
) -> np.ndarray:
    """Create a float64 parameter tensor for the ring ansatz."""
    rng = np.random.default_rng(seed)
    return np.asarray(
        init_scale * rng.standard_normal((layers, num_qubits, 2)),
        dtype=np.float64,
    )
