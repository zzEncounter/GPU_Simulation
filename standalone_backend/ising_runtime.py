"""Small Python wrapper around the custom CUDA backend."""

from __future__ import annotations

import numpy as np

from ring_ising.params import make_initial_params_array

from .config import RingIsingConfig, StrategyResolution
from .strategy import resolve_strategy

try:
    from . import _cuda_backend
except ImportError as exc:  # pragma: no cover - exercised only before build
    _cuda_backend = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def make_initial_params(
    num_qubits: int, layers: int, seed: int = 7, init_scale: float = 0.3
) -> np.ndarray:
    """Create trainable parameters matching the ring ansatz shape."""
    return make_initial_params_array(
        num_qubits=num_qubits,
        layers=layers,
        seed=seed,
        init_scale=init_scale,
    )


class RingIsingAdjointBackend:
    """Python-facing entry point for the custom CUDA backend."""

    def __init__(self, config: RingIsingConfig) -> None:
        config.validate()
        self.config = config
        self._backend = self._load_backend()
        self._resolution = resolve_strategy(config)
        self._resolved_checkpoint_interval_ops = self._resolution.checkpoint_interval_ops
        self._cuda = self._backend.RingIsingCudaBackend(
            self.config.num_qubits,
            self.config.layers,
            float(self.config.field),
            self._resolution.resolved_strategy,
            self.config.gate_fusion,
            self._resolved_checkpoint_interval_ops,
        )

    @staticmethod
    def _load_backend():
        if _cuda_backend is None:
            raise ImportError(
                "The standalone CUDA extension is not built yet. "
                "Run `.venv/bin/python setup.py build_ext --inplace` first."
            ) from _IMPORT_ERROR
        return _cuda_backend

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        array = np.asarray(params, dtype=np.float64)
        if array.shape != self.config.param_shape:
            raise ValueError(
                f"Expected params with shape {self.config.param_shape}, got {array.shape}."
            )
        return np.ascontiguousarray(array.reshape(-1))

    @property
    def strategy_resolution(self) -> StrategyResolution:
        return self._resolution

    def energy(self, params: np.ndarray) -> float:
        """Evaluate the Ising energy using the custom CUDA backend."""

        flat = self._normalize_params(params)
        raw = self._cuda.energy_and_grad(flat, False)
        return float(raw["energy"])

    def energy_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the Ising energy and gradient."""

        flat = self._normalize_params(params)
        raw = self._cuda.energy_and_grad(flat)
        return float(raw["energy"]), np.asarray(
            raw["gradient"], dtype=np.float64
        ).reshape(self.config.param_shape)
