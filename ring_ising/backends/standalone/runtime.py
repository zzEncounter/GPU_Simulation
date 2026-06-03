"""Python-facing wrapper around the custom standalone CUDA backend."""

from __future__ import annotations

import importlib

import numpy as np

from .config import StandaloneBackendConfig


class RingIsingAdjointBackend:
    """Python-facing entry point for the custom CUDA backend."""

    def __init__(self, config: StandaloneBackendConfig) -> None:
        config.validate()
        self.config = config
        self.gradient_strategy = config.normalized_gradient_strategy
        self.checkpoint_interval_ops = config.resolve_checkpoint_interval_ops(
            self.gradient_strategy
        )
        self.estimated_workspace_gib = config.estimated_gradient_workspace_gib_for(
            self.gradient_strategy,
            self.checkpoint_interval_ops,
        )
        self._backend = self._load_backend()
        self._cuda = self._backend.RingIsingCudaBackend(
            self.config.num_qubits,
            self.config.layers,
            float(self.config.field),
            self.gradient_strategy,
            self.config.gate_fusion,
            self.checkpoint_interval_ops,
        )

    @staticmethod
    def _load_backend():
        try:
            return importlib.import_module("standalone_backend._cuda_backend")
        except ImportError as exc:  # pragma: no cover - exercised only before build
            raise ImportError(
                "The standalone CUDA extension is not built yet. "
                "Run `.venv/bin/python setup.py build_ext --inplace` first."
            ) from exc

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        array = np.asarray(params, dtype=np.float64)
        if array.shape != self.config.param_shape:
            raise ValueError(
                f"Expected params with shape {self.config.param_shape}, got {array.shape}."
            )
        return np.ascontiguousarray(array.reshape(-1))

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
