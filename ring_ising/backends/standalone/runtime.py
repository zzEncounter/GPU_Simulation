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
        self.uses_pennylane_gate_structure = self.gradient_strategy in {
            "inverse_walk",
            "ryrz_fused",
            "structured_adjoint",
        }
        self.estimated_workspace_gib = config.estimated_gradient_workspace_gib_for(
            self.gradient_strategy
        )
        self._backend = self._load_backend()
        self._cuda = self._backend.RingIsingCudaBackend(
            self.config.num_qubits,
            self.config.layers,
            float(self.config.field),
            self.gradient_strategy,
            int(self.config.effective_structured_rotation_chunk_width),
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

    def profile_energy_and_grad(self, params: np.ndarray) -> dict[str, object]:
        """Evaluate energy/gradient and return a stage timing breakdown."""

        flat = self._normalize_params(params)
        raw = self._cuda.energy_and_grad(flat, True, True)
        timings = {
            str(name): float(value_ms)
            for name, value_ms in dict(raw.get("timings_ms", {})).items()
        }
        return {
            "energy": float(raw["energy"]),
            "gradient": np.asarray(raw["gradient"], dtype=np.float64).reshape(
                self.config.param_shape
            ),
            "timings_ms": timings,
        }
