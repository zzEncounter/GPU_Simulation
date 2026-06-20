"""Python-facing wrapper around the custom standalone CUDA backend."""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Mapping

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
        self.intrablock_block_size = config.resolve_intrablock_block_size(
            self.gradient_strategy
        )
        self.uses_pennylane_gate_structure = self.gradient_strategy in {
            "inverse_walk",
            "save_param_states",
        }
        self.effective_gate_fusion = (
            False if self.uses_pennylane_gate_structure else self.config.gate_fusion
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
            self.intrablock_block_size,
        )
        self._timing_totals_s: defaultdict[str, float] = defaultdict(float)
        self._timing_counts: defaultdict[str, int] = defaultdict(int)

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

    def _record_timings(self, raw: Mapping[str, object]) -> None:
        timings = raw.get("timings_s", {})
        if not isinstance(timings, Mapping):
            return
        for key, value in timings.items():
            self._timing_totals_s[str(key)] += float(value)
            self._timing_counts[str(key)] += 1

    @property
    def timing_totals_s(self) -> dict[str, float]:
        """Return cumulative backend timings recorded during gradient calls."""

        return dict(self._timing_totals_s)

    @property
    def timing_counts(self) -> dict[str, int]:
        """Return per-timing sample counts recorded during gradient calls."""

        return dict(self._timing_counts)

    def energy_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the Ising energy and gradient."""

        flat = self._normalize_params(params)
        raw = self._cuda.energy_and_grad(flat)
        self._record_timings(raw)
        return float(raw["energy"]), np.asarray(
            raw["gradient"], dtype=np.float64
        ).reshape(self.config.param_shape)
