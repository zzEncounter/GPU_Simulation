"""Small Python wrapper around the custom CUDA backend."""

from __future__ import annotations

import numpy as np

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
    """Create trainable parameters matching the original PennyLane demo."""

    rng = np.random.default_rng(seed)
    return init_scale * rng.standard_normal((layers, num_qubits, 2))


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
        return float(self._cuda.forward_energy(flat))

    def energy_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the Ising energy and its reverse-mode gradient."""

        flat = self._normalize_params(params)
        energy, grad = self._cuda.energy_and_grad(flat)
        return float(energy), np.asarray(grad, dtype=np.float64).reshape(self.config.param_shape)

    def dense_scan_experiment(self, params: np.ndarray) -> dict[str, np.ndarray | float]:
        """Run the q<=6 brute-force dense scan experiment and return diagnostics."""

        flat = self._normalize_params(params)
        raw = self._cuda.dense_scan_experiment(flat)
        return {
            "energy": float(raw["energy"]),
            "gradient": np.asarray(raw["gradient"], dtype=np.float64).reshape(
                self.config.param_shape
            ),
            "forward_states_ri": np.asarray(raw["forward_states_ri"], dtype=np.float64),
            "backward_states_ri": np.asarray(raw["backward_states_ri"], dtype=np.float64),
            "cpu_reference_ms": float(raw["cpu_reference_ms"]),
            "gpu_scan_ms": float(raw["gpu_scan_ms"]),
            "sequential_statevector_ms": float(raw["sequential_statevector_ms"]),
        }
