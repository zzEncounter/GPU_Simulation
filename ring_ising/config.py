"""User-facing run configuration for the ring-Ising experiment."""

from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("pennylane", "standalone")
STANDALONE_GRADIENT_STRATEGIES = (
    "inverse_walk_cuQuantum",
    "structured_adjoint",
    "dense_scan",
)
EXPERIMENTAL_STANDALONE_GRADIENT_STRATEGIES = ()
STANDALONE_GRADIENT_STRATEGY_ALIASES = {
    "inverse_walk_cuquantum": "inverse_walk_cuQuantum",
}
SUPPORTED_STANDALONE_GRADIENT_STRATEGIES = (
    *STANDALONE_GRADIENT_STRATEGIES,
    *EXPERIMENTAL_STANDALONE_GRADIENT_STRATEGIES,
    *STANDALONE_GRADIENT_STRATEGY_ALIASES,
)
STRUCTURED_ROTATION_CHUNK_WIDTH_MAX = 8
DEFAULT_PROGRESS_PARTITIONS = 60


@dataclass(frozen=True)
class RunConfig:
    """Unified frontend configuration for PennyLane and standalone runs."""

    backend: str = "standalone"
    num_qubits: int = 12
    layers: int = 3
    field: float = 1.0
    steps: int = 20
    stepsize: float = 0.08
    seed: int = 7
    init_scale: float = 0.3
    verbose: bool = True
    show_progress: bool = True
    report_steps: bool = False
    gpu_telemetry: bool = False
    telemetry_interval_s: float = 0.5
    telemetry_live: bool = False
    gradient_strategy: str = "structured_adjoint"
    structured_rotation_chunk_width: int = 8
    mode2_rotation_chunk_width: int | None = None
    double_buffer: bool = False

    @property
    def effective_structured_rotation_chunk_width(self) -> int:
        if self.mode2_rotation_chunk_width is not None:
            return self.mode2_rotation_chunk_width
        return self.structured_rotation_chunk_width
