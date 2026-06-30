"""User-facing run configuration for the ring-Ising experiment."""

from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("pennylane", "standalone")
STANDALONE_GRADIENT_STRATEGIES = (
    "inverse_walk",
    "mode2",
    "save_param_states",
    "dense_scan",
)
MODE2_ROTATION_CHUNK_WIDTH_MAX = 8
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
    gradient_strategy: str = "inverse_walk"
    mode2_rotation_chunk_width: int = 8
