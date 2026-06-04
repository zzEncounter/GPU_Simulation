"""User-facing run configuration for the ring-Ising experiment."""

from __future__ import annotations

from dataclasses import dataclass

BACKENDS = ("pennylane", "standalone")
STANDALONE_GRADIENT_STRATEGIES = (
    "save_param_states",
    "checkpoint",
    "dense_scan",
    "intrablock_parallel",
)
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
    gradient_strategy: str = "save_param_states"
    checkpoint_interval_ops: int | None = None
    intrablock_block_size: int | None = None
    gate_fusion: bool = True
