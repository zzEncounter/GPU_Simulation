"""Shared run-result models and helpers for training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import numpy as np

from ring_ising.cli.common import create_telemetry_monitor
from ring_ising.training.loop import StepMetric
from ring_ising.runtime import GpuTelemetrySummary

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class LoopTimingBreakdown:
    """Timing summary for one measured training run."""

    measured_loop_s: float
    gradient_wall_s: float
    final_readout_s: float
    total_compute_s: float


@dataclass
class TrainingRunResult:
    """High-level outputs shared by both PennyLane and standalone workflows."""

    backend_label: str
    final_params: np.ndarray
    final_energy: float
    step_metrics: tuple[StepMetric, ...]
    timings: LoopTimingBreakdown
    gpu_telemetry_summary: GpuTelemetrySummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_common_run_args(
    *,
    num_qubits: int,
    layers: int,
    steps: int,
    telemetry_interval_s: float,
) -> None:
    """Validate the common subset of training-loop options."""
    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2 for the ring Ising Hamiltonian.")
    if layers < 1:
        raise ValueError("layers must be at least 1.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if telemetry_interval_s <= 0:
        raise ValueError("telemetry_interval_s must be positive.")


def run_with_optional_telemetry(
    *,
    enabled: bool,
    interval_s: float,
    live: bool,
    label: str,
    body: Callable[[], ResultT],
) -> tuple[ResultT, GpuTelemetrySummary | None]:
    """Execute a callable under optional background GPU telemetry sampling."""
    monitor = create_telemetry_monitor(
        enabled=enabled,
        interval_s=interval_s,
        live=live,
        label=label,
    )
    summary: GpuTelemetrySummary | None = None
    if monitor is not None:
        monitor.start()
    try:
        result = body()
    finally:
        if monitor is not None:
            summary = monitor.stop()
    return result, summary
