"""Compatibility wrappers for moved training result models."""

from ring_ising.training.result import (
    LoopTimingBreakdown,
    TrainingRunResult,
    run_with_optional_telemetry,
    validate_common_run_args,
)

__all__ = [
    "LoopTimingBreakdown",
    "TrainingRunResult",
    "run_with_optional_telemetry",
    "validate_common_run_args",
]
