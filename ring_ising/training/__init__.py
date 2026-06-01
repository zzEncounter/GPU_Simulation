"""Shared training-loop and result models."""

from .loop import (
    LoopResult,
    StepMetric,
    run_gradient_descent_loop_from_energy_grad,
    run_gradient_descent_loop_from_grad,
)
from .result import (
    LoopTimingBreakdown,
    TrainingRunResult,
    run_with_optional_telemetry,
    validate_common_run_args,
)

__all__ = [
    "LoopResult",
    "LoopTimingBreakdown",
    "StepMetric",
    "TrainingRunResult",
    "run_gradient_descent_loop_from_energy_grad",
    "run_gradient_descent_loop_from_grad",
    "run_with_optional_telemetry",
    "validate_common_run_args",
]
