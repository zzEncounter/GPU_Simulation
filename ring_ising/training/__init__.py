"""Shared training-loop and result models."""

from .loop import (
    LoopResult,
    StepEvaluation,
    StepMetric,
    run_gradient_descent_loop,
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
    "StepEvaluation",
    "StepMetric",
    "TrainingRunResult",
    "run_gradient_descent_loop",
    "run_with_optional_telemetry",
    "validate_common_run_args",
]
