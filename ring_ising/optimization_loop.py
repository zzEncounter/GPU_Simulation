"""Compatibility wrappers for moved training-loop helpers."""

from ring_ising.training.loop import (
    LoopResult,
    StepMetric,
    run_gradient_descent_loop_from_energy_grad,
    run_gradient_descent_loop_from_grad,
)

__all__ = [
    "LoopResult",
    "StepMetric",
    "run_gradient_descent_loop_from_energy_grad",
    "run_gradient_descent_loop_from_grad",
]
