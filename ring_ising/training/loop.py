"""Shared optimization-loop helpers for both PennyLane and standalone paths."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np

from ring_ising.config import DEFAULT_PROGRESS_PARTITIONS

ParamsT = TypeVar("ParamsT")


@dataclass(frozen=True)
class StepMetric:
    """Per-step metrics recorded during the measured optimization loop."""

    step: int
    energy: float
    grad_norm: float
    grad_wall_s: float
    step_wall_s: float


@dataclass(frozen=True)
class StepEvaluation:
    """One gradient evaluation with the matching energy."""

    grad: np.ndarray
    energy: float


@dataclass(frozen=True)
class LoopResult(Generic[ParamsT]):
    """Outputs from one measured gradient-descent loop."""

    final_params: ParamsT
    step_metrics: tuple[StepMetric, ...]
    wall_s: float


def _progress_segment(
    current_step: int,
    total_steps: int,
    *,
    partitions: int,
) -> int:
    if total_steps <= 0:
        return 0
    return min(
        partitions,
        max(1, ((current_step * partitions - 1) // total_steps) + 1),
    )


def _render_progress_line(
    current_step: int,
    total_steps: int,
    *,
    elapsed_s: float,
    width: int = 24,
) -> str:
    if total_steps <= 0:
        return "Progress [------------------------] 0/0"
    ratio = current_step / total_steps
    filled = min(width, int(round(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"\rProgress [{bar}] {current_step}/{total_steps} "
        f"({100.0 * ratio:5.1f}%) elapsed={elapsed_s:.3f}s"
    )


def run_gradient_descent_loop(
    params: ParamsT,
    *,
    steps: int,
    stepsize: float,
    step_fn: Callable[[ParamsT], StepEvaluation],
    apply_gradient_step: Callable[[ParamsT, np.ndarray, float], ParamsT],
    verbose: bool,
    show_progress: bool,
    report_steps: bool,
    format_step: Callable[[StepMetric], str],
) -> LoopResult[ParamsT]:
    """Run a measured loop for one generic gradient-evaluation callback."""

    step_metrics: list[StepMetric] = []

    loop_start = time.perf_counter()
    last_progress_segment = 0
    for current_step in range(1, steps + 1):
        step_start = grad_start = time.perf_counter()
        progress_segment = _progress_segment(
            current_step,
            steps,
            partitions=DEFAULT_PROGRESS_PARTITIONS,
        )
        report_this_step = report_steps and progress_segment != last_progress_segment

        evaluation = step_fn(params)
        grad = np.asarray(evaluation.grad, dtype=np.float64)
        grad_wall_s = time.perf_counter() - grad_start

        params = apply_gradient_step(params, grad, stepsize)
        step_wall_s = time.perf_counter() - step_start
        grad_norm = float(np.linalg.norm(grad))

        if report_this_step:
            metric = StepMetric(
                step=current_step,
                energy=float(evaluation.energy),
                grad_norm=grad_norm,
                grad_wall_s=grad_wall_s,
                step_wall_s=step_wall_s,
            )
            step_metrics.append(metric)
            if verbose:
                print(format_step(metric))

        if verbose and show_progress and progress_segment != last_progress_segment:
            print(
                _render_progress_line(
                    current_step,
                    steps,
                    elapsed_s=time.perf_counter() - loop_start,
                ),
                end="",
                flush=True,
            )

        last_progress_segment = progress_segment

    if verbose and show_progress and steps > 0:
        print()

    return LoopResult(
        final_params=params,
        step_metrics=tuple(step_metrics),
        wall_s=time.perf_counter() - loop_start,
    )
