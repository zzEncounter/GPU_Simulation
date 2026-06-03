"""Shared optimization-loop helpers for both PennyLane and standalone paths."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

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
class LoopResult(Generic[ParamsT]):
    """Outputs from one measured gradient-descent loop."""

    final_params: ParamsT
    step_metrics: tuple[StepMetric, ...]
    gradient_wall_s: float
    measured_loop_s: float


def _current_step_number(
    step: int,
    *,
    one_based: bool,
) -> int:
    return step if one_based else step + 1


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


def run_gradient_descent_loop_from_grad(
    params: ParamsT,
    *,
    steps: int,
    stepsize: float,
    grad_fn: Callable[[ParamsT], np.ndarray],
    energy_fn: Callable[[ParamsT], float],
    apply_gradient_step: Callable[[ParamsT, np.ndarray, float], ParamsT],
    verbose: bool,
    show_progress: bool,
    report_steps: bool,
    one_based_steps: bool,
    format_step: Callable[[StepMetric], str],
    report_energy_before_step: bool = False,
) -> LoopResult[ParamsT]:
    """Run a measured loop when gradient and energy evaluators are separate."""

    step_metrics: list[StepMetric] = []
    gradient_wall_s = 0.0

    loop_start = time.perf_counter()
    step_iter = range(1, steps + 1) if one_based_steps else range(steps)
    last_progress_segment = 0
    for step in step_iter:
        step_start = time.perf_counter()
        current_step = _current_step_number(step, one_based=one_based_steps)
        progress_segment = _progress_segment(
            current_step,
            steps,
            partitions=DEFAULT_PROGRESS_PARTITIONS,
        )
        report_this_step = report_steps and progress_segment != last_progress_segment
        energy: float | None = None
        if report_this_step and report_energy_before_step:
            energy = float(energy_fn(params))

        grad_start = time.perf_counter()
        grad = np.asarray(grad_fn(params), dtype=np.float64)
        grad_wall_s = time.perf_counter() - grad_start
        gradient_wall_s += grad_wall_s

        params = apply_gradient_step(params, grad, stepsize)
        step_wall_s = time.perf_counter() - step_start
        grad_norm = float(np.linalg.norm(grad))

        if report_this_step:
            if energy is None:
                energy = float(energy_fn(params))
            step_wall_s = time.perf_counter() - step_start
            metric = StepMetric(
                step=current_step,
                energy=float(energy),
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
        gradient_wall_s=gradient_wall_s,
        measured_loop_s=time.perf_counter() - loop_start,
    )


def run_gradient_descent_loop_from_energy_grad(
    params: ParamsT,
    *,
    steps: int,
    stepsize: float,
    energy_grad_fn: Callable[[ParamsT], tuple[float, np.ndarray] | tuple[float, np.ndarray, Any]],
    apply_gradient_step: Callable[[ParamsT, np.ndarray, float], ParamsT],
    verbose: bool,
    show_progress: bool,
    report_steps: bool,
    one_based_steps: bool,
    format_step: Callable[[StepMetric], str],
    on_step_aux: Callable[[Any], None] | None = None,
) -> LoopResult[ParamsT]:
    """Run a measured loop when one call returns both energy and gradient."""

    step_metrics: list[StepMetric] = []
    gradient_wall_s = 0.0

    loop_start = time.perf_counter()
    step_iter = range(1, steps + 1) if one_based_steps else range(steps)
    last_progress_segment = 0
    for step in step_iter:
        step_start = grad_start = time.perf_counter()
        current_step = _current_step_number(step, one_based=one_based_steps)
        progress_segment = _progress_segment(
            current_step,
            steps,
            partitions=DEFAULT_PROGRESS_PARTITIONS,
        )
        raw = energy_grad_fn(params)
        aux = None
        if len(raw) == 3:
            energy, grad, aux = raw
        else:
            energy, grad = raw
        grad = np.asarray(grad, dtype=np.float64)
        grad_wall_s = time.perf_counter() - grad_start
        gradient_wall_s += grad_wall_s
        if on_step_aux is not None:
            on_step_aux(aux)

        params = apply_gradient_step(params, grad, stepsize)
        step_wall_s = time.perf_counter() - step_start
        grad_norm = float(np.linalg.norm(grad))

        if report_steps and progress_segment != last_progress_segment:
            metric = StepMetric(
                step=current_step,
                energy=float(energy),
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
        gradient_wall_s=gradient_wall_s,
        measured_loop_s=time.perf_counter() - loop_start,
    )
