"""Shared optimization-loop helpers for both baseline and standalone paths."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import numpy as np

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


def _should_report_step(
    step: int,
    total_steps: int,
    report_every: int,
    *,
    one_based: bool,
) -> bool:
    if total_steps <= 0:
        return False
    current_step = step if one_based else step + 1
    return (
        current_step == 1
        or current_step == total_steps
        or current_step % report_every == 0
    )


def run_gradient_descent_loop_from_grad(
    params: ParamsT,
    *,
    steps: int,
    stepsize: float,
    report_every: int,
    grad_fn: Callable[[ParamsT], np.ndarray],
    energy_fn: Callable[[ParamsT], float],
    apply_gradient_step: Callable[[ParamsT, np.ndarray, float], ParamsT],
    verbose: bool,
    one_based_steps: bool,
    format_step: Callable[[StepMetric], str],
) -> LoopResult[ParamsT]:
    """Run a measured loop when gradient and energy evaluators are separate."""

    step_metrics: list[StepMetric] = []
    gradient_wall_s = 0.0

    loop_start = time.perf_counter()
    step_iter = range(1, steps + 1) if one_based_steps else range(steps)
    for step in step_iter:
        step_start = grad_start = time.perf_counter()
        grad = np.asarray(grad_fn(params), dtype=np.float64)
        grad_wall_s = time.perf_counter() - grad_start
        gradient_wall_s += grad_wall_s

        params = apply_gradient_step(params, grad, stepsize)
        step_wall_s = time.perf_counter() - step_start
        grad_norm = float(np.linalg.norm(grad))

        if _should_report_step(step, steps, report_every, one_based=one_based_steps):
            energy = float(energy_fn(params))
            step_wall_s = time.perf_counter() - step_start
            metric = StepMetric(
                step=step,
                energy=energy,
                grad_norm=grad_norm,
                grad_wall_s=grad_wall_s,
                step_wall_s=step_wall_s,
            )
            step_metrics.append(metric)
            if verbose:
                print(format_step(metric))

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
    report_every: int,
    energy_grad_fn: Callable[[ParamsT], tuple[float, np.ndarray] | tuple[float, np.ndarray, Any]],
    apply_gradient_step: Callable[[ParamsT, np.ndarray, float], ParamsT],
    verbose: bool,
    one_based_steps: bool,
    format_step: Callable[[StepMetric], str],
    on_step_aux: Callable[[Any], None] | None = None,
) -> LoopResult[ParamsT]:
    """Run a measured loop when one call returns both energy and gradient."""

    step_metrics: list[StepMetric] = []
    gradient_wall_s = 0.0

    loop_start = time.perf_counter()
    step_iter = range(1, steps + 1) if one_based_steps else range(steps)
    for step in step_iter:
        step_start = grad_start = time.perf_counter()
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

        if _should_report_step(step, steps, report_every, one_based=one_based_steps):
            metric = StepMetric(
                step=step,
                energy=float(energy),
                grad_norm=grad_norm,
                grad_wall_s=grad_wall_s,
                step_wall_s=step_wall_s,
            )
            step_metrics.append(metric)
            if verbose:
                print(format_step(metric))

    return LoopResult(
        final_params=params,
        step_metrics=tuple(step_metrics),
        gradient_wall_s=gradient_wall_s,
        measured_loop_s=time.perf_counter() - loop_start,
    )
