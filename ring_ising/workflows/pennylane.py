"""PennyLane workflow using shared training-run abstractions."""

from __future__ import annotations

import time
from autograd import value_and_grad
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml

from ring_ising.cli.common import format_telemetry_banner
from ring_ising.config import RunConfig
from ring_ising.models import build_ring_ising_hamiltonian, hardware_efficient_ring
from ring_ising.runtime import (
    create_device,
    format_gate_types,
    package_version,
    print_gpu_telemetry_summary,
)
from ring_ising.training import (
    StepEvaluation,
    TrainingRunResult,
    run_gradient_descent_loop,
    run_with_optional_telemetry,
    validate_common_run_args,
)
from .common import (
    apply_gradient_step,
    as_pennylane_params,
    make_initial_params,
    print_problem_summary,
    print_result_summary,
)


@dataclass(frozen=True)
class PennyLaneWorkflow:
    """Compiled QNodes and device metadata for the PennyLane backend."""

    config: RunConfig
    device_name: str
    energy_qnode: Callable[[object], float]
    energy_and_grad_fn: Callable[[object], tuple[object, object]]


PennyLaneResult = TrainingRunResult


def create_pennylane_workflow(config: RunConfig) -> PennyLaneWorkflow:
    """Create the QNodes needed by the PennyLane backend."""
    selection = create_device(wires=config.num_qubits)
    hamiltonian = build_ring_ising_hamiltonian(config.num_qubits, config.field)

    @qml.qnode(selection.device, diff_method="adjoint")
    def energy_qnode(params) -> float:
        hardware_efficient_ring(params)
        return qml.expval(hamiltonian)

    energy_and_grad_fn = value_and_grad(energy_qnode)

    return PennyLaneWorkflow(
        config=config,
        device_name=selection.device_name,
        energy_qnode=energy_qnode,
        energy_and_grad_fn=energy_and_grad_fn,
    )


def print_pennylane_runtime_summary(
    workflow: PennyLaneWorkflow,
    params: np.ndarray,
) -> None:
    """Print the environment and circuit metadata for the run."""
    specs = qml.specs(workflow.energy_qnode)(as_pennylane_params(params))
    resources = specs["resources"]

    print("Runtime:")
    print("  Backend: PennyLane")
    print(f"  PennyLane: {package_version('pennylane')}")
    print(f"  PennyLane-Lightning: {package_version('pennylane-lightning')}")
    print(f"  PennyLane-Lightning-GPU: {package_version('pennylane-lightning-gpu')}")
    print(f"  Device: {workflow.device_name}")
    print()

    print_problem_summary(workflow.config, params)

    print("Backend details:")
    print(f"  Hamiltonian terms: {2 * workflow.config.num_qubits}")
    print(f"  Diff method: {specs['diff_method']}")
    print(f"  Trainable params: {specs['num_trainable_params']}")
    print(f"  Observables: {specs['num_observables']}")
    print(f"  Gates: {resources.num_gates}")
    print(f"  Depth: {resources.depth}")
    print(f"  Gate types: {format_gate_types(resources.gate_types)}")
    print()


def run_pennylane(config: RunConfig) -> TrainingRunResult:
    """Run the end-to-end PennyLane workflow."""
    if config.backend != "pennylane":
        raise ValueError(f"run_pennylane expected backend='pennylane', got {config.backend!r}.")
    validate_common_run_args(
        num_qubits=config.num_qubits,
        layers=config.layers,
        steps=config.steps,
        telemetry_interval_s=config.telemetry_interval_s,
    )
    workflow = create_pennylane_workflow(config)
    params = make_initial_params(config)

    if config.verbose:
        print_pennylane_runtime_summary(workflow, params)
        if config.gpu_telemetry:
            print(
                "GPU telemetry: "
                + format_telemetry_banner(config.telemetry_interval_s, config.telemetry_live)
            )
            print()

    def _run_body() -> TrainingRunResult:
        run_start = time.perf_counter()
        loop = run_gradient_descent_loop(
            params,
            steps=config.steps,
            stepsize=config.stepsize,
            step_fn=lambda current: (
                lambda value_grad: StepEvaluation(
                    energy=float(value_grad[0]),
                    grad=np.asarray(value_grad[1], dtype=np.float64),
                )
            )(workflow.energy_and_grad_fn(as_pennylane_params(current))),
            apply_gradient_step=apply_gradient_step,
            verbose=config.verbose,
            show_progress=config.show_progress and not config.report_steps,
            report_steps=config.report_steps,
            format_step=(
                lambda metric: (
                    f"step={metric.step:03d} "
                    f"energy={metric.energy:.10f} "
                    f"grad_norm={metric.grad_norm:.10f} "
                    f"step_ms={1000.0 * metric.step_wall_s:.3f}"
                )
            ),
        )

        final_energy = float(workflow.energy_qnode(as_pennylane_params(loop.final_params)))
        total_wall_s = time.perf_counter() - run_start

        return TrainingRunResult(
            backend_label=f"pennylane:{workflow.device_name}",
            final_params=np.asarray(loop.final_params, dtype=np.float64),
            final_energy=final_energy,
            step_metrics=loop.step_metrics,
            wall_s=total_wall_s,
            metadata={
                "device": workflow.device_name,
            },
        )

    result, telemetry_summary = run_with_optional_telemetry(
        enabled=config.gpu_telemetry,
        interval_s=config.telemetry_interval_s,
        live=config.telemetry_live,
        label="PennyLane telemetry",
        body=_run_body,
    )
    result.gpu_telemetry_summary = telemetry_summary

    if config.verbose:
        print_result_summary(result, config)
        print(f"  Energy per qubit: {result.final_energy / config.num_qubits:.7f}")
        if result.gpu_telemetry_summary is not None:
            print()
            print_gpu_telemetry_summary(result.gpu_telemetry_summary, indent="  ")

    return result


__all__ = [
    "PennyLaneResult",
    "PennyLaneWorkflow",
    "run_pennylane",
]
