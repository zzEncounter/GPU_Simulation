"""Standalone CUDA backend workflow with a function-level run interface."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ring_ising.backends.standalone import RingIsingAdjointBackend, StandaloneBackendConfig
from ring_ising.cli.common import format_telemetry_banner
from ring_ising.config import RunConfig
from ring_ising.runtime import print_gpu_telemetry_summary
from ring_ising.training import (
    StepEvaluation,
    TrainingRunResult,
    run_gradient_descent_loop,
    run_with_optional_telemetry,
    validate_common_run_args,
)
from .common import apply_gradient_step, make_initial_params, print_problem_summary, print_result_summary


@dataclass(frozen=True)
class StandaloneWorkflow:
    """Prepared standalone backend workflow."""

    config: RunConfig
    backend_config: StandaloneBackendConfig
    backend: RingIsingAdjointBackend


StandaloneResult = TrainingRunResult


def _to_backend_config(config: RunConfig) -> StandaloneBackendConfig:
    return StandaloneBackendConfig(
        num_qubits=config.num_qubits,
        layers=config.layers,
        field=config.field,
        gradient_strategy=config.gradient_strategy,
        structured_rotation_chunk_width=(
            config.effective_structured_rotation_chunk_width
        ),
        double_buffer=config.double_buffer,
    )


def create_standalone_workflow(config: RunConfig) -> StandaloneWorkflow:
    """Create a standalone backend workflow object."""
    backend_config = _to_backend_config(config)
    backend = RingIsingAdjointBackend(backend_config)
    return StandaloneWorkflow(config=config, backend_config=backend_config, backend=backend)


def print_standalone_runtime_summary(workflow: StandaloneWorkflow, params: np.ndarray) -> None:
    """Print runtime metadata for one standalone run."""
    config = workflow.config

    print("Runtime:")
    print("  Backend: Standalone CUDA")
    print(f"  Gradient strategy: {workflow.backend.gradient_strategy}")
    if workflow.backend.gradient_strategy == "structured_adjoint":
        print(
            "  Structured rotation chunk width: "
            f"{workflow.backend_config.effective_structured_rotation_chunk_width}"
        )
    if workflow.backend.uses_pennylane_gate_structure:
        print("  PennyLane-style gate structure: True")
    else:
        print("  Dense-scan fused gate structure: True")
    print(
        "  Estimated gradient workspace: "
        f"{workflow.backend.estimated_workspace_gib:.2f} GiB"
    )
    print()

    print_problem_summary(config, params)

    print("Backend details:")
    print(f"  Initial energy: {workflow.backend.energy(params):.10f}")
    if config.gpu_telemetry:
        print(
            "  GPU telemetry: "
            + format_telemetry_banner(config.telemetry_interval_s, config.telemetry_live)
        )
    print()


def run_standalone(config: RunConfig) -> TrainingRunResult:
    """Run one full standalone training workflow."""
    if config.backend != "standalone":
        raise ValueError(f"run_standalone expected backend='standalone', got {config.backend!r}.")
    validate_common_run_args(
        num_qubits=config.num_qubits,
        layers=config.layers,
        steps=config.steps,
        telemetry_interval_s=config.telemetry_interval_s,
    )

    workflow = create_standalone_workflow(config)
    params = make_initial_params(config)

    if config.verbose:
        print_standalone_runtime_summary(workflow, params)

    def _run_body() -> TrainingRunResult:
        run_start = time.perf_counter()
        loop = run_gradient_descent_loop(
            params,
            steps=config.steps,
            stepsize=config.stepsize,
            step_fn=lambda current: (
                lambda energy_grad: StepEvaluation(
                    energy=float(energy_grad[0]),
                    grad=np.asarray(energy_grad[1], dtype=np.float64),
                )
            )(workflow.backend.energy_and_grad(current)),
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

        final_energy = workflow.backend.energy(loop.final_params)
        total_wall_s = time.perf_counter() - run_start

        return TrainingRunResult(
            backend_label="standalone_cuda",
            final_params=np.asarray(loop.final_params, dtype=np.float64),
            final_energy=float(final_energy),
            step_metrics=loop.step_metrics,
            wall_s=total_wall_s,
            metadata={
                "gradient_strategy": workflow.backend.gradient_strategy,
                "structured_rotation_chunk_width": (
                    workflow.backend_config.effective_structured_rotation_chunk_width
                ),
                "estimated_workspace_gib": workflow.backend.estimated_workspace_gib,
                "pennylane_gate_structure": workflow.backend.uses_pennylane_gate_structure,
            },
        )

    result, telemetry_summary = run_with_optional_telemetry(
        enabled=config.gpu_telemetry,
        interval_s=config.telemetry_interval_s,
        live=config.telemetry_live,
        label="Standalone backend telemetry",
        body=_run_body,
    )
    result.gpu_telemetry_summary = telemetry_summary

    if config.verbose:
        print_result_summary(result, config)
        if result.gpu_telemetry_summary is not None:
            print()
            print_gpu_telemetry_summary(result.gpu_telemetry_summary, indent="  ")

    return result


__all__ = [
    "StandaloneResult",
    "StandaloneWorkflow",
    "run_standalone",
]
