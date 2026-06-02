"""Standalone CUDA backend workflow with a function-level run interface."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ring_ising.cli.common import format_telemetry_banner
from ring_ising.runtime import print_gpu_telemetry_summary
from ring_ising.training import (
    LoopTimingBreakdown,
    TrainingRunResult,
    run_gradient_descent_loop_from_energy_grad,
    run_with_optional_telemetry,
    validate_common_run_args,
)

from standalone_backend.config import RingIsingConfig
from standalone_backend.ising_runtime import RingIsingAdjointBackend, make_initial_params


@dataclass(frozen=True)
class StandaloneRunConfig:
    """Configuration for one standalone training run."""

    num_qubits: int = 12
    layers: int = 3
    field: float = 1.0
    steps: int = 20
    stepsize: float = 0.08
    seed: int = 7
    init_scale: float = 0.3
    report_every: int = 5
    gradient_strategy: str = "auto"
    checkpoint_interval_ops: int | None = None
    gate_fusion: bool = True
    auto_memory_budget_fraction: float = 0.85
    auto_memory_reserve_mib: int = 1024
    verbose: bool = True
    gpu_telemetry: bool = False
    telemetry_interval_s: float = 0.5
    telemetry_live: bool = False


@dataclass(frozen=True)
class StandaloneWorkflow:
    """Prepared standalone backend workflow."""

    config: StandaloneRunConfig
    backend_config: RingIsingConfig
    backend: RingIsingAdjointBackend


StandaloneResult = TrainingRunResult


def _to_backend_config(config: StandaloneRunConfig) -> RingIsingConfig:
    return RingIsingConfig(
        num_qubits=config.num_qubits,
        layers=config.layers,
        field=config.field,
        gradient_strategy=config.gradient_strategy,
        checkpoint_interval_ops=config.checkpoint_interval_ops,
        gate_fusion=config.gate_fusion,
        auto_memory_budget_fraction=config.auto_memory_budget_fraction,
        auto_memory_reserve_mib=config.auto_memory_reserve_mib,
    )


def create_workflow(config: StandaloneRunConfig) -> StandaloneWorkflow:
    """Create a standalone backend workflow object."""
    backend_config = _to_backend_config(config)
    backend = RingIsingAdjointBackend(backend_config)
    return StandaloneWorkflow(config=config, backend_config=backend_config, backend=backend)


def _initial_params(config: StandaloneRunConfig) -> np.ndarray:
    return np.asarray(
        make_initial_params(
            num_qubits=config.num_qubits,
            layers=config.layers,
            seed=config.seed,
            init_scale=config.init_scale,
        ),
        dtype=np.float64,
    )


def print_runtime_summary(workflow: StandaloneWorkflow, params: np.ndarray) -> None:
    """Print runtime metadata for one standalone run."""
    resolution = workflow.backend.strategy_resolution
    config = workflow.config

    print("Standalone CUDA backend")
    print(f"  Qubits: {config.num_qubits}")
    print(f"  Layers: {config.layers}")
    print(f"  Field strength: {config.field}")
    print(f"  Requested gradient strategy: {config.gradient_strategy}")
    print(f"  Resolved gradient strategy: {resolution.resolved_strategy}")
    print(f"  Gate fusion enabled: {config.gate_fusion}")
    if resolution.resolved_strategy == "checkpoint":
        print(f"  Effective checkpoint interval: {resolution.checkpoint_interval_ops} ops")
    if resolution.available_gpu_memory_mib is not None:
        print(
            "  Auto memory budget: "
            f"{resolution.memory_budget_mib:.0f} MiB "
            f"(free={resolution.available_gpu_memory_mib} MiB, "
            f"total={resolution.total_gpu_memory_mib} MiB)"
        )
    print("  Estimated gradient workspace: " f"{resolution.estimated_workspace_gib:.2f} GiB")
    if resolution.note:
        print(f"  Strategy note: {resolution.note}")
    print(f"  Initial energy: {workflow.backend.energy(params):.10f}")
    if config.gpu_telemetry:
        print(
            "  GPU telemetry: "
            + format_telemetry_banner(config.telemetry_interval_s, config.telemetry_live)
        )
    print()


def _print_result_summary(result: TrainingRunResult, config: StandaloneRunConfig) -> None:
    timings = result.timings
    avg_grad_s = timings.gradient_wall_s / config.steps if config.steps else 0.0
    avg_step_s = timings.measured_loop_s / config.steps if config.steps else 0.0

    print()
    print("Summary:")
    print(f"  Final energy: {result.final_energy:.10f}")
    print(f"  Measured loop wall time: {timings.measured_loop_s:.4f} s")
    if config.steps:
        print(f"  Average measured step: {1000.0 * avg_step_s:.3f} ms")
        print(f"  Average gradient call: {1000.0 * avg_grad_s:.3f} ms")

    if result.gpu_telemetry_summary is not None:
        print()
        print_gpu_telemetry_summary(result.gpu_telemetry_summary, indent="  ")


def run_standalone(config: StandaloneRunConfig) -> TrainingRunResult:
    """Run one full standalone training workflow."""
    validate_common_run_args(
        num_qubits=config.num_qubits,
        layers=config.layers,
        steps=config.steps,
        report_every=config.report_every,
        telemetry_interval_s=config.telemetry_interval_s,
    )

    workflow = create_workflow(config)
    params = _initial_params(config)

    if config.verbose:
        print_runtime_summary(workflow, params)

    def _energy_grad_step(current: np.ndarray) -> tuple[float, np.ndarray]:
        energy, grad = workflow.backend.energy_and_grad(current)
        return float(energy), np.asarray(grad, dtype=np.float64)

    def _run_body() -> TrainingRunResult:
        loop = run_gradient_descent_loop_from_energy_grad(
            params,
            steps=config.steps,
            stepsize=config.stepsize,
            report_every=config.report_every,
            energy_grad_fn=_energy_grad_step,
            apply_gradient_step=lambda current, grad, eta: current - eta * grad,
            verbose=config.verbose,
            one_based_steps=False,
            format_step=(
                lambda metric: (
                    f"step={metric.step:03d} "
                    f"energy={metric.energy:.10f} "
                    f"grad_norm={metric.grad_norm:.10f} "
                    f"step_ms={1000.0 * metric.step_wall_s:.3f}"
                )
            ),
        )

        final_readout_start = time.perf_counter()
        final_energy = workflow.backend.energy(loop.final_params)
        final_readout_s = time.perf_counter() - final_readout_start

        return TrainingRunResult(
            backend_label="standalone_cuda",
            final_params=np.asarray(loop.final_params, dtype=np.float64),
            final_energy=float(final_energy),
            step_metrics=loop.step_metrics,
            timings=LoopTimingBreakdown(
                measured_loop_s=loop.measured_loop_s,
                gradient_wall_s=loop.gradient_wall_s,
                final_readout_s=final_readout_s,
                total_compute_s=loop.measured_loop_s + final_readout_s,
            ),
            metadata={
                "requested_strategy": workflow.backend.strategy_resolution.requested_strategy,
                "resolved_strategy": workflow.backend.strategy_resolution.resolved_strategy,
                "checkpoint_interval_ops": workflow.backend.strategy_resolution.checkpoint_interval_ops,
                "estimated_workspace_gib": workflow.backend.strategy_resolution.estimated_workspace_gib,
                "gate_fusion": config.gate_fusion,
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
        _print_result_summary(result, config)

    return result
