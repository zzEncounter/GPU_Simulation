"""PennyLane baseline workflow using shared training-run abstractions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ring_ising.cli.common import format_telemetry_banner
from ring_ising.models import build_ring_ising_hamiltonian, hardware_efficient_ring, make_initial_params
from ring_ising.runtime import create_device, format_gate_types, package_version, print_gpu_telemetry_summary
from ring_ising.training import (
    LoopTimingBreakdown,
    TrainingRunResult,
    run_gradient_descent_loop_from_grad,
    run_with_optional_telemetry,
    validate_common_run_args,
)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for one PennyLane baseline run."""

    num_qubits: int = 12
    layers: int = 3
    field: float = 1.0
    steps: int = 20
    stepsize: float = 0.08
    seed: int = 7
    init_scale: float = 0.3
    report_every: int = 5
    device: str = "gpu"
    verbose: bool = True
    gpu_telemetry: bool = False
    telemetry_interval_s: float = 0.5
    telemetry_live: bool = False


@dataclass(frozen=True)
class BaselineWorkflow:
    """Compiled QNodes and device metadata for the baseline."""

    config: BaselineConfig
    device_name: str
    energy_qnode: Callable[[pnp.ndarray], float]
    gradient_fn: Callable[[pnp.ndarray], pnp.ndarray]
    selection_errors: tuple[str, ...]


BaselineResult = TrainingRunResult
TimingBreakdown = LoopTimingBreakdown


def create_workflow(config: BaselineConfig) -> BaselineWorkflow:
    """Create the QNodes needed by the baseline."""
    selection = create_device(requested_mode=config.device, wires=config.num_qubits)
    hamiltonian = build_ring_ising_hamiltonian(config.num_qubits, config.field)

    @qml.qnode(selection.device, diff_method="adjoint")
    def energy_qnode(params: pnp.ndarray) -> float:
        hardware_efficient_ring(params)
        return qml.expval(hamiltonian)

    gradient_fn = qml.grad(energy_qnode)

    return BaselineWorkflow(
        config=config,
        device_name=selection.device_name,
        energy_qnode=energy_qnode,
        gradient_fn=gradient_fn,
        selection_errors=tuple(selection.selection_errors),
    )


def print_runtime_summary(workflow: BaselineWorkflow, params: pnp.ndarray) -> None:
    """Print the environment and circuit metadata for the run."""
    specs = qml.specs(workflow.energy_qnode)(params)
    resources = specs["resources"]

    print("Runtime:")
    print(f"  PennyLane: {package_version('pennylane')}")
    print(f"  PennyLane-Lightning: {package_version('pennylane-lightning')}")
    print(f"  PennyLane-Lightning-GPU: {package_version('pennylane-lightning-gpu')}")
    print(f"  Requested device mode: {workflow.config.device}")
    print(f"  Selected device: {workflow.device_name}")
    if workflow.selection_errors:
        print("  Fallback notes:")
        for error in workflow.selection_errors:
            print(f"    {error}")
    print()

    print("Problem:")
    print(f"  Qubits: {workflow.config.num_qubits}")
    print(f"  Layers: {workflow.config.layers}")
    print(f"  Field strength: {workflow.config.field}")
    print(f"  Hamiltonian terms: {2 * workflow.config.num_qubits}")
    print(f"  Parameter tensor shape: {tuple(params.shape)}")
    print()

    print("Circuit specs:")
    print(f"  Diff method: {specs['diff_method']}")
    print(f"  Trainable params: {specs['num_trainable_params']}")
    print(f"  Observables: {specs['num_observables']}")
    print(f"  Gates: {resources.num_gates}")
    print(f"  Depth: {resources.depth}")
    print(f"  Gate types: {format_gate_types(resources.gate_types)}")
    print()


def _fresh_params(config: BaselineConfig) -> pnp.ndarray:
    return make_initial_params(
        num_qubits=config.num_qubits,
        layers=config.layers,
        seed=config.seed,
        init_scale=config.init_scale,
    )


def _apply_gradient_step(params: pnp.ndarray, grad: np.ndarray, stepsize: float) -> pnp.ndarray:
    updated = np.asarray(params) - stepsize * np.asarray(grad)
    return pnp.array(updated, requires_grad=True)


def _print_result_summary(result: TrainingRunResult, config: BaselineConfig) -> None:
    timings = result.timings
    avg_grad_s = timings.gradient_wall_s / config.steps if config.steps else 0.0
    avg_step_s = timings.measured_loop_s / config.steps if config.steps else 0.0
    non_gradient_s = max(0.0, timings.measured_loop_s - timings.gradient_wall_s)
    gradient_share = timings.gradient_wall_s / timings.measured_loop_s if timings.measured_loop_s else 0.0

    print()
    print("Summary:")
    print(f"  Final energy: {result.final_energy:.7f}")
    print(f"  Energy per qubit: {result.final_energy / config.num_qubits:.7f}")
    print(f"  Measured loop wall time: {timings.measured_loop_s:.4f} s")
    print(f"  Gradient wall time: {timings.gradient_wall_s:.4f} s")
    print(f"  Non-gradient measured time: {non_gradient_s:.4f} s")
    print(f"  Final readout wall time: {timings.final_readout_s:.4f} s")
    print(f"  Total compute wall time: {timings.total_compute_s:.4f} s")
    print(f"  Gradient share of measured loop: {gradient_share:.1%}")
    print(f"  Average gradient call: {avg_grad_s:.4f} s")
    print(f"  Average measured step: {avg_step_s:.4f} s")
    if result.gpu_telemetry_summary is not None:
        print()
        print_gpu_telemetry_summary(result.gpu_telemetry_summary, indent="  ")


def run_baseline(config: BaselineConfig) -> TrainingRunResult:
    """Run the end-to-end PennyLane adjoint-diff baseline."""
    validate_common_run_args(
        num_qubits=config.num_qubits,
        layers=config.layers,
        steps=config.steps,
        report_every=config.report_every,
        telemetry_interval_s=config.telemetry_interval_s,
    )
    workflow = create_workflow(config)
    params = _fresh_params(config)

    if config.verbose:
        print_runtime_summary(workflow, params)
        if config.gpu_telemetry:
            print(
                "GPU telemetry: "
                + format_telemetry_banner(config.telemetry_interval_s, config.telemetry_live)
            )
            print()

    def _run_body() -> TrainingRunResult:
        loop = run_gradient_descent_loop_from_grad(
            params,
            steps=config.steps,
            stepsize=config.stepsize,
            report_every=config.report_every,
            grad_fn=lambda current: np.asarray(workflow.gradient_fn(current), dtype=np.float64),
            energy_fn=lambda current: float(workflow.energy_qnode(current)),
            apply_gradient_step=_apply_gradient_step,
            verbose=config.verbose,
            one_based_steps=True,
            format_step=(
                lambda metric: (
                    f"Step {metric.step:>3}: "
                    f"energy={metric.energy: .7f}, "
                    f"grad_norm={metric.grad_norm: .7f}, "
                    f"grad_s={metric.grad_wall_s: .4f}, "
                    f"step_s={metric.step_wall_s: .4f}"
                )
            ),
        )

        final_readout_start = time.perf_counter()
        final_energy = float(workflow.energy_qnode(loop.final_params))
        final_readout_s = time.perf_counter() - final_readout_start

        return TrainingRunResult(
            backend_label=f"pennylane:{workflow.device_name}",
            final_params=np.asarray(loop.final_params, dtype=np.float64),
            final_energy=final_energy,
            step_metrics=loop.step_metrics,
            timings=LoopTimingBreakdown(
                measured_loop_s=loop.measured_loop_s,
                gradient_wall_s=loop.gradient_wall_s,
                final_readout_s=final_readout_s,
                total_compute_s=loop.measured_loop_s + final_readout_s,
            ),
            metadata={
                "requested_device_mode": config.device,
                "selected_device": workflow.device_name,
                "selection_errors": workflow.selection_errors,
            },
        )

    result, telemetry_summary = run_with_optional_telemetry(
        enabled=config.gpu_telemetry,
        interval_s=config.telemetry_interval_s,
        live=config.telemetry_live,
        label="PennyLane baseline telemetry",
        body=_run_body,
    )
    result.gpu_telemetry_summary = telemetry_summary

    if config.verbose:
        _print_result_summary(result, config)

    return result
