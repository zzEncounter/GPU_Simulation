"""Minimal PennyLane adjoint-diff baseline workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ring_ising.cli.common import (
    create_telemetry_monitor,
    format_telemetry_banner,
    should_report_step,
)
from ring_ising.models import (
    build_ring_ising_hamiltonian,
    hardware_efficient_ring,
    make_initial_params,
)
from ring_ising.runtime import (
    GpuTelemetrySummary,
    create_device,
    format_gate_types,
    package_version,
    print_gpu_telemetry_summary,
)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for one baseline run."""

    num_qubits: int = 12
    layers: int = 3
    field: float = 1.0
    steps: int = 20
    warmup: int = 2
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
    z_profile_qnode: Callable[[pnp.ndarray], tuple[float, ...]]
    gradient_fn: Callable[[pnp.ndarray], pnp.ndarray]
    selection_errors: tuple[str, ...]


@dataclass(frozen=True)
class StepMetric:
    """Per-step metrics recorded during the measured loop."""

    step: int
    energy: float
    grad_norm: float
    grad_wall_s: float
    step_wall_s: float


@dataclass(frozen=True)
class TimingBreakdown:
    """Compute-phase timing breakdown for one baseline run."""

    initial_forward_s: float
    warmup_s: float
    measured_loop_s: float
    gradient_wall_s: float
    final_readout_s: float
    total_compute_s: float


@dataclass
class BaselineResult:
    """High-level outputs from one baseline execution."""

    workflow: BaselineWorkflow
    final_params: pnp.ndarray
    final_energy: float
    mean_local_z: float
    z_profile: np.ndarray
    timings: TimingBreakdown
    step_metrics: tuple[StepMetric, ...]
    gpu_telemetry_summary: GpuTelemetrySummary | None = None


def create_workflow(config: BaselineConfig) -> BaselineWorkflow:
    """Create the QNodes needed by the baseline."""
    selection = create_device(requested_mode=config.device, wires=config.num_qubits)
    hamiltonian = build_ring_ising_hamiltonian(config.num_qubits, config.field)

    @qml.qnode(selection.device, diff_method="adjoint")
    def energy_qnode(params: pnp.ndarray) -> float:
        hardware_efficient_ring(params)
        return qml.expval(hamiltonian)

    @qml.qnode(selection.device, diff_method=None)
    def z_profile_qnode(params: pnp.ndarray) -> tuple[float, ...]:
        hardware_efficient_ring(params)
        return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(config.num_qubits))

    gradient_fn = qml.grad(energy_qnode)

    return BaselineWorkflow(
        config=config,
        device_name=selection.device_name,
        energy_qnode=energy_qnode,
        z_profile_qnode=z_profile_qnode,
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


def _apply_gradient_step(
    params: pnp.ndarray, grad: pnp.ndarray, stepsize: float
) -> pnp.ndarray:
    updated = np.asarray(params) - stepsize * np.asarray(grad)
    return pnp.array(updated, requires_grad=True)


def _run_warmup(workflow: BaselineWorkflow, config: BaselineConfig) -> None:
    """Warm up the device and PennyLane internals before timing."""
    params = _fresh_params(config)
    for _ in range(config.warmup):
        grad = workflow.gradient_fn(params)
        params = _apply_gradient_step(params, grad, config.stepsize)


def _validate_config(config: BaselineConfig) -> None:
    """Fail fast on invalid baseline settings."""
    if config.num_qubits < 2:
        raise ValueError("num_qubits must be at least 2 for the ring Ising Hamiltonian.")
    if config.layers < 1:
        raise ValueError("layers must be at least 1.")
    if config.steps < 0:
        raise ValueError("steps must be non-negative.")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if config.report_every < 1:
        raise ValueError("report_every must be at least 1.")
    if config.telemetry_interval_s <= 0:
        raise ValueError("telemetry_interval_s must be positive.")


def _run_measured_loop(
    workflow: BaselineWorkflow,
    config: BaselineConfig,
    params: pnp.ndarray,
    verbose: bool,
) -> tuple[pnp.ndarray, tuple[StepMetric, ...], float, float]:
    """Run the measured optimization loop and collect progress snapshots."""
    step_metrics: list[StepMetric] = []
    gradient_wall_s = 0.0

    measured_start = time.perf_counter()
    for step in range(1, config.steps + 1):
        step_start = grad_start = time.perf_counter()

        grad = workflow.gradient_fn(params)
        grad_wall_s = time.perf_counter() - grad_start
        gradient_wall_s += grad_wall_s

        params = _apply_gradient_step(params, grad, config.stepsize)
        step_wall_s = time.perf_counter() - step_start
        grad_norm = float(np.linalg.norm(np.asarray(grad)))

        if should_report_step(step, config.steps, config.report_every, one_based=True):
            energy = float(workflow.energy_qnode(params))
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
                print(
                    f"Step {step:>3}: "
                    f"energy={energy: .7f}, "
                    f"grad_norm={grad_norm: .7f}, "
                    f"grad_s={grad_wall_s: .4f}, "
                    f"step_s={step_wall_s: .4f}"
                )

    measured_loop_s = time.perf_counter() - measured_start
    return params, tuple(step_metrics), gradient_wall_s, measured_loop_s


def _final_readout(
    workflow: BaselineWorkflow,
    params: pnp.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    """Measure final observables for the summary output."""
    final_readout_start = time.perf_counter()
    final_energy = float(workflow.energy_qnode(params))
    z_profile = np.asarray(workflow.z_profile_qnode(params), dtype=float)
    final_readout_s = time.perf_counter() - final_readout_start
    mean_local_z = float(z_profile.mean())
    return final_energy, z_profile, mean_local_z, final_readout_s


def _print_result_summary(result: BaselineResult) -> None:
    """Print a compact end-of-run summary."""
    config = result.workflow.config
    timings = result.timings
    avg_grad_s = timings.gradient_wall_s / config.steps if config.steps else 0.0
    avg_step_s = timings.measured_loop_s / config.steps if config.steps else 0.0
    non_gradient_s = max(0.0, timings.measured_loop_s - timings.gradient_wall_s)
    gradient_share = (
        timings.gradient_wall_s / timings.measured_loop_s if timings.measured_loop_s else 0.0
    )
    z_preview = ", ".join(
        f"{value:+.4f}" for value in result.z_profile[: min(8, len(result.z_profile))]
    )

    print()
    print("Summary:")
    print(f"  Final energy: {result.final_energy:.7f}")
    print(f"  Energy per qubit: {result.final_energy / config.num_qubits:.7f}")
    print(f"  Mean local <Z>: {result.mean_local_z:.7f}")
    print(f"  Initial forward wall time: {timings.initial_forward_s:.4f} s")
    print(f"  Warmup wall time: {timings.warmup_s:.4f} s")
    print(f"  Measured loop wall time: {timings.measured_loop_s:.4f} s")
    print(f"  Gradient wall time: {timings.gradient_wall_s:.4f} s")
    print(f"  Non-gradient measured time: {non_gradient_s:.4f} s")
    print(f"  Final readout wall time: {timings.final_readout_s:.4f} s")
    print(f"  Total compute wall time: {timings.total_compute_s:.4f} s")
    print(f"  Gradient share of measured loop: {gradient_share:.1%}")
    print(f"  Average gradient call: {avg_grad_s:.4f} s")
    print(f"  Average measured step: {avg_step_s:.4f} s")
    print(f"  First local <Z> values: {z_preview}")
    if result.gpu_telemetry_summary is not None:
        print()
        print_gpu_telemetry_summary(result.gpu_telemetry_summary, indent="  ")


def run_baseline(
    config: BaselineConfig,
) -> BaselineResult:
    """Run the end-to-end adjoint-diff baseline."""
    _validate_config(config)
    workflow = create_workflow(config)
    params = _fresh_params(config)
    telemetry_monitor = create_telemetry_monitor(
        enabled=config.gpu_telemetry,
        interval_s=config.telemetry_interval_s,
        live=config.telemetry_live,
        label="PennyLane baseline telemetry",
    )
    gpu_telemetry_summary: GpuTelemetrySummary | None = None

    if config.verbose:
        print_runtime_summary(workflow, params)
        if telemetry_monitor is not None:
            print(
                "GPU telemetry: "
                + format_telemetry_banner(
                    config.telemetry_interval_s,
                    config.telemetry_live,
                )
            )
        if config.warmup:
            print(f"Warmup gradient calls: {config.warmup}")
        print()

    if telemetry_monitor is not None:
        telemetry_monitor.start()

    try:
        warmup_start = time.perf_counter()
        if config.warmup:
            _run_warmup(workflow, config)
        warmup_s = time.perf_counter() - warmup_start

        params = _fresh_params(config)
        params, step_metrics, gradient_wall_s, measured_loop_s = _run_measured_loop(
            workflow,
            config,
            params,
            config.verbose,
        )

        final_energy, z_profile, mean_local_z, final_readout_s = _final_readout(workflow, params)
    finally:
        if telemetry_monitor is not None:
            gpu_telemetry_summary = telemetry_monitor.stop()

    timings = TimingBreakdown(
        initial_forward_s=0,
        warmup_s=warmup_s,
        measured_loop_s=measured_loop_s,
        gradient_wall_s=gradient_wall_s,
        final_readout_s=final_readout_s,
        total_compute_s=0 + warmup_s + measured_loop_s + final_readout_s,
    )

    result = BaselineResult(
        workflow=workflow,
        final_params=params,
        final_energy=final_energy,
        mean_local_z=mean_local_z,
        z_profile=z_profile,
        timings=timings,
        step_metrics=step_metrics,
        gpu_telemetry_summary=gpu_telemetry_summary,
    )

    if config.verbose:
        _print_result_summary(result)

    return result
