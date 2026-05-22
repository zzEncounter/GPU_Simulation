"""Minimal PennyLane adjoint-diff baseline for GPU Ising experiments."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from runtime_utils import (
    DEVICE_CANDIDATES,
    GpuTelemetrySummary,
    GpuTelemetryMonitor,
    ResourceSnapshot,
    capture_resource_snapshot,
    create_device,
    format_gate_types,
    package_version,
    print_gpu_telemetry_summary,
    print_resource_snapshot,
)
from ising_model import (
    build_ring_ising_hamiltonian,
    hardware_efficient_ring,
    make_initial_params,
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


@dataclass
class BaselineWorkflow:
    """Compiled QNodes and device metadata for the baseline."""

    config: BaselineConfig
    device_name: str
    device: Any
    hamiltonian: Any
    energy_qnode: Callable[[pnp.ndarray], float]
    z_profile_qnode: Callable[[pnp.ndarray], tuple[float, ...]]
    gradient_fn: Callable[[pnp.ndarray], pnp.ndarray]
    selection_errors: list[str]


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
    resource_snapshots: tuple[ResourceSnapshot, ...]
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
        device=selection.device,
        hamiltonian=hamiltonian,
        energy_qnode=energy_qnode,
        z_profile_qnode=z_profile_qnode,
        gradient_fn=gradient_fn,
        selection_errors=list(selection.selection_errors),
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


def _capture_snapshot(
    label: str, snapshots: list[ResourceSnapshot], verbose: bool
) -> ResourceSnapshot:
    """Capture and optionally print a resource snapshot."""
    snapshot = capture_resource_snapshot(label)
    snapshots.append(snapshot)
    if verbose:
        print_resource_snapshot(snapshot)
    return snapshot


def _peak_process_rss_mib(snapshots: list[ResourceSnapshot]) -> float | None:
    """Return the peak observed process RSS across snapshots."""
    rss_values = [
        snapshot.process_rss_mib
        for snapshot in snapshots
        if snapshot.process_rss_mib is not None
    ]
    return max(rss_values) if rss_values else None


def _peak_active_gpu_memory_mib(snapshots: list[ResourceSnapshot]) -> int | None:
    """Return the peak observed GPU memory for the current compute process."""
    gpu_mem_values = [
        process.used_gpu_memory_mib
        for snapshot in snapshots
        for process in snapshot.active_compute_processes
        if process.used_gpu_memory_mib is not None
    ]
    return max(gpu_mem_values) if gpu_mem_values else None


def run_baseline(
    config: BaselineConfig,
    verbose: bool = True,
    gpu_telemetry: bool = False,
    telemetry_interval_s: float = 0.5,
    telemetry_live: bool = False,
) -> BaselineResult:
    """Run the end-to-end adjoint-diff baseline."""
    _validate_config(config)
    workflow = create_workflow(config)
    params = _fresh_params(config)
    resource_snapshots: list[ResourceSnapshot] = []
    telemetry_monitor = (
        GpuTelemetryMonitor(
            sample_interval_s=telemetry_interval_s,
            live=telemetry_live,
            label="PennyLane baseline telemetry",
        )
        if gpu_telemetry
        else None
    )

    if verbose:
        print_runtime_summary(workflow, params)
        if telemetry_monitor is not None:
            print(
                "GPU telemetry: "
                f"sampling every {telemetry_interval_s:.2f} s"
                + (" with live printing enabled." if telemetry_live else ".")
            )
            print()

    if telemetry_monitor is not None:
        telemetry_monitor.start()

    _capture_snapshot("Resource snapshot before first execution", resource_snapshots, verbose)

    initial_forward_start = time.perf_counter()
    initial_energy = float(workflow.energy_qnode(params))
    initial_forward_s = time.perf_counter() - initial_forward_start

    if verbose:
        print(f"Initial energy: {initial_energy:.7f}")
        print(f"Initial forward wall time: {initial_forward_s:.4f} s")
        if config.warmup:
            print(f"Warmup gradient calls: {config.warmup}")
        print()

    _capture_snapshot("Resource snapshot after initial execution", resource_snapshots, verbose)

    warmup_start = time.perf_counter()
    if config.warmup:
        _run_warmup(workflow, config)
    warmup_s = time.perf_counter() - warmup_start

    if config.warmup:
        _capture_snapshot("Resource snapshot after warmup", resource_snapshots, verbose)

    params = _fresh_params(config)
    step_metrics: list[StepMetric] = []
    gradient_wall_s = 0.0

    measured_start = time.perf_counter()
    for step in range(1, config.steps + 1):
        step_start = time.perf_counter()
        grad_start = time.perf_counter()
        grad = workflow.gradient_fn(params)
        grad_wall_s = time.perf_counter() - grad_start
        gradient_wall_s += grad_wall_s

        params = _apply_gradient_step(params, grad, config.stepsize)
        grad_norm = float(np.linalg.norm(np.asarray(grad)))
        step_wall_s = time.perf_counter() - step_start

        if step == 1 or step % config.report_every == 0 or step == config.steps:
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
    _capture_snapshot("Resource snapshot after measured loop", resource_snapshots, verbose)

    final_readout_start = time.perf_counter()
    final_energy = float(workflow.energy_qnode(params))
    z_profile = np.asarray(workflow.z_profile_qnode(params), dtype=float)
    final_readout_s = time.perf_counter() - final_readout_start
    mean_local_z = float(z_profile.mean())
    _capture_snapshot("Resource snapshot after final readout", resource_snapshots, verbose)

    timings = TimingBreakdown(
        initial_forward_s=initial_forward_s,
        warmup_s=warmup_s,
        measured_loop_s=measured_loop_s,
        gradient_wall_s=gradient_wall_s,
        final_readout_s=final_readout_s,
        total_compute_s=initial_forward_s + warmup_s + measured_loop_s + final_readout_s,
    )

    gpu_telemetry_summary = telemetry_monitor.stop() if telemetry_monitor is not None else None

    result = BaselineResult(
        workflow=workflow,
        final_params=params,
        final_energy=final_energy,
        mean_local_z=mean_local_z,
        z_profile=z_profile,
        timings=timings,
        step_metrics=tuple(step_metrics),
        resource_snapshots=tuple(resource_snapshots),
        gpu_telemetry_summary=gpu_telemetry_summary,
    )

    if verbose:
        avg_grad_s = timings.gradient_wall_s / config.steps if config.steps else 0.0
        avg_step_s = timings.measured_loop_s / config.steps if config.steps else 0.0
        non_gradient_s = max(0.0, timings.measured_loop_s - timings.gradient_wall_s)
        gradient_share = (
            timings.gradient_wall_s / timings.measured_loop_s if timings.measured_loop_s else 0.0
        )
        peak_rss_mib = _peak_process_rss_mib(resource_snapshots)
        peak_gpu_mem_mib = _peak_active_gpu_memory_mib(resource_snapshots)
        z_preview = ", ".join(
            f"{value:+.4f}" for value in z_profile[: min(8, len(z_profile))]
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
        if peak_rss_mib is not None:
            print(f"  Peak process RSS: {peak_rss_mib:.1f} MiB")
        if peak_gpu_mem_mib is not None:
            print(f"  Peak process GPU memory: {peak_gpu_mem_mib} MiB")
        print(f"  First local <Z> values: {z_preview}")
        if result.gpu_telemetry_summary is not None:
            print()
            print_gpu_telemetry_summary(result.gpu_telemetry_summary, indent="  ")

    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PennyLane baseline."""
    parser = argparse.ArgumentParser(
        description="Run a minimal PennyLane adjoint-diff baseline on a ring-Ising circuit."
    )
    parser.add_argument("--qubits", type=int, default=12, help="Number of qubits.")
    parser.add_argument("--layers", type=int, default=3, help="Ansatz layers.")
    parser.add_argument(
        "--field",
        type=float,
        default=1.0,
        help="Transverse-field Ising X-field strength.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Measured gradient steps.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup gradient calls before the measured loop.",
    )
    parser.add_argument(
        "--stepsize",
        type=float,
        default=0.08,
        help="Manual gradient-descent step size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for parameter initialization.",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.3,
        help="Standard deviation for the initial parameter distribution.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=5,
        help="Print a progress line every N measured steps.",
    )
    parser.add_argument(
        "--device",
        choices=tuple(DEVICE_CANDIDATES),
        default="gpu",
        help="Device mode: auto, gpu, cpu, or default.",
    )
    parser.add_argument(
        "--gpu-telemetry",
        action="store_true",
        help="Sample GPU utilization, memory, and compute-component telemetry during the run.",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds for GPU telemetry.",
    )
    parser.add_argument(
        "--telemetry-live",
        action="store_true",
        help="Print one live GPU telemetry line per sample while the run is in progress.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    """Convert parsed CLI arguments into a baseline config."""
    return BaselineConfig(
        num_qubits=args.qubits,
        layers=args.layers,
        field=args.field,
        steps=args.steps,
        warmup=args.warmup,
        stepsize=args.stepsize,
        seed=args.seed,
        init_scale=args.init_scale,
        report_every=args.report_every,
        device=args.device,
    )


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        config_from_args(args),
        gpu_telemetry=args.gpu_telemetry,
        telemetry_interval_s=args.telemetry_interval,
        telemetry_live=args.telemetry_live,
    )
