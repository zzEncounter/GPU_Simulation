"""Shared CLI helpers for the PennyLane and standalone entry scripts."""

from __future__ import annotations

import argparse

from ring_ising.runtime import DEVICE_CANDIDATES, GpuTelemetryMonitor


def add_problem_args(
    parser: argparse.ArgumentParser,
    *,
    qubits_default: int,
    layers_default: int,
    field_default: float,
) -> None:
    """Add common problem-size arguments."""
    parser.add_argument(
        "--qubits",
        dest="num_qubits",
        type=int,
        default=qubits_default,
        help="Number of qubits.",
    )
    parser.add_argument("--layers", type=int, default=layers_default, help="Ansatz layers.")
    parser.add_argument(
        "--field",
        type=float,
        default=field_default,
        help="Transverse-field Ising X-field strength.",
    )


def add_optimization_args(
    parser: argparse.ArgumentParser,
    *,
    steps_default: int,
    stepsize_default: float,
    seed_default: int,
    init_scale_default: float,
    report_every_default: int,
    include_warmup: bool = False,
    warmup_default: int | None = None,
) -> None:
    """Add common loop and initialization arguments."""
    parser.add_argument("--steps", type=int, default=steps_default, help="Measured gradient steps.")
    if include_warmup:
        if warmup_default is None:
            raise ValueError("warmup_default must be provided when include_warmup=True.")
        parser.add_argument(
            "--warmup",
            type=int,
            default=warmup_default,
            help="Warmup gradient calls before the measured loop.",
        )
    parser.add_argument(
        "--stepsize",
        type=float,
        default=stepsize_default,
        help="Manual gradient-descent step size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed_default,
        help="Random seed for parameter initialization.",
    )
    parser.add_argument(
        "--init-scale",
        dest="init_scale",
        type=float,
        default=init_scale_default,
        help="Standard deviation for the initial parameter distribution.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=report_every_default,
        help="Print a progress line every N measured steps.",
    )


def add_device_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = "gpu",
) -> None:
    """Add the PennyLane device-selection argument."""
    parser.add_argument(
        "--device",
        choices=tuple(DEVICE_CANDIDATES),
        default=default,
        help="Device mode: auto, gpu, cpu, or default.",
    )


def add_telemetry_args(
    parser: argparse.ArgumentParser,
    *,
    interval_default: float,
    gpu_telemetry_default: bool = False,
    telemetry_live_default: bool = False,
) -> None:
    """Add common GPU telemetry arguments."""
    parser.add_argument(
        "--gpu-telemetry",
        action="store_true",
        default=gpu_telemetry_default,
        help="Sample GPU utilization, memory, and compute-component telemetry during the run.",
    )
    parser.add_argument(
        "--telemetry-interval",
        dest="telemetry_interval_s",
        type=float,
        default=interval_default,
        help="Sampling interval in seconds for GPU telemetry.",
    )
    parser.add_argument(
        "--telemetry-live",
        action="store_true",
        default=telemetry_live_default,
        help="Print one live GPU telemetry line per sample while the run is in progress.",
    )


def create_telemetry_monitor(
    *,
    enabled: bool,
    interval_s: float,
    live: bool,
    label: str,
) -> GpuTelemetryMonitor | None:
    """Create a telemetry monitor only when the caller requested one."""
    if not enabled:
        return None
    return GpuTelemetryMonitor(sample_interval_s=interval_s, live=live, label=label)


def format_telemetry_banner(interval_s: float, live: bool) -> str:
    """Return a short human-readable telemetry status message."""
    return (
        f"sampling every {interval_s:.2f} s"
        + (" with live printing enabled." if live else ".")
    )


def should_report_step(
    step: int,
    total_steps: int,
    report_every: int,
    *,
    one_based: bool = False,
) -> bool:
    """Return whether a step should emit a progress line."""
    if total_steps <= 0:
        return False

    current_step = step if one_based else step + 1
    return (
        current_step == 1
        or current_step == total_steps
        or current_step % report_every == 0
    )
