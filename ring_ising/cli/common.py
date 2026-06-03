"""Shared CLI helpers for the unified workflow runner."""

from __future__ import annotations

import argparse

from ring_ising.runtime import GpuTelemetryMonitor


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
) -> None:
    """Add common loop and initialization arguments."""
    parser.add_argument("--steps", type=int, default=steps_default, help="Measured gradient steps.")
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
