"""CLI entrypoint for the PennyLane baseline."""

from __future__ import annotations

import argparse

from ring_ising.baseline import BaselineConfig, run_baseline
from ring_ising.cli.common import (
    add_device_arg,
    add_optimization_args,
    add_problem_args,
    add_telemetry_args,
)

DEFAULT_BASELINE_CONFIG = BaselineConfig()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PennyLane baseline."""
    parser = argparse.ArgumentParser(
        description="Run a minimal PennyLane adjoint-diff baseline on a ring-Ising circuit."
    )
    add_problem_args(
        parser,
        qubits_default=DEFAULT_BASELINE_CONFIG.num_qubits,
        layers_default=DEFAULT_BASELINE_CONFIG.layers,
        field_default=DEFAULT_BASELINE_CONFIG.field,
    )
    add_optimization_args(
        parser,
        steps_default=DEFAULT_BASELINE_CONFIG.steps,
        stepsize_default=DEFAULT_BASELINE_CONFIG.stepsize,
        seed_default=DEFAULT_BASELINE_CONFIG.seed,
        init_scale_default=DEFAULT_BASELINE_CONFIG.init_scale,
        report_every_default=DEFAULT_BASELINE_CONFIG.report_every,
    )
    add_device_arg(parser, default=DEFAULT_BASELINE_CONFIG.device)
    add_telemetry_args(
        parser,
        interval_default=DEFAULT_BASELINE_CONFIG.telemetry_interval_s,
        gpu_telemetry_default=DEFAULT_BASELINE_CONFIG.gpu_telemetry,
        telemetry_live_default=DEFAULT_BASELINE_CONFIG.telemetry_live,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(BaselineConfig(**vars(args)))
