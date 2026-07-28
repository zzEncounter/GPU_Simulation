"""Unified CLI entrypoint for PennyLane and standalone workflows."""

from __future__ import annotations

import argparse

from ring_ising.cli.common import (
    add_optimization_args,
    add_problem_args,
    add_telemetry_args,
)
from ring_ising.config import (
    BACKENDS,
    RunConfig,
    STRUCTURED_ROTATION_CHUNK_WIDTH_MAX,
    STANDALONE_GRADIENT_STRATEGIES,
    SUPPORTED_STANDALONE_GRADIENT_STRATEGIES,
)
from ring_ising.workflows import run

DEFAULT_RUN_CONFIG = RunConfig()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the unified workflow runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the ring-Ising experiment with either the PennyLane or "
            "standalone backend. PennyLane runs require lightning.gpu."
        )
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="standalone",
        help="Backend to execute: 'pennylane' or 'standalone'.",
    )
    add_problem_args(
        parser,
        qubits_default=DEFAULT_RUN_CONFIG.num_qubits,
        layers_default=DEFAULT_RUN_CONFIG.layers,
        field_default=DEFAULT_RUN_CONFIG.field,
    )
    add_optimization_args(
        parser,
        steps_default=DEFAULT_RUN_CONFIG.steps,
        stepsize_default=DEFAULT_RUN_CONFIG.stepsize,
        seed_default=DEFAULT_RUN_CONFIG.seed,
        init_scale_default=DEFAULT_RUN_CONFIG.init_scale,
    )
    parser.add_argument(
        "--gradient-strategy",
        choices=SUPPORTED_STANDALONE_GRADIENT_STRATEGIES,
        default=DEFAULT_RUN_CONFIG.gradient_strategy,
        help=(
            "Standalone adjoint strategy. Public strategies: "
            f"{', '.join(STANDALONE_GRADIENT_STRATEGIES)}."
        ),
    )
    parser.add_argument(
        "--structured-rotation-chunk-width",
        type=int,
        choices=range(1, STRUCTURED_ROTATION_CHUNK_WIDTH_MAX + 1),
        default=DEFAULT_RUN_CONFIG.structured_rotation_chunk_width,
        metavar=f"1..{STRUCTURED_ROTATION_CHUNK_WIDTH_MAX}",
        help=(
            "Rotation-layer fusion width for --gradient-strategy structured_adjoint."
        ),
    )
    parser.add_argument(
        "--mode2-rotation-chunk-width",
        dest="structured_rotation_chunk_width",
        type=int,
        choices=range(1, STRUCTURED_ROTATION_CHUNK_WIDTH_MAX + 1),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--double-buffer",
        dest="double_buffer",
        action="store_true",
        default=DEFAULT_RUN_CONFIG.double_buffer,
        help=(
            "Enable shared-memory double-buffer kernels for rotation layers "
            "(structured_adjoint only)."
        ),
    )
    parser.add_argument(
        "--report-steps",
        action="store_true",
        default=DEFAULT_RUN_CONFIG.report_steps,
        help=(
            "Print detailed step reports at about 60 evenly spaced checkpoints "
            "instead of only showing the progress bar."
        ),
    )
    add_telemetry_args(
        parser,
        interval_default=DEFAULT_RUN_CONFIG.telemetry_interval_s,
        gpu_telemetry_default=DEFAULT_RUN_CONFIG.gpu_telemetry,
        telemetry_live_default=DEFAULT_RUN_CONFIG.telemetry_live,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(RunConfig(**vars(args)))
