"""Unified CLI entrypoint for PennyLane and standalone workflows."""

from __future__ import annotations

import argparse

from ring_ising.cli.common import (
    add_optimization_args,
    add_problem_args,
    add_telemetry_args,
)
from ring_ising.config import BACKENDS, RunConfig, STANDALONE_GRADIENT_STRATEGIES
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
        choices=STANDALONE_GRADIENT_STRATEGIES,
        default=DEFAULT_RUN_CONFIG.gradient_strategy,
        help="Adjoint gradient memory strategy. Only used with --backend standalone.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval_ops",
        type=int,
        default=DEFAULT_RUN_CONFIG.checkpoint_interval_ops,
        help="Number of ops per checkpoint chunk. Only used with checkpoint strategy.",
    )
    parser.add_argument(
        "--intrablock-block-size",
        dest="intrablock_block_size",
        type=int,
        default=DEFAULT_RUN_CONFIG.intrablock_block_size,
        help="Ops per block for intrablock_parallel. Only used with that strategy.",
    )
    parser.add_argument(
        "--disable-gate-fusion",
        dest="gate_fusion",
        action="store_false",
        default=DEFAULT_RUN_CONFIG.gate_fusion,
        help="Disable gate fusion optimizations for standalone comparisons.",
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
