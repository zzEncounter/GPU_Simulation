"""CLI entrypoint for the standalone CUDA backend."""

from __future__ import annotations

import argparse

from ring_ising.cli.common import add_optimization_args, add_problem_args, add_telemetry_args
from standalone_backend import StandaloneRunConfig, run_standalone

DEFAULT_STANDALONE_CONFIG = StandaloneRunConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_problem_args(
        parser,
        qubits_default=DEFAULT_STANDALONE_CONFIG.num_qubits,
        layers_default=DEFAULT_STANDALONE_CONFIG.layers,
        field_default=DEFAULT_STANDALONE_CONFIG.field,
    )
    add_optimization_args(
        parser,
        steps_default=DEFAULT_STANDALONE_CONFIG.steps,
        stepsize_default=DEFAULT_STANDALONE_CONFIG.stepsize,
        seed_default=DEFAULT_STANDALONE_CONFIG.seed,
        init_scale_default=DEFAULT_STANDALONE_CONFIG.init_scale,
        report_every_default=DEFAULT_STANDALONE_CONFIG.report_every,
    )
    parser.add_argument(
        "--gradient-strategy",
        choices=("auto", "checkpoint", "save_param_states", "dense_scan"),
        default=DEFAULT_STANDALONE_CONFIG.gradient_strategy,
        help="Adjoint gradient memory strategy for the standalone backend.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval_ops",
        type=int,
        default=DEFAULT_STANDALONE_CONFIG.checkpoint_interval_ops,
        help="Number of ops per checkpoint chunk. Only used with --gradient-strategy checkpoint.",
    )
    parser.add_argument(
        "--disable-gate-fusion",
        dest="gate_fusion",
        action="store_false",
        default=DEFAULT_STANDALONE_CONFIG.gate_fusion,
        help="Disable gate fusion optimizations for A/B comparisons.",
    )
    parser.add_argument(
        "--auto-memory-budget-frac",
        dest="auto_memory_budget_fraction",
        type=float,
        default=DEFAULT_STANDALONE_CONFIG.auto_memory_budget_fraction,
        help="Fraction of currently free GPU memory that auto strategy is allowed to budget.",
    )
    parser.add_argument(
        "--auto-memory-reserve-mib",
        dest="auto_memory_reserve_mib",
        type=int,
        default=DEFAULT_STANDALONE_CONFIG.auto_memory_reserve_mib,
        help="Extra GPU memory margin reserved when auto-selecting a strategy.",
    )
    add_telemetry_args(parser, interval_default=DEFAULT_STANDALONE_CONFIG.telemetry_interval_s)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_standalone(StandaloneRunConfig(**vars(args)))
