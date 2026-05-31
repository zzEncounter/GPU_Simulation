"""CLI entrypoint for the standalone CUDA backend."""

from __future__ import annotations

import argparse
import time

import numpy as np

from ring_ising.cli.common import (
    add_optimization_args,
    add_problem_args,
    add_telemetry_args,
    create_telemetry_monitor,
    format_telemetry_banner,
    should_report_step,
)
from ring_ising.runtime import print_gpu_telemetry_summary
from standalone_backend import RingIsingAdjointBackend, RingIsingConfig, make_initial_params

DEFAULT_STANDALONE_CONFIG = RingIsingConfig()
DEFAULT_STANDALONE_STEPS = 20
DEFAULT_STANDALONE_STEPSIZE = 0.08
DEFAULT_STANDALONE_SEED = 7
DEFAULT_STANDALONE_INIT_SCALE = 0.3
DEFAULT_STANDALONE_REPORT_EVERY = 5
DEFAULT_STANDALONE_TELEMETRY_INTERVAL_S = 0.5


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
        steps_default=DEFAULT_STANDALONE_STEPS,
        stepsize_default=DEFAULT_STANDALONE_STEPSIZE,
        seed_default=DEFAULT_STANDALONE_SEED,
        init_scale_default=DEFAULT_STANDALONE_INIT_SCALE,
        report_every_default=DEFAULT_STANDALONE_REPORT_EVERY,
    )
    parser.add_argument(
        "--gradient-strategy",
        choices=("auto", "checkpoint", "save_param_states", "bruteforce_parallel_q6"),
        default=DEFAULT_STANDALONE_CONFIG.gradient_strategy,
        help="Adjoint gradient memory strategy for the standalone backend.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval_ops",
        type=int,
        default=None,
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
    add_telemetry_args(parser, interval_default=DEFAULT_STANDALONE_TELEMETRY_INTERVAL_S)
    return parser.parse_args()


def _print_runtime_summary(
    args: argparse.Namespace,
    config: RingIsingConfig,
    backend: RingIsingAdjointBackend,
    params: np.ndarray,
    initial_energy: float,
) -> None:
    resolution = backend.strategy_resolution

    print("Standalone CUDA backend")
    print(f"  Qubits: {args.num_qubits}")
    print(f"  Layers: {args.layers}")
    print(f"  Field strength: {args.field}")
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
    print(
        "  Estimated gradient workspace: "
        f"{resolution.estimated_workspace_gib:.2f} GiB"
    )
    if resolution.note:
        print(f"  Strategy note: {resolution.note}")
    print(f"  Initial energy: {initial_energy:.10f}")
    if resolution.resolved_strategy == "bruteforce_parallel_q6":
        diag = backend.dense_scan_experiment(params)
        print("  Dense q<=6 experiment snapshot:")
        print(f"    CPU reference: {diag['cpu_reference_ms']:.3f} ms")
        print(f"    GPU scan: {diag['gpu_scan_ms']:.3f} ms")
        print(f"    Sequential statevector: {diag['sequential_statevector_ms']:.3f} ms")
        print(
            "    Forward states shape (num_states, state_size, 2): "
            f"{diag['forward_states_ri'].shape}"
        )
        print(
            "    Backward states shape (num_states, state_size, 2): "
            f"{diag['backward_states_ri'].shape}"
        )
    if args.gpu_telemetry:
        print(
            "  GPU telemetry: "
            + format_telemetry_banner(args.telemetry_interval_s, args.telemetry_live)
        )
    print()


def _initial_params(args: argparse.Namespace) -> np.ndarray:
    return make_initial_params(
        num_qubits=args.num_qubits,
        layers=args.layers,
        seed=args.seed,
        init_scale=args.init_scale,
    )


def _run_measured_loop(
    backend: RingIsingAdjointBackend,
    params: np.ndarray,
    *,
    steps: int,
    stepsize: float,
    report_every: int,
) -> tuple[np.ndarray, float]:
    total_loop_start = time.perf_counter()
    for step in range(steps):
        step_start = time.perf_counter()
        energy, grad = backend.energy_and_grad(params)
        grad_norm = float(np.linalg.norm(grad))
        params = params - stepsize * grad
        step_wall_ms = (time.perf_counter() - step_start) * 1000.0

        if should_report_step(step, steps, report_every):
            print(
                f"step={step:03d} energy={energy:.10f} grad_norm={grad_norm:.10f} "
                f"step_ms={step_wall_ms:.3f}"
            )

    return params, time.perf_counter() - total_loop_start


def main() -> None:
    args = parse_args()
    args_dict = vars(args)
    config = RingIsingConfig(
        **{
            field_name: args_dict[field_name]
            for field_name in RingIsingConfig.__dataclass_fields__
        }
    )
    backend = RingIsingAdjointBackend(config)
    telemetry_monitor = create_telemetry_monitor(
        enabled=args.gpu_telemetry,
        interval_s=args.telemetry_interval_s,
        live=args.telemetry_live,
        label="Standalone backend telemetry",
    )
    params = _initial_params(args)

    initial_energy = backend.energy(params)
    _print_runtime_summary(args, config, backend, params, float(initial_energy))
    telemetry_summary = None

    if telemetry_monitor is not None:
        telemetry_monitor.start()

    try:
        params, total_loop_s = _run_measured_loop(
            backend,
            params,
            steps=args.steps,
            stepsize=args.stepsize,
            report_every=args.report_every,
        )
    finally:
        telemetry_summary = telemetry_monitor.stop() if telemetry_monitor is not None else None

    final_energy = backend.energy(params)
    print()
    print("Summary:")
    print(f"  Final energy: {final_energy:.10f}")
    print(f"  Measured loop wall time: {total_loop_s:.4f} s")
    if args.steps:
        print(f"  Average step wall time: {1000.0 * total_loop_s / args.steps:.3f} ms")

    if telemetry_summary is not None:
        print()
        print_gpu_telemetry_summary(telemetry_summary, indent="  ")
