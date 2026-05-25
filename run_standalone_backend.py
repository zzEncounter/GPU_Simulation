"""Run the standalone CUDA backend without PennyLane's QNode stack."""

from __future__ import annotations

import argparse
import time

import numpy as np

from runtime_utils import GpuTelemetryMonitor, print_gpu_telemetry_summary
from standalone_backend import RingIsingAdjointBackend, RingIsingConfig, make_initial_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stepsize", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument(
        "--gradient-strategy",
        choices=("auto", "checkpoint", "save_param_states", "bruteforce_parallel_q6"),
        default="auto",
        help="Adjoint gradient memory strategy for the standalone backend.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Number of ops per checkpoint chunk. Only used with --gradient-strategy checkpoint.",
    )
    parser.add_argument(
        "--disable-gate-fusion",
        action="store_true",
        help="Disable gate fusion optimizations for A/B comparisons.",
    )
    parser.add_argument(
        "--auto-memory-budget-frac",
        type=float,
        default=0.85,
        help="Fraction of currently free GPU memory that auto strategy is allowed to budget.",
    )
    parser.add_argument(
        "--auto-memory-reserve-mib",
        type=int,
        default=1024,
        help="Extra GPU memory margin reserved when auto-selecting a strategy.",
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


def main() -> None:
    args = parse_args()
    config = RingIsingConfig(
        num_qubits=args.qubits,
        layers=args.layers,
        field=args.field,
        gradient_strategy=args.gradient_strategy,
        checkpoint_interval_ops=args.checkpoint_interval,
        gate_fusion=not args.disable_gate_fusion,
        auto_memory_budget_fraction=args.auto_memory_budget_frac,
        auto_memory_reserve_mib=args.auto_memory_reserve_mib,
    )
    backend = RingIsingAdjointBackend(config)
    resolution = backend.strategy_resolution
    telemetry_monitor = (
        GpuTelemetryMonitor(
            sample_interval_s=args.telemetry_interval,
            live=args.telemetry_live,
            label="Standalone backend telemetry",
        )
        if args.gpu_telemetry
        else None
    )

    params = make_initial_params(
        num_qubits=args.qubits,
        layers=args.layers,
        seed=args.seed,
        init_scale=args.init_scale,
    )

    initial_energy = backend.energy(params)
    print("Standalone CUDA backend")
    print(f"  Qubits: {args.qubits}")
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
    if telemetry_monitor is not None:
        print(
            "  GPU telemetry: "
            f"sampling every {args.telemetry_interval:.2f} s"
            + (" with live printing enabled." if args.telemetry_live else "")
        )
    print()

    total_loop_start = time.perf_counter()
    if telemetry_monitor is not None:
        telemetry_monitor.start()

    for step in range(args.steps):
        step_start = time.perf_counter()
        energy, grad = backend.energy_and_grad(params)
        grad_norm = float(np.linalg.norm(grad))
        params = params - args.stepsize * grad
        step_wall_ms = (time.perf_counter() - step_start) * 1000.0
        if step % args.report_every == 0 or step == args.steps - 1:
            print(
                f"step={step:03d} energy={energy:.10f} grad_norm={grad_norm:.10f} "
                f"step_ms={step_wall_ms:.3f}"
            )

    total_loop_s = time.perf_counter() - total_loop_start
    final_energy = backend.energy(params)
    print()
    print("Summary:")
    print(f"  Final energy: {final_energy:.10f}")
    print(f"  Measured loop wall time: {total_loop_s:.4f} s")
    if args.steps:
        print(f"  Average step wall time: {1000.0 * total_loop_s / args.steps:.3f} ms")

    if telemetry_monitor is not None:
        print()
        print_gpu_telemetry_summary(telemetry_monitor.stop(), indent="  ")


if __name__ == "__main__":
    main()
