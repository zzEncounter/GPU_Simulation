"""Compare block_fused_adjoint against save_param_states on large cases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising import RunConfig, run_standalone
from ring_ising.backends.standalone.config import StandaloneBackendConfig
from ring_ising.runtime._nvidia_smi import parse_int_or_none, query_process_gpu_memory_rows

MODES = ("save_param_states", "block_fused_adjoint")
MIB_PER_GIB = 1024
DEFAULT_MAX_GPU_MEMORY_GIB = 20.0


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


@dataclass(frozen=True)
class MemoryEstimate:
    mode: str
    checkpoint_interval_ops: int
    workspace_gib: float


class GpuMemoryGuard:
    def __init__(self, *, limit_mib: int, interval_s: float) -> None:
        self.limit_mib = limit_mib
        self.interval_s = interval_s
        self.pid = os.getpid()
        self.peak_mib = 0
        self.note: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> GpuMemoryGuard:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            rows, note = query_process_gpu_memory_rows()
            if note is not None:
                self.note = note
            used_mib = 0
            for row in rows:
                if row.get("pid") != str(self.pid):
                    continue
                value = parse_int_or_none(row.get("used_gpu_memory", ""))
                if value is not None:
                    used_mib += value
            self.peak_mib = max(self.peak_mib, used_mib)
            if used_mib > self.limit_mib:
                print(
                    "\nGPU memory guard exceeded "
                    f"{self.limit_mib / MIB_PER_GIB:.2f} GiB "
                    f"(current={used_mib / MIB_PER_GIB:.2f} GiB). "
                    "Terminating this benchmark process.",
                    flush=True,
                )
                os._exit(137)
            self._stop.wait(self.interval_s)


DEFAULT_QUBITS = tuple(range(7, 25))
DEFAULT_LAYERS = (32, 128, 512, 1024, 2048)


def default_steps_for(case: BenchmarkCase) -> int:
    base_steps_by_layers = {32: 80, 128: 40, 512: 12, 1024: 6, 2048: 3}
    base = base_steps_by_layers.get(case.layers, max(2, 960 // max(case.layers, 1)))
    return max(2, int(round(base * 4.0 / case.num_qubits)))


def build_cases(qubits: list[int], layers: list[int]) -> tuple[BenchmarkCase, ...]:
    return tuple(
        BenchmarkCase(num_qubits=num_qubits, layers=layer_count)
        for num_qubits in qubits
        for layer_count in layers
    )


def estimate_mode_memory(
    mode: str,
    *,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> MemoryEstimate:
    config = StandaloneBackendConfig(
        num_qubits=case.num_qubits,
        layers=case.layers,
        field=args.field,
        gradient_strategy=mode,
        checkpoint_interval_ops=args.checkpoint_interval_ops,
        gate_fusion=args.gate_fusion,
    )
    config.validate()
    interval = config.resolve_checkpoint_interval_ops(mode)
    return MemoryEstimate(
        mode=mode,
        checkpoint_interval_ops=interval,
        workspace_gib=config.estimated_gradient_workspace_gib_for(mode, interval),
    )


def estimate_case_memory(
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, MemoryEstimate]:
    return {
        mode: estimate_mode_memory(mode, case=case, args=args)
        for mode in MODES
    }


def case_allowed(
    estimates: dict[str, MemoryEstimate],
    *,
    max_workspace_gib: float,
) -> bool:
    return all(
        estimate.workspace_gib <= max_workspace_gib
        for estimate in estimates.values()
    )


def run_config_memory_estimate(row_mode: str, estimates: dict[str, MemoryEstimate]) -> float:
    return estimates[row_mode].workspace_gib


def build_run_config(
    mode: str,
    *,
    case: BenchmarkCase,
    steps: int,
    args: argparse.Namespace,
) -> RunConfig:
    return RunConfig(
        backend="standalone",
        num_qubits=case.num_qubits,
        layers=case.layers,
        field=args.field,
        steps=steps,
        stepsize=args.stepsize,
        seed=args.seed,
        init_scale=args.init_scale,
        verbose=False,
        show_progress=False,
        report_steps=False,
        gpu_telemetry=False,
        gradient_strategy=mode,
        checkpoint_interval_ops=args.checkpoint_interval_ops,
        gate_fusion=args.gate_fusion,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=list(DEFAULT_QUBITS),
        help="Qubit counts to test. Defaults to 7 through 24.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS),
        help="Layer counts to test. Defaults to 32 128 512 1024 2048.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override steps for every case. Defaults scale by layer count.",
    )
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--stepsize", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval_ops",
        type=int,
        default=None,
        help="Ops per checkpoint block for block_fused_adjoint.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=float,
        default=DEFAULT_MAX_GPU_MEMORY_GIB,
        help="Hard per-process GPU memory limit. Defaults to 20 GiB.",
    )
    parser.add_argument(
        "--memory-guard-interval",
        type=float,
        default=0.01,
        help="Seconds between nvidia-smi process-memory checks.",
    )
    parser.add_argument(
        "--skip-estimated-over-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip cases when either mode's estimated workspace exceeds the "
            "GPU memory limit. Enabled by default."
        ),
    )
    parser.add_argument(
        "--disable-gate-fusion",
        dest="gate_fusion",
        action="store_false",
        default=True,
        help="Disable existing gate fusion for both modes.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT
        / "benchmarks"
        / "results"
        / "block_fused_adjoint_memory_capped_grid.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def terminate(message: str) -> NoReturn:
    raise SystemExit(message)


def main() -> None:
    args = parse_args()
    if args.max_gpu_memory_gib > DEFAULT_MAX_GPU_MEMORY_GIB:
        terminate(
            "Refusing to use a GPU memory limit above 20 GiB. "
            "Pass --max-gpu-memory-gib <= 20."
        )
    if args.max_gpu_memory_gib <= 0:
        terminate("--max-gpu-memory-gib must be positive.")

    cases = build_cases(args.qubits, args.layers)
    rows: list[dict[str, object]] = []
    limit_mib = int(args.max_gpu_memory_gib * MIB_PER_GIB)

    print("Block fused adjoint comparison")
    print(f"  Modes: {', '.join(MODES)}")
    print(f"  Qubits: {', '.join(str(value) for value in args.qubits)}")
    print(f"  Layers: {', '.join(str(value) for value in args.layers)}")
    print(f"  Cases: {len(cases)}")
    print(f"  Gate fusion enabled: {args.gate_fusion}")
    print(f"  GPU memory hard limit: {args.max_gpu_memory_gib:.2f} GiB")
    print(f"  Skip estimated over-limit cases: {args.skip_estimated_over_limit}")
    if args.checkpoint_interval_ops is not None:
        print(f"  Checkpoint interval: {args.checkpoint_interval_ops} ops")
    print()

    with GpuMemoryGuard(
        limit_mib=limit_mib,
        interval_s=args.memory_guard_interval,
    ) as guard:
        for case in cases:
            estimates = estimate_case_memory(case, args)
            max_estimated_gib = max(
                estimate.workspace_gib for estimate in estimates.values()
            )
            if args.skip_estimated_over_limit and not case_allowed(
                estimates, max_workspace_gib=args.max_gpu_memory_gib
            ):
                print(
                    f"Skipping q{case.num_qubits}_l{case.layers}: "
                    f"estimated workspace max={max_estimated_gib:.3f} GiB "
                    f"> limit={args.max_gpu_memory_gib:.3f} GiB"
                )
                for mode, estimate in estimates.items():
                    rows.append(
                        {
                            "num_qubits": case.num_qubits,
                            "layers": case.layers,
                            "steps": 0,
                            "mode": mode,
                            "avg_step_ms": None,
                            "total_wall_s": None,
                            "speedup_vs_save_param_states": None,
                            "final_energy": None,
                            "final_energy_abs_diff_vs_save_param_states": None,
                            "checkpoint_interval_ops": estimate.checkpoint_interval_ops,
                            "estimated_workspace_gib": estimate.workspace_gib,
                            "peak_process_memory_gib": guard.peak_mib / MIB_PER_GIB,
                            "status": "skipped_estimated_over_limit",
                        }
                    )
                continue

            steps = args.steps if args.steps is not None else default_steps_for(case)
            case_rows: list[dict[str, object]] = []

            for mode in MODES:
                result = run_standalone(
                    build_run_config(mode, case=case, steps=steps, args=args)
                )
                case_rows.append(
                    {
                        "num_qubits": case.num_qubits,
                        "layers": case.layers,
                        "steps": steps,
                        "mode": mode,
                        "avg_step_ms": result.timings.wall_s * 1000.0 / steps,
                        "total_wall_s": result.timings.wall_s,
                        "final_energy": float(result.final_energy),
                        "checkpoint_interval_ops": result.metadata.get(
                            "checkpoint_interval_ops"
                        ),
                        "estimated_workspace_gib": run_config_memory_estimate(
                            mode, estimates
                        ),
                        "peak_process_memory_gib": guard.peak_mib / MIB_PER_GIB,
                        "status": "ok",
                    }
                )

            reference = next(row for row in case_rows if row["mode"] == "save_param_states")
            ref_step_ms = float(reference["avg_step_ms"])
            ref_energy = float(reference["final_energy"])

            for row in case_rows:
                avg_step_ms = float(row["avg_step_ms"])
                row["speedup_vs_save_param_states"] = (
                    ref_step_ms / avg_step_ms if avg_step_ms > 0.0 else None
                )
                row["final_energy_abs_diff_vs_save_param_states"] = abs(
                    float(row["final_energy"]) - ref_energy
                )
                rows.append(row)

            print(f"q{case.num_qubits}_l{case.layers}, steps={steps}")
            for row in case_rows:
                print(
                    f"  {row['mode']:<20} "
                    f"avg_step_ms={float(row['avg_step_ms']):.3f} "
                    f"speedup={float(row['speedup_vs_save_param_states']):.3f}x "
                    f"energy_diff={float(row['final_energy_abs_diff_vs_save_param_states']):.3e} "
                    f"workspace_gib={float(row['estimated_workspace_gib']):.4f} "
                    f"peak_process_gib={float(row['peak_process_memory_gib']):.4f}"
                )
            print()

        if guard.note is not None:
            print(f"GPU memory guard note: {guard.note}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "num_qubits",
                    "layers",
                    "steps",
                    "mode",
                    "avg_step_ms",
                    "total_wall_s",
                    "speedup_vs_save_param_states",
                    "final_energy",
                    "final_energy_abs_diff_vs_save_param_states",
                    "checkpoint_interval_ops",
                    "estimated_workspace_gib",
                    "peak_process_memory_gib",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to: {args.csv_out}")


if __name__ == "__main__":
    main()
