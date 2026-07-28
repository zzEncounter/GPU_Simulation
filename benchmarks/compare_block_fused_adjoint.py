"""Compare standalone strategies and PennyLane lightning.gpu on selected cases."""

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

from ring_ising import RunConfig, run_pennylane, run_standalone
from ring_ising.backends.standalone.config import StandaloneBackendConfig
from ring_ising.runtime._nvidia_smi import parse_int_or_none, query_process_gpu_memory_rows

PENNYLANE_MODE = "pennylane_lightning_gpu"
MODES = (
    "save_param_states",
    "block_fused_adjoint",
    "inverse_walk",
    "inverse_walk_cuQuantum",
    "dense_scan",
    PENNYLANE_MODE,
)
DEFAULT_MODES = (
    "save_param_states",
    "block_fused_adjoint",
    "inverse_walk",
    "inverse_walk_cuQuantum",
    PENNYLANE_MODE,
)
STANDALONE_MODES = (
    "save_param_states",
    "block_fused_adjoint",
    "inverse_walk",
    "inverse_walk_cuQuantum",
    "dense_scan",
)
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


def parse_mode(text: str) -> str:
    mode = text.strip()
    if mode == "reverse_walk":
        return "inverse_walk"
    if mode not in MODES:
        choices = ", ".join((*MODES, "reverse_walk"))
        raise argparse.ArgumentTypeError(
            f"Invalid mode {text!r}. Expected one of: {choices}"
        )
    return mode


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
        intrablock_block_size=None,
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
        for mode in args.modes
        if mode in STANDALONE_MODES
        if mode != "dense_scan" or case.num_qubits <= 6
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


def run_config_memory_estimate(row_mode: str, estimates: dict[str, MemoryEstimate]) -> float | None:
    estimate = estimates.get(row_mode)
    if estimate is None:
        return None
    return estimate.workspace_gib


def build_run_config(
    mode: str,
    *,
    case: BenchmarkCase,
    steps: int,
    args: argparse.Namespace,
) -> RunConfig:
    backend = "pennylane" if mode == PENNYLANE_MODE else "standalone"
    gradient_strategy = "save_param_states" if backend == "pennylane" else mode
    return RunConfig(
        backend=backend,
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
        gradient_strategy=gradient_strategy,
        checkpoint_interval_ops=args.checkpoint_interval_ops,
        gate_fusion=args.gate_fusion,
    )


def run_mode(mode: str, *, case: BenchmarkCase, steps: int, args: argparse.Namespace):
    config = build_run_config(mode, case=case, steps=steps, args=args)
    if mode == PENNYLANE_MODE:
        return run_pennylane(config)
    return run_standalone(config)


def timing_avg_ms(result, key: str, steps: int) -> float | None:
    totals = result.metadata.get("backend_timing_totals_s", {})
    if not isinstance(totals, dict):
        return None
    value = totals.get(key)
    if value is None or steps <= 0:
        return None
    return float(value) * 1000.0 / steps


def phase_timing_row(result, *, steps: int) -> dict[str, float | None]:
    dense_matrix_ms = timing_avg_ms(result, "dense_vector_to_matrix_s", steps)
    dense_buffer_setup_ms = timing_avg_ms(result, "dense_buffer_setup_s", steps)
    dense_upload_ms = timing_avg_ms(result, "dense_upload_s", steps)
    forward_ms = timing_avg_ms(result, "forward_s", steps)
    backward_ms = timing_avg_ms(result, "backward_s", steps)
    gradient_ms = timing_avg_ms(result, "gradient_s", steps)
    return {
        "dense_vector_to_matrix_avg_ms": dense_matrix_ms,
        "dense_buffer_setup_avg_ms": dense_buffer_setup_ms,
        "dense_upload_avg_ms": dense_upload_ms,
        "forward": forward_ms,
        "backward": backward_ms,
        "gradient": gradient_ms,
    }


def format_optional_float(value: object, fmt: str, *, fallback: str = "") -> str:
    if value is None:
        return fallback
    return format(float(value), fmt)


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
    parser.add_argument(
        "--modes",
        nargs="+",
        type=parse_mode,
        default=list(DEFAULT_MODES),
        help="Modes to compare.",
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

    print("Block fused adjoint / dense scan / PennyLane comparison")
    print(f"  Modes: {', '.join(args.modes)}")
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
                (estimate.workspace_gib for estimate in estimates.values()),
                default=0.0,
            )
            if args.skip_estimated_over_limit and not case_allowed(
                estimates, max_workspace_gib=args.max_gpu_memory_gib
            ):
                print(
                    f"Skipping q{case.num_qubits}_l{case.layers}: "
                    f"estimated workspace max={max_estimated_gib:.3f} GiB "
                    f"> limit={args.max_gpu_memory_gib:.3f} GiB"
                )
                for mode in args.modes:
                    estimate = estimates.get(mode)
                    rows.append(
                        {
                            "num_qubits": case.num_qubits,
                            "layers": case.layers,
                            "steps": 0,
                            "mode": mode,
                            "avg_step_ms": None,
                            "total_wall_s": None,
                            "dense_vector_to_matrix_avg_ms": None,
                            "dense_buffer_setup_avg_ms": None,
                            "dense_upload_avg_ms": None,
                            "forward": None,
                            "backward": None,
                            "gradient": None,
                            "speedup_vs_inverse_walk": None,
                            "speedup_vs_pennylane_lighting_gpu": None,
                            "final_energy": None,
                            "final_energy_abs_diff_vs_save_param_states": None,
                            "checkpoint_interval_ops": (
                                None if estimate is None else estimate.checkpoint_interval_ops
                            ),
                            "estimated_workspace_gib": (
                                None if estimate is None else estimate.workspace_gib
                            ),
                            "peak_process_memory_gib": guard.peak_mib / MIB_PER_GIB,
                            "status": "skipped_estimated_over_limit",
                        }
                    )
                continue

            steps = args.steps if args.steps is not None else default_steps_for(case)
            case_rows: list[dict[str, object]] = []

            for mode in args.modes:
                if mode == "dense_scan" and case.num_qubits > 6:
                    case_rows.append(
                        {
                            "num_qubits": case.num_qubits,
                            "layers": case.layers,
                            "steps": 0,
                            "mode": mode,
                            "avg_step_ms": None,
                            "total_wall_s": None,
                            "dense_vector_to_matrix_avg_ms": None,
                            "dense_buffer_setup_avg_ms": None,
                            "dense_upload_avg_ms": None,
                            "forward": None,
                            "backward": None,
                            "gradient": None,
                            "speedup_vs_pennylane_lighting_gpu": None,
                            "final_energy": None,
                            "checkpoint_interval_ops": None,
                            "estimated_workspace_gib": None,
                            "peak_process_memory_gib": guard.peak_mib / MIB_PER_GIB,
                            "status": "skipped_dense_scan_qubits_gt_6",
                        }
                    )
                    continue

                run_start = time.perf_counter()
                result = run_mode(mode, case=case, steps=steps, args=args)
                total_wall_s = time.perf_counter() - run_start
                timing_row = phase_timing_row(result, steps=steps)
                case_rows.append(
                    {
                        "num_qubits": case.num_qubits,
                        "layers": case.layers,
                        "steps": steps,
                        "mode": mode,
                        "avg_step_ms": total_wall_s * 1000.0 / steps,
                        "total_wall_s": total_wall_s,
                        **timing_row,
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

            speedup_reference = next(
                (
                    row
                    for row in case_rows
                    if row["mode"] == "inverse_walk" and row["status"] == "ok"
                ),
                None,
            )
            speedup_ref_step_ms = (
                None
                if speedup_reference is None
                else float(speedup_reference["avg_step_ms"])
            )
            energy_reference = next(
                (
                    row
                    for row in case_rows
                    if row["mode"] == "save_param_states" and row["status"] == "ok"
                ),
                None,
            )
            ref_energy = (
                None
                if energy_reference is None
                else float(energy_reference["final_energy"])
            )
            pennylane_reference = next(
                (
                    row
                    for row in case_rows
                    if row["mode"] == PENNYLANE_MODE and row["status"] == "ok"
                ),
                None,
            )
            pennylane_ref_step_ms = (
                None
                if pennylane_reference is None
                else float(pennylane_reference["avg_step_ms"])
            )

            for row in case_rows:
                if row["status"] != "ok":
                    row["speedup_vs_inverse_walk"] = None
                    row["speedup_vs_pennylane_lighting_gpu"] = None
                    row["final_energy_abs_diff_vs_save_param_states"] = None
                    rows.append(row)
                    continue
                avg_step_ms = float(row["avg_step_ms"])
                row["speedup_vs_inverse_walk"] = (
                    speedup_ref_step_ms / avg_step_ms
                    if speedup_ref_step_ms is not None and avg_step_ms > 0.0
                    else None
                )
                row["speedup_vs_pennylane_lighting_gpu"] = (
                    pennylane_ref_step_ms / avg_step_ms
                    if pennylane_ref_step_ms is not None and avg_step_ms > 0.0
                    else None
                )
                row["final_energy_abs_diff_vs_save_param_states"] = (
                    abs(float(row["final_energy"]) - ref_energy)
                    if ref_energy is not None
                    else None
                )
                rows.append(row)

            print(f"q{case.num_qubits}_l{case.layers}, steps={steps}")
            for row in case_rows:
                if row["status"] != "ok":
                    print(f"  {row['mode']:<28} {row['status']}")
                    continue
                dense_matrix_ms = row["dense_vector_to_matrix_avg_ms"]
                dense_detail = ""
                if dense_matrix_ms is not None:
                    dense_detail = f" dense_matrix_ms={float(dense_matrix_ms):.3f}"
                if row["dense_buffer_setup_avg_ms"] is not None:
                    dense_detail += (
                        f" dense_buffer_ms="
                        f"{float(row['dense_buffer_setup_avg_ms']):.3f}"
                    )
                if row["dense_upload_avg_ms"] is not None:
                    dense_detail += (
                        f" dense_upload_ms={float(row['dense_upload_avg_ms']):.3f}"
                    )
                phase_detail = ""
                if row["forward"] is not None:
                    phase_detail += f" forward={float(row['forward']):.3f}"
                if row["backward"] is not None:
                    phase_detail += f" backward={float(row['backward']):.3f}"
                if row["gradient"] is not None:
                    phase_detail += f" gradient={float(row['gradient']):.3f}"
                speedup_text = format_optional_float(
                    row["speedup_vs_inverse_walk"], ".3f", fallback="nan"
                )
                energy_diff_text = format_optional_float(
                    row["final_energy_abs_diff_vs_save_param_states"],
                    ".3e",
                    fallback="nan",
                )
                print(
                    f"  {row['mode']:<28} "
                    f"avg_step_ms={float(row['avg_step_ms']):.3f} "
                    f"speedup={speedup_text}x "
                    f"energy_diff={energy_diff_text} "
                    f"workspace_gib={row['estimated_workspace_gib']} "
                    f"peak_process_gib={float(row['peak_process_memory_gib']):.4f}"
                    f"{dense_detail}"
                    f"{phase_detail}"
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
                    "dense_vector_to_matrix_avg_ms",
                    "dense_buffer_setup_avg_ms",
                    "dense_upload_avg_ms",
                    "forward",
                    "backward",
                    "gradient",
                    "speedup_vs_inverse_walk",
                    "speedup_vs_pennylane_lighting_gpu",
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
