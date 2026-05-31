"""Compare PennyLane and standalone adjoint strategies."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising.baseline import BaselineConfig, create_workflow
from ring_ising.runtime import (
    GpuTelemetrySummary,
    capture_telemetry_window,
    derive_timing_breakdown,
    median_runtime_ms,
    median_timing_fields,
)
from standalone_backend import RingIsingAdjointBackend, RingIsingConfig, make_initial_params


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


@dataclass(frozen=True)
class ReferenceData:
    energy: float
    grad: np.ndarray


@dataclass(frozen=True)
class ModeResult:
    mode: str
    ok: bool
    note: str | None
    energy_abs_diff: float | None
    grad_max_abs_diff: float | None
    grad_l2_diff: float | None
    forward_ms: float | None
    back_ms: float | None
    gradient_ms: float | None
    total_ms: float | None
    workspace_gib: float | None
    cpu_reference_ms: float | None
    gpu_scan_ms: float | None
    sequential_statevector_ms: float | None
    peak_fb_memory_mib: int | None
    peak_process_memory_mib: int | None
    telemetry_samples: int
    gate_fusion_enabled: bool = True


def _flatten_row(case: BenchmarkCase, result: ModeResult) -> dict[str, object]:
    return {
        "num_qubits": case.num_qubits,
        "layers": case.layers,
        "mode": result.mode,
        "ok": result.ok,
        "note": result.note or "",
        "energy_abs_diff": result.energy_abs_diff,
        "grad_max_abs_diff": result.grad_max_abs_diff,
        "grad_l2_diff": result.grad_l2_diff,
        "fwd_ms": result.forward_ms,
        "back_ms": result.back_ms,
        "gradient_ms": result.gradient_ms,
        "total_ms": result.total_ms,
        "workspace_gib": result.workspace_gib,
        "cpu_reference_ms": result.cpu_reference_ms,
        "gpu_scan_ms": result.gpu_scan_ms,
        "sequential_statevector_ms": result.sequential_statevector_ms,
        "peak_fb_memory_mib": result.peak_fb_memory_mib,
        "peak_process_memory_mib": result.peak_process_memory_mib,
        "telemetry_samples": result.telemetry_samples,
        "gate_fusion_enabled": result.gate_fusion_enabled,
    }


def parse_case(text: str) -> BenchmarkCase:
    try:
        qubits_text, layers_text = text.lower().split("x", maxsplit=1)
        return BenchmarkCase(num_qubits=int(qubits_text), layers=int(layers_text))
    except Exception as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid case {text!r}. Expected format like 24x2."
        ) from exc


_CUDA_RUNTIME: ctypes.CDLL | None | bool = None


def _cuda_synchronize() -> None:
    """Best-effort CUDA sync for timing; unavailable runtimes are ignored."""

    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is False:
        return
    if _CUDA_RUNTIME is None:
        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.13",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                _CUDA_RUNTIME = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        else:
            _CUDA_RUNTIME = False
            return
    try:
        _CUDA_RUNTIME.cudaDeviceSynchronize()
    except AttributeError:
        _CUDA_RUNTIME = False


def _peak_memory_from_summary(summary: GpuTelemetrySummary) -> tuple[int | None, int | None]:
    if not summary.gpus:
        return None, None
    peak_fb = max(
        (
            gpu.peak_fb_memory_used_mib
            for gpu in summary.gpus
            if gpu.peak_fb_memory_used_mib is not None
        ),
        default=None,
    )
    peak_proc = max(
        (
            gpu.peak_process_used_gpu_memory_mib
            for gpu in summary.gpus
            if gpu.peak_process_used_gpu_memory_mib is not None
        ),
        default=None,
    )
    return peak_fb, peak_proc


def _build_reference(case: BenchmarkCase, field: float, seed: int, init_scale: float) -> tuple[np.ndarray, ReferenceData]:
    params_np = make_initial_params(
        num_qubits=case.num_qubits,
        layers=case.layers,
        seed=seed,
        init_scale=init_scale,
    )
    params_np = np.asarray(params_np, dtype=np.float64)
    params_pl = pnp.array(params_np, requires_grad=True)

    workflow = create_workflow(
        BaselineConfig(
            num_qubits=case.num_qubits,
            layers=case.layers,
            field=field,
            warmup=0,
            steps=0,
            device="gpu",
        )
    )
    energy = float(workflow.energy_qnode(params_pl))
    grad = np.asarray(workflow.gradient_fn(params_pl), dtype=np.float64)
    return params_np, ReferenceData(energy=energy, grad=grad)


def _benchmark_pennylane(
    case: BenchmarkCase,
    params_np: np.ndarray,
    ref: ReferenceData,
    field: float,
    repeats: int,
    warmup: int,
    telemetry_interval_s: float,
    telemetry_target_ms: float,
    gate_fusion_enabled: bool,
) -> ModeResult:
    params_pl = pnp.array(params_np, requires_grad=True)
    workflow = create_workflow(
        BaselineConfig(
            num_qubits=case.num_qubits,
            layers=case.layers,
            field=field,
            warmup=0,
            steps=0,
            device="gpu",
        )
    )

    try:
        forward_ms = median_runtime_ms(
            lambda: float(workflow.energy_qnode(params_pl)),
            repeats,
            warmup,
            synchronize=_cuda_synchronize,
        )
        gradient_ms = median_runtime_ms(
            lambda: np.asarray(workflow.gradient_fn(params_pl), dtype=np.float64),
            repeats,
            warmup,
            synchronize=_cuda_synchronize,
        )
        back_ms, total_ms = derive_timing_breakdown(forward_ms, gradient_ms)
        telemetry_summary = capture_telemetry_window(
            lambda: np.asarray(workflow.gradient_fn(params_pl), dtype=np.float64),
            target_ms=telemetry_target_ms,
            measured_ms=gradient_ms,
            interval_s=telemetry_interval_s,
        )
        peak_fb, peak_proc = _peak_memory_from_summary(telemetry_summary)
        return ModeResult(
            mode="pennylane",
            ok=True,
            note=None,
            energy_abs_diff=0.0,
            grad_max_abs_diff=0.0,
            grad_l2_diff=0.0,
            forward_ms=forward_ms,
            back_ms=back_ms,
            gradient_ms=gradient_ms,
            total_ms=total_ms,
            workspace_gib=None,
            cpu_reference_ms=None,
            gpu_scan_ms=None,
            sequential_statevector_ms=None,
            peak_fb_memory_mib=peak_fb,
            peak_process_memory_mib=peak_proc,
            telemetry_samples=telemetry_summary.snapshots_collected,
            gate_fusion_enabled=gate_fusion_enabled,
        )
    except Exception as exc:  # pragma: no cover - surfaced in output
        return ModeResult(
            mode="pennylane",
            ok=False,
            note=f"{type(exc).__name__}: {exc}",
            energy_abs_diff=None,
            grad_max_abs_diff=None,
            grad_l2_diff=None,
            forward_ms=None,
            back_ms=None,
            gradient_ms=None,
            total_ms=None,
            workspace_gib=None,
            cpu_reference_ms=None,
            gpu_scan_ms=None,
            sequential_statevector_ms=None,
            peak_fb_memory_mib=None,
            peak_process_memory_mib=None,
            telemetry_samples=0,
            gate_fusion_enabled=gate_fusion_enabled,
        )


def _benchmark_standalone(
    mode: str,
    case: BenchmarkCase,
    params_np: np.ndarray,
    ref: ReferenceData | None,
    field: float,
    repeats: int,
    warmup: int,
    telemetry_interval_s: float,
    telemetry_target_ms: float,
    gate_fusion_enabled: bool,
) -> ModeResult:
    config = RingIsingConfig(
        num_qubits=case.num_qubits,
        layers=case.layers,
        field=field,
        gradient_strategy=mode,
        gate_fusion=gate_fusion_enabled,
    )

    try:
        backend = RingIsingAdjointBackend(config)
        resolution = backend.strategy_resolution
        dense_diag = (
            backend.dense_scan_experiment(params_np)
            if mode == "bruteforce_parallel_q6"
            else None
        )
        timed = backend.energy_and_grad_with_timings(params_np)
        energy = float(timed["energy"])
        grad = np.asarray(timed["gradient"], dtype=np.float64)
        if ref is not None:
            grad_diff = ref.grad - grad
            energy_abs_diff = abs(ref.energy - energy)
            grad_max_abs_diff = float(np.max(np.abs(grad_diff)))
            grad_l2_diff = float(np.linalg.norm(grad_diff))
        else:
            energy_abs_diff = None
            grad_max_abs_diff = None
            grad_l2_diff = None

        timings = median_timing_fields(
            lambda: backend.energy_and_grad_with_timings(params_np),
            ("forward_ms", "back_ms", "gradient_ms", "total_ms"),
            repeats,
            warmup,
        )
        forward_ms = timings["forward_ms"]
        back_ms = timings["back_ms"]
        gradient_ms = timings["gradient_ms"]
        total_ms = timings["total_ms"]
        heavy_call = lambda: backend.energy_and_grad(params_np)
        telemetry_summary = capture_telemetry_window(
            heavy_call,
            target_ms=telemetry_target_ms,
            measured_ms=gradient_ms,
            interval_s=telemetry_interval_s,
        )
        peak_fb, peak_proc = _peak_memory_from_summary(telemetry_summary)
        return ModeResult(
            mode=mode,
            ok=True,
            note=(
                None
                if resolution.requested_strategy == resolution.resolved_strategy
                else f"resolved={resolution.resolved_strategy}"
            ),
            energy_abs_diff=energy_abs_diff,
            grad_max_abs_diff=grad_max_abs_diff,
            grad_l2_diff=grad_l2_diff,
            forward_ms=forward_ms,
            back_ms=back_ms,
            gradient_ms=gradient_ms,
            total_ms=total_ms,
            workspace_gib=resolution.estimated_workspace_gib,
            cpu_reference_ms=(
                float(dense_diag["cpu_reference_ms"]) if dense_diag is not None else None
            ),
            gpu_scan_ms=(
                float(dense_diag["gpu_scan_ms"]) if dense_diag is not None else None
            ),
            sequential_statevector_ms=(
                float(dense_diag["sequential_statevector_ms"])
                if dense_diag is not None
                else None
            ),
            peak_fb_memory_mib=peak_fb,
            peak_process_memory_mib=peak_proc,
            telemetry_samples=telemetry_summary.snapshots_collected,
            gate_fusion_enabled=gate_fusion_enabled,
        )
    except Exception as exc:
        return ModeResult(
            mode=mode,
            ok=False,
            note=f"{type(exc).__name__}: {exc}",
            energy_abs_diff=None,
            grad_max_abs_diff=None,
            grad_l2_diff=None,
            forward_ms=None,
            back_ms=None,
            gradient_ms=None,
            total_ms=None,
            workspace_gib=None,
            cpu_reference_ms=None,
            gpu_scan_ms=None,
            sequential_statevector_ms=None,
            peak_fb_memory_mib=None,
            peak_process_memory_mib=None,
            telemetry_samples=0,
            gate_fusion_enabled=gate_fusion_enabled,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[
            parse_case("16x1"),
            parse_case("16x2"),
            parse_case("16x4"),
            parse_case("20x1"),
            parse_case("20x2"),
            parse_case("20x4"),
            parse_case("24x1"),
            parse_case("24x2"),
            parse_case("24x4"),
        ],
        help="Problem sizes in QxL format.",
    )
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--telemetry-interval", type=float, default=0.2)
    parser.add_argument(
        "--telemetry-target-ms",
        type=float,
        default=800.0,
        help="Try to keep the memory-profiling region at least this long by repeating the heavy call.",
    )
    parser.add_argument(
        "--standalone-modes",
        nargs="+",
        choices=("save_param_states", "checkpoint", "auto", "bruteforce_parallel_q6"),
        default=["save_param_states", "checkpoint"],
    )
    parser.add_argument(
        "--worker-mode",
        choices=(
            "pennylane",
            "save_param_states",
            "checkpoint",
            "auto",
            "bruteforce_parallel_q6",
        ),
        default=None,
    )
    parser.add_argument("--worker-case", type=parse_case, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument(
        "--disable-gate-fusion",
        action="store_true",
        help="Disable gate fusion optimizations for A/B comparisons.",
    )
    parser.add_argument(
        "--skip-pennylane",
        action="store_true",
        help="Skip PennyLane baseline and only run standalone modes.",
    )
    return parser.parse_args()


def _worker_run(args: argparse.Namespace) -> ModeResult:
    if args.worker_case is None or args.worker_mode is None:
        raise ValueError("worker-mode and worker-case must be provided in worker mode.")

    case = args.worker_case
    params_np = make_initial_params(
        num_qubits=case.num_qubits,
        layers=case.layers,
        seed=args.seed,
        init_scale=args.init_scale,
    )
    params_np = np.asarray(params_np, dtype=np.float64)
    ref = None
    if not args.skip_pennylane or args.worker_mode == "pennylane":
        _, ref = _build_reference(case, args.field, args.seed, args.init_scale)
    if args.worker_mode == "pennylane":
        return _benchmark_pennylane(
            case,
            params_np,
            ref,
            args.field,
            args.repeats,
            args.warmup,
            args.telemetry_interval,
            args.telemetry_target_ms,
            not args.disable_gate_fusion,
        )
    return _benchmark_standalone(
        args.worker_mode,
        case,
        params_np,
        ref,
        args.field,
        args.repeats,
        args.warmup,
        args.telemetry_interval,
        args.telemetry_target_ms,
        not args.disable_gate_fusion,
    )


def _run_worker_subprocess(
    mode: str,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> ModeResult:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-case",
        f"{case.num_qubits}x{case.layers}",
        "--field",
        str(args.field),
        "--seed",
        str(args.seed),
        "--init-scale",
        str(args.init_scale),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--telemetry-interval",
        str(args.telemetry_interval),
        "--telemetry-target-ms",
        str(args.telemetry_target_ms),
    ]
    if args.skip_pennylane:
        command.append("--skip-pennylane")
    if args.disable_gate_fusion:
        command.append("--disable-gate-fusion")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return ModeResult(**json.loads(completed.stdout))


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _print_case(case: BenchmarkCase, results: list[ModeResult]) -> None:
    print(f"{case.num_qubits} qubits x {case.layers} layers")
    header = (
        "  mode         ok   fwd_ms   back_ms  gradient_ms   total_ms  "
        "energy_diff   grad_max_diff "
        "workspace_GiB  cpu_ref_ms  gpu_scan_ms  seq_sv_ms  peak_fb_MiB "
        "peak_proc_MiB  samples  fusion  note"
    )
    print(header)
    for result in results:
        print(
            f"  {result.mode:<12} "
            f"{('yes' if result.ok else 'no'):<4} "
            f"{_fmt_float(result.forward_ms):>8} "
            f"{_fmt_float(result.back_ms):>8} "
            f"{_fmt_float(result.gradient_ms):>12} "
            f"{_fmt_float(result.total_ms):>10} "
            f"{_fmt_float(result.energy_abs_diff, 3):>12} "
            f"{_fmt_float(result.grad_max_abs_diff, 3):>15} "
            f"{_fmt_float(result.workspace_gib, 2):>13} "
            f"{_fmt_float(result.cpu_reference_ms):>11} "
            f"{_fmt_float(result.gpu_scan_ms):>12} "
            f"{_fmt_float(result.sequential_statevector_ms):>10} "
            f"{_fmt_int(result.peak_fb_memory_mib):>12} "
            f"{_fmt_int(result.peak_process_memory_mib):>14} "
            f"{result.telemetry_samples:>7}  "
            f"{('on' if result.gate_fusion_enabled else 'off'):>6}  "
            f"{result.note or '-'}"
        )
    print()


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "num_qubits",
        "layers",
        "mode",
        "ok",
        "note",
        "energy_abs_diff",
        "grad_max_abs_diff",
        "grad_l2_diff",
        "fwd_ms",
        "back_ms",
        "gradient_ms",
        "total_ms",
        "workspace_gib",
        "cpu_reference_ms",
        "gpu_scan_ms",
        "sequential_statevector_ms",
        "peak_fb_memory_mib",
        "peak_process_memory_mib",
        "telemetry_samples",
        "gate_fusion_enabled",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| qubits | layers | mode | ok | fwd_ms | back_ms | gradient_ms | total_ms | energy_diff | grad_max_diff | workspace_gib | cpu_ref_ms | gpu_scan_ms | seq_sv_ms | peak_fb_mib | peak_proc_mib | samples | fusion | note |",
        "|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["num_qubits"]),
                    str(row["layers"]),
                    str(row["mode"]),
                    "yes" if row["ok"] else "no",
                    _fmt_float(row["fwd_ms"]),
                    _fmt_float(row["back_ms"]),
                    _fmt_float(row["gradient_ms"]),
                    _fmt_float(row["total_ms"]),
                    _fmt_float(row["energy_abs_diff"], 3),
                    _fmt_float(row["grad_max_abs_diff"], 3),
                    _fmt_float(row["workspace_gib"], 2),
                    _fmt_float(row["cpu_reference_ms"]),
                    _fmt_float(row["gpu_scan_ms"]),
                    _fmt_float(row["sequential_statevector_ms"]),
                    _fmt_int(row["peak_fb_memory_mib"]),
                    _fmt_int(row["peak_process_memory_mib"]),
                    str(row["telemetry_samples"]),
                    "on" if row["gate_fusion_enabled"] else "off",
                    str(row["note"] or "-"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.worker_mode is not None:
        result = _worker_run(args)
        print(json.dumps(asdict(result)))
        return

    standalone_modes_text = ", ".join(args.standalone_modes)
    print(f"PennyLane vs standalone modes: {standalone_modes_text}")
    print(f"  Field strength: {args.field}")
    print(f"  Repeats per timing metric: {args.repeats}")
    print(f"  Warmup calls per timing metric: {args.warmup}")
    print(f"  Telemetry interval: {args.telemetry_interval:.2f} s")
    print(f"  Telemetry target window: {args.telemetry_target_ms:.0f} ms")
    print(f"  Gate fusion enabled: {not args.disable_gate_fusion}")
    print()

    rows: list[dict[str, object]] = []
    for case in args.cases:
        results: list[ModeResult] = []
        if not args.skip_pennylane:
            results.append(_run_worker_subprocess("pennylane", case, args))
        results.extend(
            _run_worker_subprocess(mode, case, args) for mode in args.standalone_modes
        )
        rows.extend(_flatten_row(case, result) for result in results)
        _print_case(case, results)

    if args.csv_out is not None:
        _write_csv(rows, args.csv_out)
        print(f"CSV written to {args.csv_out}")
    if args.md_out is not None:
        _write_markdown(rows, args.md_out)
        print(f"Markdown written to {args.md_out}")


if __name__ == "__main__":
    main()
