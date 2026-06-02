"""Compare save_param_states vs dense_scan with timing breakdown and GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising.runtime import (
    GpuTelemetrySummary,
    capture_telemetry_window,
    median_runtime_ms,
)
from standalone_backend import RingIsingAdjointBackend, RingIsingConfig, make_initial_params


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


@dataclass(frozen=True)
class ModeResult:
    mode: str
    ok: bool
    note: str | None
    energy_abs_diff: float | None
    grad_max_abs_diff: float | None
    grad_l2_diff: float | None
    forward_ms: float | None
    total_ms: float | None
    backward_estimate_ms: float | None
    dense_call_ms: float | None
    workspace_gib: float | None
    cpu_reference_ms: float | None
    gpu_scan_ms: float | None
    sequential_statevector_ms: float | None
    peak_fb_memory_mib: int | None
    peak_process_memory_mib: int | None
    avg_gpu_util_pct: float | None
    peak_gpu_util_pct: float | None
    avg_mem_util_pct: float | None
    peak_mem_util_pct: float | None
    avg_sm_activity_pct: float | None
    peak_sm_activity_pct: float | None
    avg_sm_occupancy_pct: float | None
    peak_sm_occupancy_pct: float | None
    telemetry_samples: int
    gate_fusion_enabled: bool = True


@dataclass(frozen=True)
class TelemetryRollup:
    peak_fb_memory_mib: int | None
    peak_process_memory_mib: int | None
    avg_gpu_util_pct: float | None
    peak_gpu_util_pct: float | None
    avg_mem_util_pct: float | None
    peak_mem_util_pct: float | None
    avg_sm_activity_pct: float | None
    peak_sm_activity_pct: float | None
    avg_sm_occupancy_pct: float | None
    peak_sm_occupancy_pct: float | None


def parse_case(text: str) -> BenchmarkCase:
    try:
        qubits_text, layers_text = text.lower().split("x", maxsplit=1)
        return BenchmarkCase(num_qubits=int(qubits_text), layers=int(layers_text))
    except Exception as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid case {text!r}. Expected format like 5x8."
        ) from exc


def _max_or_none(values: list[float | int | None]) -> float | int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _mean_or_none(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _telemetry_rollup(summary: GpuTelemetrySummary) -> TelemetryRollup:
    gpus = summary.gpus
    return TelemetryRollup(
        peak_fb_memory_mib=_max_or_none([gpu.peak_fb_memory_used_mib for gpu in gpus]),
        peak_process_memory_mib=_max_or_none(
            [gpu.peak_process_used_gpu_memory_mib for gpu in gpus]
        ),
        avg_gpu_util_pct=_mean_or_none([gpu.avg_utilization_gpu_pct for gpu in gpus]),
        peak_gpu_util_pct=_max_or_none([gpu.peak_utilization_gpu_pct for gpu in gpus]),
        avg_mem_util_pct=_mean_or_none([gpu.avg_utilization_memory_pct for gpu in gpus]),
        peak_mem_util_pct=_max_or_none([gpu.peak_utilization_memory_pct for gpu in gpus]),
        avg_sm_activity_pct=_mean_or_none([gpu.avg_sm_activity_pct for gpu in gpus]),
        peak_sm_activity_pct=_max_or_none([gpu.peak_sm_activity_pct for gpu in gpus]),
        avg_sm_occupancy_pct=_mean_or_none([gpu.avg_sm_occupancy_pct for gpu in gpus]),
        peak_sm_occupancy_pct=_max_or_none([gpu.peak_sm_occupancy_pct for gpu in gpus]),
    )


def _run_single_mode(
    mode: str,
    case: BenchmarkCase,
    params_np: np.ndarray,
    field: float,
    repeats: int,
    warmup: int,
    telemetry_interval_s: float,
    telemetry_target_ms: float,
    gate_fusion_enabled: bool,
    collect_dense_diagnostics: bool,
    dense_diag_repeats: int,
    dense_diag_warmup: int,
) -> ModeResult:
    config = RingIsingConfig(
        num_qubits=case.num_qubits,
        layers=case.layers,
        field=field,
        gradient_strategy=mode,
        gate_fusion=gate_fusion_enabled,
    )
    backend = RingIsingAdjointBackend(config)
    resolution = backend.strategy_resolution

    dense_diag: dict[str, np.ndarray | float] | None = None
    dense_call_ms: float | None = None
    if collect_dense_diagnostics and mode == "dense_scan":
        for _ in range(dense_diag_warmup):
            backend.dense_scan_experiment(params_np)

        dense_call_samples: list[float] = []
        cpu_ref_samples: list[float] = []
        gpu_scan_samples: list[float] = []
        seq_sv_samples: list[float] = []
        for _ in range(dense_diag_repeats):
            start = time.perf_counter()
            diag = backend.dense_scan_experiment(params_np)
            dense_call_samples.append((time.perf_counter() - start) * 1000.0)
            cpu_ref_samples.append(float(diag["cpu_reference_ms"]))
            gpu_scan_samples.append(float(diag["gpu_scan_ms"]))
            seq_sv_samples.append(float(diag["sequential_statevector_ms"]))

        dense_call_ms = statistics.median(dense_call_samples)
        dense_diag = {
            "cpu_reference_ms": statistics.median(cpu_ref_samples),
            "gpu_scan_ms": statistics.median(gpu_scan_samples),
            "sequential_statevector_ms": statistics.median(seq_sv_samples),
        }

    # Use save_param_states result as correctness reference for both modes.
    reference_backend = RingIsingAdjointBackend(
        RingIsingConfig(
            num_qubits=case.num_qubits,
            layers=case.layers,
            field=field,
            gradient_strategy="save_param_states",
            gate_fusion=gate_fusion_enabled,
        )
    )
    ref_energy, ref_grad = reference_backend.energy_and_grad(params_np)

    energy, grad = backend.energy_and_grad(params_np)
    grad_diff = np.asarray(ref_grad, dtype=np.float64) - np.asarray(grad, dtype=np.float64)

    forward_ms = median_runtime_ms(lambda: backend.energy(params_np), repeats, warmup)
    total_ms = median_runtime_ms(lambda: backend.energy_and_grad(params_np), repeats, warmup)
    backward_estimate_ms = max(0.0, total_ms - forward_ms)

    telemetry_summary = capture_telemetry_window(
        lambda: backend.energy_and_grad(params_np),
        target_ms=telemetry_target_ms,
        measured_ms=total_ms,
        interval_s=telemetry_interval_s,
    )
    telemetry = _telemetry_rollup(telemetry_summary)

    return ModeResult(
        mode=mode,
        ok=True,
        note=(
            None
            if resolution.requested_strategy == resolution.resolved_strategy
            else f"resolved={resolution.resolved_strategy}"
        ),
        energy_abs_diff=abs(float(ref_energy) - float(energy)),
        grad_max_abs_diff=float(np.max(np.abs(grad_diff))),
        grad_l2_diff=float(np.linalg.norm(grad_diff)),
        forward_ms=forward_ms,
        total_ms=total_ms,
        backward_estimate_ms=backward_estimate_ms,
        dense_call_ms=dense_call_ms,
        workspace_gib=resolution.estimated_workspace_gib,
        cpu_reference_ms=(
            float(dense_diag["cpu_reference_ms"]) if dense_diag is not None else None
        ),
        gpu_scan_ms=(float(dense_diag["gpu_scan_ms"]) if dense_diag is not None else None),
        sequential_statevector_ms=(
            float(dense_diag["sequential_statevector_ms"])
            if dense_diag is not None
            else None
        ),
        peak_fb_memory_mib=(
            int(telemetry.peak_fb_memory_mib)
            if telemetry.peak_fb_memory_mib is not None
            else None
        ),
        peak_process_memory_mib=(
            int(telemetry.peak_process_memory_mib)
            if telemetry.peak_process_memory_mib is not None
            else None
        ),
        avg_gpu_util_pct=telemetry.avg_gpu_util_pct,
        peak_gpu_util_pct=(
            float(telemetry.peak_gpu_util_pct)
            if telemetry.peak_gpu_util_pct is not None
            else None
        ),
        avg_mem_util_pct=telemetry.avg_mem_util_pct,
        peak_mem_util_pct=(
            float(telemetry.peak_mem_util_pct)
            if telemetry.peak_mem_util_pct is not None
            else None
        ),
        avg_sm_activity_pct=telemetry.avg_sm_activity_pct,
        peak_sm_activity_pct=(
            float(telemetry.peak_sm_activity_pct)
            if telemetry.peak_sm_activity_pct is not None
            else None
        ),
        avg_sm_occupancy_pct=telemetry.avg_sm_occupancy_pct,
        peak_sm_occupancy_pct=(
            float(telemetry.peak_sm_occupancy_pct)
            if telemetry.peak_sm_occupancy_pct is not None
            else None
        ),
        telemetry_samples=telemetry_summary.snapshots_collected,
        gate_fusion_enabled=gate_fusion_enabled,
    )


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
        "forward_ms": result.forward_ms,
        "total_ms": result.total_ms,
        "backward_estimate_ms": result.backward_estimate_ms,
        "dense_call_ms": result.dense_call_ms,
        "workspace_gib": result.workspace_gib,
        "cpu_reference_ms": result.cpu_reference_ms,
        "gpu_scan_ms": result.gpu_scan_ms,
        "sequential_statevector_ms": result.sequential_statevector_ms,
        "peak_fb_memory_mib": result.peak_fb_memory_mib,
        "peak_process_memory_mib": result.peak_process_memory_mib,
        "avg_gpu_util_pct": result.avg_gpu_util_pct,
        "peak_gpu_util_pct": result.peak_gpu_util_pct,
        "avg_mem_util_pct": result.avg_mem_util_pct,
        "peak_mem_util_pct": result.peak_mem_util_pct,
        "avg_sm_activity_pct": result.avg_sm_activity_pct,
        "peak_sm_activity_pct": result.peak_sm_activity_pct,
        "avg_sm_occupancy_pct": result.avg_sm_occupancy_pct,
        "peak_sm_occupancy_pct": result.peak_sm_occupancy_pct,
        "telemetry_samples": result.telemetry_samples,
        "gate_fusion_enabled": result.gate_fusion_enabled,
    }


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _render_ascii_table(
    headers: list[str],
    rows: list[list[str]],
    aligns: list[str] | None = None,
) -> str:
    if aligns is None:
        aligns = ["left"] * len(headers)
    if len(aligns) != len(headers):
        raise ValueError("aligns length must match headers length")

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _line(char: str = "-") -> str:
        return "+" + "+".join(char * (width + 2) for width in widths) + "+"

    def _row(cells: list[str]) -> str:
        padded: list[str] = []
        for index, cell in enumerate(cells):
            if aligns[index] == "right":
                padded.append(cell.rjust(widths[index]))
            else:
                padded.append(cell.ljust(widths[index]))
        return "| " + " | ".join(padded) + " |"

    lines = [_line("-"), _row(headers), _line("=")]
    for row in rows:
        lines.append(_row(row))
    lines.append(_line("-"))
    return "\n".join(lines)


def _print_case(case: BenchmarkCase, results: list[ModeResult]) -> None:
    print(f"{case.num_qubits} qubits x {case.layers} layers")
    save_result = next((result for result in results if result.mode == "save_param_states"), None)
    save_total_ms = save_result.total_ms if save_result is not None else None

    timing_headers = [
        "mode",
        "ok",
        "fwd_ms",
        "total_ms",
        "speedup_vs_save",
        "bwd_est_ms",
        "dense_call_ms",
        "cpu_ref_ms",
        "gpu_scan_ms",
        "seq_sv_ms",
        "energy_diff",
        "grad_max_diff",
        "workspace_gib",
    ]
    timing_rows: list[list[str]] = []
    for result in results:
        speedup_text = "-"
        if (
            save_total_ms is not None
            and result.total_ms is not None
            and result.total_ms > 0.0
        ):
            speedup_text = f"{save_total_ms / result.total_ms:.3f}x"
        timing_rows.append(
            [
                result.mode,
                "yes" if result.ok else "no",
                _fmt_float(result.forward_ms),
                _fmt_float(result.total_ms),
                speedup_text,
                _fmt_float(result.backward_estimate_ms),
                _fmt_float(result.dense_call_ms),
                _fmt_float(result.cpu_reference_ms),
                _fmt_float(result.gpu_scan_ms),
                _fmt_float(result.sequential_statevector_ms),
                _fmt_float(result.energy_abs_diff, 3),
                _fmt_float(result.grad_max_abs_diff, 3),
                _fmt_float(result.workspace_gib, 3),
            ]
        )
    timing_aligns = [
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
    ]
    print(_render_ascii_table(timing_headers, timing_rows, timing_aligns))

    resource_headers = [
        "mode",
        "peak_fb_mib",
        "peak_proc_mib",
        "avg_gpu_%",
        "peak_gpu_%",
        "avg_mem_%",
        "peak_mem_%",
        "avg_sm_act_%",
        "peak_sm_act_%",
        "avg_sm_occ_%",
        "peak_sm_occ_%",
        "samples",
        "note",
    ]
    resource_rows = [
        [
            result.mode,
            _fmt_int(result.peak_fb_memory_mib),
            _fmt_int(result.peak_process_memory_mib),
            _fmt_float(result.avg_gpu_util_pct, 1),
            _fmt_float(result.peak_gpu_util_pct, 1),
            _fmt_float(result.avg_mem_util_pct, 1),
            _fmt_float(result.peak_mem_util_pct, 1),
            _fmt_float(result.avg_sm_activity_pct, 1),
            _fmt_float(result.peak_sm_activity_pct, 1),
            _fmt_float(result.avg_sm_occupancy_pct, 1),
            _fmt_float(result.peak_sm_occupancy_pct, 1),
            str(result.telemetry_samples),
            result.note or "-",
        ]
        for result in results
    ]
    resource_aligns = [
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
    ]
    print(_render_ascii_table(resource_headers, resource_rows, resource_aligns))
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
        "forward_ms",
        "total_ms",
        "backward_estimate_ms",
        "dense_call_ms",
        "workspace_gib",
        "cpu_reference_ms",
        "gpu_scan_ms",
        "sequential_statevector_ms",
        "peak_fb_memory_mib",
        "peak_process_memory_mib",
        "avg_gpu_util_pct",
        "peak_gpu_util_pct",
        "avg_mem_util_pct",
        "peak_mem_util_pct",
        "avg_sm_activity_pct",
        "peak_sm_activity_pct",
        "avg_sm_occupancy_pct",
        "peak_sm_occupancy_pct",
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
        "| qubits | layers | mode | ok | fwd_ms | total_ms | bwd_est_ms | dense_call_ms | cpu_ref_ms | gpu_scan_ms | seq_sv_ms | energy_diff | grad_max_diff | peak_fb_mib | peak_proc_mib | avg_gpu_util_pct | peak_gpu_util_pct | avg_sm_act_pct | peak_sm_occ_pct | note |",
        "|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
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
                    _fmt_float(row["forward_ms"]),
                    _fmt_float(row["total_ms"]),
                    _fmt_float(row["backward_estimate_ms"]),
                    _fmt_float(row["dense_call_ms"]),
                    _fmt_float(row["cpu_reference_ms"]),
                    _fmt_float(row["gpu_scan_ms"]),
                    _fmt_float(row["sequential_statevector_ms"]),
                    _fmt_float(row["energy_abs_diff"]),
                    _fmt_float(row["grad_max_abs_diff"]),
                    _fmt_int(row["peak_fb_memory_mib"]),
                    _fmt_int(row["peak_process_memory_mib"]),
                    _fmt_float(row["avg_gpu_util_pct"], 1),
                    _fmt_float(row["peak_gpu_util_pct"], 1),
                    _fmt_float(row["avg_sm_activity_pct"], 1),
                    _fmt_float(row["peak_sm_occupancy_pct"], 1),
                    str(row["note"] or "-"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[parse_case("5x4"), parse_case("5x8"), parse_case("6x8")],
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
        help="Try to keep telemetry window at least this long by repeating energy_and_grad.",
    )
    parser.add_argument(
        "--worker-mode",
        choices=("save_param_states", "dense_scan"),
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
        "--skip-dense-diagnostics",
        action="store_true",
        help="Skip dense_scan_experiment breakdown fields for dense_scan mode.",
    )
    parser.add_argument(
        "--dense-diag-repeats",
        type=int,
        default=1,
        help="Repeats used for dense_scan_experiment timing breakdown (dense_scan mode only).",
    )
    parser.add_argument(
        "--dense-diag-warmup",
        type=int,
        default=1,
        help="Warmup calls before dense_scan_experiment timing breakdown.",
    )
    return parser.parse_args()


def _worker_run(args: argparse.Namespace) -> ModeResult:
    if args.worker_case is None or args.worker_mode is None:
        raise ValueError("worker-mode and worker-case must be provided in worker mode.")

    params_np = make_initial_params(
        num_qubits=args.worker_case.num_qubits,
        layers=args.worker_case.layers,
        seed=args.seed,
        init_scale=args.init_scale,
    )
    params_np = np.asarray(params_np, dtype=np.float64)

    try:
        return _run_single_mode(
            mode=args.worker_mode,
            case=args.worker_case,
            params_np=params_np,
            field=args.field,
            repeats=args.repeats,
            warmup=args.warmup,
            telemetry_interval_s=args.telemetry_interval,
            telemetry_target_ms=args.telemetry_target_ms,
            gate_fusion_enabled=not args.disable_gate_fusion,
            collect_dense_diagnostics=not args.skip_dense_diagnostics,
            dense_diag_repeats=max(1, args.dense_diag_repeats),
            dense_diag_warmup=max(0, args.dense_diag_warmup),
        )
    except Exception as exc:  # pragma: no cover - surfaced in output
        return ModeResult(
            mode=args.worker_mode,
            ok=False,
            note=f"{type(exc).__name__}: {exc}",
            energy_abs_diff=None,
            grad_max_abs_diff=None,
            grad_l2_diff=None,
            forward_ms=None,
            total_ms=None,
            backward_estimate_ms=None,
            dense_call_ms=None,
            workspace_gib=None,
            cpu_reference_ms=None,
            gpu_scan_ms=None,
            sequential_statevector_ms=None,
            peak_fb_memory_mib=None,
            peak_process_memory_mib=None,
            avg_gpu_util_pct=None,
            peak_gpu_util_pct=None,
            avg_mem_util_pct=None,
            peak_mem_util_pct=None,
            avg_sm_activity_pct=None,
            peak_sm_activity_pct=None,
            avg_sm_occupancy_pct=None,
            peak_sm_occupancy_pct=None,
            telemetry_samples=0,
            gate_fusion_enabled=not args.disable_gate_fusion,
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
        "--dense-diag-repeats",
        str(args.dense_diag_repeats),
        "--dense-diag-warmup",
        str(args.dense_diag_warmup),
    ]
    if args.disable_gate_fusion:
        command.append("--disable-gate-fusion")
    if args.skip_dense_diagnostics:
        command.append("--skip-dense-diagnostics")

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return ModeResult(**json.loads(completed.stdout))


def main() -> None:
    args = parse_args()
    if args.worker_mode is not None:
        result = _worker_run(args)
        print(json.dumps(asdict(result)))
        return

    print("Standalone modes: save_param_states vs dense_scan")
    print(f"  Field strength: {args.field}")
    print(f"  Repeats per timing metric: {args.repeats}")
    print(f"  Warmup calls per timing metric: {args.warmup}")
    print(f"  Telemetry interval: {args.telemetry_interval:.2f} s")
    print(f"  Telemetry target window: {args.telemetry_target_ms:.0f} ms")
    print(f"  Gate fusion enabled: {not args.disable_gate_fusion}")
    print(f"  Dense diagnostics enabled: {not args.skip_dense_diagnostics}")
    print()

    rows: list[dict[str, object]] = []
    for case in args.cases:
        results = [
            _run_worker_subprocess("save_param_states", case, args),
            _run_worker_subprocess("dense_scan", case, args),
        ]
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
