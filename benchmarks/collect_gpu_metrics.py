"""Collect comprehensive GPU metrics for every benchmark method and circuit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ring_ising import RunConfig, run_standalone
from ring_ising.config import STANDALONE_GRADIENT_STRATEGIES
from ring_ising.runtime import GpuTelemetryMonitor
from ring_ising.runtime._nvidia_smi import (
    max_or_none,
    mean_or_none,
    parse_float_or_none,
    parse_nvidia_csv,
)


ALL_METHODS = STANDALONE_GRADIENT_STRATEGIES
DEFAULT_METHODS = (
    "inverse_walk",
    "reverse_walk",
    "save_param_states",
    "checkpoint",
    "dense_scan",
    "block_fused_adjoint",
    "intrablock_parallel",
)
DEFAULT_CASES_TEXT = (
    "4x8",
    "4x32",
    "4x128",
    "4x512",
    "4x2048",
    "5x8",
    "5x32",
    "5x128",
    "5x512",
    "5x2048",
    "6x8",
    "6x32",
    "6x128",
    "6x512",
    "6x2048",
)
PCIE_DMON_FIELDS = ["gpu", "rxpci", "txpci"]
NCU_METRICS = (
    "launch__grid_size",
    "launch__block_size",
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
)


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


@dataclass(frozen=True)
class PcieSample:
    index: str
    rx_kbps: float | None
    tx_kbps: float | None


@dataclass(frozen=True)
class PcieSummary:
    index: str
    sample_count: int
    avg_pcie_rx_kbps: float | None
    peak_pcie_rx_kbps: float | None
    avg_pcie_tx_kbps: float | None
    peak_pcie_tx_kbps: float | None


class PcieDmonMonitor:
    """Read PCIe RX/TX throughput from nvidia-smi dmon while a run is active."""

    def __init__(self, *, interval_s: float) -> None:
        self.interval_s = max(1.0, interval_s)
        self.note: str | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._samples: list[PcieSample] = []

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.note = "nvidia-smi not found on PATH"
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[PcieSummary, ...]:
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summaries()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def summaries(self) -> tuple[PcieSummary, ...]:
        with self._lock:
            samples = tuple(self._samples)
        by_gpu: dict[str, list[PcieSample]] = {}
        for sample in samples:
            by_gpu.setdefault(sample.index, []).append(sample)
        return tuple(
            PcieSummary(
                index=index,
                sample_count=len(gpu_samples),
                avg_pcie_rx_kbps=mean_or_none([sample.rx_kbps for sample in gpu_samples]),
                peak_pcie_rx_kbps=max_or_none([sample.rx_kbps for sample in gpu_samples]),
                avg_pcie_tx_kbps=mean_or_none([sample.tx_kbps for sample in gpu_samples]),
                peak_pcie_tx_kbps=max_or_none([sample.tx_kbps for sample in gpu_samples]),
            )
            for index, gpu_samples in sorted(by_gpu.items())
        )

    def _run(self) -> None:
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "dmon",
                    "-s",
                    "t",
                    "-d",
                    str(int(round(self.interval_s))),
                    "--format",
                    "csv,nounit,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.note = str(exc)
            return

        if self._process.stdout is None:
            self.note = "Unable to read nvidia-smi PCIe telemetry output."
            return

        for line in self._process.stdout:
            if self._stop_event.is_set():
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = next(iter(parse_nvidia_csv(stripped, PCIE_DMON_FIELDS)), None)
            if row is None:
                continue
            sample = PcieSample(
                index=row["gpu"].strip(),
                rx_kbps=parse_float_or_none(row.get("rxpci", "")),
                tx_kbps=parse_float_or_none(row.get("txpci", "")),
            )
            with self._lock:
                self._samples.append(sample)


def parse_case(text: str) -> BenchmarkCase:
    try:
        qubits_text, layers_text = text.lower().split("x", maxsplit=1)
        return BenchmarkCase(num_qubits=int(qubits_text), layers=int(layers_text))
    except Exception as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid case {text!r}. Expected format like 6x128."
        ) from exc


def parse_method(text: str) -> str:
    method = text.strip()
    if method not in ALL_METHODS:
        choices = ", ".join(ALL_METHODS)
        raise argparse.ArgumentTypeError(
            f"Invalid method {text!r}. Expected one of: {choices}"
        )
    return method


def default_steps_for(case: BenchmarkCase) -> int:
    base_steps_by_layers = {8: 120, 32: 80, 128: 40, 512: 12, 1024: 6, 2048: 4}
    base = base_steps_by_layers.get(case.layers, max(2, 960 // max(case.layers, 1)))
    return max(2, int(round(base * 4.0 / case.num_qubits)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[parse_case(text) for text in DEFAULT_CASES_TEXT],
        help="Circuit sizes in QxL format.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_method,
        default=list(DEFAULT_METHODS),
        help="Methods to run and profile.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override optimization steps for every case.",
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
        help="Ops per checkpoint block for checkpoint/block_fused_adjoint methods.",
    )
    parser.add_argument(
        "--intrablock-block-size",
        type=int,
        default=64,
        help="Ops per block for intrablock_parallel.",
    )
    parser.add_argument(
        "--disable-gate-fusion",
        dest="gate_fusion",
        action="store_false",
        default=True,
        help="Disable gate fusion in standalone runs.",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=0.5,
        help="Sampling interval for one-shot nvidia-smi metrics.",
    )
    parser.add_argument(
        "--pcie-interval",
        type=float,
        default=1.0,
        help="Sampling interval for nvidia-smi dmon PCIe throughput.",
    )
    parser.add_argument(
        "--telemetry-live",
        action="store_true",
        help="Print live GPU telemetry samples during each run.",
    )
    parser.add_argument(
        "--min-telemetry-samples",
        type=int,
        default=1,
        help="Repeat a method/circuit until at least this many GPU samples are collected.",
    )
    parser.add_argument(
        "--max-repeat-runs",
        type=int,
        default=1,
        help="Maximum repeats per method/circuit while collecting telemetry.",
    )
    parser.add_argument(
        "--report-steps",
        action="store_true",
        help="Write per-step optimization metrics.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write failed run rows and continue with remaining methods/circuits.",
    )
    parser.add_argument(
        "--kernel-profile-ncu",
        action="store_true",
        help=(
            "Run NVIDIA Nsight Compute once per method/circuit and write "
            "kernel-level metrics that can explain blank dmon/GPM columns."
        ),
    )
    parser.add_argument(
        "--ncu-path",
        default="ncu",
        help="Path to the Nsight Compute CLI executable.",
    )
    parser.add_argument(
        "--ncu-launch-count",
        type=int,
        default=64,
        help="Maximum CUDA kernel launches profiled per Nsight Compute run.",
    )
    parser.add_argument(
        "--ncu-metrics",
        nargs="+",
        default=list(NCU_METRICS),
        help="Nsight Compute metrics to collect.",
    )
    parser.add_argument(
        "--allow-incomplete-kernel-profile",
        action="store_true",
        help="Write diagnostic Nsight Compute rows instead of failing on profiler errors.",
    )
    parser.add_argument(
        "--internal-ncu-run-case",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-num-qubits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to a timestamped name.",
    )
    return parser.parse_args()


def build_run_config(
    method: str,
    *,
    case: BenchmarkCase,
    steps: int,
    seed: int,
    args: argparse.Namespace,
) -> RunConfig:
    return RunConfig(
        backend="standalone",
        num_qubits=case.num_qubits,
        layers=case.layers,
        field=args.field,
        steps=steps,
        stepsize=args.stepsize,
        seed=seed,
        init_scale=args.init_scale,
        verbose=False,
        show_progress=False,
        report_steps=args.report_steps,
        gpu_telemetry=False,
        telemetry_interval_s=args.telemetry_interval,
        telemetry_live=False,
        gradient_strategy=method,
        checkpoint_interval_ops=args.checkpoint_interval_ops,
        intrablock_block_size=args.intrablock_block_size,
        gate_fusion=args.gate_fusion,
    )


def run_method(method: str, config: RunConfig):
    _ = method
    return run_standalone(config)


def run_internal_ncu_case(args: argparse.Namespace) -> None:
    if args.internal_method is None:
        raise SystemExit("--internal-method is required.")
    if args.internal_num_qubits is None:
        raise SystemExit("--internal-num-qubits is required.")
    if args.internal_layers is None:
        raise SystemExit("--internal-layers is required.")
    if args.internal_steps is None:
        raise SystemExit("--internal-steps is required.")

    case = BenchmarkCase(
        num_qubits=args.internal_num_qubits,
        layers=args.internal_layers,
    )
    config = build_run_config(
        args.internal_method,
        case=case,
        steps=args.internal_steps,
        seed=args.seed,
        args=args,
    )
    run_method(args.internal_method, config)


def value_or_empty(value: Any) -> Any:
    return "" if value is None else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value_or_empty(row.get(key)) for key in fieldnames})


def ensure_ncu_available(ncu_path: str) -> str:
    resolved = shutil.which(ncu_path) if Path(ncu_path).name == ncu_path else ncu_path
    if resolved is None:
        raise RuntimeError(
            "Nsight Compute CLI was not found. Install NVIDIA Nsight Compute "
            "or pass --ncu-path."
        )
    return resolved


def excerpt_text(text: str, limit: int = 500) -> str | None:
    stripped = " ".join(text.split())
    if not stripped:
        return None
    return stripped[:limit]


def parse_ncu_raw_csv(output: str) -> list[dict[str, str]]:
    lines = [line for line in output.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        try:
            parsed = next(csv.reader([line]))
        except csv.Error:
            continue
        if (
            ("Metric Name" in parsed and "Metric Value" in parsed)
            or ("ID" in parsed and "Kernel Name" in parsed)
        ):
            header_index = index
            break
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def normalize_ncu_metric_value(value: str) -> Any:
    normalized = value.strip().replace(",", "")
    if normalized in {"", "N/A", "nan", "NaN", "-"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return value.strip()


def ncu_diagnostic_row(
    *,
    run_id: str,
    method: str,
    case: BenchmarkCase,
    steps: int,
    status: str,
    message: str,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "method": method,
        "num_qubits": case.num_qubits,
        "layers": case.layers,
        "steps": steps,
        "launch_index": None,
        "kernel_name": None,
        "ncu_status": status,
        "ncu_message": message,
        "ncu_stdout_excerpt": excerpt_text(stdout),
        "ncu_stderr_excerpt": excerpt_text(stderr),
    }


def build_ncu_kernel_rows(
    *,
    run_id: str,
    method: str,
    case: BenchmarkCase,
    steps: int,
    metric_rows: list[dict[str, str]],
    metrics: list[str],
) -> list[dict[str, Any]]:
    if metric_rows and "Metric Name" not in metric_rows[0]:
        return build_ncu_wide_kernel_rows(
            run_id=run_id,
            method=method,
            case=case,
            steps=steps,
            profile_rows=metric_rows,
            metrics=metrics,
        )

    by_launch: dict[tuple[str, str], dict[str, Any]] = {}
    for metric_row in metric_rows:
        metric_name = metric_row.get("Metric Name", "").strip()
        if metric_name not in metrics:
            continue
        launch_id = metric_row.get("ID", "").strip()
        kernel_name = metric_row.get("Kernel Name", "").strip()
        if not launch_id or not kernel_name:
            continue
        key = (launch_id, kernel_name)
        row = by_launch.setdefault(
            key,
            {
                "run_id": run_id,
                "method": method,
                "num_qubits": case.num_qubits,
                "layers": case.layers,
                "steps": steps,
                "launch_index": launch_id,
                "kernel_name": kernel_name,
                "ncu_status": "ok",
                "ncu_message": None,
                "ncu_stdout_excerpt": None,
                "ncu_stderr_excerpt": None,
            },
        )
        row[metric_name] = normalize_ncu_metric_value(metric_row.get("Metric Value", ""))

    rows = list(by_launch.values())
    for row in rows:
        missing = [metric for metric in metrics if row.get(metric) in (None, "")]
        row["missing_ncu_metrics"] = "/".join(missing) if missing else None
        for metric in missing:
            row[metric] = None
    return rows


def build_ncu_wide_kernel_rows(
    *,
    run_id: str,
    method: str,
    case: BenchmarkCase,
    steps: int,
    profile_rows: list[dict[str, str]],
    metrics: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_row in profile_rows:
        launch_id = profile_row.get("ID", "").strip()
        kernel_name = profile_row.get("Kernel Name", "").strip()
        if not launch_id or not kernel_name:
            continue
        row: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "num_qubits": case.num_qubits,
            "layers": case.layers,
            "steps": steps,
            "launch_index": launch_id,
            "kernel_name": kernel_name,
            "ncu_status": "ok",
            "ncu_message": None,
            "ncu_stdout_excerpt": None,
            "ncu_stderr_excerpt": None,
        }
        missing: list[str] = []
        for metric in metrics:
            if metric in profile_row:
                row[metric] = normalize_ncu_metric_value(profile_row.get(metric, ""))
            else:
                row[metric] = None
                missing.append(metric)
        row["missing_ncu_metrics"] = "/".join(missing) if missing else None
        rows.append(row)
    return rows


def run_ncu_profile_case(
    *,
    args: argparse.Namespace,
    run_id: str,
    method: str,
    case: BenchmarkCase,
    steps: int,
) -> list[dict[str, Any]]:
    ncu_path = ensure_ncu_available(args.ncu_path)
    command = [
        ncu_path,
        "--csv",
        "--page",
        "raw",
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--launch-count",
        str(args.ncu_launch_count),
        "--metrics",
        ",".join(args.ncu_metrics),
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-ncu-run-case",
        "--internal-method",
        method,
        "--internal-num-qubits",
        str(case.num_qubits),
        "--internal-layers",
        str(case.layers),
        "--internal-steps",
        str(steps),
        "--field",
        str(args.field),
        "--stepsize",
        str(args.stepsize),
        "--seed",
        str(args.seed),
        "--init-scale",
        str(args.init_scale),
        "--intrablock-block-size",
        str(args.intrablock_block_size),
    ]
    if args.checkpoint_interval_ops is not None:
        command.extend(["--checkpoint-interval", str(args.checkpoint_interval_ops)])
    if not args.gate_fusion:
        command.append("--disable-gate-fusion")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if "ERR_NVGPUCTRPERM" in completed.stdout or "ERR_NVGPUCTRPERM" in completed.stderr:
            row = ncu_diagnostic_row(
                run_id=run_id,
                method=method,
                case=case,
                steps=steps,
                status="permission_denied",
                message=(
                    "Nsight Compute cannot access NVIDIA GPU Performance Counters. "
                    "Enable profiling permissions or run with sufficient privileges."
                ),
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        else:
            row = ncu_diagnostic_row(
                run_id=run_id,
                method=method,
                case=case,
                steps=steps,
                status="failed",
                message=f"Nsight Compute exited with code {completed.returncode}.",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        if args.allow_incomplete_kernel_profile:
            return [row]
        raise RuntimeError(row["ncu_message"])

    rows = build_ncu_kernel_rows(
        run_id=run_id,
        method=method,
        case=case,
        steps=steps,
        metric_rows=parse_ncu_raw_csv(completed.stdout),
        metrics=args.ncu_metrics,
    )
    if not rows:
        row = ncu_diagnostic_row(
            run_id=run_id,
            method=method,
            case=case,
            steps=steps,
            status="no_metric_rows",
            message="Nsight Compute produced no matching metric rows.",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if args.allow_incomplete_kernel_profile:
            return [row]
        raise RuntimeError(row["ncu_message"])
    return rows


def numeric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (float, int)):
            values.append(float(value))
    return values


def summarize_ncu_rows(
    *,
    run_id: str,
    method: str,
    case: BenchmarkCase,
    steps: int,
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ncu_status") == "ok"]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "method": method,
        "num_qubits": case.num_qubits,
        "layers": case.layers,
        "steps": steps,
        "profiled_kernel_launches": len(ok_rows),
        "ncu_status": "ok" if ok_rows else (rows[0].get("ncu_status") if rows else "empty"),
        "ncu_message": None if ok_rows else (rows[0].get("ncu_message") if rows else None),
    }
    for metric in metrics:
        values = numeric_values(ok_rows, metric)
        summary[f"avg_{metric}"] = mean_or_none(values)
        summary[f"peak_{metric}"] = max_or_none(values)

    summary["ncu_avg_sm_activity_pct"] = summary.get(
        "avg_sm__throughput.avg.pct_of_peak_sustained_elapsed"
    )
    summary["ncu_peak_sm_activity_pct"] = summary.get(
        "peak_sm__throughput.avg.pct_of_peak_sustained_elapsed"
    )
    summary["ncu_avg_sm_occupancy_pct"] = summary.get(
        "avg_sm__warps_active.avg.pct_of_peak_sustained_active"
    )
    summary["ncu_peak_sm_occupancy_pct"] = summary.get(
        "peak_sm__warps_active.avg.pct_of_peak_sustained_active"
    )
    summary["ncu_avg_dram_activity_pct"] = summary.get(
        "avg_dram__throughput.avg.pct_of_peak_sustained_elapsed"
    )
    summary["ncu_peak_dram_activity_pct"] = summary.get(
        "peak_dram__throughput.avg.pct_of_peak_sustained_elapsed"
    )
    return summary


def pcie_by_index(summaries: tuple[PcieSummary, ...]) -> dict[str, PcieSummary]:
    return {summary.index: summary for summary in summaries}


def result_metadata_row(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", {}) or {}
    return {
        "backend_label": getattr(result, "backend_label", None),
        "gradient_strategy": metadata.get("gradient_strategy"),
        "device": metadata.get("device"),
        "checkpoint_interval_ops": metadata.get("checkpoint_interval_ops"),
        "intrablock_block_size": metadata.get("intrablock_block_size"),
        "estimated_workspace_gib": metadata.get("estimated_workspace_gib"),
        "effective_gate_fusion": metadata.get("gate_fusion"),
        "pennylane_gate_structure": metadata.get("pennylane_gate_structure"),
    }


def backend_timing_row(result: Any, steps: int, repeats: int) -> dict[str, Any]:
    metadata = getattr(result, "metadata", {}) or {}
    totals = metadata.get("backend_timing_totals_s", {})
    counts = metadata.get("backend_timing_counts", {})
    if not isinstance(totals, dict):
        totals = {}
    if not isinstance(counts, dict):
        counts = {}

    row: dict[str, Any] = {}
    total_steps = max(1, steps * max(1, repeats))
    for key, value in sorted(totals.items()):
        row[f"backend_timing_total_{key}"] = value
        row[f"backend_timing_avg_ms_per_step_{key}"] = float(value) * 1000.0 / total_steps
    for key, value in sorted(counts.items()):
        row[f"backend_timing_count_{key}"] = value
    return row


def append_gpu_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    method: str,
    case: BenchmarkCase,
    steps: int,
    repeats: int,
    summary,
    pcie_summaries: tuple[PcieSummary, ...],
    pcie_note: str | None,
) -> None:
    base = {
        "run_id": run_id,
        "method": method,
        "num_qubits": case.num_qubits,
        "layers": case.layers,
        "steps": steps,
        "repeats": repeats,
        "total_steps": steps * repeats,
        "sample_interval_s": summary.sample_interval_s,
        "telemetry_elapsed_wall_s": summary.elapsed_wall_s,
        "snapshots_collected": summary.snapshots_collected,
        "telemetry_note": summary.note,
        "pcie_note": pcie_note,
    }
    pcie = pcie_by_index(pcie_summaries)
    if not summary.gpus:
        base.update(
            {
                "pcie_sample_count_total": sum(item.sample_count for item in pcie_summaries),
            }
        )
        rows.append(base)
        return

    for gpu in summary.gpus:
        pcie_summary = pcie.get(gpu.index)
        row = {**base, **asdict(gpu)}
        if pcie_summary is not None:
            row.update(
                {
                    "pcie_sample_count": pcie_summary.sample_count,
                    "avg_pcie_rx_kbps": pcie_summary.avg_pcie_rx_kbps,
                    "peak_pcie_rx_kbps": pcie_summary.peak_pcie_rx_kbps,
                    "avg_pcie_tx_kbps": pcie_summary.avg_pcie_tx_kbps,
                    "peak_pcie_tx_kbps": pcie_summary.peak_pcie_tx_kbps,
                }
            )
        else:
            row.update(
                {
                    "pcie_sample_count": 0,
                    "avg_pcie_rx_kbps": None,
                    "peak_pcie_rx_kbps": None,
                    "avg_pcie_tx_kbps": None,
                    "peak_pcie_tx_kbps": None,
                }
            )
        rows.append(row)


def main() -> None:
    args = parse_args()
    if args.internal_ncu_run_case:
        run_internal_ncu_case(args)
        return
    if args.steps is not None and args.steps < 0:
        raise SystemExit("--steps must be non-negative.")
    if args.max_repeat_runs < 1:
        raise SystemExit("--max-repeat-runs must be at least 1.")
    if args.kernel_profile_ncu:
        ensure_ncu_available(args.ncu_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"gpu_metrics_{timestamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    ncu_kernel_rows: list[dict[str, Any]] = []
    ncu_summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    total_runs = len(args.cases) * len(args.methods)
    print("GPU metrics collection")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Cases: {', '.join(f'{c.num_qubits}x{c.layers}' for c in args.cases)}")
    print(f"  Total method/circuit runs: {total_runs}")
    print(f"  Telemetry interval: {args.telemetry_interval:.2f} s")
    print(f"  PCIe interval: {args.pcie_interval:.2f} s")
    print(f"  Nsight Compute kernel profile: {'enabled' if args.kernel_profile_ncu else 'disabled'}")
    if args.kernel_profile_ncu:
        print(f"  Nsight Compute path: {args.ncu_path}")
        print(f"  Nsight Compute launch count: {args.ncu_launch_count}")
    print(f"  Output directory: {args.out_dir}")
    print()

    run_index = 0
    for case in args.cases:
        steps = args.steps if args.steps is not None else default_steps_for(case)
        for method in args.methods:
            run_index += 1
            run_id = f"q{case.num_qubits}_l{case.layers}_{method}"
            print(f"[{run_index}/{total_runs}] {run_id}, steps={steps}", flush=True)

            gpu_monitor = GpuTelemetryMonitor(
                sample_interval_s=args.telemetry_interval,
                live=args.telemetry_live,
                label=run_id,
            )
            pcie_monitor = PcieDmonMonitor(interval_s=args.pcie_interval)
            results: list[Any] = []
            status = "ok"
            error_message: str | None = None
            total_wall_s = 0.0

            gpu_monitor.start()
            pcie_monitor.start()
            try:
                while len(results) < args.max_repeat_runs:
                    repeat = len(results) + 1
                    config = build_run_config(
                        method,
                        case=case,
                        steps=steps,
                        seed=args.seed + repeat - 1,
                        args=args,
                    )
                    run_start = time.perf_counter()
                    result = run_method(method, config)
                    total_wall_s += time.perf_counter() - run_start
                    results.append(result)

                    if gpu_monitor.summary().snapshots_collected >= args.min_telemetry_samples:
                        break
            except Exception as exc:
                status = "failed"
                error_message = str(exc)
                if not args.continue_on_error:
                    raise
            finally:
                gpu_summary = gpu_monitor.stop()
                pcie_summaries = pcie_monitor.stop()

            repeats = len(results)
            result = results[-1] if results else None
            total_steps = steps * repeats
            avg_step_ms = total_wall_s * 1000.0 / total_steps if total_steps else None
            run_row: dict[str, Any] = {
                "run_id": run_id,
                "method": method,
                "num_qubits": case.num_qubits,
                "layers": case.layers,
                "steps": steps,
                "repeats": repeats,
                "total_steps": total_steps,
                "field": args.field,
                "stepsize": args.stepsize,
                "seed": args.seed,
                "init_scale": args.init_scale,
                "gate_fusion": args.gate_fusion,
                "status": status,
                "error_message": error_message,
                "total_wall_s": total_wall_s if repeats else None,
                "avg_step_ms": avg_step_ms,
                "telemetry_samples": gpu_summary.snapshots_collected,
                "pcie_samples": sum(item.sample_count for item in pcie_summaries),
                "pcie_note": pcie_monitor.note,
            }
            if result is not None:
                run_row.update(
                    {
                        "final_energy": float(result.final_energy),
                        **result_metadata_row(result),
                        **backend_timing_row(result, steps, repeats),
                    }
                )
            run_rows.append(run_row)

            append_gpu_rows(
                gpu_rows,
                run_id=run_id,
                method=method,
                case=case,
                steps=steps,
                repeats=repeats,
                summary=gpu_summary,
                pcie_summaries=pcie_summaries,
                pcie_note=pcie_monitor.note,
            )

            if args.kernel_profile_ncu:
                print(f"  Nsight Compute profiling {run_id}", flush=True)
                profile_rows = run_ncu_profile_case(
                    args=args,
                    run_id=run_id,
                    method=method,
                    case=case,
                    steps=steps,
                )
                ncu_kernel_rows.extend(profile_rows)
                ncu_summary_rows.append(
                    summarize_ncu_rows(
                        run_id=run_id,
                        method=method,
                        case=case,
                        steps=steps,
                        rows=profile_rows,
                        metrics=args.ncu_metrics,
                    )
                )

            if result is not None and args.report_steps:
                for repeat_index, repeat_result in enumerate(results, start=1):
                    for metric in repeat_result.step_metrics:
                        step_rows.append(
                            {
                                "run_id": run_id,
                                "method": method,
                                "num_qubits": case.num_qubits,
                                "layers": case.layers,
                                "repeat": repeat_index,
                                "step": metric.step,
                                "energy": metric.energy,
                                "grad_norm": metric.grad_norm,
                                "grad_wall_s": metric.grad_wall_s,
                                "step_wall_s": metric.step_wall_s,
                            }
                        )

            print(
                "  "
                f"status={status} repeats={repeats} "
                f"samples={gpu_summary.snapshots_collected} "
                f"pcie_samples={sum(item.sample_count for item in pcie_summaries)} "
                f"avg_step_ms={avg_step_ms if avg_step_ms is not None else 'NA'}",
                flush=True,
            )

    run_path = args.out_dir / f"{prefix}_runs.csv"
    gpu_path = args.out_dir / f"{prefix}_gpu_metrics.csv"
    write_csv(run_path, run_rows)
    write_csv(gpu_path, gpu_rows)

    print()
    print(f"Run results written to: {run_path}")
    print(f"GPU metrics written to: {gpu_path}")

    if args.kernel_profile_ncu:
        ncu_kernel_path = args.out_dir / f"{prefix}_ncu_kernel_metrics.csv"
        ncu_summary_path = args.out_dir / f"{prefix}_ncu_summary.csv"
        write_csv(ncu_kernel_path, ncu_kernel_rows)
        write_csv(ncu_summary_path, ncu_summary_rows)
        print(f"Nsight Compute kernel metrics written to: {ncu_kernel_path}")
        print(f"Nsight Compute summary written to: {ncu_summary_path}")

    if step_rows:
        step_path = args.out_dir / f"{prefix}_steps.csv"
        write_csv(step_path, step_rows)
        print(f"Step metrics written to: {step_path}")


if __name__ == "__main__":
    main()
