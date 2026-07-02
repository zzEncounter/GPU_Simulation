"""Run standalone gradient-strategy grid benchmarks with GPU telemetry output."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
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
from ring_ising.runtime._nvidia_smi import (
    max_or_none,
    mean_or_none,
    parse_float_or_none,
    parse_int_or_none,
    parse_nvidia_csv,
    query_gpu_telemetry_rows,
    query_process_gpu_memory_rows,
)


MODES = (
    "inverse_walk",
    "ryrz_fused",
    "structured_adjoint",
    "dense_scan",
)
QUBITS = (4, 5, 6)
LAYERS = (8, 32, 128, 512, 2048)
UTIL_DMON_FIELDS = ["gpu", "sm", "mem", "enc", "dec", "jpg", "ofa"]
PCIE_DMON_FIELDS = ["gpu", "rxpci", "txpci"]
NCU_METRICS = (
    "launch__grid_size",
    "launch__block_size",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "gpu__time_duration.sum",
)
NCU_BASE_FIELDS = (
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Section Name",
    "Metric Name",
    "Metric Unit",
    "Metric Value",
)
REQUIRED_GPU_METRICS = (
    "peak_gpu_util_pct",
    "avg_sm_util_pct",
    "peak_sm_util_pct",
    "avg_pcie_tx_kbps",
    "peak_pcie_tx_kbps",
    "avg_pcie_rx_kbps",
    "peak_pcie_rx_kbps",
)
REQUIRED_POSITIVE_GPU_METRICS = (
    "peak_gpu_util_pct",
    "peak_sm_util_pct",
)
DMON_MIN_INTERVAL_S = 1.0


@dataclass(frozen=True)
class BenchmarkCase:
    num_qubits: int
    layers: int


@dataclass(frozen=True)
class BenchmarkGpuSample:
    index: str
    name: str
    uuid: str | None
    fb_memory_used_mib: int | None
    fb_memory_total_mib: int | None
    process_memory_mib: int | None
    gpu_util_pct: float | None
    sm_util_pct: float | None
    pcie_tx_kbps: float | None
    pcie_rx_kbps: float | None


@dataclass(frozen=True)
class BenchmarkGpuSummary:
    index: str
    name: str
    uuid: str | None
    sample_count: int
    peak_fb_memory_mib: int | None
    avg_fb_memory_used_pct: float | None
    peak_process_memory_mib: int | None
    avg_gpu_util_pct: float | None
    peak_gpu_util_pct: float | None
    avg_sm_util_pct: float | None
    peak_sm_util_pct: float | None
    avg_pcie_tx_kbps: float | None
    peak_pcie_tx_kbps: float | None
    avg_pcie_rx_kbps: float | None
    peak_pcie_rx_kbps: float | None


def parse_case(text: str) -> BenchmarkCase:
    try:
        qubits_text, layers_text = text.lower().split("x", maxsplit=1)
        return BenchmarkCase(num_qubits=int(qubits_text), layers=int(layers_text))
    except Exception as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"Invalid case {text!r}. Expected format like 12x16."
        ) from exc


def parse_mode(text: str) -> str:
    mode = text.strip()
    if mode not in MODES:
        choices = ", ".join(MODES)
        raise argparse.ArgumentTypeError(
            f"Invalid mode {text!r}. Expected one of: {choices}"
        )
    return mode


def default_steps_for(case: BenchmarkCase) -> int:
    base_steps_by_layers = {8: 120, 32: 80, 128: 40, 512: 12, 2048: 4}
    base = base_steps_by_layers.get(case.layers, max(2, 960 // max(case.layers, 1)))
    return max(2, int(round(base * 4.0 / case.num_qubits)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=parse_case,
        default=[
            BenchmarkCase(num_qubits=q, layers=l)
            for q in QUBITS
            for l in LAYERS
        ],
        help="Problem sizes in QxL format.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        type=parse_mode,
        default=list(MODES),
        help="Gradient strategies to benchmark.",
    )
    parser.add_argument(
        "--structured-widths",
        "--mode2-widths",
        dest="structured_widths",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 8],
        help="structured_adjoint rotation chunk widths to benchmark.",
    )
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--stepsize", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=0.3)
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=0.5,
        help="GPU telemetry sampling interval in seconds.",
    )
    parser.add_argument(
        "--telemetry-live",
        action="store_true",
        help="Print live GPU telemetry samples during each run.",
    )
    parser.add_argument(
        "--min-telemetry-samples",
        type=int,
        default=5,
        help=(
            "Repeat the same run until at least this many GPU "
            "samples are collected, capped by --max-repeat-runs."
        ),
    )
    parser.add_argument(
        "--max-repeat-runs",
        type=int,
        default=0,
        help=(
            "Maximum repeats while waiting for required GPU metrics. Use 0 for "
            "no repeat cap."
        ),
    )
    parser.add_argument(
        "--telemetry-warmup-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for nvidia-smi dmon streams before running a circuit.",
    )
    parser.add_argument(
        "--allow-incomplete-telemetry",
        action="store_true",
        help="Write incomplete GPU rows instead of failing when required metrics are missing.",
    )
    parser.add_argument(
        "--repeat-small-layers-up-to",
        type=int,
        default=2048,
        help="Cases with layers at or below this value may be repeated for telemetry.",
    )
    parser.add_argument(
        "--report-steps",
        action="store_true",
        help="Store roughly 60 detailed step metrics per run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to a timestamped benchmark name.",
    )
    parser.add_argument(
        "--kernel-profile-ncu",
        action="store_true",
        help="Run Nsight Compute once per circuit mode and write kernel-level metrics.",
    )
    parser.add_argument(
        "--ncu-path",
        default="ncu",
        help="Path to the Nsight Compute CLI executable.",
    )
    parser.add_argument(
        "--ncu-profile-count",
        type=int,
        default=1,
        help="Number of Nsight Compute profile runs per circuit mode.",
    )
    parser.add_argument(
        "--ncu-launch-count",
        type=int,
        default=64,
        help="Maximum profiled CUDA kernel launches per Nsight Compute run.",
    )
    parser.add_argument(
        "--allow-incomplete-kernel-profile",
        action="store_true",
        help="Write partial Nsight Compute rows instead of failing on missing metrics.",
    )
    parser.add_argument(
        "--internal-ncu-profile-case",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-num-qubits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-structured-width", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-mode2-width",
        dest="internal_structured_width",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def value_or_empty(value: Any) -> Any:
    return "" if value is None else value


def value_or_na(value: Any) -> Any:
    return "NA" if value is None else value


def should_repeat_for_telemetry(case: BenchmarkCase, args: argparse.Namespace) -> bool:
    return (
        args.min_telemetry_samples > 0
        and case.layers <= args.repeat_small_layers_up_to
    )


class BenchmarkGpuTelemetryMonitor:
    """Collect the small GPU metric set needed by this benchmark."""

    def __init__(self, interval_s: float, live: bool, label: str) -> None:
        self.interval_s = max(0.2, interval_s)
        self.dmon_interval_s = max(DMON_MIN_INTERVAL_S, interval_s)
        self.live = live
        self.label = label
        self._current_pid = os.getpid()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._util_thread: threading.Thread | None = None
        self._util_process: subprocess.Popen[str] | None = None
        self._pcie_thread: threading.Thread | None = None
        self._pcie_process: subprocess.Popen[str] | None = None
        self._samples: list[BenchmarkGpuSample] = []
        self._latest_util_rows: dict[str, dict[str, str]] = {}
        self._latest_pcie_rows: dict[str, dict[str, str]] = {}
        self._start_time_s: float | None = None
        self._end_time_s: float | None = None
        self.note: str | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.note = "nvidia-smi not found on PATH"
            return
        self._start_time_s = time.perf_counter()
        self._util_thread = threading.Thread(target=self._run_util_stream, daemon=True)
        self._util_thread.start()
        self._pcie_thread = threading.Thread(target=self._run_pcie_stream, daemon=True)
        self._pcie_thread.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._util_process is not None and self._util_process.poll() is None:
            self._util_process.terminate()
            try:
                self._util_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._util_process.kill()
        if self._pcie_process is not None and self._pcie_process.poll() is None:
            self._pcie_process.terminate()
            try:
                self._pcie_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._pcie_process.kill()
        if self._thread is not None:
                self._thread.join(timeout=max(2.0, 2 * self.interval_s))
        if self._util_thread is not None:
            self._util_thread.join(timeout=2.0)
        if self._pcie_thread is not None:
            self._pcie_thread.join(timeout=2.0)
        self._end_time_s = time.perf_counter()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def streams_ready(self) -> bool:
        with self._lock:
            return bool(self._latest_util_rows) and bool(self._latest_pcie_rows)

    @property
    def elapsed_wall_s(self) -> float:
        if self._start_time_s is None:
            return 0.0
        end_time_s = self._end_time_s if self._end_time_s is not None else time.perf_counter()
        return max(0.0, end_time_s - self._start_time_s)

    def summaries(self) -> tuple[BenchmarkGpuSummary, ...]:
        with self._lock:
            samples = tuple(self._samples)

        by_gpu: dict[str, list[BenchmarkGpuSample]] = {}
        for sample in samples:
            by_gpu.setdefault(sample.index, []).append(sample)
        return tuple(
            summarize_benchmark_gpu_samples(gpu_samples)
            for _, gpu_samples in sorted(by_gpu.items())
        )

    def wait_until_streams_ready(self, timeout_s: float) -> bool:
        deadline_s = time.perf_counter() + max(0.0, timeout_s)
        while time.perf_counter() < deadline_s:
            if self.streams_ready:
                return True
            if self.note is not None:
                return False
            time.sleep(0.05)
        return self.streams_ready

    def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_start_s = time.perf_counter()
            samples, note = capture_benchmark_gpu_samples(
                current_pid=self._current_pid,
                util_rows_by_index=self._latest_util_rows_snapshot(),
                pcie_rows_by_index=self._latest_pcie_rows_snapshot(),
            )
            if note is not None and self.note is None:
                self.note = note
            with self._lock:
                self._samples.extend(samples)
            if self.live:
                for sample in samples:
                    print(
                        f"[{self.label}] GPU {sample.index}: "
                        f"fb={value_or_na(sample.fb_memory_used_mib)} MiB, "
                        f"proc={value_or_na(sample.process_memory_mib)} MiB, "
                        f"gpu={value_or_na(sample.gpu_util_pct)}%, "
                        f"sm={value_or_na(sample.sm_util_pct)}%, "
                        f"tx={value_or_na(sample.pcie_tx_kbps)} KB/s, "
                        f"rx={value_or_na(sample.pcie_rx_kbps)} KB/s"
                    )
            elapsed_s = time.perf_counter() - loop_start_s
            self._stop_event.wait(max(0.0, self.interval_s - elapsed_s))

    def _latest_util_rows_snapshot(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {gpu_index: dict(row) for gpu_index, row in self._latest_util_rows.items()}

    def _latest_pcie_rows_snapshot(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {gpu_index: dict(row) for gpu_index, row in self._latest_pcie_rows.items()}

    def _run_util_stream(self) -> None:
        try:
            self._util_process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "dmon",
                    "-s",
                    "u",
                    "-d",
                    str(int(round(self.dmon_interval_s))),
                    "--format",
                    "csv,nounit,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            if self.note is None:
                self.note = str(exc)
            return

        if self._util_process.stdout is None:
            if self.note is None:
                self.note = "Unable to read nvidia-smi SM utilization output."
            return

        for line in self._util_process.stdout:
            if self._stop_event.is_set():
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = next(iter(parse_nvidia_csv(stripped, UTIL_DMON_FIELDS)), None)
            if row is None:
                continue
            gpu_index = row["gpu"].strip()
            with self._lock:
                self._latest_util_rows[gpu_index] = row

    def _run_pcie_stream(self) -> None:
        try:
            self._pcie_process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "dmon",
                    "-s",
                    "t",
                    "-d",
                    str(int(round(self.dmon_interval_s))),
                    "--format",
                    "csv,nounit,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            if self.note is None:
                self.note = str(exc)
            return

        if self._pcie_process.stdout is None:
            if self.note is None:
                self.note = "Unable to read nvidia-smi PCIe telemetry output."
            return

        for line in self._pcie_process.stdout:
            if self._stop_event.is_set():
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = next(iter(parse_nvidia_csv(stripped, PCIE_DMON_FIELDS)), None)
            if row is None:
                continue
            gpu_index = row["gpu"].strip()
            with self._lock:
                self._latest_pcie_rows[gpu_index] = row


def capture_benchmark_gpu_samples(
    *,
    current_pid: int,
    util_rows_by_index: dict[str, dict[str, str]],
    pcie_rows_by_index: dict[str, dict[str, str]],
) -> tuple[tuple[BenchmarkGpuSample, ...], str | None]:
    gpu_rows, gpu_note = query_gpu_telemetry_rows()
    process_rows, process_note = query_process_gpu_memory_rows()

    process_mem_by_uuid: dict[str, int] = {}
    for row in process_rows:
        if parse_int_or_none(row["pid"]) != current_pid:
            continue
        gpu_uuid = row.get("gpu_uuid", "").strip()
        used_gpu_memory = parse_int_or_none(row["used_gpu_memory"])
        if not gpu_uuid or used_gpu_memory is None:
            continue
        process_mem_by_uuid[gpu_uuid] = process_mem_by_uuid.get(gpu_uuid, 0) + used_gpu_memory

    samples: list[BenchmarkGpuSample] = []
    for row in gpu_rows:
        gpu_index = row["index"].strip()
        gpu_uuid = row["uuid"].strip() if row.get("uuid") else None
        gpu_util_pct = parse_float_or_none(row["utilization.gpu"])
        util_row = util_rows_by_index.get(gpu_index, {})
        pcie_row = pcie_rows_by_index.get(gpu_index, {})

        samples.append(
            BenchmarkGpuSample(
                index=gpu_index,
                name=row["name"],
                uuid=gpu_uuid,
                fb_memory_used_mib=parse_int_or_none(row["memory.used"]),
                fb_memory_total_mib=parse_int_or_none(row["memory.total"]),
                process_memory_mib=process_mem_by_uuid.get(gpu_uuid or ""),
                gpu_util_pct=gpu_util_pct,
                sm_util_pct=parse_float_or_none(util_row.get("sm", "")),
                pcie_tx_kbps=parse_float_or_none(pcie_row.get("txpci", "")),
                pcie_rx_kbps=parse_float_or_none(pcie_row.get("rxpci", "")),
            )
        )

    notes = [note for note in (gpu_note, process_note) if note]
    return tuple(samples), "; ".join(notes) if notes else None


def summarize_benchmark_gpu_samples(samples: list[BenchmarkGpuSample]) -> BenchmarkGpuSummary:
    sample0 = samples[0]
    fb_memory_used_pct = [
        sample.fb_memory_used_mib * 100.0 / sample.fb_memory_total_mib
        if sample.fb_memory_used_mib is not None
        and sample.fb_memory_total_mib not in (None, 0)
        else None
        for sample in samples
    ]
    return BenchmarkGpuSummary(
        index=sample0.index,
        name=sample0.name,
        uuid=sample0.uuid,
        sample_count=len(samples),
        peak_fb_memory_mib=max_or_none([sample.fb_memory_used_mib for sample in samples]),
        avg_fb_memory_used_pct=mean_or_none(fb_memory_used_pct),
        peak_process_memory_mib=max_or_none([sample.process_memory_mib for sample in samples]),
        avg_gpu_util_pct=mean_or_none([sample.gpu_util_pct for sample in samples]),
        peak_gpu_util_pct=max_or_none([sample.gpu_util_pct for sample in samples]),
        avg_sm_util_pct=mean_or_none([sample.sm_util_pct for sample in samples]),
        peak_sm_util_pct=max_or_none([sample.sm_util_pct for sample in samples]),
        avg_pcie_tx_kbps=mean_or_none([sample.pcie_tx_kbps for sample in samples]),
        peak_pcie_tx_kbps=max_or_none([sample.pcie_tx_kbps for sample in samples]),
        avg_pcie_rx_kbps=mean_or_none([sample.pcie_rx_kbps for sample in samples]),
        peak_pcie_rx_kbps=max_or_none([sample.pcie_rx_kbps for sample in samples]),
    )


def required_metrics_complete(summary: BenchmarkGpuSummary) -> bool:
    if any(getattr(summary, metric) is None for metric in REQUIRED_GPU_METRICS):
        return False
    return all((getattr(summary, metric) or 0) > 0 for metric in REQUIRED_POSITIVE_GPU_METRICS)


def telemetry_complete(monitor: BenchmarkGpuTelemetryMonitor, min_samples: int) -> bool:
    summaries = monitor.summaries()
    if not summaries:
        return False
    return all(
        summary.sample_count >= min_samples and required_metrics_complete(summary)
        for summary in summaries
    )


def missing_required_metrics(monitor: BenchmarkGpuTelemetryMonitor) -> str | None:
    summaries = monitor.summaries()
    if not summaries:
        return "no_gpu_summary"

    missing_parts: list[str] = []
    for summary in summaries:
        missing = [
            metric
            for metric in REQUIRED_GPU_METRICS
            if getattr(summary, metric) is None
        ]
        not_positive = [
            metric
            for metric in REQUIRED_POSITIVE_GPU_METRICS
            if getattr(summary, metric) is not None and (getattr(summary, metric) or 0) <= 0
        ]
        missing.extend(f"{metric}_not_positive" for metric in not_positive)
        if missing:
            missing_parts.append(f"gpu{summary.index}:{'/'.join(missing)}")
    return "; ".join(missing_parts) if missing_parts else None


def repeat_cap_reached(results: list[Any], args: argparse.Namespace) -> bool:
    return args.max_repeat_runs > 0 and len(results) >= args.max_repeat_runs


def run_internal_ncu_profile_case(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in (
            ("--internal-num-qubits", args.internal_num_qubits),
            ("--internal-layers", args.internal_layers),
            ("--internal-steps", args.internal_steps),
            ("--internal-mode", args.internal_mode),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing internal Nsight Compute arguments: {', '.join(missing)}")

    config = RunConfig(
        backend="standalone",
        num_qubits=args.internal_num_qubits,
        layers=args.internal_layers,
        field=args.field,
        steps=args.internal_steps,
        stepsize=args.stepsize,
        seed=args.seed,
        init_scale=args.init_scale,
        verbose=False,
        show_progress=False,
        report_steps=False,
        gpu_telemetry=False,
        telemetry_interval_s=args.telemetry_interval,
        telemetry_live=False,
        gradient_strategy=args.internal_mode,
        structured_rotation_chunk_width=args.internal_structured_width,
    )
    result = run_standalone(config)
    print(f"ncu_profile_final_energy={result.final_energy}", file=sys.stderr)


def ensure_ncu_available(ncu_path: str) -> str:
    if os.path.basename(ncu_path) != ncu_path:
        candidate = Path(ncu_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError(
            f"Nsight Compute CLI executable was not found at {candidate}. "
            "Install NVIDIA Nsight Compute or pass a valid path with --ncu-path."
        )

    candidates: list[Path] = []

    which_result = shutil.which(ncu_path)
    if which_result is not None:
        candidates.append(Path(which_result))

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_raw = os.environ.get(env_name)
        if env_raw:
            candidates.append(Path(env_raw).expanduser() / "bin" / ncu_path)

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        candidates.append(Path(nvcc).resolve().parent.parent / "bin" / ncu_path)

    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        candidates.append(Path("/home") / sudo_user / "cuda" / "bin" / ncu_path)

    candidates.extend(
        [
            Path.home() / "cuda" / "bin" / ncu_path,
            Path("/usr/local/cuda/bin") / ncu_path,
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return str(resolved)

    searched = ", ".join(str(path) for path in seen)
    raise RuntimeError(
        "Nsight Compute CLI executable was not found. Install NVIDIA Nsight Compute, "
        "pass its path with --ncu-path, or make sure one of these paths exists: "
        f"{searched}"
    )


def run_ncu_profile_case(
    *,
    args: argparse.Namespace,
    run_id: str,
    mode: str,
    mode_label: str,
    structured_width: int,
    case: BenchmarkCase,
    steps: int,
    profile_index: int,
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
        ",".join(NCU_METRICS),
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-ncu-profile-case",
        "--internal-num-qubits",
        str(case.num_qubits),
        "--internal-layers",
        str(case.layers),
        "--internal-steps",
        str(steps),
        "--internal-mode",
        mode,
        "--internal-structured-width",
        str(structured_width),
        "--field",
        str(args.field),
        "--stepsize",
        str(args.stepsize),
        "--seed",
        str(args.seed + profile_index),
        "--init-scale",
        str(args.init_scale),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if "ERR_NVGPUCTRPERM" in completed.stdout or "ERR_NVGPUCTRPERM" in completed.stderr:
            return [
                ncu_diagnostic_row(
                    run_id=run_id,
                    mode=mode_label,
                    case=case,
                    steps=steps,
                    profile_index=profile_index,
                    status="permission_denied",
                    message=(
                        "Nsight Compute cannot access NVIDIA GPU Performance Counters. "
                        "Enable profiling permissions or run with sufficient privileges. "
                        "See https://developer.nvidia.com/ERR_NVGPUCTRPERM"
                    ),
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            ]
        if args.allow_incomplete_kernel_profile:
            return [
                ncu_diagnostic_row(
                    run_id=run_id,
                    mode=mode_label,
                    case=case,
                    steps=steps,
                    profile_index=profile_index,
                    status="failed",
                    message=f"Nsight Compute exited with code {completed.returncode}.",
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            ]
        raise RuntimeError(
            "Nsight Compute profiling failed for "
            f"{run_id} (profile {profile_index + 1}).\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    rows = parse_ncu_raw_csv(completed.stdout)
    if not rows:
        rows = parse_ncu_raw_csv(completed.stderr)
    kernel_rows = build_kernel_profile_rows(
        run_id=run_id,
        mode=mode_label,
        case=case,
        steps=steps,
        profile_index=profile_index,
        metric_rows=rows,
        allow_incomplete=args.allow_incomplete_kernel_profile,
    )
    if not kernel_rows and not args.allow_incomplete_kernel_profile:
        raise RuntimeError(
            f"Nsight Compute produced no kernel metric rows for {run_id}. "
            "Check ncu permissions, metric availability, whether no kernels were "
            "captured, and --ncu-launch-count.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return kernel_rows


def ncu_diagnostic_row(
    *,
    run_id: str,
    mode: str,
    case: BenchmarkCase,
    steps: int,
    profile_index: int,
    status: str,
    message: str,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "num_qubits": case.num_qubits,
        "layers": case.layers,
        "steps": steps,
        "profile_index": profile_index + 1,
        "launch_index": None,
        "kernel_name": None,
        "ncu_status": status,
        "ncu_message": message,
        "ncu_stdout_excerpt": excerpt_text(stdout),
        "ncu_stderr_excerpt": excerpt_text(stderr),
        "missing_ncu_metrics": "/".join(NCU_METRICS),
    }
    for metric in NCU_METRICS:
        row[metric] = None
    return row


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
        if ("Metric Name" in parsed and "Metric Value" in parsed) or (
            "Kernel Name" in parsed and any(metric in parsed for metric in NCU_METRICS)
        ):
            header_index = index
            break
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def build_kernel_profile_rows(
    *,
    run_id: str,
    mode: str,
    case: BenchmarkCase,
    steps: int,
    profile_index: int,
    metric_rows: list[dict[str, str]],
    allow_incomplete: bool,
) -> list[dict[str, Any]]:
    by_launch: dict[tuple[str, str], dict[str, Any]] = {}
    for metric_row in metric_rows:
        launch_id = metric_row.get("ID", "").strip()
        kernel_name = (
            metric_row.get("Kernel Name", "").strip()
            or metric_row.get("launch__kernel_name", "").strip()
        )
        if not launch_id or not kernel_name:
            continue
        key = (launch_id, kernel_name)
        row = by_launch.setdefault(
            key,
            {
                "run_id": run_id,
                "mode": mode,
                "num_qubits": case.num_qubits,
                "layers": case.layers,
                "steps": steps,
                "profile_index": profile_index + 1,
                "launch_index": launch_id,
                "kernel_name": kernel_name,
                "ncu_status": "ok",
                "ncu_message": None,
                "ncu_stdout_excerpt": None,
                "ncu_stderr_excerpt": None,
            },
        )

        metric_name = metric_row.get("Metric Name", "").strip()
        if metric_name:
            if metric_name in NCU_METRICS:
                row[metric_name] = normalize_ncu_metric_value(metric_row.get("Metric Value", ""))
            continue

        for column_metric in NCU_METRICS:
            if column_metric in metric_row:
                row[column_metric] = normalize_ncu_metric_value(metric_row.get(column_metric, ""))

    kernel_rows = list(by_launch.values())
    for row in kernel_rows:
        missing = [metric for metric in NCU_METRICS if row.get(metric) in (None, "")]
        row["missing_ncu_metrics"] = "/".join(missing) if missing else None
        if missing and not allow_incomplete:
            raise RuntimeError(
                f"Nsight Compute row for {run_id} kernel {row['kernel_name']} "
                f"launch {row['launch_index']} is missing metrics: {', '.join(missing)}"
            )
        for metric in missing:
            row[metric] = None
    return kernel_rows


def normalize_ncu_metric_value(value: str) -> Any:
    normalized = value.strip().replace(",", "")
    if normalized in {"", "N/A", "nan", "NaN", "-"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return value.strip()


def append_gpu_summary_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    mode: str,
    case: BenchmarkCase,
    steps: int,
    repeats: int,
    monitor: BenchmarkGpuTelemetryMonitor,
    telemetry_complete_for_run: bool,
) -> None:
    summaries = monitor.summaries()
    if not summaries:
        rows.append(
            base_gpu_row(
                run_id=run_id,
                mode=mode,
                case=case,
                steps=steps,
                repeats=repeats,
                monitor=monitor,
                telemetry_complete_for_run=telemetry_complete_for_run,
            )
        )
        return

    for gpu in summaries:
        row = base_gpu_row(
            run_id=run_id,
            mode=mode,
            case=case,
            steps=steps,
            repeats=repeats,
            monitor=monitor,
            telemetry_complete_for_run=telemetry_complete_for_run,
        )
        row.update(
            {
                "gpu_index": gpu.index,
                "gpu_name": gpu.name,
                "gpu_uuid": gpu.uuid,
                "gpu_sample_count": gpu.sample_count,
                "peak_fb_memory_mib": gpu.peak_fb_memory_mib,
                "avg_fb_memory_used_pct": gpu.avg_fb_memory_used_pct,
                "peak_process_memory_mib": gpu.peak_process_memory_mib,
                "avg_gpu_util_pct": gpu.avg_gpu_util_pct,
                "peak_gpu_util_pct": gpu.peak_gpu_util_pct,
                "avg_sm_util_pct": gpu.avg_sm_util_pct,
                "peak_sm_util_pct": gpu.peak_sm_util_pct,
                "avg_pcie_tx_kbps": gpu.avg_pcie_tx_kbps,
                "peak_pcie_tx_kbps": gpu.peak_pcie_tx_kbps,
                "avg_pcie_rx_kbps": gpu.avg_pcie_rx_kbps,
                "peak_pcie_rx_kbps": gpu.peak_pcie_rx_kbps,
            }
        )
        rows.append(row)


def base_gpu_row(
    *,
    run_id: str,
    mode: str,
    case: BenchmarkCase,
    steps: int,
    repeats: int,
    monitor: BenchmarkGpuTelemetryMonitor,
    telemetry_complete_for_run: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "num_qubits": case.num_qubits,
        "layers": case.layers,
        "steps": steps,
        "repeats": repeats,
        "total_steps": steps * repeats,
        "sample_interval_s": monitor.interval_s,
        "telemetry_elapsed_wall_s": monitor.elapsed_wall_s,
        "snapshots_collected": monitor.sample_count,
        "required_gpu_metrics_complete": telemetry_complete_for_run,
        "missing_required_gpu_metrics": missing_required_metrics(monitor),
        "telemetry_note": monitor.note,
    }


def mode_specs(
    modes: list[str], structured_widths: list[int]
) -> list[tuple[str, str, int]]:
    specs: list[tuple[str, str, int]] = []
    for mode in modes:
        if mode in {"structured_adjoint", "mode2"}:
            specs.extend(
                (f"structured_w{width}", "structured_adjoint", width)
                for width in structured_widths
            )
        else:
            specs.append((mode, mode, 1))
    return specs


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    serialize_value=value_or_empty,
) -> None:
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
            writer.writerow({key: serialize_value(row.get(key)) for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.internal_ncu_profile_case:
        run_internal_ncu_profile_case(args)
        return

    specs = mode_specs(list(args.modes), list(args.structured_widths))
    labels = [label for label, _, _ in specs]
    invalid_dense_cases = [
        case
        for case in args.cases
        if case.num_qubits > 6 and "dense_scan" in labels
    ]
    if invalid_dense_cases:
        invalid_text = ", ".join(
            f"{case.num_qubits}x{case.layers}" for case in invalid_dense_cases
        )
        raise ValueError(
            "dense_scan requires num_qubits <= 6. "
            f"Unsupported cases: {invalid_text}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"standalone_gpu_telemetry_grid_{timestamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    kernel_profile_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    cases = list(args.cases)
    total_runs = len(cases) * len(specs)

    print("Standalone GPU telemetry benchmark grid")
    print(f"  Cases: {', '.join(f'{case.num_qubits}x{case.layers}' for case in cases)}")
    print(f"  Modes: {', '.join(labels)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Telemetry interval: {args.telemetry_interval:.2f} s")
    print(f"  Min telemetry samples per circuit: {args.min_telemetry_samples}")
    print(
        "  Max repeat runs for telemetry: "
        f"{args.max_repeat_runs if args.max_repeat_runs > 0 else 'unlimited'}"
    )
    print(f"  Nsight Compute kernel profile: {'enabled' if args.kernel_profile_ncu else 'disabled'}")
    if args.kernel_profile_ncu:
        ensure_ncu_available(args.ncu_path)
        print(f"  Nsight Compute path: {args.ncu_path}")
        print(f"  Nsight Compute profile count: {args.ncu_profile_count}")
        print(f"  Nsight Compute launch count: {args.ncu_launch_count}")
    print(f"  Output directory: {args.out_dir}")
    print()

    run_index = 0
    for case in cases:
        steps = default_steps_for(case)
        for mode_label, mode, structured_width in specs:
            run_index += 1
            run_id = f"q{case.num_qubits}_l{case.layers}_{mode_label}"
            print(f"[{run_index}/{total_runs}] {run_id}, steps={steps}")

            monitor = BenchmarkGpuTelemetryMonitor(
                interval_s=args.telemetry_interval,
                live=args.telemetry_live,
                label=run_id,
            )
            monitor.start()
            streams_ready = monitor.wait_until_streams_ready(args.telemetry_warmup_timeout)
            if not streams_ready:
                message = (
                    f"{run_id}: nvidia-smi dmon streams were not ready within "
                    f"{args.telemetry_warmup_timeout:.1f}s. note={monitor.note or 'none'}"
                )
                if args.allow_incomplete_telemetry:
                    print(f"  Warning: {message}")
                else:
                    monitor.stop()
                    raise RuntimeError(message)
            results: list[Any] = []
            try:
                while True:
                    repeat_index = len(results) + 1
                    config = RunConfig(
                        backend="standalone",
                        num_qubits=case.num_qubits,
                        layers=case.layers,
                        field=args.field,
                        steps=steps,
                        stepsize=args.stepsize,
                        seed=args.seed + repeat_index - 1,
                        init_scale=args.init_scale,
                        verbose=False,
                        show_progress=False,
                        report_steps=args.report_steps,
                        gpu_telemetry=False,
                        telemetry_interval_s=args.telemetry_interval,
                        telemetry_live=False,
                        gradient_strategy=mode,
                        structured_rotation_chunk_width=structured_width,
                    )
                    results.append(run_standalone(config))

                    if not should_repeat_for_telemetry(case, args):
                        break
                    if telemetry_complete(monitor, args.min_telemetry_samples):
                        break
                    if repeat_cap_reached(results, args):
                        break
            finally:
                monitor.stop()

            telemetry_complete_for_run = telemetry_complete(monitor, args.min_telemetry_samples)
            if not telemetry_complete_for_run and not args.allow_incomplete_telemetry:
                raise RuntimeError(
                    f"{run_id}: required GPU metrics were not fully collected after "
                    f"{len(results)} repeat(s). Missing: "
                    f"{missing_required_metrics(monitor) or 'unknown'}"
                )

            result = results[-1]
            repeats = len(results)
            total_wall_s = sum(item.wall_s for item in results)
            avg_step_ms = total_wall_s * 1000.0 / (steps * repeats) if steps and repeats else 0.0

            run_rows.append(
                {
                    "run_id": run_id,
                    "mode": mode_label,
                    "gradient_strategy": mode,
                    "structured_rotation_chunk_width": structured_width,
                    "num_qubits": case.num_qubits,
                    "layers": case.layers,
                    "steps": steps,
                    "repeats": repeats,
                    "total_steps": steps * repeats,
                    "field": args.field,
                    "stepsize": args.stepsize,
                    "seed": args.seed,
                    "init_scale": args.init_scale,
                    "final_energy": result.final_energy,
                    "total_wall_s": total_wall_s,
                    "avg_step_ms": avg_step_ms,
                    "backend_label": result.backend_label,
                    "estimated_workspace_gib": result.metadata.get("estimated_workspace_gib"),
                }
            )
            append_gpu_summary_rows(
                telemetry_rows,
                run_id=run_id,
                mode=mode_label,
                case=case,
                steps=steps,
                repeats=repeats,
                monitor=monitor,
                telemetry_complete_for_run=telemetry_complete_for_run,
            )
            if args.kernel_profile_ncu:
                print(
                    f"  Nsight Compute profiling {run_id} "
                    f"({args.ncu_profile_count} run(s), launch_count={args.ncu_launch_count})"
                )
                for profile_index in range(max(0, args.ncu_profile_count)):
                    kernel_profile_rows.extend(
                        run_ncu_profile_case(
                            args=args,
                            run_id=run_id,
                            mode=mode,
                            mode_label=mode_label,
                            structured_width=structured_width,
                            case=case,
                            steps=steps,
                            profile_index=profile_index,
                        )
                    )
            for repeat_index, repeat_result in enumerate(results, start=1):
                for metric in repeat_result.step_metrics:
                    step_rows.append(
                        {
                            "run_id": run_id,
                            "mode": mode_label,
                            "gradient_strategy": mode,
                            "structured_rotation_chunk_width": structured_width,
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

    run_path = args.out_dir / f"{prefix}_runs.csv"
    telemetry_path = args.out_dir / f"{prefix}_gpu_telemetry.csv"
    write_csv(run_path, run_rows)
    write_csv(telemetry_path, telemetry_rows, serialize_value=value_or_na)

    print()
    print(f"Run results written to: {run_path}")
    print(f"GPU telemetry written to: {telemetry_path}")

    if args.kernel_profile_ncu:
        kernel_profile_path = args.out_dir / f"{prefix}_kernel_profile.csv"
        write_csv(kernel_profile_path, kernel_profile_rows, serialize_value=value_or_na)
        print(f"Nsight Compute kernel profile written to: {kernel_profile_path}")

    if step_rows:
        step_path = args.out_dir / f"{prefix}_steps.csv"
        write_csv(step_path, step_rows)
        print(f"Step metrics written to: {step_path}")


if __name__ == "__main__":
    main()
