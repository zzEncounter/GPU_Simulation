"""Shared utilities for the adjoint-diff baseline."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pennylane as qml


DEVICE_CANDIDATES = {
    "auto": ("lightning.gpu", "lightning.qubit", "default.qubit"),
    "gpu": ("lightning.gpu",),
    "cpu": ("lightning.qubit", "default.qubit"),
    "default": ("default.qubit",),
}


@dataclass(frozen=True)
class DeviceSelection:
    """Selected PennyLane device plus any fallback notes."""

    requested_mode: str
    device_name: str
    device: Any
    selection_errors: tuple[str, ...]


@dataclass(frozen=True)
class GpuSnapshot:
    """One GPU-level snapshot row from nvidia-smi."""

    index: str
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    utilization_gpu_pct: int | None
    temperature_c: int | None


@dataclass(frozen=True)
class ComputeProcessSnapshot:
    """One compute-process row from nvidia-smi."""

    pid: int
    process_name: str
    used_gpu_memory_mib: int | None


@dataclass(frozen=True)
class ResourceSnapshot:
    """Combined CPU and GPU resource snapshot for the current process."""

    label: str
    process_rss_mib: float | None
    gpus: tuple[GpuSnapshot, ...]
    active_compute_processes: tuple[ComputeProcessSnapshot, ...]
    note: str | None = None


@dataclass(frozen=True)
class GpuTelemetrySample:
    """One merged GPU telemetry sample."""

    index: str
    uuid: str | None
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    process_used_gpu_memory_mib: int | None
    utilization_gpu_pct: float | None
    utilization_memory_pct: float | None
    utilization_encoder_pct: float | None
    utilization_decoder_pct: float | None
    utilization_jpeg_pct: float | None
    utilization_ofa_pct: float | None
    power_draw_w: float | None
    temperature_c: float | None
    clocks_sm_mhz: float | None
    clocks_mem_mhz: float | None
    pstate: str | None
    sm_activity_pct: float | None
    sm_occupancy_pct: float | None
    tensor_activity_pct: float | None
    dram_activity_pct: float | None
    fp64_activity_pct: float | None
    fp32_activity_pct: float | None
    fp16_activity_pct: float | None


@dataclass(frozen=True)
class GpuTelemetrySnapshot:
    """All GPU telemetry collected at one sampling point."""

    timestamp_s: float
    gpus: tuple[GpuTelemetrySample, ...]
    note: str | None = None


@dataclass(frozen=True)
class GpuTelemetryGpuSummary:
    """Aggregate summary for one GPU across many telemetry samples."""

    index: str
    name: str
    uuid: str | None
    sample_count: int
    fb_memory_total_mib: int | None
    avg_utilization_gpu_pct: float | None
    peak_utilization_gpu_pct: float | None
    avg_utilization_memory_pct: float | None
    peak_utilization_memory_pct: float | None
    avg_utilization_encoder_pct: float | None
    peak_utilization_encoder_pct: float | None
    avg_utilization_decoder_pct: float | None
    peak_utilization_decoder_pct: float | None
    avg_utilization_jpeg_pct: float | None
    peak_utilization_jpeg_pct: float | None
    avg_utilization_ofa_pct: float | None
    peak_utilization_ofa_pct: float | None
    avg_power_draw_w: float | None
    peak_power_draw_w: float | None
    peak_temperature_c: float | None
    peak_fb_memory_used_mib: int | None
    peak_process_used_gpu_memory_mib: int | None
    peak_clocks_sm_mhz: float | None
    peak_clocks_mem_mhz: float | None
    avg_sm_activity_pct: float | None
    peak_sm_activity_pct: float | None
    avg_sm_occupancy_pct: float | None
    peak_sm_occupancy_pct: float | None
    avg_tensor_activity_pct: float | None
    peak_tensor_activity_pct: float | None
    avg_dram_activity_pct: float | None
    peak_dram_activity_pct: float | None
    avg_fp64_activity_pct: float | None
    peak_fp64_activity_pct: float | None
    avg_fp32_activity_pct: float | None
    peak_fp32_activity_pct: float | None
    avg_fp16_activity_pct: float | None
    peak_fp16_activity_pct: float | None


@dataclass(frozen=True)
class GpuTelemetrySummary:
    """Summary of sampled GPU telemetry across the whole run."""

    sample_interval_s: float
    elapsed_wall_s: float
    snapshots_collected: int
    gpus: tuple[GpuTelemetryGpuSummary, ...]
    note: str | None = None


class GpuTelemetryMonitor:
    """Lightweight background sampler for GPU utilization and memory telemetry."""

    def __init__(
        self,
        sample_interval_s: float = 0.5,
        live: bool = False,
        label: str = "GPU telemetry",
    ) -> None:
        self.sample_interval_s = max(0.2, sample_interval_s)
        self.live = live
        self.label = label
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gpm_thread: threading.Thread | None = None
        self._gpm_process: subprocess.Popen[str] | None = None
        self._snapshots: list[GpuTelemetrySnapshot] = []
        self._start_time_s: float | None = None
        self._end_time_s: float | None = None
        self._note: str | None = None
        self._current_pid = os.getpid()
        self._latest_gpm_rows: dict[str, dict[str, str]] = {}

    def start(self) -> None:
        """Start sampling GPU telemetry in the background."""
        if shutil.which("nvidia-smi") is None:
            self._note = "nvidia-smi not found on PATH"
            return
        if self._thread is not None:
            return
        self._start_time_s = time.perf_counter()
        self._gpm_thread = threading.Thread(target=self._run_gpm_stream, daemon=True)
        self._gpm_thread.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> GpuTelemetrySummary:
        """Stop sampling and return the collected telemetry summary."""
        self._stop_event.set()
        if self._gpm_process is not None and self._gpm_process.poll() is None:
            self._gpm_process.terminate()
            try:
                self._gpm_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._gpm_process.kill()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 2 * self.sample_interval_s))
        if self._gpm_thread is not None:
            self._gpm_thread.join(timeout=2.0)
        self._end_time_s = time.perf_counter()
        return self.summary()

    def summary(self) -> GpuTelemetrySummary:
        """Return a summary of the samples collected so far."""
        with self._lock:
            snapshots = tuple(self._snapshots)

        elapsed_wall_s = 0.0
        if self._start_time_s is not None:
            end_time_s = self._end_time_s if self._end_time_s is not None else time.perf_counter()
            elapsed_wall_s = max(0.0, end_time_s - self._start_time_s)

        note = self._note
        if not snapshots and note is None:
            note = "No GPU telemetry samples were collected."

        by_gpu: dict[str, list[GpuTelemetrySample]] = {}
        for snapshot in snapshots:
            for sample in snapshot.gpus:
                by_gpu.setdefault(sample.index, []).append(sample)

        gpu_summaries = tuple(
            _summarize_gpu_telemetry_samples(samples) for _, samples in sorted(by_gpu.items())
        )

        return GpuTelemetrySummary(
            sample_interval_s=self.sample_interval_s,
            elapsed_wall_s=elapsed_wall_s,
            snapshots_collected=len(snapshots),
            gpus=gpu_summaries,
            note=note,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_start_s = time.perf_counter()
            with self._lock:
                gpm_rows = {
                    gpu_index: dict(row) for gpu_index, row in self._latest_gpm_rows.items()
                }
            snapshot = capture_gpu_telemetry_snapshot(
                current_pid=self._current_pid,
                gpm_rows_by_index=gpm_rows,
            )
            if snapshot.note is not None and self._note is None:
                self._note = snapshot.note
            with self._lock:
                self._snapshots.append(snapshot)
            if self.live:
                print_live_gpu_telemetry_snapshot(snapshot, prefix=f"[{self.label}] ")
            elapsed_s = time.perf_counter() - loop_start_s
            remaining_s = max(0.0, self.sample_interval_s - elapsed_s)
            self._stop_event.wait(remaining_s)

    def _run_gpm_stream(self) -> None:
        gpm_fields = [
            "gpu",
            "pwr",
            "gtemp",
            "mtemp",
            "sm",
            "mem",
            "enc",
            "dec",
            "jpg",
            "ofa",
            "mclk",
            "pclk",
            "smutil",
            "smocc",
            "mmaact",
            "dram",
            "fp64",
            "fp32",
            "fp16",
            "nvenc0",
        ]
        try:
            self._gpm_process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "dmon",
                    "--gpm-options",
                    "d",
                    "--gpm-metrics",
                    "2,3,5,10,11,12,13,166",
                    "-d",
                    "1",
                    "--format",
                    "csv,nounit,noheader",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            if self._note is None:
                self._note = str(exc)
            return

        if self._gpm_process.stdout is None:
            if self._note is None:
                self._note = "Unable to read nvidia-smi dmon output."
            return

        for line in self._gpm_process.stdout:
            if self._stop_event.is_set():
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = next(iter(_parse_nvidia_csv(stripped, gpm_fields)), None)
            if row is None:
                continue
            gpu_index = row["gpu"].strip()
            with self._lock:
                self._latest_gpm_rows[gpu_index] = row


def _is_missing_value(value: str) -> bool:
    normalized = value.strip()
    return normalized in {"", "-", "N/A", "[Not Supported]", "Not Supported"}


def create_device(requested_mode: str, wires: int) -> DeviceSelection:
    """Create the requested PennyLane device with optional fallback."""
    if requested_mode not in DEVICE_CANDIDATES:
        choices = ", ".join(sorted(DEVICE_CANDIDATES))
        raise ValueError(f"Unknown device mode {requested_mode!r}. Expected one of: {choices}")

    selection_errors: list[str] = []
    for device_name in DEVICE_CANDIDATES[requested_mode]:
        try:
            return DeviceSelection(
                requested_mode=requested_mode,
                device_name=device_name,
                device=qml.device(device_name, wires=wires),
                selection_errors=tuple(selection_errors),
            )
        except Exception as exc:
            selection_errors.append(f"{device_name}: {type(exc).__name__}: {exc}")

    error_text = "\n".join(selection_errors) or "No device candidates were configured."
    raise RuntimeError(
        f"Unable to initialize a PennyLane device for mode {requested_mode!r}.\n{error_text}"
    )


def package_version(package_name: str) -> str:
    """Return an installed package version or a readable fallback."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def format_gate_types(gate_types: dict[str, int]) -> str:
    """Format gate counts into a compact human-readable string."""
    return ", ".join(
        f"{gate_name}:{count}" for gate_name, count in sorted(gate_types.items())
    )


def _parse_nvidia_csv(output: str, fields: list[str]) -> list[dict[str, str]]:
    """Parse a CSV table returned by nvidia-smi."""
    if not output.strip():
        return []

    rows: list[dict[str, str]] = []
    for row in csv.reader(output.splitlines()):
        values = [value.strip() for value in row]
        rows.append(dict(zip(fields, values)))
    return rows


def _parse_int_or_none(value: str) -> int | None:
    """Convert a string to int when possible."""
    if _is_missing_value(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float_or_none(value: str) -> float | None:
    """Convert a string to float when possible."""
    if _is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _max_or_none(values: list[float | int | None]) -> float | int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def process_rss_mib() -> float | None:
    """Read the resident set size for the current process on Linux."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                    return rss_kib / 1024
    except (OSError, ValueError):
        return None
    return None


def _run_nvidia_smi_command(command: list[str]) -> tuple[str | None, str | None]:
    """Run one nvidia-smi command and return stdout or an error note."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, str(exc)
    return result.stdout, None


def _query_gpu_telemetry_rows() -> tuple[list[dict[str, str]], str | None]:
    """Return one-shot GPU telemetry rows from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found on PATH"

    gpu_fields = [
        "index",
        "uuid",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "utilization.encoder",
        "utilization.decoder",
        "utilization.jpeg",
        "utilization.ofa",
        "power.draw",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "pstate",
    ]
    output, note = _run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return [], note
    return _parse_nvidia_csv(output, gpu_fields), None


def _query_process_gpu_memory_rows() -> tuple[list[dict[str, str]], str | None]:
    """Return active compute-process GPU memory rows from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found on PATH"

    app_fields = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
    output, note = _run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-compute-apps={','.join(app_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return [], note
    return _parse_nvidia_csv(output, app_fields), None


def _query_gpm_rows() -> tuple[list[dict[str, str]], str | None]:
    """Return one-shot GPM rows from nvidia-smi dmon when available."""
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found on PATH"

    gpm_fields = [
        "gpu",
        "pwr",
        "gtemp",
        "mtemp",
        "sm",
        "mem",
        "enc",
        "dec",
        "jpg",
        "ofa",
        "mclk",
        "pclk",
        "smutil",
        "smocc",
        "mmaact",
        "dram",
        "fp64",
        "fp32",
        "fp16",
        "nvenc0",
    ]
    output, note = _run_nvidia_smi_command(
        [
            "nvidia-smi",
            "dmon",
            "--gpm-options",
            "d",
            "--gpm-metrics",
            "2,3,5,10,11,12,13,166",
            "-d",
            "1",
            "-c",
            "1",
            "--format",
            "csv,nounit,noheader",
        ]
    )
    if output is None:
        return [], note
    return _parse_nvidia_csv(output, gpm_fields), None


def capture_gpu_telemetry_snapshot(
    current_pid: int | None = None,
    gpm_rows_by_index: dict[str, dict[str, str]] | None = None,
) -> GpuTelemetrySnapshot:
    """Collect one merged GPU telemetry sample across base, GPM, and process-memory views."""
    current_pid = os.getpid() if current_pid is None else current_pid
    timestamp_s = time.perf_counter()

    gpu_rows, gpu_note = _query_gpu_telemetry_rows()
    process_rows, process_note = _query_process_gpu_memory_rows()
    if gpm_rows_by_index is None:
        gpm_rows, gpm_note = _query_gpm_rows()
        gpm_rows_by_index = {row["gpu"].strip(): row for row in gpm_rows}
    else:
        gpm_note = None

    notes = [note for note in (gpu_note, process_note, gpm_note) if note]
    note_text = "; ".join(notes) if notes else None

    process_mem_by_uuid: dict[str, int] = {}
    for row in process_rows:
        if _parse_int_or_none(row["pid"]) != current_pid:
            continue
        uuid = row.get("gpu_uuid", "").strip()
        used_gpu_memory = _parse_int_or_none(row["used_gpu_memory"])
        if not uuid or used_gpu_memory is None:
            continue
        process_mem_by_uuid[uuid] = process_mem_by_uuid.get(uuid, 0) + used_gpu_memory

    samples: list[GpuTelemetrySample] = []
    for row in gpu_rows:
        gpu_index = row["index"].strip()
        gpu_uuid = row["uuid"].strip() if row.get("uuid") else None
        gpm_row = gpm_rows_by_index.get(gpu_index, {})

        samples.append(
            GpuTelemetrySample(
                index=gpu_index,
                uuid=gpu_uuid,
                name=row["name"],
                memory_used_mib=_parse_int_or_none(row["memory.used"]),
                memory_total_mib=_parse_int_or_none(row["memory.total"]),
                process_used_gpu_memory_mib=process_mem_by_uuid.get(gpu_uuid or ""),
                utilization_gpu_pct=_parse_float_or_none(row["utilization.gpu"]),
                utilization_memory_pct=_parse_float_or_none(row["utilization.memory"]),
                utilization_encoder_pct=_parse_float_or_none(row["utilization.encoder"]),
                utilization_decoder_pct=_parse_float_or_none(row["utilization.decoder"]),
                utilization_jpeg_pct=_parse_float_or_none(row["utilization.jpeg"]),
                utilization_ofa_pct=_parse_float_or_none(row["utilization.ofa"]),
                power_draw_w=_parse_float_or_none(row["power.draw"]),
                temperature_c=_parse_float_or_none(row["temperature.gpu"]),
                clocks_sm_mhz=_parse_float_or_none(row["clocks.sm"]),
                clocks_mem_mhz=_parse_float_or_none(row["clocks.mem"]),
                pstate=None if _is_missing_value(row["pstate"]) else row["pstate"].strip(),
                sm_activity_pct=_parse_float_or_none(gpm_row.get("smutil", "")),
                sm_occupancy_pct=_parse_float_or_none(gpm_row.get("smocc", "")),
                tensor_activity_pct=_parse_float_or_none(gpm_row.get("mmaact", "")),
                dram_activity_pct=_parse_float_or_none(gpm_row.get("dram", "")),
                fp64_activity_pct=_parse_float_or_none(gpm_row.get("fp64", "")),
                fp32_activity_pct=_parse_float_or_none(gpm_row.get("fp32", "")),
                fp16_activity_pct=_parse_float_or_none(gpm_row.get("fp16", "")),
            )
        )

    return GpuTelemetrySnapshot(
        timestamp_s=timestamp_s,
        gpus=tuple(samples),
        note=note_text,
    )


def query_nvidia_smi() -> tuple[list[dict[str, str]], list[dict[str, str]], str | None]:
    """Return GPU and compute-process snapshots from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return [], [], "nvidia-smi not found on PATH"

    gpu_fields = [
        "index",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "temperature.gpu",
    ]
    app_fields = ["pid", "process_name", "used_gpu_memory"]

    gpu_output, gpu_note = _run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    app_output, app_note = _run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-compute-apps={','.join(app_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_output is None or app_output is None:
        note = "; ".join(note for note in (gpu_note, app_note) if note)
        return [], [], note

    return (
        _parse_nvidia_csv(gpu_output, gpu_fields),
        _parse_nvidia_csv(app_output, app_fields),
        None,
    )


def capture_resource_snapshot(label: str) -> ResourceSnapshot:
    """Capture CPU RSS and the current process's visible GPU usage."""
    gpu_rows, app_rows, note = query_nvidia_smi()
    current_pid = os.getpid()

    gpus = tuple(
        GpuSnapshot(
            index=row["index"],
            name=row["name"],
            memory_used_mib=_parse_int_or_none(row["memory.used"]),
            memory_total_mib=_parse_int_or_none(row["memory.total"]),
            utilization_gpu_pct=_parse_int_or_none(row["utilization.gpu"]),
            temperature_c=_parse_int_or_none(row["temperature.gpu"]),
        )
        for row in gpu_rows
    )
    active_compute_processes = tuple(
        ComputeProcessSnapshot(
            pid=int(row["pid"]),
            process_name=row["process_name"],
            used_gpu_memory_mib=_parse_int_or_none(row["used_gpu_memory"]),
        )
        for row in app_rows
        if _parse_int_or_none(row["pid"]) == current_pid
    )

    return ResourceSnapshot(
        label=label,
        process_rss_mib=process_rss_mib(),
        gpus=gpus,
        active_compute_processes=active_compute_processes,
        note=note,
    )


def _summarize_gpu_telemetry_samples(
    samples: list[GpuTelemetrySample],
) -> GpuTelemetryGpuSummary:
    sample0 = samples[0]
    return GpuTelemetryGpuSummary(
        index=sample0.index,
        name=sample0.name,
        uuid=sample0.uuid,
        sample_count=len(samples),
        fb_memory_total_mib=sample0.memory_total_mib,
        avg_utilization_gpu_pct=_mean_or_none([sample.utilization_gpu_pct for sample in samples]),
        peak_utilization_gpu_pct=_max_or_none([sample.utilization_gpu_pct for sample in samples]),
        avg_utilization_memory_pct=_mean_or_none(
            [sample.utilization_memory_pct for sample in samples]
        ),
        peak_utilization_memory_pct=_max_or_none(
            [sample.utilization_memory_pct for sample in samples]
        ),
        avg_utilization_encoder_pct=_mean_or_none(
            [sample.utilization_encoder_pct for sample in samples]
        ),
        peak_utilization_encoder_pct=_max_or_none(
            [sample.utilization_encoder_pct for sample in samples]
        ),
        avg_utilization_decoder_pct=_mean_or_none(
            [sample.utilization_decoder_pct for sample in samples]
        ),
        peak_utilization_decoder_pct=_max_or_none(
            [sample.utilization_decoder_pct for sample in samples]
        ),
        avg_utilization_jpeg_pct=_mean_or_none(
            [sample.utilization_jpeg_pct for sample in samples]
        ),
        peak_utilization_jpeg_pct=_max_or_none(
            [sample.utilization_jpeg_pct for sample in samples]
        ),
        avg_utilization_ofa_pct=_mean_or_none(
            [sample.utilization_ofa_pct for sample in samples]
        ),
        peak_utilization_ofa_pct=_max_or_none([sample.utilization_ofa_pct for sample in samples]),
        avg_power_draw_w=_mean_or_none([sample.power_draw_w for sample in samples]),
        peak_power_draw_w=_max_or_none([sample.power_draw_w for sample in samples]),
        peak_temperature_c=_max_or_none([sample.temperature_c for sample in samples]),
        peak_fb_memory_used_mib=_max_or_none([sample.memory_used_mib for sample in samples]),
        peak_process_used_gpu_memory_mib=_max_or_none(
            [sample.process_used_gpu_memory_mib for sample in samples]
        ),
        peak_clocks_sm_mhz=_max_or_none([sample.clocks_sm_mhz for sample in samples]),
        peak_clocks_mem_mhz=_max_or_none([sample.clocks_mem_mhz for sample in samples]),
        avg_sm_activity_pct=_mean_or_none([sample.sm_activity_pct for sample in samples]),
        peak_sm_activity_pct=_max_or_none([sample.sm_activity_pct for sample in samples]),
        avg_sm_occupancy_pct=_mean_or_none([sample.sm_occupancy_pct for sample in samples]),
        peak_sm_occupancy_pct=_max_or_none([sample.sm_occupancy_pct for sample in samples]),
        avg_tensor_activity_pct=_mean_or_none([sample.tensor_activity_pct for sample in samples]),
        peak_tensor_activity_pct=_max_or_none([sample.tensor_activity_pct for sample in samples]),
        avg_dram_activity_pct=_mean_or_none([sample.dram_activity_pct for sample in samples]),
        peak_dram_activity_pct=_max_or_none([sample.dram_activity_pct for sample in samples]),
        avg_fp64_activity_pct=_mean_or_none([sample.fp64_activity_pct for sample in samples]),
        peak_fp64_activity_pct=_max_or_none([sample.fp64_activity_pct for sample in samples]),
        avg_fp32_activity_pct=_mean_or_none([sample.fp32_activity_pct for sample in samples]),
        peak_fp32_activity_pct=_max_or_none([sample.fp32_activity_pct for sample in samples]),
        avg_fp16_activity_pct=_mean_or_none([sample.fp16_activity_pct for sample in samples]),
        peak_fp16_activity_pct=_max_or_none([sample.fp16_activity_pct for sample in samples]),
    )


def _format_ratio_or_unknown(numerator: float | int | None, denominator: float | int | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "?"
    return f"{numerator}/{denominator}"


def _format_float_metric(value: float | int | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "?"
    if isinstance(value, int):
        return f"{value}{unit}"
    return f"{value:.{digits}f}{unit}"


def print_live_gpu_telemetry_snapshot(
    snapshot: GpuTelemetrySnapshot,
    prefix: str = "",
) -> None:
    """Print one concise live telemetry line per GPU sample."""
    if snapshot.note is not None and not snapshot.gpus:
        print(f"{prefix}GPU telemetry unavailable ({snapshot.note})")
        return

    for gpu in snapshot.gpus:
        print(
            f"{prefix}GPU {gpu.index} {gpu.name}: "
            f"fb={_format_ratio_or_unknown(gpu.memory_used_mib, gpu.memory_total_mib)} MiB, "
            f"proc_fb={_format_float_metric(gpu.process_used_gpu_memory_mib, ' MiB', digits=0)}, "
            f"gpu={_format_float_metric(gpu.utilization_gpu_pct, '%')}, "
            f"memctl={_format_float_metric(gpu.utilization_memory_pct, '%')}, "
            f"smact={_format_float_metric(gpu.sm_activity_pct, '%')}, "
            f"dram={_format_float_metric(gpu.dram_activity_pct, '%')}, "
            f"fp32={_format_float_metric(gpu.fp32_activity_pct, '%')}, "
            f"fp16={_format_float_metric(gpu.fp16_activity_pct, '%')}, "
            f"tensor={_format_float_metric(gpu.tensor_activity_pct, '%')}, "
            f"pwr={_format_float_metric(gpu.power_draw_w, ' W')}, "
            f"temp={_format_float_metric(gpu.temperature_c, ' C')}"
        )


def print_gpu_telemetry_summary(
    summary: GpuTelemetrySummary,
    indent: str = "",
) -> None:
    """Print a human-readable summary of sampled GPU telemetry."""
    print(f"{indent}GPU telemetry summary:")
    print(
        f"{indent}  Sample interval: {summary.sample_interval_s:.2f} s, "
        f"snapshots: {summary.snapshots_collected}, "
        f"elapsed wall time: {summary.elapsed_wall_s:.2f} s"
    )
    if summary.note is not None:
        print(f"{indent}  Note: {summary.note}")

    if not summary.gpus:
        return

    for gpu in summary.gpus:
        has_detailed_gpm = any(
            value is not None
            for value in (
                gpu.avg_sm_activity_pct,
                gpu.peak_sm_activity_pct,
                gpu.avg_sm_occupancy_pct,
                gpu.peak_sm_occupancy_pct,
                gpu.avg_tensor_activity_pct,
                gpu.peak_tensor_activity_pct,
                gpu.avg_dram_activity_pct,
                gpu.peak_dram_activity_pct,
                gpu.avg_fp64_activity_pct,
                gpu.peak_fp64_activity_pct,
                gpu.avg_fp32_activity_pct,
                gpu.peak_fp32_activity_pct,
                gpu.avg_fp16_activity_pct,
                gpu.peak_fp16_activity_pct,
            )
        )
        print(f"{indent}  GPU {gpu.index} {gpu.name}:")
        print(
            f"{indent}    Board util avg/peak: "
            f"{_format_float_metric(gpu.avg_utilization_gpu_pct, '%')} / "
            f"{_format_float_metric(gpu.peak_utilization_gpu_pct, '%')}"
        )
        print(
            f"{indent}    Mem util avg/peak: "
            f"{_format_float_metric(gpu.avg_utilization_memory_pct, '%')} / "
            f"{_format_float_metric(gpu.peak_utilization_memory_pct, '%')}"
        )
        print(
            f"{indent}    FB memory peak: "
            f"{_format_ratio_or_unknown(gpu.peak_fb_memory_used_mib, gpu.fb_memory_total_mib)} MiB, "
            f"process peak: {_format_float_metric(gpu.peak_process_used_gpu_memory_mib, ' MiB', digits=0)}"
        )
        print(
            f"{indent}    Power avg/peak: "
            f"{_format_float_metric(gpu.avg_power_draw_w, ' W')} / "
            f"{_format_float_metric(gpu.peak_power_draw_w, ' W')}, "
            f"temp peak: {_format_float_metric(gpu.peak_temperature_c, ' C')}"
        )
        if has_detailed_gpm:
            print(
                f"{indent}    SM activity avg/peak: "
                f"{_format_float_metric(gpu.avg_sm_activity_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_sm_activity_pct, '%')}, "
                f"SM occupancy avg/peak: "
                f"{_format_float_metric(gpu.avg_sm_occupancy_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_sm_occupancy_pct, '%')}"
            )
            print(
                f"{indent}    DRAM avg/peak: "
                f"{_format_float_metric(gpu.avg_dram_activity_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_dram_activity_pct, '%')}, "
                f"Tensor avg/peak: "
                f"{_format_float_metric(gpu.avg_tensor_activity_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_tensor_activity_pct, '%')}"
            )
            print(
                f"{indent}    FP32 avg/peak: "
                f"{_format_float_metric(gpu.avg_fp32_activity_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_fp32_activity_pct, '%')}, "
                f"FP16 avg/peak: "
                f"{_format_float_metric(gpu.avg_fp16_activity_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_fp16_activity_pct, '%')}, "
                f"FP64 avg/peak: "
                f"{_format_float_metric(gpu.avg_fp64_activity_pct, '%')} / "
                f"{_format_float_metric(gpu.peak_fp64_activity_pct, '%')}"
            )
        else:
            print(
                f"{indent}    Detailed GPM metrics: unavailable from nvidia-smi dmon "
                "on this device/driver for the sampled window."
            )
        print(
            f"{indent}    Encoder/decoder avg/peak: "
            f"enc {_format_float_metric(gpu.avg_utilization_encoder_pct, '%')}/"
            f"{_format_float_metric(gpu.peak_utilization_encoder_pct, '%')}, "
            f"dec {_format_float_metric(gpu.avg_utilization_decoder_pct, '%')}/"
            f"{_format_float_metric(gpu.peak_utilization_decoder_pct, '%')}, "
            f"jpg {_format_float_metric(gpu.avg_utilization_jpeg_pct, '%')}/"
            f"{_format_float_metric(gpu.peak_utilization_jpeg_pct, '%')}, "
            f"ofa {_format_float_metric(gpu.avg_utilization_ofa_pct, '%')}/"
            f"{_format_float_metric(gpu.peak_utilization_ofa_pct, '%')}"
        )
        print(
            f"{indent}    Clock peaks: "
            f"SM {_format_float_metric(gpu.peak_clocks_sm_mhz, ' MHz')}, "
            f"MEM {_format_float_metric(gpu.peak_clocks_mem_mhz, ' MHz')}"
        )


def print_resource_snapshot(snapshot: ResourceSnapshot) -> None:
    """Print a compact resource snapshot."""
    print(f"{snapshot.label}:")

    if snapshot.process_rss_mib is None:
        print("  Process RSS: unavailable")
    else:
        print(f"  Process RSS: {snapshot.process_rss_mib:.1f} MiB")

    if snapshot.note is not None:
        print(f"  GPU snapshot: unavailable ({snapshot.note})")
        print()
        return

    if not snapshot.gpus:
        print("  GPU snapshot: no devices reported")
        print()
        return

    for gpu in snapshot.gpus:
        memory_text = "unavailable"
        if gpu.memory_used_mib is not None and gpu.memory_total_mib is not None:
            memory_text = f"{gpu.memory_used_mib}/{gpu.memory_total_mib} MiB"

        util_text = "?"
        if gpu.utilization_gpu_pct is not None:
            util_text = f"{gpu.utilization_gpu_pct}%"

        temp_text = "?"
        if gpu.temperature_c is not None:
            temp_text = f"{gpu.temperature_c} C"

        print(
            f"  GPU {gpu.index} {gpu.name}: "
            f"mem {memory_text}, util {util_text}, temp {temp_text}"
        )

    if snapshot.active_compute_processes:
        for process in snapshot.active_compute_processes:
            gpu_mem_text = "unavailable"
            if process.used_gpu_memory_mib is not None:
                gpu_mem_text = f"{process.used_gpu_memory_mib} MiB"
            print(
                "  Active compute process: "
                f"pid={process.pid}, name={process.process_name}, gpu_mem={gpu_mem_text}"
            )
    else:
        print("  Active compute process: this Python process is not listed right now")

    print()
