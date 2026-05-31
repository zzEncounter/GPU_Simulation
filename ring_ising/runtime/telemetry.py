"""GPU telemetry sampling and summary helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from ._nvidia_smi import (
    is_missing_value,
    max_or_none,
    mean_or_none,
    parse_float_or_none,
    parse_int_or_none,
    parse_nvidia_csv,
    query_gpu_telemetry_rows,
    query_gpm_rows,
    query_process_gpu_memory_rows,
)


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
            row = next(iter(parse_nvidia_csv(stripped, gpm_fields)), None)
            if row is None:
                continue
            gpu_index = row["gpu"].strip()
            with self._lock:
                self._latest_gpm_rows[gpu_index] = row


def capture_gpu_telemetry_snapshot(
    current_pid: int | None = None,
    gpm_rows_by_index: dict[str, dict[str, str]] | None = None,
) -> GpuTelemetrySnapshot:
    """Collect one merged GPU telemetry sample across base, GPM, and process-memory views."""
    current_pid = os.getpid() if current_pid is None else current_pid
    timestamp_s = time.perf_counter()

    gpu_rows, gpu_note = query_gpu_telemetry_rows()
    process_rows, process_note = query_process_gpu_memory_rows()
    if gpm_rows_by_index is None:
        gpm_rows, gpm_note = query_gpm_rows()
        gpm_rows_by_index = {row["gpu"].strip(): row for row in gpm_rows}
    else:
        gpm_note = None

    notes = [note for note in (gpu_note, process_note, gpm_note) if note]
    note_text = "; ".join(notes) if notes else None

    process_mem_by_uuid: dict[str, int] = {}
    for row in process_rows:
        if parse_int_or_none(row["pid"]) != current_pid:
            continue
        uuid = row.get("gpu_uuid", "").strip()
        used_gpu_memory = parse_int_or_none(row["used_gpu_memory"])
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
                memory_used_mib=parse_int_or_none(row["memory.used"]),
                memory_total_mib=parse_int_or_none(row["memory.total"]),
                process_used_gpu_memory_mib=process_mem_by_uuid.get(gpu_uuid or ""),
                utilization_gpu_pct=parse_float_or_none(row["utilization.gpu"]),
                utilization_memory_pct=parse_float_or_none(row["utilization.memory"]),
                utilization_encoder_pct=parse_float_or_none(row["utilization.encoder"]),
                utilization_decoder_pct=parse_float_or_none(row["utilization.decoder"]),
                utilization_jpeg_pct=parse_float_or_none(row["utilization.jpeg"]),
                utilization_ofa_pct=parse_float_or_none(row["utilization.ofa"]),
                power_draw_w=parse_float_or_none(row["power.draw"]),
                temperature_c=parse_float_or_none(row["temperature.gpu"]),
                clocks_sm_mhz=parse_float_or_none(row["clocks.sm"]),
                clocks_mem_mhz=parse_float_or_none(row["clocks.mem"]),
                pstate=None if is_missing_value(row["pstate"]) else row["pstate"].strip(),
                sm_activity_pct=parse_float_or_none(gpm_row.get("smutil", "")),
                sm_occupancy_pct=parse_float_or_none(gpm_row.get("smocc", "")),
                tensor_activity_pct=parse_float_or_none(gpm_row.get("mmaact", "")),
                dram_activity_pct=parse_float_or_none(gpm_row.get("dram", "")),
                fp64_activity_pct=parse_float_or_none(gpm_row.get("fp64", "")),
                fp32_activity_pct=parse_float_or_none(gpm_row.get("fp32", "")),
                fp16_activity_pct=parse_float_or_none(gpm_row.get("fp16", "")),
            )
        )

    return GpuTelemetrySnapshot(
        timestamp_s=timestamp_s,
        gpus=tuple(samples),
        note=note_text,
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
        avg_utilization_gpu_pct=mean_or_none([sample.utilization_gpu_pct for sample in samples]),
        peak_utilization_gpu_pct=max_or_none([sample.utilization_gpu_pct for sample in samples]),
        avg_utilization_memory_pct=mean_or_none(
            [sample.utilization_memory_pct for sample in samples]
        ),
        peak_utilization_memory_pct=max_or_none(
            [sample.utilization_memory_pct for sample in samples]
        ),
        avg_utilization_encoder_pct=mean_or_none(
            [sample.utilization_encoder_pct for sample in samples]
        ),
        peak_utilization_encoder_pct=max_or_none(
            [sample.utilization_encoder_pct for sample in samples]
        ),
        avg_utilization_decoder_pct=mean_or_none(
            [sample.utilization_decoder_pct for sample in samples]
        ),
        peak_utilization_decoder_pct=max_or_none(
            [sample.utilization_decoder_pct for sample in samples]
        ),
        avg_utilization_jpeg_pct=mean_or_none(
            [sample.utilization_jpeg_pct for sample in samples]
        ),
        peak_utilization_jpeg_pct=max_or_none(
            [sample.utilization_jpeg_pct for sample in samples]
        ),
        avg_utilization_ofa_pct=mean_or_none(
            [sample.utilization_ofa_pct for sample in samples]
        ),
        peak_utilization_ofa_pct=max_or_none([sample.utilization_ofa_pct for sample in samples]),
        avg_power_draw_w=mean_or_none([sample.power_draw_w for sample in samples]),
        peak_power_draw_w=max_or_none([sample.power_draw_w for sample in samples]),
        peak_temperature_c=max_or_none([sample.temperature_c for sample in samples]),
        peak_fb_memory_used_mib=max_or_none([sample.memory_used_mib for sample in samples]),
        peak_process_used_gpu_memory_mib=max_or_none(
            [sample.process_used_gpu_memory_mib for sample in samples]
        ),
        peak_clocks_sm_mhz=max_or_none([sample.clocks_sm_mhz for sample in samples]),
        peak_clocks_mem_mhz=max_or_none([sample.clocks_mem_mhz for sample in samples]),
        avg_sm_activity_pct=mean_or_none([sample.sm_activity_pct for sample in samples]),
        peak_sm_activity_pct=max_or_none([sample.sm_activity_pct for sample in samples]),
        avg_sm_occupancy_pct=mean_or_none([sample.sm_occupancy_pct for sample in samples]),
        peak_sm_occupancy_pct=max_or_none([sample.sm_occupancy_pct for sample in samples]),
        avg_tensor_activity_pct=mean_or_none([sample.tensor_activity_pct for sample in samples]),
        peak_tensor_activity_pct=max_or_none([sample.tensor_activity_pct for sample in samples]),
        avg_dram_activity_pct=mean_or_none([sample.dram_activity_pct for sample in samples]),
        peak_dram_activity_pct=max_or_none([sample.dram_activity_pct for sample in samples]),
        avg_fp64_activity_pct=mean_or_none([sample.fp64_activity_pct for sample in samples]),
        peak_fp64_activity_pct=max_or_none([sample.fp64_activity_pct for sample in samples]),
        avg_fp32_activity_pct=mean_or_none([sample.fp32_activity_pct for sample in samples]),
        peak_fp32_activity_pct=max_or_none([sample.fp32_activity_pct for sample in samples]),
        avg_fp16_activity_pct=mean_or_none([sample.fp16_activity_pct for sample in samples]),
        peak_fp16_activity_pct=max_or_none([sample.fp16_activity_pct for sample in samples]),
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
