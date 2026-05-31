"""CPU/GPU snapshot helpers for one process."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ._nvidia_smi import parse_int_or_none, run_nvidia_smi_command, parse_nvidia_csv


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


def query_nvidia_smi() -> tuple[list[dict[str, str]], list[dict[str, str]], str | None]:
    """Return GPU and compute-process snapshots from nvidia-smi."""
    gpu_fields = [
        "index",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "temperature.gpu",
    ]
    app_fields = ["pid", "process_name", "used_gpu_memory"]

    gpu_output, gpu_note = run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    app_output, app_note = run_nvidia_smi_command(
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
        parse_nvidia_csv(gpu_output, gpu_fields),
        parse_nvidia_csv(app_output, app_fields),
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
            memory_used_mib=parse_int_or_none(row["memory.used"]),
            memory_total_mib=parse_int_or_none(row["memory.total"]),
            utilization_gpu_pct=parse_int_or_none(row["utilization.gpu"]),
            temperature_c=parse_int_or_none(row["temperature.gpu"]),
        )
        for row in gpu_rows
    )
    active_compute_processes = tuple(
        ComputeProcessSnapshot(
            pid=int(row["pid"]),
            process_name=row["process_name"],
            used_gpu_memory_mib=parse_int_or_none(row["used_gpu_memory"]),
        )
        for row in app_rows
        if parse_int_or_none(row["pid"]) == current_pid
    )

    return ResourceSnapshot(
        label=label,
        process_rss_mib=process_rss_mib(),
        gpus=gpus,
        active_compute_processes=active_compute_processes,
        note=note,
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
