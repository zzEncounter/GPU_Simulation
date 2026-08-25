"""Low-overhead memory snapshots kept outside the timed regions.

Identical in structure to the pennylane-lightning baseline version.
"""

from __future__ import annotations

import os
import resource
import subprocess
from dataclasses import dataclass

MIB = 1024 * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    gpu_process_used_mib: float | None
    host_rss_mib: float
    host_peak_rss_mib: float


def _host_rss_mib() -> float:
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / MIB
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _host_peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value > 10 * MIB:
        return value / MIB
    return value / 1024.0


def _gpu_memory_from_nvml() -> float | None:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        current_pid = os.getpid()
        total_bytes = 0
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError_NotSupported:
                continue
            for process in processes:
                if process.pid == current_pid and process.usedGpuMemory is not None:
                    total_bytes += int(process.usedGpuMemory)
        return total_bytes / MIB
    except Exception:  # noqa: BLE001
        return None


def _gpu_memory_from_nvidia_smi() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    current_pid = str(os.getpid())
    total_mib = 0.0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[0] != current_pid:
            continue
        try:
            total_mib += float(fields[1])
            found = True
        except ValueError:
            continue
    return total_mib if found else 0.0


def take_memory_snapshot() -> MemorySnapshot:
    gpu_used = _gpu_memory_from_nvml()
    if gpu_used is None:
        gpu_used = _gpu_memory_from_nvidia_smi()
    host_rss = _host_rss_mib()
    return MemorySnapshot(
        gpu_process_used_mib=gpu_used,
        host_rss_mib=host_rss,
        host_peak_rss_mib=max(host_rss, _host_peak_rss_mib()),
    )
