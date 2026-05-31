"""Low-level helpers for parsing and querying nvidia-smi output."""

from __future__ import annotations

import csv
import shutil
import subprocess


def is_missing_value(value: str) -> bool:
    """Return whether a raw nvidia-smi field should be treated as missing."""
    normalized = value.strip()
    return normalized in {"", "-", "N/A", "[Not Supported]", "Not Supported"}


def parse_nvidia_csv(output: str, fields: list[str]) -> list[dict[str, str]]:
    """Parse a CSV table returned by nvidia-smi."""
    if not output.strip():
        return []

    rows: list[dict[str, str]] = []
    for row in csv.reader(output.splitlines()):
        values = [value.strip() for value in row]
        rows.append(dict(zip(fields, values)))
    return rows


def parse_int_or_none(value: str) -> int | None:
    """Convert a string to int when possible."""
    if is_missing_value(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float_or_none(value: str) -> float | None:
    """Convert a string to float when possible."""
    if is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float | int | None]) -> float | None:
    """Return the mean of the non-missing values."""
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def max_or_none(values: list[float | int | None]) -> float | int | None:
    """Return the max of the non-missing values."""
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def run_nvidia_smi_command(command: list[str]) -> tuple[str | None, str | None]:
    """Run one nvidia-smi command and return stdout or an error note."""
    executable = command[0]
    if shutil.which(executable) is None:
        return None, f"{executable} not found on PATH"
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


def query_gpu_telemetry_rows() -> tuple[list[dict[str, str]], str | None]:
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
    output, note = run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return [], note
    return parse_nvidia_csv(output, gpu_fields), None


def query_process_gpu_memory_rows() -> tuple[list[dict[str, str]], str | None]:
    """Return active compute-process GPU memory rows from nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found on PATH"

    app_fields = ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
    output, note = run_nvidia_smi_command(
        [
            "nvidia-smi",
            f"--query-compute-apps={','.join(app_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return [], note
    return parse_nvidia_csv(output, app_fields), None


def query_gpm_rows() -> tuple[list[dict[str, str]], str | None]:
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
    output, note = run_nvidia_smi_command(
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
    return parse_nvidia_csv(output, gpm_fields), None
