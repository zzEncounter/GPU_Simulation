"""Runtime helpers for device selection, telemetry, and timing."""

from .device import DEVICE_CANDIDATES, DeviceSelection, create_device
from .meta import format_gate_types, package_version
from .telemetry import (
    GpuTelemetryGpuSummary,
    GpuTelemetryMonitor,
    GpuTelemetrySample,
    GpuTelemetrySnapshot,
    GpuTelemetrySummary,
    capture_gpu_telemetry_snapshot,
    print_gpu_telemetry_summary,
    print_live_gpu_telemetry_snapshot,
)
from .timing import (
    capture_telemetry_window,
    median_runtime_ms,
    median_timing_fields,
)

__all__ = [
    "DEVICE_CANDIDATES",
    "DeviceSelection",
    "GpuTelemetryGpuSummary",
    "GpuTelemetryMonitor",
    "GpuTelemetrySample",
    "GpuTelemetrySnapshot",
    "GpuTelemetrySummary",
    "capture_telemetry_window",
    "create_device",
    "format_gate_types",
    "median_runtime_ms",
    "median_timing_fields",
    "package_version",
    "print_gpu_telemetry_summary",
    "print_live_gpu_telemetry_snapshot",
]
