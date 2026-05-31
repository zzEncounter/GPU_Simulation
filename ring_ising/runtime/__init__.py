"""Runtime helpers for device selection, snapshots, telemetry, and timing."""

from .device import DEVICE_CANDIDATES, DeviceSelection, create_device
from .meta import format_gate_types, package_version
from .resources import (
    ComputeProcessSnapshot,
    GpuSnapshot,
    ResourceSnapshot,
    capture_resource_snapshot,
    print_resource_snapshot,
    process_rss_mib,
    query_nvidia_smi,
)
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
    derive_timing_breakdown,
    median_runtime_ms,
    median_timing_fields,
)

__all__ = [
    "DEVICE_CANDIDATES",
    "ComputeProcessSnapshot",
    "DeviceSelection",
    "GpuSnapshot",
    "GpuTelemetryGpuSummary",
    "GpuTelemetryMonitor",
    "GpuTelemetrySample",
    "GpuTelemetrySnapshot",
    "GpuTelemetrySummary",
    "ResourceSnapshot",
    "capture_gpu_telemetry_snapshot",
    "capture_resource_snapshot",
    "capture_telemetry_window",
    "create_device",
    "derive_timing_breakdown",
    "format_gate_types",
    "median_runtime_ms",
    "median_timing_fields",
    "package_version",
    "print_gpu_telemetry_summary",
    "print_live_gpu_telemetry_snapshot",
    "print_resource_snapshot",
    "process_rss_mib",
    "query_nvidia_smi",
]
