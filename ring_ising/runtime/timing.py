"""Small timing helpers shared by benchmarks and scripts."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

from .telemetry import GpuTelemetryMonitor, GpuTelemetrySummary


def median_runtime_ms(
    fn: Callable[[], Any],
    repeats: int,
    warmup: int,
    *,
    synchronize: Callable[[], None] | None = None,
) -> float:
    """Return the median runtime of a callable in milliseconds."""
    for _ in range(warmup):
        fn()
        if synchronize is not None:
            synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def median_timing_fields(
    fn: Callable[[], dict[str, float | int]],
    field_names: Sequence[str],
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    """Return median values for a fixed set of numeric timing fields."""
    for _ in range(warmup):
        fn()

    samples: dict[str, list[float]] = {field_name: [] for field_name in field_names}
    for _ in range(repeats):
        raw = fn()
        for field_name in field_names:
            samples[field_name].append(float(raw[field_name]))
    return {
        field_name: statistics.median(field_samples)
        for field_name, field_samples in samples.items()
    }

def capture_telemetry_window(
    fn: Callable[[], Any],
    *,
    target_ms: float,
    measured_ms: float,
    interval_s: float,
) -> GpuTelemetrySummary:
    """Sample GPU telemetry while repeating a callable for a target wall-time window."""
    repeats = 1
    if measured_ms > 0:
        repeats = max(1, min(100, int(math.ceil(target_ms / measured_ms))))
    monitor = GpuTelemetryMonitor(sample_interval_s=interval_s, live=False)
    monitor.start()
    for _ in range(repeats):
        fn()
    return monitor.stop()
