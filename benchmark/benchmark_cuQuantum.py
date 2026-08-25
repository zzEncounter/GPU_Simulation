"""Batch benchmark for the correctness-first cuQuantum inverse-walk backend.

Edit the global constants below to select the circuit and sweep range.  The
layout intentionally matches ``benchmark_pennylane_lightning.py``.
"""

from __future__ import annotations

import csv
import gc
import json
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cuQuantum" / "python"))

from sad_cuquantum.runner import (  # noqa: E402
    expected_parameter_count,
    random_parameters,
    run,
)

# ---------------------------------------------------------------------------
# Benchmark configuration (edit here; no argparse by design)
# ---------------------------------------------------------------------------
# Select one circuit, or several circuits, by editing this tuple.
# The eight primary SAD circuits (qaoa-ns is benchmarked separately).
CIRCUITS = (
    "ra-hea",
    "su2-hea",
    "rzz-hea",
    "qaoa",
    "qaoa-ns",
    "xxz-hva",
    "mera",
    "equivariant-qnn",
    "data-reuploading",
)
QUBITS = tuple(range(4, 29, 2))
LAYERS = 8
RANDOM_SEED = 42
BATCHES = 1
PRECISION = "float64"
WARMUP_STEPS = 1
BACKEND_NAME = "custatevec-inverse-walk"
OUTPUT_CSV = Path(__file__).resolve().parent / "results" / "cuquantum.csv"
OVERWRITE_OUTPUT = True

# Match benchmark_pennylane_lightning.py: larger state vectors get fewer
# measured repetitions so the full sweep remains practical.
STEP_SCHEDULE = (
    (8, 20),
    (14, 10),
    (20, 5),
    (24, 3),
    (28, 2),
)

CSV_FIELDS = (
    "timestamp_utc",
    "status",
    "backend",
    "circuit",
    "qubits",
    "layers",
    "parameter_count",
    "precision",
    "random_seed",
    "batches",
    "warmup_steps",
    "steps",
    "energy",
    "grad_json",
    "grad_l2_norm",
    "grad_max_abs",
    "time_mean_s",
    "time_median_s",
    "time_std_s",
    "time_min_s",
    "time_max_s",
    "step_times_s_json",
    "error",
)


def layers_for_circuit(circuit: str, qubits: int) -> int:
    """MERA's layer count is fixed by its topology, as in SAD."""

    return (qubits - 1).bit_length() if circuit == "mera" else LAYERS


def steps_for_qubits(qubits: int) -> int:
    for maximum_qubits, steps in STEP_SCHEDULE:
        if qubits <= maximum_qubits:
            return steps
    return STEP_SCHEDULE[-1][1]


def _empty_row(
    circuit: str, qubits: int, layers: int, steps: int
) -> dict[str, object]:
    return {field: "" for field in CSV_FIELDS} | {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "backend": BACKEND_NAME,
        "circuit": circuit,
        "qubits": qubits,
        "layers": layers,
        "parameter_count": expected_parameter_count(circuit, qubits, layers),
        "precision": PRECISION,
        "random_seed": RANDOM_SEED,
        "batches": BATCHES,
        "warmup_steps": WARMUP_STEPS,
        "steps": steps,
    }


def _success_row(
    circuit: str,
    qubits: int,
    layers: int,
    result: dict[str, object],
    step_times: list[float],
) -> dict[str, object]:
    gradient = np.asarray(result["grad"], dtype=float)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "backend": BACKEND_NAME,
        "circuit": circuit,
        "qubits": qubits,
        "layers": layers,
        "parameter_count": len(gradient),
        "precision": PRECISION,
        "random_seed": RANDOM_SEED,
        "batches": BATCHES,
        "warmup_steps": WARMUP_STEPS,
        "steps": len(step_times),
        "energy": result["energy"],
        "grad_json": json.dumps(gradient.tolist(), separators=(",", ":")),
        "grad_l2_norm": float(np.linalg.norm(gradient)),
        "grad_max_abs": float(np.max(np.abs(gradient))),
        "time_mean_s": statistics.fmean(step_times),
        "time_median_s": statistics.median(step_times),
        "time_std_s": statistics.pstdev(step_times),
        "time_min_s": min(step_times),
        "time_max_s": max(step_times),
        "step_times_s_json": json.dumps(step_times),
        "error": "",
    }


def _run_case(
    circuit: str, qubits: int, layers: int, steps: int
) -> tuple[dict[str, object], list[float]]:
    params = random_parameters(circuit, qubits, layers, RANDOM_SEED)

    for _ in range(WARMUP_STEPS):
        run(circuit, qubits, layers, params, PRECISION)

    times: list[float] = []
    result: dict[str, object] | None = None
    for _ in range(steps):
        started = time.perf_counter()
        result = run(circuit, qubits, layers, params, PRECISION)
        times.append(time.perf_counter() - started)
    assert result is not None
    return result, times


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if OVERWRITE_OUTPUT else "a"
    needs_header = (
        OVERWRITE_OUTPUT or not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    )

    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
            stream.flush()

        total = len(CIRCUITS) * len(QUBITS)
        run_index = 0
        for circuit in CIRCUITS:
            for qubits in QUBITS:
                run_index += 1
                layers = layers_for_circuit(circuit, qubits)
                steps = steps_for_qubits(qubits)
                label = f"[{run_index:02d}/{total:02d}] {circuit} {qubits}q x {layers}l"
                print(f"{label}: {steps} measured steps", flush=True)
                row = _empty_row(circuit, qubits, layers, steps)
                try:
                    result, times = _run_case(circuit, qubits, layers, steps)
                    row = _success_row(circuit, qubits, layers, result, times)
                    print(
                        f"{label}: median={statistics.median(times):.6f}s, "
                        f"energy={float(result['energy']):.10g}, "
                        f"grad_max_abs={row['grad_max_abs']:.6g}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - keep sweeps recoverable.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"{label}: ERROR: {row['error']}", file=sys.stderr, flush=True
                    )
                    traceback.print_exc()
                finally:
                    writer.writerow(row)
                    stream.flush()
                    gc.collect()

    print(f"CSV written to {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
