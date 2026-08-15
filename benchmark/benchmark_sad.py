"""Batch benchmark for the custom SAD CUDA adjoint implementation.

Edit the global constants below; this script intentionally has no CLI arguments.
"""

from __future__ import annotations

import csv
import gc
import importlib
import json
import os
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark configuration (edit here; no argparse by design)
# ---------------------------------------------------------------------------
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea", "qaoa", "xxz-hva")
QUBITS = tuple(range(4, 29, 2))
LAYERS = 8
RANDOM_SEED = 42
BATCHES = 1
PRECISION = "float64"
DEVICE_NAME = "sad.cuda"
EXECUTION_MODE = "optimized"
OUTPUT_CSV = Path(__file__).resolve().parent / "results" / "sad_optimized_gpu.csv"
OVERWRITE_OUTPUT = True

STEP_SCHEDULE = (
    (8, 20),
    (14, 10),
    (20, 5),
    (24, 3),
    (28, 2),
)

WARMUP_SCHEDULE = (
    (20, 5),
    (28, 1),
)


SAD_PYTHON = Path(__file__).resolve().parents[1] / "sad" / "python"
sys.path.insert(0, str(SAD_PYTHON))
energy_and_grad = importlib.import_module("sad_baseline").energy_and_grad

CSV_FIELDS = (
    "timestamp_utc",
    "status",
    "backend",
    "execution_mode",
    "kernel_variant",
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
    "forward_mean_s",
    "hamiltonian_mean_s",
    "backward_mean_s",
    "forward_times_s_json",
    "hamiltonian_times_s_json",
    "backward_times_s_json",
    "step_times_s_json",
    "gpu_peak_observed_mib",
    "gpu_delta_observed_mib",
    "state_vector_mib",
    "total_workspace_mib",
    "device_total_mib",
    "host_rss_before_mib",
    "host_rss_after_mib",
    "host_peak_rss_mib",
    "error",
)


def steps_for_qubits(qubits: int) -> int:
    for maximum_qubits, steps in STEP_SCHEDULE:
        if qubits <= maximum_qubits:
            return steps
    return STEP_SCHEDULE[-1][1]


def warmups_for_qubits(qubits: int) -> int:
    for maximum_qubits, warmups in WARMUP_SCHEDULE:
        if qubits <= maximum_qubits:
            return warmups
    return WARMUP_SCHEDULE[-1][1]


def _empty_row(
    circuit: str, qubits: int, steps: int, warmup_steps: int
) -> dict[str, object]:
    return {field: "" for field in CSV_FIELDS} | {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "backend": DEVICE_NAME,
        "execution_mode": EXECUTION_MODE,
        "kernel_variant": "",
        "circuit": circuit,
        "qubits": qubits,
        "layers": LAYERS,
        "precision": PRECISION,
        "random_seed": RANDOM_SEED,
        "batches": BATCHES,
        "warmup_steps": warmup_steps,
        "steps": steps,
    }


def _success_row(result: object, steps: int) -> dict[str, object]:
    times = result.step_times_s
    memory = result.memory
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "backend": result.device_name,
        "execution_mode": result.execution_mode,
        "kernel_variant": result.kernel_variant,
        "circuit": result.circuit,
        "qubits": result.qubits,
        "layers": result.layers,
        "parameter_count": result.parameter_count,
        "precision": result.precision,
        "random_seed": result.random_seed,
        "batches": result.batches,
        "warmup_steps": result.warmup_steps,
        "steps": steps,
        "energy": result.energy,
        "grad_json": json.dumps(result.grad.tolist(), separators=(",", ":")),
        "grad_l2_norm": float(np.linalg.norm(result.grad)),
        "grad_max_abs": float(np.max(np.abs(result.grad))),
        "time_mean_s": statistics.fmean(times),
        "time_median_s": statistics.median(times),
        "time_std_s": statistics.pstdev(times),
        "time_min_s": min(times),
        "time_max_s": max(times),
        "forward_mean_s": statistics.fmean(result.forward_times_s),
        "hamiltonian_mean_s": statistics.fmean(result.hamiltonian_times_s),
        "backward_mean_s": statistics.fmean(result.backward_times_s),
        "forward_times_s_json": json.dumps(result.forward_times_s),
        "hamiltonian_times_s_json": json.dumps(result.hamiltonian_times_s),
        "backward_times_s_json": json.dumps(result.backward_times_s),
        "step_times_s_json": json.dumps(times),
        "gpu_peak_observed_mib": memory.gpu_peak_observed_mib,
        "gpu_delta_observed_mib": memory.gpu_delta_observed_mib,
        "state_vector_mib": memory.state_vector_mib,
        "total_workspace_mib": memory.total_workspace_mib,
        "device_total_mib": memory.device_total_mib,
        "host_rss_before_mib": memory.host_rss_before_mib,
        "host_rss_after_mib": memory.host_rss_after_mib,
        "host_peak_rss_mib": memory.host_peak_rss_mib,
        "error": "",
    }


def main() -> None:
    os.environ["SAD_EXECUTION_MODE"] = EXECUTION_MODE
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if OVERWRITE_OUTPUT else "a"
    needs_header = (
        OVERWRITE_OUTPUT or not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    )
    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        if needs_header:
            writer.writeheader()
            stream.flush()

        total = len(CIRCUITS) * len(QUBITS)
        run_index = 0
        for circuit in CIRCUITS:
            for qubits in QUBITS:
                run_index += 1
                steps = steps_for_qubits(qubits)
                warmup_steps = warmups_for_qubits(qubits)
                label = f"[{run_index:02d}/{total:02d}] {circuit} {qubits}q x {LAYERS}l"
                print(
                    f"{label}: {warmup_steps} warmups + {steps} measured steps",
                    flush=True,
                )
                row = _empty_row(circuit, qubits, steps, warmup_steps)
                try:
                    result = energy_and_grad(
                        circuit=circuit,
                        random_seed=RANDOM_SEED,
                        scalability=(qubits, LAYERS),
                        batches=BATCHES,
                        precision=PRECISION,
                        steps=steps,
                        warmup_steps=warmup_steps,
                        device_name=DEVICE_NAME,
                    )
                    row = _success_row(result, steps)
                    print(
                        f"{label}: median={result.median_step_time_s:.6f}s "
                        f"(fwd={statistics.fmean(result.forward_times_s):.6f}s, "
                        f"H={statistics.fmean(result.hamiltonian_times_s):.6f}s, "
                        f"bwd={statistics.fmean(result.backward_times_s):.6f}s), "
                        f"energy={result.energy:.10g}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - preserve long sweep progress.
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
