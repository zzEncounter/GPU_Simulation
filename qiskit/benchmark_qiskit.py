"""Batch benchmark for the Qiskit AerSimulator (GPU) parameter-shift baseline.

Mirrors the structure of benchmark/benchmark_pennylane_lightning.py so that
results can be compared column-by-column.  Edit the constants below; this
script intentionally has no CLI arguments.

Key difference vs PennyLane:
  Each gradient step requires  2 * n_circuit_params  expectation-value
  evaluations (parameter shift rule), whereas PennyLane's adjoint method
  needs only O(1) circuit executions regardless of parameter count.
"""

from __future__ import annotations

import csv
import gc
import json
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration (edit here; no argparse)
# ---------------------------------------------------------------------------
CIRCUITS = ("ra-hea", "su2-hea", "rzz-hea")
QUBITS = tuple(range(4, 29, 2))
LAYERS = 8
RANDOM_SEED = 42
BATCHES = 1
PRECISION = "float64"
WARMUP_STEPS = 1
GPU = True
OUTPUT_CSV = Path(__file__).resolve().parent / "results" / "qiskit_aer_gpu.csv"
OVERWRITE_OUTPUT = True

# Fewer steps per size than PennyLane because each step is O(N) more expensive.
STEP_SCHEDULE = (
    (8,  10),
    (14,  5),
    (20,  3),
    (24,  2),
    (28,  1),
)

# ---------------------------------------------------------------------------
# Import the baseline module
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))

from qiskit_baseline import energy_and_grad  # noqa: E402

# ---------------------------------------------------------------------------
# CSV schema (matches pennylane_lightning_gpu.csv for easy comparison)
# ---------------------------------------------------------------------------
CSV_FIELDS = (
    "timestamp_utc",
    "status",
    "backend",
    "circuit",
    "qubits",
    "layers",
    "parameter_count",
    "n_circuit_params",   # >= parameter_count for shared-param circuits
    "precision",
    "random_seed",
    "batches",
    "warmup_steps",
    "steps",
    "evaluations_per_step",   # = 2 * n_circuit_params
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
    "gpu_before_device_mib",
    "gpu_after_warmup_mib",
    "gpu_peak_observed_mib",
    "gpu_delta_observed_mib",
    "host_rss_before_mib",
    "host_rss_after_mib",
    "host_peak_rss_mib",
    "error",
)


def _steps_for_qubits(qubits: int) -> int:
    for max_q, s in STEP_SCHEDULE:
        if qubits <= max_q:
            return s
    return STEP_SCHEDULE[-1][1]


def _empty_row(circuit: str, qubits: int, steps: int) -> dict[str, object]:
    return {f: "" for f in CSV_FIELDS} | {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "backend": "aer_simulator_gpu" if GPU else "aer_simulator",
        "circuit": circuit,
        "qubits": qubits,
        "layers": LAYERS,
        "precision": PRECISION,
        "random_seed": RANDOM_SEED,
        "batches": BATCHES,
        "warmup_steps": WARMUP_STEPS,
        "steps": steps,
    }


def _success_row(result, steps: int, n_circuit_params: int) -> dict[str, object]:
    times = result.step_times_s
    mem = result.memory
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "backend": "aer_simulator_gpu" if GPU else "aer_simulator",
        "circuit": result.circuit,
        "qubits": result.qubits,
        "layers": result.layers,
        "parameter_count": result.parameter_count,
        "n_circuit_params": n_circuit_params,
        "precision": result.precision,
        "random_seed": result.random_seed,
        "batches": result.batches,
        "warmup_steps": result.warmup_steps,
        "steps": steps,
        "evaluations_per_step": 2 * n_circuit_params,
        "energy": result.energy,
        "grad_json": json.dumps(result.grad.tolist(), separators=(",", ":")),
        "grad_l2_norm": float(np.linalg.norm(result.grad)),
        "grad_max_abs": float(np.max(np.abs(result.grad))),
        "time_mean_s": statistics.fmean(times),
        "time_median_s": statistics.median(times),
        "time_std_s": statistics.pstdev(times),
        "time_min_s": min(times),
        "time_max_s": max(times),
        "step_times_s_json": json.dumps(list(times)),
        "gpu_before_device_mib": mem.gpu_before_device_mib,
        "gpu_after_warmup_mib": mem.gpu_after_warmup_mib,
        "gpu_peak_observed_mib": mem.gpu_peak_observed_mib,
        "gpu_delta_observed_mib": mem.gpu_delta_observed_mib,
        "host_rss_before_mib": mem.host_rss_before_mib,
        "host_rss_after_mib": mem.host_rss_after_mib,
        "host_peak_rss_mib": mem.host_peak_rss_mib,
        "error": "",
    }


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if OVERWRITE_OUTPUT else "a"
    needs_header = (
        OVERWRITE_OUTPUT or not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    )

    # Pre-compute circuit param counts (needed for n_circuit_params column)
    from qiskit_baseline.circuits import build_circuit

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
                steps = _steps_for_qubits(qubits)
                label = f"[{run_index:02d}/{total:02d}] {circuit} {qubits}q x {LAYERS}l"
                print(f"{label}: {steps} measured steps", flush=True)
                row = _empty_row(circuit, qubits, steps)
                try:
                    bundle = build_circuit(circuit, qubits, LAYERS)
                    n_circuit_params = len(bundle.param_list)

                    result = energy_and_grad(
                        circuit=circuit,
                        random_seed=RANDOM_SEED,
                        scalability=(qubits, LAYERS),
                        batches=BATCHES,
                        precision=PRECISION,
                        steps=steps,
                        warmup_steps=WARMUP_STEPS,
                        gpu=GPU,
                    )
                    row = _success_row(result, steps, n_circuit_params)
                    print(
                        f"{label}: median={result.median_step_time_s:.6f}s, "
                        f"energy={result.energy:.10g}, "
                        f"n_evals_per_step={2 * n_circuit_params}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"{label}: ERROR: {row['error']}", file=sys.stderr, flush=True)
                    traceback.print_exc()
                finally:
                    writer.writerow(row)
                    stream.flush()
                    gc.collect()

    print(f"CSV written to {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
