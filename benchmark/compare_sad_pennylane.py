"""Join SAD and PennyLane CSVs and compare energy/full gradients.

Configuration is intentionally expressed as globals rather than CLI arguments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PENNYLANE_CSV = RESULTS_DIR / "pennylane_lightning_gpu.csv"
SAD_CSV = RESULTS_DIR / "sad_gpu.csv"
OUTPUT_CSV = RESULTS_DIR / "sad_vs_pennylane.csv"

JOIN_FIELDS = ("circuit", "qubits", "layers", "precision", "random_seed", "batches")
FLOAT32_ENERGY_ATOL = 2e-5
FLOAT32_GRAD_ATOL = 2e-5
FLOAT64_ENERGY_ATOL = 1e-10
FLOAT64_GRAD_ATOL = 1e-9

OUTPUT_FIELDS = (
    *JOIN_FIELDS,
    "status",
    "parameter_count",
    "pennylane_energy",
    "sad_energy",
    "energy_abs_error",
    "grad_max_abs_error",
    "grad_l2_error",
    "grad_allclose",
    "pennylane_time_mean_s",
    "sad_time_mean_s",
    "speedup",
    "sad_forward_mean_s",
    "sad_hamiltonian_mean_s",
    "sad_backward_mean_s",
    "error",
)


def _key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in JOIN_FIELDS)


def _read_success_rows(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate benchmark key in {path}: {key}")
        result[key] = row
    return result


def _empty_output(key: tuple[str, ...]) -> dict[str, object]:
    return {field: "" for field in OUTPUT_FIELDS} | dict(
        zip(JOIN_FIELDS, key, strict=True)
    )


def main() -> None:
    pennylane = _read_success_rows(PENNYLANE_CSV)
    sad = _read_success_rows(SAD_CSV)
    keys = sorted(set(pennylane) | set(sad), key=lambda key: (key[0], int(key[1])))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for key in keys:
            output = _empty_output(key)
            pl_row = pennylane.get(key)
            sad_row = sad.get(key)
            if pl_row is None or sad_row is None:
                output["status"] = "missing"
                output["error"] = (
                    "missing PennyLane row" if pl_row is None else "missing SAD row"
                )
                failures += 1
                writer.writerow(output)
                continue
            if not pl_row.get("grad_json") or not sad_row.get("grad_json"):
                output["status"] = "missing-gradient"
                output["error"] = (
                    "grad_json is absent; rerun both updated benchmark scripts before comparing"
                )
                failures += 1
                writer.writerow(output)
                continue

            pl_grad = np.asarray(json.loads(pl_row["grad_json"]), dtype=np.float64)
            sad_grad = np.asarray(json.loads(sad_row["grad_json"]), dtype=np.float64)
            if pl_grad.shape != sad_grad.shape:
                output["status"] = "shape-mismatch"
                output["error"] = f"PennyLane {pl_grad.shape} vs SAD {sad_grad.shape}"
                failures += 1
                writer.writerow(output)
                continue

            precision = key[3]
            if precision == "float32":
                energy_atol = FLOAT32_ENERGY_ATOL
                grad_atol = FLOAT32_GRAD_ATOL
            else:
                energy_atol = FLOAT64_ENERGY_ATOL
                grad_atol = FLOAT64_GRAD_ATOL
            pl_energy = float(pl_row["energy"])
            sad_energy = float(sad_row["energy"])
            energy_error = abs(pl_energy - sad_energy)
            grad_difference = sad_grad - pl_grad
            grad_max_error = float(np.max(np.abs(grad_difference)))
            close = energy_error <= energy_atol and grad_max_error <= grad_atol
            if not close:
                failures += 1

            pl_time = float(pl_row["time_mean_s"])
            sad_time = float(sad_row["time_mean_s"])
            output |= {
                "status": "match" if close else "mismatch",
                "parameter_count": sad_row["parameter_count"],
                "pennylane_energy": pl_energy,
                "sad_energy": sad_energy,
                "energy_abs_error": energy_error,
                "grad_max_abs_error": grad_max_error,
                "grad_l2_error": float(np.linalg.norm(grad_difference)),
                "grad_allclose": close,
                "pennylane_time_mean_s": pl_time,
                "sad_time_mean_s": sad_time,
                "speedup": pl_time / sad_time,
                "sad_forward_mean_s": sad_row["forward_mean_s"],
                "sad_hamiltonian_mean_s": sad_row["hamiltonian_mean_s"],
                "sad_backward_mean_s": sad_row["backward_mean_s"],
                "error": "",
            }
            writer.writerow(output)

    print(
        f"Compared {len(keys)} configurations: {len(keys) - failures} matched, {failures} failed"
    )
    print(f"CSV written to {OUTPUT_CSV}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
